"""Confere o pipeline supervisionado e gera casos verificáveis para a integração.

Execute da raiz: python -m src.validar_inferencia
Opcional: python -m src.validar_inferencia --dsi-root ../dsi-grupo4
Com API: CHURNGUARD_ID_TOKEN=<token> python -m src.validar_inferencia --api-url http://localhost:8000

Esta rotina mede somente o processo Python local. O limite de três segundos do
aplicativo exige outra medição no dispositivo e na rede da demonstração.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from urllib.request import Request, urlopen

import joblib
import numpy as np
import pandas as pd
import shap

from src.common import ROOT, fingerprint, load_data


TABLES = ROOT / "reports" / "tabelas"
MODEL_PATH = ROOT / "models" / "modelo_final.joblib"
EXAMPLES_PATH = TABLES / "casos_referencia_pisi.json"
EPSILON = 1e-6


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def positive_values(explainer: shap.TreeExplainer, transformed: np.ndarray, index: int) -> tuple[float, np.ndarray]:
    """Lê a classe churn nos formatos de retorno do SHAP 0.52 e anteriores."""

    values = explainer.shap_values(transformed)
    if isinstance(values, list):
        vector = np.asarray(values[index])[0]
    else:
        array = np.asarray(values)
        vector = array[0, :, index] if array.ndim == 3 else array[0]

    expected = np.asarray(explainer.expected_value)
    base = float(expected[index] if expected.ndim else expected)
    return base, np.asarray(vector, dtype=float)


def source_payload(row: pd.Series) -> dict:
    """Mantém as 13 entradas no domínio original para a API."""

    payload = {}
    for name, value in row.items():
        if name == "churn":
            continue
        payload[name] = value.item() if hasattr(value, "item") else value
    return payload


def consultar_api(api_url: str, payload: dict) -> dict:
    token = os.getenv("CHURNGUARD_ID_TOKEN")
    if not token:
        raise ValueError("Defina CHURNGUARD_ID_TOKEN para testar a API autenticada.")
    request = Request(
        api_url.rstrip("/") + "/predict",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    with urlopen(request, timeout=15) as response:
        return json.load(response)


def run(dsi_root: Path | None = None, api_url: str | None = None) -> dict:
    metadata = json.loads((TABLES / "model_metadata.json").read_text(encoding="utf-8"))
    if metadata["sha256"] != fingerprint():
        raise AssertionError("O CSV não corresponde ao usado no treinamento.")

    model_sha = digest(MODEL_PATH)
    if metadata.get("modelo_sha256") != model_sha:
        raise AssertionError("O modelo exportado não corresponde ao metadado.")

    load_start = perf_counter()
    model = joblib.load(MODEL_PATH)
    load_seconds = perf_counter() - load_start
    df, features, target, groups, train, test = load_data()

    if list(features.columns) != metadata["features"] or "age_group" in features:
        raise AssertionError("A ordem dos 12 atributos ou a exclusão de age_group mudou.")
    classes = list(model.classes_)
    positive_index = classes.index(metadata["classe_churn"])
    if classes != [0, 1] or positive_index != 1:
        raise AssertionError("A classe positiva do modelo não corresponde a churn=1.")

    published = pd.read_csv(TABLES / "predicoes.csv")
    published = published.loc[
        (published.experimento == metadata["vencedor"])
        & (published.conjunto == "teste")
    ]
    if len(published) != len(test) or set(published.row_id) != set(test):
        raise AssertionError("As predições publicadas não cobrem o holdout inteiro.")
    selected = features.loc[published.row_id.to_numpy(), metadata["features"]]
    scores = model.predict_proba(selected)[:, positive_index]
    max_score_error = float(np.max(np.abs(scores - published.score.to_numpy())))
    if max_score_error >= 1e-9:
        raise AssertionError(f"Predições publicadas divergentes: {max_score_error:.3g}.")
    if not np.array_equal(target.loc[published.row_id.to_numpy()], published.real):
        raise AssertionError("Os rótulos reais não correspondem aos row_id publicados.")

    # O estimador recebe a ordem de saída do ColumnTransformer, que difere da
    # ordem de entrada. Atribuir SHAP na ordem errada inverteria alguns fatores.
    preprocessor = model.named_steps["prep"]
    transformed_names = list(preprocessor.get_feature_names_out())
    if len(transformed_names) != model.named_steps["model"].n_features_in_:
        raise AssertionError("A quantidade de nomes transformados está incorreta.")
    explainer = shap.TreeExplainer(model.named_steps["model"])

    # Casos estáveis: diferentes faixas, reclamação, status e cobrança máxima.
    cases = [110, 516, 399, 1624, 72]
    max_charge_rows = df.index[df.charge_amount == 10].tolist()
    if len(max_charge_rows) != 7:
        raise AssertionError("A base deveria conter sete registros com charge_amount=10.")
    cases.append(max_charge_rows[0])
    cases = list(dict.fromkeys(cases))

    exported = []
    elapsed = []
    http_elapsed = []
    for row_id in cases:
        started = perf_counter()
        row = features.loc[[row_id], metadata["features"]]
        probability = float(model.predict_proba(row)[0, positive_index])
        transformed = preprocessor.transform(row)
        base, values = positive_values(explainer, transformed, positive_index)
        if len(values) != len(transformed_names):
            raise AssertionError("O vetor SHAP não corresponde aos nomes dos atributos.")
        if not np.isclose(base + values.sum(), probability, atol=1e-8):
            raise AssertionError(f"SHAP não reconstrói a saída do caso {row_id}.")

        order = sorted(range(len(values)), key=lambda i: (-abs(values[i]), i))
        relevant = [i for i in order if abs(values[i]) >= EPSILON][:3]
        maximum = abs(values[relevant[0]]) if relevant else 0.0
        factors = [
            {
                "campo": transformed_names[i],
                "impacto": "aumenta" if values[i] > 0 else "reduz",
                "peso": round(float(abs(values[i]) / maximum), 4),
                "valor_shap": float(values[i]),
            }
            for i in relevant
        ]
        if any(factors[i]["peso"] < factors[i + 1]["peso"] for i in range(len(factors) - 1)):
            raise AssertionError("Os fatores não estão ordenados por contribuição.")

        exported.append(
            {
                "row_id": int(row_id),
                "entrada_api": source_payload(df.loc[row_id]),
                "classe_real": int(target.loc[row_id]),
                "probabilidade_churn": probability,
                "faixa": "baixo" if probability < .30 else "medio" if probability <= .65 else "alto",
                "fatores": factors,
            }
        )
        elapsed.append(perf_counter() - started)

        if api_url:
            http_started = perf_counter()
            response = consultar_api(api_url, exported[-1]["entrada_api"])
            http_elapsed.append(perf_counter() - http_started)
            if abs(response["probabilidade"] - round(probability, 4)) > 1e-4:
                raise AssertionError(f"A API divergiu do modelo no caso {row_id}.")
            version = response["modelo_versao"]
            if response["faixa"] != exported[-1]["faixa"] or not version or version.startswith("stub"):
                raise AssertionError(f"A faixa ou a versão da API divergiu no caso {row_id}.")
            received = response["fatores"]
            if len(received) != len(factors):
                raise AssertionError(f"A quantidade de fatores da API divergiu no caso {row_id}.")
            for actual, expected in zip(received, factors):
                if actual["campo"] != expected["campo"] or actual["impacto"] != expected["impacto"]:
                    raise AssertionError(f"A ordem ou o sinal dos fatores divergiu no caso {row_id}.")
                if abs(actual["peso"] - expected["peso"]) > 1e-4:
                    raise AssertionError(f"O peso de um fator divergiu no caso {row_id}.")
                if not actual["rotulo"] or not actual["sugestao"]:
                    raise AssertionError(f"Faltam rótulo ou sugestão no caso {row_id}.")

    if dsi_root is not None:
        dsi_model = dsi_root / "ml" / "artefatos" / "modelo.joblib"
        if not dsi_model.is_file():
            raise FileNotFoundError(f"Modelo integrado não encontrado: {dsi_model}")
        if digest(dsi_model) != model_sha:
            raise AssertionError("O modelo de DSI está desatualizado; Lucas precisa reimportar o pacote.")
        dsi_meta = json.loads((dsi_root / "ml" / "artefatos" / "metadados.json").read_text(encoding="utf-8"))
        if dsi_meta["atributos"] != metadata["features"] or dsi_meta["classe_churn"] != 1:
            raise AssertionError("O contrato da API diverge em atributos ou classe positiva.")
        if dsi_meta["atributos_ignorados"] != ["age_group"]:
            raise AssertionError("A API não declara a exclusão de age_group.")
        if dsi_meta["explicacao"]["atributos_transformados"] != transformed_names:
            raise AssertionError("A ordem dos fatores na API diverge da ordem do estimador.")

    reference = {
        "modelo_sha256": model_sha,
        "dataset_sha256": metadata["sha256"],
        "vencedor": metadata["vencedor"],
        "classe_churn": 1,
        "holdout_conferido": len(published),
        "erro_maximo_scores": max_score_error,
        "casos": exported,
    }
    EXAMPLES_PATH.write_text(json.dumps(reference, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        **reference,
        "tempo_carregamento_s": load_seconds,
        "tempo_primeiro_caso_s": elapsed[0],
        "tempo_maximo_caso_local_s": max(elapsed),
        "tempo_maximo_http_s": max(http_elapsed) if http_elapsed else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsi-root", type=Path, help="raiz do repositório DSI para comparação do pacote importado")
    parser.add_argument("--api-url", help="endereço da API autenticada para comparar os seis casos")
    args = parser.parse_args()
    result = run(args.dsi_root.resolve() if args.dsi_root else None, args.api_url)
    print(f"Modelo: {result['vencedor']} ({result['modelo_sha256'][:12]})")
    print(f"Holdout: {result['holdout_conferido']} scores; erro máximo {result['erro_maximo_scores']:.2g}")
    print(f"Casos exportados: {len(result['casos'])} em {EXAMPLES_PATH.relative_to(ROOT)}")
    print(f"Carregamento local: {result['tempo_carregamento_s']:.3f}s; "
          f"maior caso local com SHAP: {result['tempo_maximo_caso_local_s']:.3f}s")
    if result["tempo_maximo_http_s"] is not None:
        print(f"Maior chamada HTTP: {result['tempo_maximo_http_s']:.3f}s; "
              "o tempo no aplicativo ainda exige medição no dispositivo e na rede final.")


if __name__ == "__main__":
    main()
