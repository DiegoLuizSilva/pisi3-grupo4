"""Camada de dados do painel ChurnGuard.

Todas as páginas importam os dados daqui, e de nenhum outro lugar:

    from dashboard.dados import carteira, tabela, metadados

O painel só LÊ resultados já gerados pelo projeto (data/raw, reports/tabelas
e models). Nada aqui treina, reajusta ou recalcula modelo: o escore de cada
cliente vem de reports/tabelas/predicoes.csv, produzido pelo modelo final
(o mesmo arquivo models/modelo_final.joblib usado pela API do aplicativo).

Precisa de um campo que não está aqui? Peça ao Lucas, dono deste arquivo.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
DADOS_BRUTOS = RAIZ / "data" / "raw" / "iranian_churn.csv"
TABELAS = RAIZ / "reports" / "tabelas"
FIGURAS = RAIZ / "reports" / "figuras"
MODELOS = RAIZ / "models"

TOTAL_CLIENTES = 3150

# Experimento escolhido na Seção 7.2 do artigo (maior F1 de churn na validação cruzada).
EXPERIMENTO_FINAL = "Random Forest | SMOTENC"

# Faixas de risco: mesmos cortes da API (dsi-grupo4/api/faixas.py) e do artigo.
CORTE_BAIXO = 0.30
CORTE_ALTO = 0.65
ORDEM_FAIXAS = ["Risco baixo", "Risco médio", "Risco alto"]
CORES_FAIXA = {"Risco baixo": "#1BAF7A", "Risco médio": "#E8955A", "Risco alto": "#C2551F"}

# Atributos que o modelo recebe, na ordem do treino (age_group fica de fora).
ATRIBUTOS_MODELO = [
    "call_failure", "complains", "subscription_length", "charge_amount",
    "seconds_of_use", "frequency_of_use", "frequency_of_sms",
    "distinct_called_numbers", "tariff_plan", "status", "age", "customer_value",
]

# Rótulo em português de cada coluna, para eixos, legendas e tabelas.
ROTULOS = {
    "call_failure": "Falhas de chamada",
    "complains": "Reclamação registrada",
    "subscription_length": "Tempo de assinatura (meses)",
    "charge_amount": "Faixa de cobrança (0 a 10)",
    "seconds_of_use": "Segundos de uso",
    "frequency_of_use": "Frequência de uso (chamadas)",
    "frequency_of_sms": "Frequência de SMS",
    "distinct_called_numbers": "Números distintos chamados",
    "age_group": "Faixa etária",
    "tariff_plan": "Plano",
    "status": "Situação da linha",
    "age": "Idade",
    "customer_value": "Valor do cliente",
    "churn": "Cancelou",
    "conjunto": "Conjunto",
    "perfil": "Perfil",
    "escore": "Escore de risco",
    "faixa": "Faixa de risco",
}

# Códigos do domínio original da base (UCI), antes da recodificação do treino.
PLANO = {1: "Pré-pago", 2: "Contrato"}
SITUACAO = {1: "Ativa", 2: "Inativa"}
SIM_NAO = {0: "Não", 1: "Sim"}
DESFECHO = {0: "Permaneceu", 1: "Cancelou"}


def faixa_de(escore: float) -> str:
    """Mesma regra da API: abaixo de 0,30 baixo; até 0,65 médio; acima, alto."""
    if escore < CORTE_BAIXO:
        return "Risco baixo"
    if escore <= CORTE_ALTO:
        return "Risco médio"
    return "Risco alto"


def _snake_case(colunas: pd.Index) -> pd.Index:
    # Mesma normalização de src/common.py ("Call  Failure" -> "call_failure").
    return colunas.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)


@st.cache_data(show_spinner=False)
def base_bruta() -> pd.DataFrame:
    """As 3.150 linhas do CSV original, com nomes em snake_case e row_id."""
    df = pd.read_csv(DADOS_BRUTOS)
    df.columns = _snake_case(df.columns)
    df.insert(0, "row_id", range(len(df)))
    return df


@st.cache_data(show_spinner="Carregando a carteira de clientes...")
def carteira() -> pd.DataFrame:
    """Uma linha por cliente, pronta para filtros, indicadores e gráficos.

    Colunas:
      row_id                      posição do cliente no CSV original
      13 atributos + churn        valores do domínio original (plano e situação em 1/2)
      plano, situacao, reclamacao, faixa_etaria, desfecho   rótulos em português
      conjunto                    "treino" ou "teste" (divisão da Seção 4.3)
      perfil_id, perfil           grupo do K-Means e seu nome (Seção 7.1)
      escore                      probabilidade de churn do modelo final
      previsto                    classe prevista pelo modelo (0/1, limiar 0,5)
      faixa                       Risco baixo / médio / alto

    Atenção: para clientes do conjunto de treino o escore é otimista, porque o
    modelo já viu esses clientes. Indicadores de desempenho devem usar o teste.
    """
    base = base_bruta()

    clusters = pd.read_csv(TABELAS / "clientes_clusters.csv",
                           usecols=["row_id", "cluster", "churn", "conjunto"])
    nomes = pd.read_csv(TABELAS / "perfil_tamanho.csv", usecols=["grupo", "perfil"])
    predicoes = pd.read_csv(TABELAS / "predicoes.csv")
    predicoes = predicoes.loc[predicoes["experimento"] == EXPERIMENTO_FINAL,
                              ["row_id", "real", "score", "previsto"]]

    df = (base
          .merge(clusters.rename(columns={"cluster": "perfil_id", "churn": "_churn_cluster"}),
                 on="row_id", how="left", validate="one_to_one")
          .merge(nomes.rename(columns={"grupo": "perfil_id"}),
                 on="perfil_id", how="left", validate="many_to_one")
          .merge(predicoes.rename(columns={"score": "escore"}),
                 on="row_id", how="left", validate="one_to_one"))

    # Os três arquivos precisam descrever os mesmos clientes, na mesma ordem.
    if len(df) != TOTAL_CLIENTES:
        raise ValueError(f"carteira com {len(df)} linhas; esperado {TOTAL_CLIENTES}")
    faltando = df[["perfil_id", "perfil", "conjunto", "escore"]].isna().sum()
    if faltando.any():
        raise ValueError(f"clientes sem perfil, conjunto ou escore: {faltando[faltando > 0].to_dict()}")
    if not ((df["churn"] == df["_churn_cluster"]) & (df["churn"] == df["real"])).all():
        raise ValueError("o churn do CSV original não bate com clientes_clusters.csv ou predicoes.csv")

    idade_por_faixa = base.groupby("age_group")["age"].agg(["min", "max"])
    rotulo_idade = {g: (f"{a} anos" if a == b else f"{a} a {b} anos")
                    for g, (a, b) in idade_por_faixa.iterrows()}

    df["plano"] = df["tariff_plan"].map(PLANO)
    df["situacao"] = df["status"].map(SITUACAO)
    df["reclamacao"] = df["complains"].map(SIM_NAO)
    df["faixa_etaria"] = pd.Categorical(df["age_group"].map(rotulo_idade),
                                        categories=[rotulo_idade[g] for g in sorted(rotulo_idade)],
                                        ordered=True)
    df["desfecho"] = df["churn"].map(DESFECHO)
    df["perfil_id"] = df["perfil_id"].astype(int)
    df["previsto"] = df["previsto"].astype(int)
    df["faixa"] = pd.Categorical(df["escore"].map(faixa_de), categories=ORDEM_FAIXAS, ordered=True)

    return df.drop(columns=["_churn_cluster", "real"])


def tabelas_disponiveis() -> list[str]:
    """Nomes aceitos por tabela() e metadados()."""
    return sorted(p.stem for p in TABELAS.glob("*.csv")) + sorted(p.stem for p in TABELAS.glob("*.json"))


@st.cache_data(show_spinner=False)
def tabela(nome: str) -> pd.DataFrame:
    """Lê reports/tabelas/<nome>.csv. Ex.: tabela("metricas_treino_teste")."""
    caminho = TABELAS / f"{nome}.csv"
    if not caminho.exists():
        raise FileNotFoundError(f"reports/tabelas/{nome}.csv não existe. Disponíveis: {tabelas_disponiveis()}")
    return pd.read_csv(caminho)


@st.cache_data(show_spinner=False)
def metadados(nome: str) -> dict:
    """Lê reports/tabelas/<nome>.json. Ex.: metadados("model_metadata")."""
    caminho = TABELAS / f"{nome}.json"
    if not caminho.exists():
        raise FileNotFoundError(f"reports/tabelas/{nome}.json não existe. Disponíveis: {tabelas_disponiveis()}")
    return json.loads(caminho.read_text(encoding="utf-8"))


def figura(nome: str) -> Path:
    """Caminho de reports/figuras/<nome>.png, para st.image()."""
    return FIGURAS / f"{nome}.png"


@st.cache_resource(show_spinner="Carregando o modelo final...")
def modelo():
    """Pipeline final (models/modelo_final.joblib), só para predizer.

    Usado pelo simulador da página de Explicabilidade. Exige scikit-learn e
    imbalanced-learn nas versões do treino; o import fica aqui dentro para as
    demais páginas abrirem mesmo sem essas bibliotecas.

    Antes de chamar predict_proba, passe os dados por preparar_para_modelo().
    """
    import joblib

    return joblib.load(MODELOS / "modelo_final.joblib")


def preparar_para_modelo(clientes: pd.DataFrame) -> pd.DataFrame:
    """Converte linhas no domínio original (como em carteira()) para a entrada do modelo.

    Aplica o que o treino fez (Seção 4.3), lendo de model_metadata.json:
    descarta age_group, recodifica tariff_plan e status (1 -> 0, 2 -> 1) e
    põe os 12 atributos na ordem do treino. Esquecer a recodificação muda a
    faixa de 738 dos 3.150 clientes (Seção 4.6), sem nenhuma mensagem de erro.
    """
    meta = metadados("model_metadata")
    entrada = clientes[meta["features"]].copy()
    for coluna, mapa in meta["mapeamentos"].items():
        entrada[coluna] = entrada[coluna].map({int(k): v for k, v in mapa.items()})
    if entrada.isna().any().any():
        raise ValueError("valor fora do domínio original em tariff_plan ou status (use 1 ou 2)")
    return entrada