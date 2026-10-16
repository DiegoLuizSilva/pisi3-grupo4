"""Página "Explicabilidade": SHAP global, por cliente e simulador.

Dono: Diego Luiz. Dados e modelo vêm de dashboard/dados.py; os fatores usam o
mesmo método de src/validar_inferencia.py (e da API).
Rodar pelo painel, a partir da raiz: streamlit run dashboard/app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st

from dashboard.dados import (EXPERIMENTO_FINAL, ROTULOS, base_bruta, faixa_de, metadados, modelo,
                             preparar_para_modelo, tabela)
from src.validar_inferencia import EPSILON, positive_values


@st.cache_resource(show_spinner=False)
def carregar_explicador():
    pipeline = modelo()
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    transformed_names = list(pipeline.named_steps["prep"].get_feature_names_out())
    return pipeline, explainer, transformed_names


def nome(campo: str) -> str:
    return ROTULOS.get(campo, campo)


def idades_por_faixa() -> dict[int, int]:
    """No conjunto original, cada faixa etária corresponde a uma única idade."""
    pares = base_bruta()[["age", "age_group"]].drop_duplicates().sort_values("age")
    return {int(i): int(g) for i, g in zip(pares.age, pares.age_group)}


def explicar(entrada: dict, meta: dict) -> dict:
    """Escore, faixa e três fatores com o mesmo método de src/validar_inferencia.py."""
    model, explainer, transformed_names = carregar_explicador()
    positive_index = list(model.classes_).index(meta["classe_churn"])
    row = preparar_para_modelo(pd.DataFrame([entrada])).astype(float)
    probability = float(model.predict_proba(row)[0, positive_index])
    transformed = model.named_steps["prep"].transform(row)
    base, values = positive_values(explainer, transformed, positive_index)

    order = sorted(range(len(values)), key=lambda i: (-abs(values[i]), i))
    relevant = [i for i in order if abs(values[i]) >= EPSILON][:3]
    maximum = abs(values[relevant[0]]) if relevant else 0.0
    fatores = [
        {"campo": transformed_names[i], "impacto": "aumenta" if values[i] > 0 else "reduz",
         "peso": round(float(abs(values[i]) / maximum), 4), "valor_shap": float(values[i])}
        for i in relevant
    ]
    return {"probabilidade": probability, "faixa": faixa_de(probability), "base": base,
            "valores": dict(zip(transformed_names, values)), "fatores": fatores}


def barras_contribuicao(contribuicoes: pd.Series, titulo: str) -> go.Figure:
    ordem = contribuicoes.reindex(contribuicoes.abs().sort_values().index)
    fig = go.Figure(go.Bar(
        x=ordem.values, y=[nome(c) for c in ordem.index], orientation="h",
        marker_color=np.where(ordem.values > 0, "#E45756", "#4C78A8"),
        hovertemplate="%{y}: %{x:+.4f}<extra></extra>"))
    fig.add_vline(x=0, line_color="gray")
    fig.update_layout(title=titulo, xaxis_title="Contribuição SHAP para o escore de churn",
                      height=420, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def main() -> None:
    st.title("Explicabilidade (SHAP)")

    meta = metadados("model_metadata")
    shap_meta = metadados("shap_metadata")
    importancia = tabela("shap_importancia")
    valores = tabela("shap_valores")
    amostra = tabela("shap_amostra")

    st.info(
        f"Explicações do modelo final (**{EXPERIMENTO_FINAL}**). Cada valor SHAP indica quanto um "
        "atributo empurra o escore de churn para cima (vermelho) ou para baixo (azul) em relação à "
        "média do modelo. O escore é uma probabilidade não calibrada e as contribuições estão na "
        "mesma unidade. SHAP descreve o que o modelo aprendeu, não relações de causa."
    )

    # Importância global
    st.header("Importância global")
    imp = importancia.sort_values("shap_absoluto_medio")
    fig = px.bar(imp, x="shap_absoluto_medio", y=imp.variavel.map(nome), orientation="h",
                 labels={"shap_absoluto_medio": "Média de |SHAP|", "y": ""}, color_discrete_sequence=["#4C78A8"])
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Média do valor absoluto em {shap_meta['n_explicados']} clientes explicados, "
               f"com {shap_meta['n_background']} clientes de referência.")

    # Distribuição das contribuições
    st.header("Distribuição das contribuições")
    longo = valores.melt(id_vars="row_id", var_name="variavel", value_name="shap").merge(
        amostra.melt(id_vars="row_id", var_name="variavel", value_name="valor"), on=["row_id", "variavel"])
    faixas = longo.groupby("variavel").valor.agg(["min", "max"])
    amplitude = (faixas["max"] - faixas["min"]).replace(0, 1)
    longo["valor_relativo"] = (longo.valor - longo.variavel.map(faixas["min"])) / longo.variavel.map(amplitude)
    ordem = importancia.sort_values("shap_absoluto_medio").variavel.tolist()
    posicao = {v: i for i, v in enumerate(ordem)}
    gerador = np.random.default_rng(0)
    longo["y"] = longo.variavel.map(posicao) + gerador.uniform(-0.3, 0.3, len(longo))
    fig = go.Figure(go.Scatter(
        x=longo.shap, y=longo.y, mode="markers",
        marker=dict(size=6, color=longo.valor_relativo, colorscale="RdBu_r", cmin=0, cmax=1, opacity=0.8,
                    colorbar=dict(title="Valor do<br>atributo", tickvals=[0, 1], ticktext=["baixo", "alto"])),
        customdata=np.stack([longo.variavel.map(nome), longo.valor, longo.row_id], axis=1),
        hovertemplate="%{customdata[0]} = %{customdata[1]}<br>SHAP %{x:+.4f}<br>cliente %{customdata[2]}<extra></extra>"))
    fig.add_vline(x=0, line_color="gray")
    fig.update_layout(height=520, xaxis_title="Contribuição SHAP", margin=dict(l=10, r=10, t=10, b=10),
                      yaxis=dict(tickvals=list(range(len(ordem))), ticktext=[nome(v) for v in ordem]))
    st.plotly_chart(fig, width="stretch")
    st.caption("Cada ponto é um cliente da amostra. A cor indica se o valor do atributo é baixo ou alto "
               "dentro da amostra; plano tarifário e status estão codificados como 0 (pré-pago, ativa) "
               "e 1 (contrato, inativa).")

    # Cliente da amostra
    st.header("Explicação de um cliente da amostra")
    model, _, _ = carregar_explicador()
    cliente = st.selectbox("Cliente (row_id)", amostra.row_id.tolist(), key="cliente")
    linha = amostra.loc[amostra.row_id == cliente, meta["features"]]
    score = float(model.predict_proba(linha.astype(float))[0, list(model.classes_).index(meta["classe_churn"])])
    contrib = valores.loc[valores.row_id == cliente, meta["features"]].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Escore de churn", f"{score:.3f}".replace(".", ","))
    c2.metric("Faixa", faixa_de(score))
    c3.metric("Média do modelo (base)", f"{score - contrib.sum():.3f}".replace(".", ","))
    col_g, col_t = st.columns([3, 2])
    with col_g:
        st.plotly_chart(barras_contribuicao(contrib, f"Contribuições do cliente {cliente}"), width="stretch")
    with col_t:
        st.dataframe(pd.DataFrame({"Atributo": [nome(c) for c in meta["features"]],
                                   "Valor no modelo": linha.iloc[0].values,
                                   "SHAP": contrib.values}).sort_values("SHAP", key=abs, ascending=False),
                     hide_index=True, width="stretch",
                     column_config={"SHAP": st.column_config.NumberColumn(format="%+.4f")})
    st.caption("Base mais a soma das contribuições reconstrói o escore do cliente.")

    # Simulador
    st.header("Simulador")
    st.caption("Informe os 13 campos no formato original da base. A entrada passa pela mesma "
               "preparação do pipeline: age_group é descartado e plano e status são recodificados.")
    referencias = metadados("casos_referencia_pisi")["casos"]
    idade_para_faixa = idades_por_faixa()
    opcoes = ["Valores padrão"] + [f"Caso {c['row_id']} (real: {'churn' if c['classe_real'] else 'fica'})"
                                   for c in referencias]
    escolha = st.selectbox("Preencher com", opcoes, key="preencher")
    base_entrada = {"call_failure": 5, "complains": 0, "subscription_length": 32, "charge_amount": 1,
                    "seconds_of_use": 4500, "frequency_of_use": 70, "frequency_of_sms": 70,
                    "distinct_called_numbers": 23, "tariff_plan": 1, "status": 1, "age": 30,
                    "customer_value": 470.0}
    if escolha != opcoes[0]:
        base_entrada = referencias[opcoes.index(escolha) - 1]["entrada_api"]
    sufixo = f"_{opcoes.index(escolha)}"

    with st.form("simulador"):
        a, b, c = st.columns(3)
        entrada = {
            "call_failure": a.number_input("Falhas de chamada", 0, 100, int(base_entrada["call_failure"]), key="call_failure" + sufixo),
            "complains": b.selectbox("Reclamação", [0, 1], index=int(base_entrada["complains"]),
                                     format_func={0: "0 = não", 1: "1 = sim"}.get, key="complains" + sufixo),
            "subscription_length": c.number_input("Meses de assinatura", 0, 120, int(base_entrada["subscription_length"]), key="subscription_length" + sufixo),
            "charge_amount": a.number_input("Faixa de cobrança (0 a 10)", 0, 10, int(base_entrada["charge_amount"]), key="charge_amount" + sufixo),
            "seconds_of_use": b.number_input("Segundos de uso", 0, 100000, int(base_entrada["seconds_of_use"]), key="seconds_of_use" + sufixo),
            "frequency_of_use": c.number_input("Frequência de uso", 0, 1000, int(base_entrada["frequency_of_use"]), key="frequency_of_use" + sufixo),
            "frequency_of_sms": a.number_input("Frequência de SMS", 0, 2000, int(base_entrada["frequency_of_sms"]), key="frequency_of_sms" + sufixo),
            "distinct_called_numbers": b.number_input("Números distintos chamados", 0, 500, int(base_entrada["distinct_called_numbers"]), key="distinct_called_numbers" + sufixo),
            "tariff_plan": c.selectbox("Plano tarifário", [1, 2], index=int(base_entrada["tariff_plan"]) - 1,
                                       format_func={1: "1 = pré-pago", 2: "2 = contrato"}.get, key="tariff_plan" + sufixo),
            "status": a.selectbox("Status da linha", [1, 2], index=int(base_entrada["status"]) - 1,
                                  format_func={1: "1 = ativa", 2: "2 = inativa"}.get, key="status" + sufixo),
            "age": b.selectbox("Idade", list(idade_para_faixa), index=list(idade_para_faixa).index(int(base_entrada["age"])),
                               key="age" + sufixo),
            "customer_value": c.number_input("Valor do cliente", 0.0, 10000.0, float(base_entrada["customer_value"]),
                                             step=10.0, format="%.3f", key="customer_value" + sufixo),
        }
        entrada["age_group"] = idade_para_faixa[entrada["age"]]
        st.caption(f"Faixa etária derivada da idade: {entrada['age_group']}. A base só tem as idades "
                   f"{', '.join(map(str, idade_para_faixa))}, uma por faixa.")
        st.form_submit_button("Calcular escore")

    resultado = explicar(entrada, meta)
    c1, c2, c3 = st.columns(3)
    c1.metric("Escore de churn", f"{resultado['probabilidade']:.3f}".replace(".", ","))
    c2.metric("Faixa", resultado["faixa"])
    c3.metric("Média do modelo (base)", f"{resultado['base']:.3f}".replace(".", ","))
    st.subheader("Três fatores de maior contribuição")
    if resultado["fatores"]:
        st.dataframe(pd.DataFrame([{"Atributo": nome(f["campo"]), "Efeito": f["impacto"],
                                    "Peso relativo": f["peso"], "SHAP": f["valor_shap"]}
                                   for f in resultado["fatores"]]),
                     hide_index=True, width="stretch",
                     column_config={"Peso relativo": st.column_config.NumberColumn(format="%.4f"),
                                    "SHAP": st.column_config.NumberColumn(format="%+.4f")})
    else:
        st.write("Nenhum atributo alterou o escore em relação à média do modelo.")
    st.plotly_chart(barras_contribuicao(pd.Series(resultado["valores"]), "Contribuições da entrada simulada"),
                    width="stretch")
    st.caption("Os fatores usam o TreeExplainer sobre o estimador final, na ordem de colunas que sai do "
               "pré-processamento, como em src/validar_inferencia.py e na API. Por isso podem diferir "
               "levemente dos valores da amostra acima, que usam clientes de referência.")


main()
