
"""Visão geral (BI) do ChurnGuard.

Indicadores, filtros globais, gráficos e tabela de priorização.
Todos os dados são provenientes de dashboard.dados.carteira().
Nenhum modelo é treinado nesta página.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.dados import CORES_FAIXA, ORDEM_FAIXAS, carteira


def formatar_numero(valor):
    return f"{valor:,.0f}".replace(",", ".")


def formatar_percentual(valor):
    return f"{valor:.1%}".replace(".", ",")


def aplicar_filtros(df):
    """Aplica os sete filtros globais de forma cumulativa."""
    filtros = [
        ("Situação da linha", "situacao"),
        ("Plano", "plano"),
        ("Reclamação", "reclamacao"),
        ("Faixa etária", "faixa_etaria"),
        ("Perfil", "perfil"),
        ("Faixa de risco", "faixa"),
        ("Conjunto", "conjunto"),
    ]

    selecionados = {}

    with st.sidebar:
        st.divider()
        st.subheader("Filtros globais")
        st.caption("As seleções afetam todos os indicadores e gráficos.")

        for titulo, coluna in filtros:
            opcoes = df[coluna].dropna().unique().tolist()

            if coluna == "faixa":
                opcoes = [x for x in ORDEM_FAIXAS if x in opcoes]
            else:
                opcoes = sorted(opcoes)

            selecionados[coluna] = st.multiselect(
                titulo,
                options=opcoes,
                default=opcoes,
                key=f"bi_filtro_{coluna}",
            )

    resultado = df.copy()

    for coluna, valores in selecionados.items():
        resultado = resultado[resultado[coluna].isin(valores)]

    return resultado


def mostrar_indicadores(df):
    total = len(df)
    cancelamentos = int(df["churn"].sum())
    alto = df[df["faixa"] == "Risco alto"]

    # Proxy baseado no atributo customer_value do dataset.
    # Não representa receita monetária real.
    valor_em_risco = alto["customer_value"].sum()

    taxa = cancelamentos / total if total else 0

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Clientes",
        formatar_numero(total),
        help="Quantidade de registros após os filtros.",
    )

    c2.metric(
        "Churn observado",
        formatar_percentual(taxa),
        help=f"{cancelamentos} cancelamentos observados em {total} clientes.",
    )

    c3.metric(
        "Clientes em risco alto",
        formatar_numero(len(alto)),
        help="Clientes com escore superior a 0,65.",
    )

    c4.metric(
        "Valor em risco",
        formatar_numero(valor_em_risco),
        help=(
            "Soma de customer_value dos clientes em risco alto. "
            "É um indicador relativo da base, não uma quantia em reais."
        ),
    )

    st.caption(
        "Valor em risco: soma do atributo customer_value para clientes "
        "classificados como risco alto. Não é uma estimativa financeira "
        "de receita perdida."
    )


def grafico_faixas(df):
    contagens = (
        df.groupby("faixa", observed=False)
        .size()
        .reindex(ORDEM_FAIXAS, fill_value=0)
        .reset_index(name="Clientes")
    )

    fig = px.bar(
        contagens,
        x="faixa",
        y="Clientes",
        color="faixa",
        color_discrete_map=CORES_FAIXA,
        text="Clientes",
        category_orders={"faixa": ORDEM_FAIXAS},
    )

    fig.update_layout(
        title="Clientes por faixa de risco",
        xaxis_title=None,
        yaxis_title="Quantidade de clientes",
        showlegend=False,
        height=380,
        margin=dict(l=10, r=10, t=55, b=10),
    )

    fig.update_traces(textposition="outside")
    return fig


def grafico_churn_perfil(df):
    resumo = (
        df.groupby("perfil", observed=True)
        .agg(
            Clientes=("churn", "size"),
            Cancelamentos=("churn", "sum"),
        )
        .reset_index()
    )

    resumo["Taxa de churn"] = (
        resumo["Cancelamentos"] / resumo["Clientes"]
    )

    resumo = resumo.sort_values("Taxa de churn", ascending=False)

    fig = px.bar(
        resumo,
        x="perfil",
        y="Taxa de churn",
        text="Taxa de churn",
        custom_data=["Clientes", "Cancelamentos"],
        color_discrete_sequence=["#C2551F"],
    )

    fig.update_traces(
        texttemplate="%{y:.1%}",
        textposition="outside",
        hovertemplate=(
            "Perfil: %{x}<br>"
            "Churn observado: %{y:.1%}<br>"
            "Clientes: %{customdata[0]}<br>"
            "Cancelamentos: %{customdata[1]}"
            "<extra></extra>"
        ),
    )

    fig.update_layout(
        title="Churn observado por perfil",
        xaxis_title=None,
        yaxis_title="Taxa de churn",
        yaxis_tickformat=".0%",
        yaxis_range=[0, max(0.05, resumo["Taxa de churn"].max() * 1.2)],
        height=380,
        margin=dict(l=10, r=10, t=55, b=10),
    )

    return fig


def matriz_perfil_faixa(df):
    matriz = pd.crosstab(
        df["perfil"],
        df["faixa"],
    ).reindex(columns=ORDEM_FAIXAS, fill_value=0)

    fig = go.Figure(
        data=go.Heatmap(
            z=matriz.values,
            x=matriz.columns.tolist(),
            y=matriz.index.tolist(),
            text=matriz.values,
            texttemplate="%{text}",
            colorscale="Blues",
            hovertemplate=(
                "Perfil: %{y}<br>"
                "Faixa: %{x}<br>"
                "Clientes: %{z}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Matriz perfil × faixa de risco",
        xaxis_title="Faixa de risco",
        yaxis_title="Perfil",
        height=380,
        margin=dict(l=10, r=10, t=55, b=10),
    )

    return fig


def tabela_detalhada(df):
    colunas = [
        "row_id",
        "perfil",
        "situacao",
        "plano",
        "reclamacao",
        "faixa_etaria",
        "conjunto",
        "churn",
        "escore",
        "faixa",
        "customer_value",
    ]

    tabela = df[colunas].copy()
    tabela = tabela.sort_values(
        ["escore", "row_id"],
        ascending=[False, True],
    )

    tabela["churn"] = tabela["churn"].map({
        0: "Permaneceu",
        1: "Cancelou",
    })

    tabela = tabela.rename(columns={
        "row_id": "ID do registro",
        "perfil": "Perfil",
        "situacao": "Situação",
        "plano": "Plano",
        "reclamacao": "Reclamação",
        "faixa_etaria": "Faixa etária",
        "conjunto": "Conjunto",
        "churn": "Churn observado",
        "escore": "Escore",
        "faixa": "Faixa de risco",
        "customer_value": "Valor do cliente",
    })

    st.dataframe(
        tabela,
        hide_index=True,
        width="stretch",
        column_config={
            "Escore": st.column_config.ProgressColumn(
                "Escore",
                min_value=0,
                max_value=1,
                format="%.3f",
            ),
            "Valor do cliente": st.column_config.NumberColumn(
                "Valor do cliente",
                format="%.2f",
            ),
        },
    )

    st.download_button(
        "Exportar tabela filtrada (CSV)",
        data=tabela.to_csv(index=False).encode("utf-8-sig"),
        file_name="churnguard_visao_geral.csv",
        mime="text/csv",
    )


def main():
    st.title("Visão geral (BI)")
    st.caption(
        "Panorama da carteira de clientes, churn observado "
        "e priorização por escore de risco."
    )

    try:
        dados = carteira()
    except (FileNotFoundError, ValueError, KeyError) as erro:
        st.error(f"Não foi possível carregar a carteira: {erro}")
        st.stop()

    df = aplicar_filtros(dados)

    if df.empty:
        st.warning("Nenhum cliente corresponde aos filtros selecionados.")
        st.stop()

    st.subheader("Indicadores gerais")
    mostrar_indicadores(df)

    st.divider()
    esquerda, direita = st.columns(2)

    with esquerda:
        st.plotly_chart(grafico_faixas(df), width="stretch")

    with direita:
        st.plotly_chart(grafico_churn_perfil(df), width="stretch")

    st.divider()
    st.plotly_chart(matriz_perfil_faixa(df), width="stretch")

    st.divider()
    st.subheader("Detalhamento dos clientes")
    st.caption(
        "Registros ordenados pelo escore de churn, do maior para o menor. "
        "IDs representam linhas do dataset, não identificadores reais de assinantes."
    )

    tabela_detalhada(df)

    st.info(
        "Os escores de churn não são probabilidades calibradas. "
        "No conjunto de treino, o modelo já observou os registros; "
        "para avaliar generalização, utilize o filtro Conjunto = teste."
    )


main()
