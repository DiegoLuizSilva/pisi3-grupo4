"""Análise exploratória interativa do ChurnGuard.

Versão navegável das Figuras 1 a 5 da Seção 5 do artigo. Sem nenhum filtro
aplicado, todos os números desta página reproduzem os do artigo; os filtros
recalculam tudo sobre o subconjunto escolhido.

As contas são as mesmas de eda_visual.py: assimetria, d de Cohen com desvio
agrupado, Mann-Whitney, Spearman, qui-quadrado e V de Cramér.

Nada aqui treina modelo nem usa o escore: esta página descreve a base
observada, antes de qualquer predição.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from dashboard.dados import ROTULOS, carteira

# Mesmas listas de eda_visual.py, para os valores baterem com as figuras.
# age_group entra nas duas: é ordinal e foi analisado dos dois modos no artigo.
NUMERICAS = [
    "call_failure", "subscription_length", "charge_amount", "seconds_of_use",
    "frequency_of_use", "frequency_of_sms", "distinct_called_numbers",
    "age_group", "age", "customer_value",
]

# (coluna numérica original, coluna já rotulada em dados.py)
CATEGORICAS = [
    ("complains", "reclamacao"),
    ("tariff_plan", "plano"),
    ("status", "situacao"),
    ("age_group", "faixa_etaria"),
]

COR_FICOU = "#4783BC"
COR_CANCELOU = "#C2551F"
CORES_DESFECHO = {"Permaneceu": COR_FICOU, "Cancelou": COR_CANCELOU}


def virgula(valor, casas=2):
    """Número no padrão brasileiro: 0.874 -> 0,874."""
    if pd.isna(valor):
        return "—"
    return f"{valor:,.{casas}f}".replace(",", "@").replace(".", ",").replace("@", ".")


def percentual(valor):
    if pd.isna(valor):
        return "—"
    return f"{valor:.1%}".replace(".", ",")


def rotulo(coluna):
    return ROTULOS.get(coluna, coluna)


# --------------------------------------------------------------- estatística

def cohen_d(ficou, cancelou):
    """d de Cohen com desvio-padrão agrupado. Negativo = menor em quem cancelou."""
    n1, n2 = len(ficou), len(cancelou)
    if n1 < 2 or n2 < 2:
        return np.nan
    dp = np.sqrt(((n1 - 1) * ficou.var() + (n2 - 1) * cancelou.var()) / (n1 + n2 - 2))
    if dp == 0:
        return np.nan
    return (cancelou.mean() - ficou.mean()) / dp


def comparar_classes(df, colunas):
    """Média por desfecho, d de Cohen e p de Mann-Whitney, por atributo."""
    ficou_todo = df[df["churn"] == 0]
    cancelou_todo = df[df["churn"] == 1]

    linhas = []
    for coluna in colunas:
        a = ficou_todo[coluna].dropna()
        b = cancelou_todo[coluna].dropna()

        if len(a) < 2 or len(b) < 2:
            p = np.nan
        else:
            try:
                p = stats.mannwhitneyu(a, b).pvalue
            except ValueError:  # todos os valores idênticos
                p = np.nan

        linhas.append({
            "coluna": coluna,
            "atributo": rotulo(coluna),
            "media_ficou": a.mean() if len(a) else np.nan,
            "media_cancelou": b.mean() if len(b) else np.nan,
            "cohen_d": cohen_d(a, b),
            "p_valor": p,
        })

    return pd.DataFrame(linhas)


def correlacao_spearman(df, colunas):
    """Matriz de Spearman entre os atributos numéricos e o churn."""
    return df[colunas + ["churn"]].corr(method="spearman")


def teste_categorico(df, coluna):
    """Qui-quadrado de independência com o churn e V de Cramér.

    Devolve None quando o filtro deixou uma única categoria ou uma única
    classe de desfecho: nesse caso o teste não tem o que comparar.
    """
    tabela = pd.crosstab(df[coluna], df["churn"])
    if tabela.shape[0] < 2 or tabela.shape[1] < 2:
        return None
    if (tabela.values.sum(axis=1) == 0).any():
        return None

    chi2, p, gl, _ = stats.chi2_contingency(tabela)
    n = tabela.values.sum()
    v = np.sqrt(chi2 / (n * (min(tabela.shape) - 1)))
    return {"chi2": chi2, "p": p, "gl": gl, "v": v, "n": n}


# ------------------------------------------------------------------- filtros

def aplicar_filtros(df):
    """Filtros por plano, situação da linha e faixa etária, cumulativos."""
    filtros = [
        ("Plano", "plano"),
        ("Situação da linha", "situacao"),
        ("Faixa etária", "faixa_etaria"),
    ]

    selecionados = {}
    with st.sidebar:
        st.divider()
        st.subheader("Filtros da exploratória")
        st.caption(
            "Sem nenhuma alteração aqui, os valores da página são os das "
            "Figuras 1 a 5 do artigo."
        )

        for titulo, coluna in filtros:
            opcoes = df[coluna].dropna().unique().tolist()
            if coluna != "faixa_etaria":
                opcoes = sorted(opcoes)
            selecionados[coluna] = st.multiselect(
                titulo, options=opcoes, default=opcoes, key=f"eda_filtro_{coluna}"
            )

    resultado = df.copy()
    for coluna, valores in selecionados.items():
        resultado = resultado[resultado[coluna].isin(valores)]

    filtrado = any(len(v) < df[c].dropna().nunique() for c, v in selecionados.items())
    return resultado, filtrado


# ------------------------------------------------------------------- gráficos

def grafico_histograma(df, coluna):
    fig = px.histogram(
        df, x=coluna, color="desfecho", nbins=30, barmode="overlay",
        color_discrete_map=CORES_DESFECHO, opacity=0.7,
    )

    media, mediana = df[coluna].mean(), df[coluna].median()
    fig.add_vline(x=media, line_dash="dash", line_color="#253B4B",
                  annotation_text=f"média {virgula(media)}",
                  annotation_position="top right")
    fig.add_vline(x=mediana, line_dash="dot", line_color="#6A7A2E",
                  annotation_text=f"mediana {virgula(mediana)}",
                  annotation_position="bottom right")

    fig.update_layout(
        title=f"Distribuição · {rotulo(coluna)}",
        xaxis_title=rotulo(coluna), yaxis_title="Clientes",
        legend_title=None, height=400,
        margin=dict(l=10, r=10, t=55, b=10),
    )
    return fig


def grafico_boxplot(df, coluna):
    fig = px.box(
        df, x="desfecho", y=coluna, color="desfecho",
        color_discrete_map=CORES_DESFECHO,
        category_orders={"desfecho": ["Permaneceu", "Cancelou"]},
        points=False,
    )
    fig.update_layout(
        title=f"Por desfecho · {rotulo(coluna)}",
        xaxis_title=None, yaxis_title=rotulo(coluna),
        showlegend=False, height=400,
        margin=dict(l=10, r=10, t=55, b=10),
    )
    return fig


def grafico_efeitos(efeitos):
    ordem = efeitos.dropna(subset=["cohen_d"]).sort_values("cohen_d")
    if ordem.empty:
        return None

    fig = go.Figure(go.Bar(
        x=ordem["cohen_d"], y=ordem["atributo"], orientation="h",
        marker_color=[COR_CANCELOU if v > 0 else COR_FICOU for v in ordem["cohen_d"]],
        text=[virgula(v, 3) for v in ordem["cohen_d"]], textposition="outside",
        hovertemplate="%{y}<br>d de Cohen: %{x:.3f}<extra></extra>",
    ))

    for corte in (-0.8, -0.5, -0.2, 0.2, 0.5, 0.8):
        fig.add_vline(x=corte, line_dash="dot", line_color="#C9D2D9", line_width=1)
    fig.add_vline(x=0, line_color="#253B4B", line_width=1)

    limite = max(1.0, float(ordem["cohen_d"].abs().max()) * 1.35)
    fig.update_layout(
        title="Tamanho do efeito por atributo (d de Cohen)",
        xaxis_title="d de Cohen · positivo = maior em quem cancelou",
        yaxis_title=None, xaxis_range=[-limite, limite],
        height=440, margin=dict(l=10, r=10, t=55, b=10),
    )
    return fig


def grafico_correlacao(corr):
    rotulos = [rotulo(c) for c in corr.columns]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=rotulos, y=rotulos,
        zmin=-1, zmax=1, colorscale="RdBu_r",
        text=np.round(corr.values, 2), texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate="%{y}<br>%{x}<br>rho = %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="Correlação de Spearman",
        height=620, margin=dict(l=10, r=10, t=55, b=10),
        xaxis_tickangle=-40,
    )
    return fig


def grafico_churn_categoria(df, coluna_rotulada, media_geral):
    resumo = (df.groupby(coluna_rotulada, observed=True)["churn"]
              .agg(clientes="size", cancelaram="sum", taxa="mean")
              .reset_index())
    if resumo.empty:
        return None

    fig = px.bar(
        resumo, x=coluna_rotulada, y="taxa",
        custom_data=["clientes", "cancelaram"],
        color_discrete_sequence=[COR_CANCELOU],
    )
    fig.update_traces(
        texttemplate="%{y:.1%}", textposition="outside",
        hovertemplate=("%{x}<br>Churn observado: %{y:.1%}<br>"
                       "Clientes: %{customdata[0]}<br>"
                       "Cancelamentos: %{customdata[1]}<extra></extra>"),
    )
    fig.add_hline(
        y=media_geral, line_dash="dash", line_color="#253B4B",
        annotation_text=f"média da seleção {percentual(media_geral)}",
        annotation_position="top left",
    )
    fig.update_layout(
        title=f"Churn por {rotulo(coluna_rotulada).lower()}",
        xaxis_title=None, yaxis_title="Taxa de churn",
        yaxis_tickformat=".0%",
        yaxis_range=[0, max(0.1, resumo["taxa"].max() * 1.25)],
        height=360, margin=dict(l=10, r=10, t=55, b=10),
    )
    return fig


# ----------------------------------------------------------------- seções

def secao_atributo(df):
    st.subheader("Distribuição de um atributo")
    st.caption(
        "Escolha o atributo para ver a distribuição e a comparação entre quem "
        "permaneceu e quem cancelou. Corresponde às Figuras 1 e 2 do artigo."
    )

    coluna = st.selectbox(
        "Atributo", options=NUMERICAS, format_func=rotulo, key="eda_atributo",
    )

    ficou = df[df["churn"] == 0][coluna].dropna()
    cancelou = df[df["churn"] == 1][coluna].dropna()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Média de quem permaneceu", virgula(ficou.mean()),
              help=f"{len(ficou)} clientes.")
    c2.metric("Média de quem cancelou", virgula(cancelou.mean()),
              help=f"{len(cancelou)} clientes.")
    c3.metric("d de Cohen", virgula(cohen_d(ficou, cancelou), 3),
              help="0,2 pequeno · 0,5 médio · 0,8 grande. Negativo: menor em quem cancelou.")
    c4.metric("Assimetria", virgula(df[coluna].skew(), 2),
              help="Acima de 1 indica cauda longa à direita; perto de 0, simetria.")

    esquerda, direita = st.columns(2)
    with esquerda:
        st.plotly_chart(grafico_histograma(df, coluna), width="stretch")
    with direita:
        st.plotly_chart(grafico_boxplot(df, coluna), width="stretch")


def secao_efeitos(df):
    st.subheader("Comparação entre quem permaneceu e quem cancelou")
    st.caption(
        "Diferença de cada atributo entre as duas classes, medida pelo d de "
        "Cohen. Corresponde à Figura 4 do artigo."
    )

    efeitos = comparar_classes(df, NUMERICAS)
    figura = grafico_efeitos(efeitos)
    if figura is None:
        st.warning("A seleção não tem as duas classes de desfecho; sem comparação a fazer.")
        return

    st.plotly_chart(figura, width="stretch")

    with st.expander("Ver a tabela com médias e p-valor"):
        exibir = efeitos.sort_values("cohen_d", key=abs, ascending=False).copy()
        exibir["Média · permaneceu"] = exibir["media_ficou"].map(lambda v: virgula(v))
        exibir["Média · cancelou"] = exibir["media_cancelou"].map(lambda v: virgula(v))
        exibir["d de Cohen"] = exibir["cohen_d"].map(lambda v: virgula(v, 3))
        exibir["p (Mann-Whitney)"] = exibir["p_valor"].map(
            lambda v: "—" if pd.isna(v) else f"{v:.2e}"
        )
        st.dataframe(
            exibir[["atributo", "Média · permaneceu", "Média · cancelou",
                    "d de Cohen", "p (Mann-Whitney)"]]
            .rename(columns={"atributo": "Atributo"}),
            hide_index=True, width="stretch",
        )
        st.caption(
            "O p-valor indica se a diferença é distinguível do acaso, não o "
            "tamanho dela. Com 3.150 registros, diferenças pequenas saem com "
            "p baixo: por isso o d de Cohen é o critério de leitura."
        )


def secao_correlacao(df):
    st.subheader("Correlação entre atributos")
    st.caption(
        "Spearman, que mede relação monotônica e não exige distribuição "
        "normal. Corresponde à Figura 3 do artigo."
    )

    corr = correlacao_spearman(df, NUMERICAS)
    esquerda, direita = st.columns([3, 2])

    with esquerda:
        st.plotly_chart(grafico_correlacao(corr), width="stretch")

    with direita:
        com_churn = (corr["churn"].drop("churn")
                     .sort_values(key=abs, ascending=False)
                     .reset_index())
        com_churn.columns = ["coluna", "rho"]
        com_churn["Atributo"] = com_churn["coluna"].map(rotulo)
        com_churn["rho (Spearman)"] = com_churn["rho"].map(lambda v: virgula(v, 3))
        st.markdown("**Correlação com o cancelamento**")
        st.dataframe(com_churn[["Atributo", "rho (Spearman)"]],
                     hide_index=True, width="stretch", height=420)

        pares = []
        for i in range(len(NUMERICAS)):
            for j in range(i + 1, len(NUMERICAS)):
                v = corr.loc[NUMERICAS[i], NUMERICAS[j]]
                if abs(v) > 0.6:
                    pares.append(f"{rotulo(NUMERICAS[i])} × {rotulo(NUMERICAS[j])}: "
                                 f"{virgula(v, 3)}")
        if pares:
            st.caption("**Pares com |rho| acima de 0,6:** " + " · ".join(pares))


def secao_categoricas(df):
    st.subheader("Churn por categoria")
    st.caption(
        "Taxa observada em cada categoria, com o teste qui-quadrado de "
        "independência e o V de Cramér. Corresponde à Figura 5 do artigo."
    )

    media_geral = df["churn"].mean()

    linhas = []
    for coluna, rotulada in CATEGORICAS:
        teste = teste_categorico(df, coluna)
        linhas.append({
            "Atributo": rotulo(rotulada),
            "Qui-quadrado": "—" if teste is None else virgula(teste["chi2"], 2),
            "p": "—" if teste is None else f"{teste['p']:.2e}",
            "Graus de liberdade": "—" if teste is None else str(teste["gl"]),
            "V de Cramér": "—" if teste is None else virgula(teste["v"], 3),
        })

    st.dataframe(pd.DataFrame(linhas), hide_index=True, width="stretch")
    st.caption(
        "V de Cramér vai de 0 a 1 e mede a força da associação: 0,1 fraca, "
        "0,3 moderada, 0,5 forte. Um traço significa que o filtro deixou uma "
        "única categoria, e aí não há o que comparar."
    )

    colunas_grade = st.columns(2)
    for indice, (coluna, rotulada) in enumerate(CATEGORICAS):
        figura = grafico_churn_categoria(df, rotulada, media_geral)
        if figura is not None:
            with colunas_grade[indice % 2]:
                st.plotly_chart(figura, width="stretch")


# --------------------------------------------------------------------- main

def main():
    st.title("Exploratória")
    st.caption(
        "Versão interativa das Figuras 1 a 5 do artigo. Descreve a base "
        "observada: nenhum modelo é treinado nem consultado nesta página."
    )

    try:
        dados = carteira()
    except (FileNotFoundError, ValueError, KeyError) as erro:
        st.error(f"Não foi possível carregar a carteira: {erro}")
        st.stop()

    df, filtrado = aplicar_filtros(dados)

    if df.empty:
        st.warning("Nenhum cliente corresponde aos filtros selecionados.")
        st.stop()

    if len(df) < 30:
        st.warning(
            f"A seleção tem apenas {len(df)} clientes. Testes e tamanhos de "
            "efeito ficam instáveis com amostras desse tamanho.",
            icon=":material/warning:",
        )

    c1, c2, c3 = st.columns(3)
    c1.metric("Clientes na seleção", virgula(len(df), 0))
    c2.metric("Cancelamentos", virgula(int(df["churn"].sum()), 0))
    c3.metric("Churn observado", percentual(df["churn"].mean()))

    if filtrado:
        st.info(
            "Há filtros ativos: os valores abaixo descrevem o subconjunto "
            "selecionado e não são os do artigo. Limpe os filtros para "
            "reproduzir as Figuras 1 a 5.",
            icon=":material/filter_alt:",
        )

    st.divider()
    secao_atributo(df)

    st.divider()
    secao_efeitos(df)

    st.divider()
    secao_correlacao(df)

    st.divider()
    secao_categoricas(df)

    st.divider()
    st.info(
        "Todas as medidas desta página são associações observadas na base. "
        "Nenhuma delas estabelece causa: um atributo associado ao "
        "cancelamento não é, por isso, aquilo que o provoca, e atuar sobre "
        "ele não implica reduzir o churn.",
        icon=":material/info:",
    )


main()