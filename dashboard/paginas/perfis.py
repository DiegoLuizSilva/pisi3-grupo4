"""Página "Perfis": agrupamento K-Means dos clientes (Seção 7.1 do artigo).

Todos os dados vêm de dashboard/dados.py. A página só lê os resultados do
agrupamento já gerados pelo projeto; nada é reajustado aqui.
Rodar pelo painel, a partir da raiz: streamlit run dashboard/app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard.dados import CORES_FAIXA, ORDEM_FAIXAS, ROTULOS, carteira, metadados, tabela

Z_95 = 1.959963984540054


def br(valor: float, casas: int = 1) -> str:
    """Número no formato brasileiro: 1.875 / 25,9."""
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "_").replace(".", ",").replace("_", ".")


def pct(valor: float, casas: int = 2) -> str:
    return br(100 * valor, casas) + "%"


def wilson(cancelaram: np.ndarray, clientes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Intervalo de Wilson a 95%, o mesmo do notebook 07 (statsmodels, method="wilson")."""
    p = cancelaram / clientes
    denominador = 1 + Z_95**2 / clientes
    centro = (p + Z_95**2 / (2 * clientes)) / denominador
    margem = Z_95 * np.sqrt(p * (1 - p) / clientes + Z_95**2 / (4 * clientes**2)) / denominador
    return centro - margem, centro + margem


def resumo_perfis(clientes: pd.DataFrame, ordem: list[str]) -> pd.DataFrame:
    resumo = (clientes.groupby("perfil").agg(clientes=("row_id", "size"), cancelaram=("churn", "sum"))
                      .reindex(ordem).reset_index())
    resumo["participacao"] = resumo.clientes / resumo.clientes.sum()
    resumo["taxa"] = resumo.cancelaram / resumo.clientes
    resumo["ic_inf"], resumo["ic_sup"] = wilson(resumo.cancelaram.to_numpy(), resumo.clientes.to_numpy())
    return resumo


def main() -> None:
    st.title("Perfis de clientes")

    clientes = carteira()
    tamanho = tabela("perfil_tamanho")
    ordem = tamanho.sort_values("grupo").perfil.tolist()
    nome_do_grupo = dict(zip(tamanho.grupo, tamanho.perfil))
    # Cores próprias dos perfis, distintas das cores das faixas de risco.
    cores = dict(zip(ordem, ["#2E6F95", "#7B5EA7", "#3A9D8F"]))
    meta = metadados("cluster_metadata")
    taxa_geral = clientes.churn.mean()

    st.info(
        f"O K-Means com k = {meta['k']} foi ajustado só com os clientes de treino e depois aplicado aos "
        f"{br(len(clientes), 0)} clientes, usando os 12 atributos de entrada padronizados. O cancelamento "
        "não participou da formação dos grupos nem da escolha de k: as taxas abaixo são frequências "
        "observadas, consultadas depois, e descrevem associações, não causas."
    )

    # Escolha de k
    st.header("Escolha da quantidade de grupos")
    qualidade = tabela("cluster_qualidade")
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Método do cotovelo (inércia)", "Coeficiente de silhueta"))
    fig.add_scatter(x=qualidade.k, y=qualidade.inercia, mode="lines+markers", line_color="#2E6F95",
                    showlegend=False, hovertemplate="k = %{x}<br>inércia %{y:,.0f}<extra></extra>", row=1, col=1)
    validos = qualidade.dropna(subset=["silhueta"])
    fig.add_scatter(x=validos.k, y=validos.silhueta, mode="lines+markers", line_color="#2E6F95",
                    showlegend=False, hovertemplate="k = %{x}<br>silhueta %{y:.4f}<extra></extra>", row=1, col=2)
    escolhido = qualidade.loc[qualidade.k == meta["k"]].iloc[0]
    for col, valor in ((1, escolhido.inercia), (2, escolhido.silhueta)):
        fig.add_scatter(x=[meta["k"]], y=[valor], mode="markers", marker=dict(size=14, color="#C2551F"),
                        showlegend=False, hoverinfo="skip", row=1, col=col)
    fig.update_xaxes(title_text="k", dtick=1)
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")

    revisao = tabela("revisao_k")
    criterios = revisao.loc[revisao.k.isin([2, 3, 4])].set_index("k")
    tabela_criterios = pd.DataFrame({
        f"k = {k}": [br(linha.desaceleracao, 2), br(linha.silhueta, 3), br(linha["mal_alocados_%"], 1) + "%",
                     f"{br(linha.menor_grupo, 0)} ({br(linha['menor_grupo_%'], 1)}%)", br(linha.dist_min_centroides, 2)]
        for k, linha in criterios.iterrows()
    }, index=["Desaceleração da inércia", "Coeficiente de silhueta", "Observações mal alocadas",
              "Menor grupo", "Distância mínima entre centroides"])
    st.dataframe(tabela_criterios, width="stretch")
    st.caption(f"Calculado sobre o treino ({br((clientes.conjunto == 'treino').sum(), 0)} clientes). "
               "A silhueta quase empata k = 2 e k = 3; o k = 3 é o único com desaceleração acima de 2, "
               "tem menos observações mal alocadas e centroides mais afastados (Tabela 1 do artigo).")

    with st.expander("Robustez da solução: sementes, escala, atributos e outros algoritmos"):
        sementes = tabela("sens_sementes").query("k == 3").set_index("n_init")
        st.markdown(
            f"- **Sementes:** com uma única inicialização, a silhueta em k = 3 variou de "
            f"{br(sementes.loc[1, 'silhueta_min'], 4)} a {br(sementes.loc[1, 'silhueta_max'], 4)}. Com as 20 "
            f"inicializações adotadas, o desvio-padrão caiu para {br(sementes.loc[20, 'silhueta_dp'], 4)} e o "
            f"índice Rand ajustado médio entre execuções chegou a {br(sementes.loc[20, 'ari_medio'], 4)}.")
        resumo_sementes = metadados("sens_resumo")["k_vencedor_por_semente"]
        st.markdown(
            f"- **k vencedor em 20 sementes:** com 1 inicialização, k = 3 em {resumo_sementes['1']['3']}, "
            f"k = 4 em {resumo_sementes['1']['4']} e k = 2 em {resumo_sementes['1']['2']}; com 20 "
            f"inicializações, k = 3 em {resumo_sementes['20']['3']}.")
        atributos = tabela("sens_atributos")
        melhores = atributos.loc[atributos.groupby("conjunto").silhueta.idxmax(), ["conjunto", "k", "silhueta"]]
        st.markdown("- **Conjunto de atributos:** melhor k por silhueta em cada subconjunto testado. "
                    "Sem o valor do cliente, a estrutura de três grupos se desfaz.")
        st.dataframe(melhores.rename(columns={"conjunto": "Atributos", "k": "Melhor k", "silhueta": "Silhueta"}),
                     hide_index=True, width="stretch",
                     column_config={"Silhueta": st.column_config.NumberColumn(format="%.4f")})
        algoritmos = tabela("sens_algoritmos").query("k == 3")
        st.markdown("- **Outros algoritmos (k = 3, Tabela 2 do artigo):** concordância medida pelo índice "
                    "Rand ajustado em relação ao K-Means.")
        st.dataframe(algoritmos.drop(columns="k").replace({"Ward": "Ward (hierárquico)", "GMM": "Mistura de gaussianas"})
                     .rename(columns={"algoritmo": "Algoritmo", "silhueta": "Silhueta", "menor_grupo": "Menor grupo",
                                      "maior_grupo": "Maior grupo", "ari_vs_kmeans": "Concordância (ARI)"}),
                     hide_index=True, width="stretch",
                     column_config={"Silhueta": st.column_config.NumberColumn(format="%.3f"),
                                    "Concordância (ARI)": st.column_config.NumberColumn(format="%.3f")})

    # Tamanho e cancelamento
    st.header("Tamanho e cancelamento por perfil")
    conjunto = st.radio("Clientes considerados", ["Todos", "Treino", "Teste"], horizontal=True, key="perfis_conjunto")
    base = clientes if conjunto == "Todos" else clientes.loc[clientes.conjunto == conjunto.lower()]
    resumo = resumo_perfis(base, ordem)

    colunas = st.columns(len(resumo))
    for coluna, linha in zip(colunas, resumo.itertuples()):
        coluna.metric(linha.perfil, pct(linha.taxa), help="Taxa de cancelamento observada no perfil")
        coluna.caption(f"{br(linha.clientes, 0)} clientes · {br(linha.cancelaram, 0)} cancelamentos")


    fig = go.Figure(go.Bar(
        x=resumo.perfil, y=100 * resumo.taxa, marker_color=[cores[p] for p in resumo.perfil],
        error_y=dict(type="data", symmetric=False, array=100 * (resumo.ic_sup - resumo.taxa),
                     arrayminus=100 * (resumo.taxa - resumo.ic_inf), color="#14213D"),
        text=[pct(t) for t in resumo.taxa], textposition="outside",
        customdata=np.stack([100 * resumo.ic_inf, 100 * resumo.ic_sup], axis=1),
        hovertemplate="%{x}<br>taxa %{y:.2f}%<br>IC 95%: %{customdata[0]:.2f}% a %{customdata[1]:.2f}%<extra></extra>"))
    media = base.churn.mean()
    fig.add_hline(y=100 * media, line_dash="dash", line_color="gray",
                  annotation_text=f"média {pct(media)}", annotation_position="top left")
    fig.update_layout(yaxis_title="Taxa de cancelamento observada (%)", height=380,
                      margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(
        pd.DataFrame({
            "Perfil": resumo.perfil,
            "Clientes": resumo.clientes,
            "Participação": [pct(v, 1) for v in resumo.participacao],
            "Cancelamentos": resumo.cancelaram,
            "Taxa": [pct(v) for v in resumo.taxa],
            "IC 95% (Wilson)": [f"{pct(a)} a {pct(b)}" for a, b in zip(resumo.ic_inf, resumo.ic_sup)],
        }),
        hide_index=True, width="stretch",
    )
    st.caption("Os perfis de alto engajamento têm só 5 e 4 cancelamentos na base completa, por isso o "
               "intervalo de Wilson é largo em proporção à taxa. Com todos os clientes, a tabela reproduz "
               "a Tabela 3 do artigo.")

    # Desvios por atributo
    st.header("O que distingue cada perfil")
    desvios = tabela("perfil_desvios").set_index("atributo")[ordem]
    limite = float(np.abs(desvios.values).max())
    fig = go.Figure(go.Heatmap(
        z=desvios.values, x=desvios.columns, y=desvios.index, zmin=-limite, zmax=limite,
        colorscale="RdBu_r", text=np.vectorize(lambda v: br(v, 2))(desvios.values), texttemplate="%{text}",
        colorbar=dict(title="Desvios-<br>padrão"),
        hovertemplate="%{y}<br>%{x}: %{z:+.2f} desvios-padrão<extra></extra>"))
    fig.update_layout(height=480, yaxis_autorange="reversed", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("Distância da média de cada perfil em relação à média geral, em desvios-padrão (Figura 8 do "
               "artigo). Vermelho indica atributo acima da média; azul, abaixo.")

    # PCA
    st.header("Projeção dos grupos (PCA)")
    pca = tabela("cluster_pca").assign(perfil=lambda d: d.cluster.map(nome_do_grupo))
    fig = px.scatter(pca, x="PC1", y="PC2", color="perfil", color_discrete_map=cores,
                     category_orders={"perfil": ordem}, opacity=0.55,
                     labels={"PC1": "Componente principal 1", "PC2": "Componente principal 2", "perfil": "Perfil"})
    fig.update_traces(marker_size=5)
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Clientes de treino projetados nas duas primeiras componentes, que retêm "
               f"{br(100 * meta['pca_variancia'], 0)}% da variância dos 12 atributos (Figura 9 do artigo). A "
               "projeção resume as posições relativas, mas não reproduz as distâncias usadas pelo K-Means: grupos "
               "que se tocam aqui podem estar separados no espaço completo.")

    # Detalhamento
    st.header("Detalhamento de um perfil")
    perfil = st.selectbox("Perfil", ordem, index=ordem.index("Baixo engajamento") if "Baixo engajamento" in ordem else 0,
                          key="perfil_detalhe")
    do_perfil = clientes.loc[clientes.perfil == perfil]
    linha = resumo_perfis(clientes, ordem).set_index("perfil").loc[perfil]
    c1, c2, c3 = st.columns(3)
    c1.metric("Clientes", br(linha.clientes, 0))
    c1.caption(f"{pct(linha.participacao, 1)} da base")
    c2.metric("Cancelamentos", br(linha.cancelaram, 0))
    c3.metric("Taxa observada", pct(linha.taxa))
    c3.caption(f"IC 95% (Wilson): {pct(linha.ic_inf)} a {pct(linha.ic_sup)} · média geral {pct(taxa_geral)}")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        caracteristicas = tabela("perfil_caracteristicas")
        st.dataframe(
            caracteristicas[["atributo", perfil, "geral", "unidade"]].rename(columns={
                "atributo": "Atributo", "unidade": "Unidade", perfil: "Média do perfil", "geral": "Média geral"}),
            hide_index=True, width="stretch",
            column_config={"Média do perfil": st.column_config.NumberColumn(format="%.2f"),
                           "Média geral": st.column_config.NumberColumn(format="%.2f")},
        )
    with col_b:
        st.markdown("**Faixa de risco dos clientes do perfil**")
        faixas = (do_perfil.groupby("faixa", observed=False).size().reindex(ORDEM_FAIXAS, fill_value=0)
                           .rename("clientes").reset_index())
        fig = px.bar(faixas, x="faixa", y="clientes", color="faixa", color_discrete_map=CORES_FAIXA,
                     labels={"faixa": "Faixa de risco do modelo", "clientes": "Clientes"}, text="clientes")
        fig.update_layout(showlegend=False, height=330, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("Faixas do modelo final (Seção 7.2), que não usou os perfis como entrada.")

    st.subheader("Clientes do perfil")
    colunas_lista = ["row_id", "conjunto", "desfecho", "escore", "faixa", "situacao", "plano", "reclamacao",
                     "frequency_of_use", "seconds_of_use", "frequency_of_sms", "customer_value"]
    lista = do_perfil[colunas_lista].sort_values("escore", ascending=False)
    st.dataframe(
        lista.rename(columns={c: ROTULOS.get(c, c) for c in colunas_lista} | {
            "row_id": "Cliente (row_id)", "desfecho": "Desfecho real", "situacao": "Situação da linha",
            "plano": "Plano", "reclamacao": "Reclamação"}),
        hide_index=True, width="stretch", height=360,
        column_config={ROTULOS["escore"]: st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1)},
    )
    st.caption("Ordenados pelo escore do modelo final. Para clientes de treino o escore é otimista, porque o "
               "modelo já os viu; use o teste para avaliar o desempenho.")


main()
