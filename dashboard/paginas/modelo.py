"""Página "Modelo": desempenho das combinações e do modelo final.

Dono: Diego Luiz. Todos os dados vêm de dashboard/dados.py.
Rodar pelo painel, a partir da raiz: streamlit run dashboard/app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from dashboard.dados import (CORES_FAIXA, CORTE_ALTO, CORTE_BAIXO, EXPERIMENTO_FINAL, ORDEM_FAIXAS,
                             carteira, metadados, tabela)

CORTES_FAIXA = {"Risco baixo": "< 0,30", "Risco médio": "0,30 a 0,65", "Risco alto": "> 0,65"}


def metricas_no_limiar(real: np.ndarray, score: np.ndarray, limiar: float) -> dict:
    # predict() do Random Forest só marca churn quando a probabilidade passa de 0,5;
    # usar ">" mantém o limiar padrão idêntico à coluna "previsto" publicada.
    previsto = (score > limiar).astype(int)
    vp = int(((previsto == 1) & (real == 1)).sum())
    fp = int(((previsto == 1) & (real == 0)).sum())
    fn = int(((previsto == 0) & (real == 1)).sum())
    vn = int(((previsto == 0) & (real == 0)).sum())
    precisao = vp / (vp + fp) if vp + fp else 0.0
    revocacao = vp / (vp + fn) if vp + fn else 0.0
    f1 = 2 * precisao * revocacao / (precisao + revocacao) if precisao + revocacao else 0.0
    return {"vp": vp, "fp": fp, "fn": fn, "vn": vn,
            "precisao": precisao, "revocacao": revocacao, "f1": f1}


def fmt(valor: float, casas: int = 3) -> str:
    return f"{valor:.{casas}f}".replace(".", ",")


def tabela_comparacao(cv: pd.DataFrame, treino_teste: pd.DataFrame) -> pd.DataFrame:
    comparacao = cv[["experimento", "modelo", "balanceamento", "cv_f1", "cv_f1_std",
                 "cv_precision", "cv_recall", "cv_ap"]].copy()
    teste = treino_teste.loc[treino_teste.conjunto == "teste",
                             ["experimento", "f1_churn", "precisao_churn", "recall_churn", "roc_auc"]]
    treino = treino_teste.loc[treino_teste.conjunto == "treino", ["experimento", "f1_churn"]]
    comparacao = (comparacao.merge(treino.rename(columns={"f1_churn": "treino_f1"}), on="experimento", how="left")
                    .merge(teste.rename(columns={"f1_churn": "teste_f1", "precisao_churn": "teste_precisao",
                                                 "recall_churn": "teste_revocacao", "roc_auc": "teste_roc_auc"}),
                           on="experimento", how="left"))
    return comparacao.sort_values("cv_f1", ascending=False).reset_index(drop=True)


def main() -> None:
    st.title("Desempenho do modelo")

    meta = metadados("model_metadata")
    vencedor = EXPERIMENTO_FINAL
    if meta["vencedor"] != vencedor:
        st.error(f"model_metadata.json aponta {meta['vencedor']}, mas o painel usa {vencedor}.")
    cv = tabela("comparacao_cv")
    treino_teste = tabela("metricas_treino_teste")
    predicoes = tabela("predicoes")
    classes = tabela("relatorio_classes")

    st.info(
        f"O modelo final (**{vencedor}**) foi escolhido pelo F1 de churn médio na validação cruzada "
        f"agrupada de 5 folds, só com o conjunto de treino ({meta['n_treino']} clientes). "
        f"O conjunto de teste ({meta['n_teste']} clientes) não foi usado para escolher o modelo. "
        "O escore é uma probabilidade **não calibrada**: serve para ordenar clientes por risco, "
        "não como chance exata de cancelamento."
    )

    # Comparação das combinações
    st.header("Comparação das combinações")
    comparacao = tabela_comparacao(cv, treino_teste)
    st.caption(f"{len(comparacao)} combinações de modelo e balanceamento, ordenadas pelo F1 médio na validação cruzada.")
    st.dataframe(
        comparacao.drop(columns=["modelo", "balanceamento"]).rename(columns={
            "experimento": "Combinação", "cv_f1": "F1 CV", "cv_f1_std": "Desvio F1 CV",
            "cv_precision": "Precisão CV", "cv_recall": "Revocação CV", "cv_ap": "AP CV",
            "treino_f1": "F1 treino", "teste_f1": "F1 teste", "teste_precisao": "Precisão teste",
            "teste_revocacao": "Revocação teste", "teste_roc_auc": "ROC AUC teste"}),
        hide_index=True, width="stretch",
        column_config={c: st.column_config.NumberColumn(format="%.3f") for c in [
            "F1 CV", "Desvio F1 CV", "Precisão CV", "Revocação CV", "AP CV", "F1 treino",
            "F1 teste", "Precisão teste", "Revocação teste", "ROC AUC teste"]},
    )

    grafico = comparacao.sort_values("cv_f1")
    fig = go.Figure()
    fig.add_bar(y=grafico.experimento, x=grafico.cv_f1, orientation="h", name="F1 CV (média ± desvio)",
                error_x=dict(type="data", array=grafico.cv_f1_std), marker_color="#4C78A8")
    fig.add_scatter(y=grafico.experimento, x=grafico.cv_precision, mode="markers", name="Precisão CV",
                    marker=dict(symbol="diamond", size=10, color="#F58518"))
    fig.add_scatter(y=grafico.experimento, x=grafico.cv_recall, mode="markers", name="Revocação CV",
                    marker=dict(symbol="circle", size=10, color="#54A24B"))
    fig.update_layout(xaxis_title="Valor na validação cruzada", yaxis_title=None, height=420,
                      xaxis_range=[0, 1.05], legend=dict(orientation="h", y=-0.2), margin=dict(l=10, r=10, t=10))
    st.plotly_chart(fig, width="stretch")

    if treino_teste.experimento.nunique() < len(comparacao):
        st.caption("As métricas de treino e teste foram publicadas só para as combinações em "
                   "reports/tabelas/metricas_treino_teste.csv; as demais aparecem em branco.")

    # Detalhe de uma combinação no teste
    st.header("Detalhe no conjunto de teste")
    disponiveis = [e for e in comparacao.experimento
                   if ((predicoes.experimento == e) & (predicoes.conjunto == "teste")).any()]
    escolhido = st.selectbox("Combinação", disponiveis,
                             index=disponiveis.index(vencedor) if vencedor in disponiveis else 0,
                             key="combinacao")
    teste = predicoes.loc[(predicoes.experimento == escolhido) & (predicoes.conjunto == "teste")]
    real, score = teste.real.to_numpy(), teste.score.to_numpy()

    # O SVM publica a função de decisão (corte natural em 0), não uma probabilidade.
    if score.min() < 0 or score.max() > 1:
        limiar = st.slider("Limiar sobre a função de decisão (churn quando o escore passa do limiar)",
                           float(np.floor(score.min())), float(np.ceil(score.max())), 0.0, 0.05,
                           key="limiar_decisao")
        st.caption("Esta combinação publica a distância à margem do SVM, não uma probabilidade; "
                   "o limiar padrão equivalente é 0.")
    else:
        limiar = st.slider("Limiar de decisão (churn quando o escore passa do limiar)",
                           0.05, 0.95, 0.50, 0.01, key="limiar")
    m = metricas_no_limiar(real, score, limiar)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precisão", fmt(m["precisao"]))
    c2.metric("Revocação", fmt(m["revocacao"]))
    c3.metric("F1", fmt(m["f1"]))
    c4.metric("Clientes no teste", len(teste))

    col_matriz, col_roc, col_pr = st.columns(3)
    with col_matriz:
        matriz = np.array([[m["vn"], m["fp"]], [m["fn"], m["vp"]]])
        rotulos = np.array([[f"VN<br>{m['vn']}", f"FP<br>{m['fp']}"], [f"FN<br>{m['fn']}", f"VP<br>{m['vp']}"]])
        fig = go.Figure(go.Heatmap(z=matriz, x=["Previsto: fica", "Previsto: churn"],
                                   y=["Real: fica", "Real: churn"], text=rotulos, texttemplate="%{text}",
                                   colorscale="Blues", showscale=False))
        fig.update_layout(title=f"Matriz de confusão (limiar {fmt(limiar, 2)})", yaxis_autorange="reversed",
                          height=360, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width="stretch")
    with col_roc:
        fpr, tpr, cortes = roc_curve(real, score)
        fig = go.Figure()
        fig.add_scatter(x=fpr, y=tpr, mode="lines", name="ROC", line=dict(color="#4C78A8"))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), showlegend=False)
        taxa_fp = m["fp"] / (m["fp"] + m["vn"]) if m["fp"] + m["vn"] else 0.0
        fig.add_scatter(x=[taxa_fp], y=[m["revocacao"]], mode="markers", name="Limiar atual",
                        marker=dict(size=11, color="#E45756"))
        fig.update_layout(title=f"Curva ROC (AUC {fmt(auc(fpr, tpr))})", xaxis_title="Taxa de falsos positivos",
                          yaxis_title="Revocação", height=360, showlegend=False, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width="stretch")
    with col_pr:
        precisao, revocacao, _ = precision_recall_curve(real, score)
        fig = go.Figure()
        fig.add_scatter(x=revocacao, y=precisao, mode="lines", name="Precisão-revocação", line=dict(color="#4C78A8"))
        fig.add_hline(y=real.mean(), line_dash="dash", line_color="gray",
                      annotation_text=f"proporção de churn {fmt(real.mean(), 2)}")
        fig.add_scatter(x=[m["revocacao"]], y=[m["precisao"]], mode="markers", name="Limiar atual",
                        marker=dict(size=11, color="#E45756"))
        fig.update_layout(title="Curva precisão-revocação", xaxis_title="Revocação", yaxis_title="Precisão",
                          height=360, showlegend=False, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width="stretch")

    st.caption("O ponto vermelho nas curvas marca o limiar escolhido no controle acima.")

    # Relatório por classe
    st.subheader("Relatório por classe")
    conjunto = st.radio("Conjunto", ["teste", "treino"], horizontal=True, key="conjunto_classes")
    por_classe = classes.loc[(classes.experimento == escolhido) & (classes.conjunto == conjunto)]
    if por_classe.empty:
        st.warning("Relatório por classe não publicado para esta combinação.")
    else:
        nomes = {"0": "0 (fica)", "1": "1 (churn)", "macro avg": "média macro",
                 "weighted avg": "média ponderada", "accuracy": "acurácia"}
        por_classe = por_classe.assign(classe_ou_media=por_classe.classe_ou_media.astype(str).map(lambda v: nomes.get(v, v)))
        st.dataframe(
            por_classe.drop(columns=["experimento", "conjunto"]).rename(columns={
                "classe_ou_media": "Classe", "precision": "Precisão", "recall": "Revocação",
                "f1-score": "F1", "support": "Suporte"}),
            hide_index=True, width="stretch",
            column_config={"Precisão": st.column_config.NumberColumn(format="%.3f"),
                           "Revocação": st.column_config.NumberColumn(format="%.3f"),
                           "F1": st.column_config.NumberColumn(format="%.3f"),
                           "Suporte": st.column_config.NumberColumn(format="%d")},
        )
        st.caption("Valores no limiar padrão de 0,5, como publicados pelo pipeline.")

    # Faixas de risco do modelo final
    st.header("Faixas de risco do modelo final no teste")
    final = carteira().query("conjunto == 'teste'")
    resumo = (final.groupby("faixa", observed=False)
                   .agg(clientes=("row_id", "size"), churn_real=("churn", "sum"))
                   .reindex(ORDEM_FAIXAS, fill_value=0).reset_index())
    resumo["taxa_churn"] = resumo.churn_real / resumo.clientes.where(resumo.clientes > 0)
    resumo["rotulo"] = [f"{f} ({CORTES_FAIXA[f]})" for f in resumo.faixa]

    col_a, col_b = st.columns([3, 2])
    with col_a:
        fig = px.histogram(final, x="escore", color="faixa", pattern_shape="desfecho", nbins=40,
                           category_orders={"faixa": ORDEM_FAIXAS, "desfecho": ["Permaneceu", "Cancelou"]},
                           labels={"escore": "Escore de churn", "faixa": "Faixa", "desfecho": "Desfecho real"},
                           color_discrete_map=CORES_FAIXA)
        for corte in (CORTE_BAIXO, CORTE_ALTO):
            fig.add_vline(x=corte, line_dash="dash", line_color="gray")
        fig.update_layout(yaxis_title="Clientes", height=360, margin=dict(l=10, r=10, t=10))
        st.plotly_chart(fig, width="stretch")
    with col_b:
        st.dataframe(
            resumo[["rotulo", "clientes", "churn_real", "taxa_churn"]].rename(columns={
                "rotulo": "Faixa", "clientes": "Clientes", "churn_real": "Churn real", "taxa_churn": "Taxa de churn"}),
            hide_index=True, width="stretch",
            column_config={"Taxa de churn": st.column_config.NumberColumn(format="%.3f")},
        )
        st.caption(f"Modelo final: {vencedor}. As linhas tracejadas marcam os cortes 0,30 e 0,65; "
                   "as barras hachuradas são clientes que cancelaram. "
                   "Como o escore não é calibrado, a taxa de churn observada em cada faixa é a melhor "
                   "referência de risco real.")


main()
