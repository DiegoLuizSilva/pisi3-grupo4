"""Página inicial: o problema, as perguntas de pesquisa e como navegar no painel.

Dono: Lucas Guerra. Todos os números vêm de dashboard/dados.py.
"""

import streamlit as st

from dashboard.dados import CORTE_ALTO, CORTE_BAIXO, EXPERIMENTO_FINAL, carteira, metadados, tabela


def br(valor: float, casas: int = 0) -> str:
    """Número no formato brasileiro: 3.150 / 0,870."""
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "_").replace(".", ",").replace("_", ".")


clientes = carteira()
perfis = tabela("perfil_tamanho")
metricas = tabela("metricas_treino_teste")
modelo = metadados("model_metadata")

total = len(clientes)
cancelaram = int(clientes["churn"].sum())
perfil_critico = perfis.loc[perfis["taxa_churn"].idxmax()]
teste = clientes[clientes["conjunto"] == "teste"]
acertos_teste = int(((teste["churn"] == 1) & (teste["previsto"] == 1)).sum())
cancel_teste = int(teste["churn"].sum())
f1_teste = metricas.query("experimento == @EXPERIMENTO_FINAL and conjunto == 'teste'")["f1_churn"].iloc[0]

st.title("ChurnGuard")
st.markdown("##### Quem está prestes a cancelar, e por quê")

st.markdown(
    "A operadora costuma descobrir o cancelamento depois que ele acontece: os relatórios "
    "dizem quantos clientes saíram, mas não quem está para sair nem o motivo. O ChurnGuard "
    "combina **agrupamento**, **classificação** e **explicabilidade por SHAP** sobre o "
    "Iranian Churn Dataset para apontar os clientes em risco e os fatores por trás de cada "
    "predição. Este painel reapresenta, de forma explorável, os resultados do artigo."
)

neutro = {"delta_color": "off", "delta_arrow": "off", "border": True}
c1, c2, c3, c4 = st.columns(4)
c1.metric("Clientes na base", br(total), "Iranian Churn Dataset", **neutro)
c2.metric("Cancelaram", br(cancelaram), f"{br(100 * cancelaram / total, 2)}% da base", **neutro)
c3.metric(
    "Num único perfil",
    f"{br(perfil_critico['cancelaram'])} de {br(cancelaram)}",
    f"taxa de churn {br(100 * perfil_critico['taxa_churn'], 2)}%",
    help=f"Cancelamentos no perfil “{perfil_critico['perfil']}”, o de maior taxa "
         "entre os três grupos do K-Means (Seção 7.1).",
    **neutro,
)
c4.metric(
    "Detectados no teste",
    f"{acertos_teste} de {cancel_teste}",
    f"F1 de churn {br(f1_teste, 3)}",
    help=f"Cancelamentos do conjunto de teste reservado que o modelo {EXPERIMENTO_FINAL} "
         "identificou (Seção 7.2).",
    **neutro,
)

st.divider()

st.subheader("Perguntas de pesquisa")
p1, p2, p3 = st.columns(3)
with p1:
    st.markdown("**PP1 · Perfis**")
    st.markdown(
        "Que perfis de clientes emergem do agrupamento não supervisionado dos dados de uso, "
        "e esses perfis têm propensões distintas ao cancelamento?"
    )
    st.caption("K-Means · Seção 7.1 · páginas Exploratória e Perfis")
with p2:
    st.markdown("**PP2 · Classificação**")
    st.markdown(
        "Qual algoritmo identifica melhor os clientes propensos a cancelar, com métricas "
        "adequadas ao desbalanceamento, e quais atributos mais influenciam as predições?"
    )
    st.caption(f"{modelo['vencedor']} · Seção 7.2 · páginas Modelo e Explicabilidade")
with p3:
    st.markdown("**PP3 · Apresentação**")
    st.markdown(
        "Como apresentar os fatores de cada predição de forma interpretável e associada a "
        "possíveis ações de retenção?"
    )
    st.caption("Aplicativo ChurnGuard · Seção 6 · páginas Visão geral e Explicabilidade")

st.divider()

st.subheader("Como navegar")
st.markdown(
    """
| Página | O que mostra |
|---|---|
| **Base de dados** | Dicionário das 14 colunas, qualidade dos dados e decisões de pré-processamento |
| **Visão geral (BI)** | Indicadores da carteira, filtros globais, cruzamento perfil × faixa de risco e lista de clientes |
| **Exploratória** | Distribuições por classe, churn por categoria, correlações e tamanho de efeito |
| **Perfis** | Escolha do número de grupos, tamanho e churn de cada perfil, projeção e detalhamento |
| **Modelo** | As nove combinações de algoritmo e balanceamento, matriz de confusão, curvas e limiar |
| **Explicabilidade** | Importância SHAP, explicação de um cliente e simulador de novo cliente |
"""
)

st.info(
    "O painel só lê resultados já gerados no repositório (dados, tabelas e modelo final) "
    "e não retreina nenhum modelo. O escore exibido para clientes do conjunto de treino é "
    "otimista, porque o modelo já viu esses clientes; o desempenho é medido no conjunto de teste.",
    icon=":material/info:",
)
st.caption(
    f"Faixas de risco: abaixo de {br(CORTE_BAIXO, 2)} baixo; de {br(CORTE_BAIXO, 2)} a "
    f"{br(CORTE_ALTO, 2)} médio; acima de {br(CORTE_ALTO, 2)} alto. São regras de "
    "apresentação, não probabilidades calibradas (Seção 4.5)."
)