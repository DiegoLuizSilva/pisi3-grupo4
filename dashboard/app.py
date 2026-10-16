"""Painel de análise do ChurnGuard (PISI3).

Rodar a partir da raiz do repositório:
    streamlit run dashboard/app.py

Cada página é um arquivo em dashboard/paginas/, com um dono. Enquanto o
arquivo de uma página não existir, o menu mostra um aviso de "em construção"
no lugar dela; quando o dono fizer o commit, a página entra sozinha.
"""

import sys
from pathlib import Path

import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))  # permite "from dashboard.dados import ..." nas páginas

from dashboard.dados import metadados  # noqa: E402

PASTA_PAGINAS = Path(__file__).resolve().parent / "paginas"

st.set_page_config(
    page_title="ChurnGuard | Painel de análise",
    page_icon=":material/monitoring:",
    layout="wide",
)

# (arquivo, título no menu, ícone, dono) na ordem do raciocínio do artigo.
SECOES = {
    "Apresentação": [
        ("inicio.py", "Início", ":material/home:", "Lucas Guerra"),
        ("base_de_dados.py", "Base de dados", ":material/table_chart:", "Pedro Poroca"),
    ],
    "Análise": [
        ("visao_geral.py", "Visão geral (BI)", ":material/dashboard:", "Pedro Poroca"),
        ("exploratoria.py", "Exploratória", ":material/query_stats:", "Artur Rio"),
        ("perfis.py", "Perfis", ":material/groups:", "Arthur Schmitz"),
    ],
    "Modelo": [
        ("modelo.py", "Modelo", ":material/model_training:", "Diego Luiz"),
        ("explicabilidade.py", "Explicabilidade", ":material/psychology:", "Diego Luiz"),
    ],
}


def _em_construcao(titulo: str, dono: str):
    def pagina():
        st.title(titulo)
        st.info(f"Página em construção. Responsável: {dono}.", icon=":material/construction:")

    return pagina


def _montar_paginas() -> dict:
    menu = {}
    for secao, paginas in SECOES.items():
        menu[secao] = []
        for arquivo, titulo, icone, dono in paginas:
            caminho = PASTA_PAGINAS / arquivo
            if caminho.exists():
                pagina = st.Page(caminho, title=titulo, icon=icone,
                                 default=(arquivo == "inicio.py"))
            else:
                pagina = st.Page(_em_construcao(titulo, dono), title=titulo, icon=icone,
                                 url_path=Path(arquivo).stem,
                                 default=(arquivo == "inicio.py"))
            menu[secao].append(pagina)
    return menu


navegacao = st.navigation(_montar_paginas())

with st.sidebar:
    st.markdown("**ChurnGuard**")
    st.caption("Predição de churn em telecomunicações · Iranian Churn Dataset (UCI)")
    try:
        meta = metadados("model_metadata")
        st.caption(f"Modelo: {meta['vencedor']} · F1 de churn em validação cruzada: "
                   f"{meta['cv_f1']:.3f}".replace(".", ","))
    except (FileNotFoundError, KeyError):
        pass
    st.caption("Grupo 4 · PISI3 · BSI/UFRPE")

navegacao.run()