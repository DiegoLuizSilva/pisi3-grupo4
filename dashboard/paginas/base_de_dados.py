
"""Página Base de dados — ChurnGuard.

Documenta o Iranian Churn Dataset, sua qualidade,
estatísticas descritivas e decisões de pré-processamento.

Os dados são lidos exclusivamente de dashboard.dados.
Nenhuma transformação altera a base original.
"""

import pandas as pd
import streamlit as st

from dashboard.dados import (
    ROTULOS,
    PLANO,
    SITUACAO,
    SIM_NAO,
    DESFECHO,
    base_bruta,
    metadados,
)


def numero_br(valor):
    return f"{valor:,.0f}".replace(",", ".")


def dicionario_colunas(df):
    """Dicionário das 14 colunas originais do dataset."""

    definicoes = [
        (
            "Call  Failure", "call_failure", "Discreta",
            "Falhas", "Inteiros não negativos",
            "Quantidade de falhas de chamadas."
        ),
        (
            "Complains", "complains", "Binária",
            "Não se aplica", "0 = Não; 1 = Sim",
            "Indica se existe reclamação registrada."
        ),
        (
            "Subscription  Length", "subscription_length",
            "Discreta", "Meses", "Inteiros positivos",
            "Tempo de assinatura do cliente."
        ),
        (
            "Charge  Amount", "charge_amount", "Ordinal",
            "Faixa de cobrança", "Inteiros de 0 a 10",
            "Categoria ordenada de cobrança, não valor em reais."
        ),
        (
            "Seconds of Use", "seconds_of_use", "Discreta",
            "Segundos", "Inteiros não negativos",
            "Tempo total de utilização."
        ),
        (
            "Frequency of use", "frequency_of_use", "Discreta",
            "Ocorrências", "Inteiros não negativos",
            "Frequência de uso registrada na base."
        ),
        (
            "Frequency of SMS", "frequency_of_sms", "Discreta",
            "Mensagens", "Inteiros não negativos",
            "Frequência de mensagens SMS."
        ),
        (
            "Distinct Called Numbers", "distinct_called_numbers",
            "Discreta", "Números", "Inteiros não negativos",
            "Quantidade de números distintos chamados."
        ),
        (
            "Age Group", "age_group", "Ordinal",
            "Código de faixa", "1, 2, 3, 4, 5",
            "Grupo etário, redundante com age nesta base."
        ),
        (
            "Tariff Plan", "tariff_plan", "Categórica binária",
            "Não se aplica", "1 = Pré-pago; 2 = Contrato",
            "Modalidade do plano contratado."
        ),
        (
            "Status", "status", "Categórica binária",
            "Não se aplica", "1 = Ativa; 2 = Inativa",
            "Situação registrada da linha."
        ),
        (
            "Age", "age", "Discreta",
            "Anos", "15, 25, 30, 45, 55 nesta base",
            "Idade codificada do cliente."
        ),
        (
            "Customer Value", "customer_value", "Quantitativa",
            "Unidade não informada", "Valores não negativos",
            "Indicador de valor do cliente; moeda não especificada."
        ),
        (
            "Churn", "churn", "Binária (alvo)",
            "Não se aplica", "0 = Permaneceu; 1 = Cancelou",
            "Variável-alvo de cancelamento."
        ),
    ]

    registros = []

    for original, codigo, natureza, unidade, dominio, descricao in definicoes:
        registros.append({
            "Nome original": original,
            "Nome no código": codigo,
            "Tipo de dado": str(df[codigo].dtype),
            "Natureza": natureza,
            "Unidade": unidade,
            "Domínio": dominio,
            "Descrição": descricao,
        })

    return pd.DataFrame(registros)


def mostrar_indicadores(df):
    total = len(df)
    ausentes = int(df.isna().sum().sum())
    duplicatas = int(df.duplicated().sum())

    minimo_cobranca = int(df["charge_amount"].min())
    maximo_cobranca = int(df["charge_amount"].max())

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Registros",
        numero_br(total),
        help="Quantidade de linhas do CSV original."
    )

    c2.metric(
        "Cópias completas",
        numero_br(duplicatas),
        help="Linhas excedentes idênticas nas 14 colunas."
    )

    c3.metric(
        "Valores ausentes",
        numero_br(ausentes),
        help="Total de células nulas nas 14 colunas."
    )

    c4.metric(
        "Faixa de cobrança",
        f"{minimo_cobranca}–{maximo_cobranca}",
        help="Charge Amount é uma variável ordinal, não monetária."
    )

    criterios = {
        "3.150 registros": total == 3150,
        "300 cópias completas": duplicatas == 300,
        "0 valores ausentes": ausentes == 0,
        "Cobrança de 0 a 10": (
            minimo_cobranca == 0
            and maximo_cobranca == 10
            and df["charge_amount"].between(0, 10).all()
        ),
    }

    if all(criterios.values()):
        st.success(
            "Os quatro números de qualidade conferem com "
            "os valores documentados na Seção 4 do artigo."
        )
    else:
        divergencias = [
            nome for nome, correto in criterios.items()
            if not correto
        ]
        st.error(
            "Divergências em relação à Seção 4: "
            + ", ".join(divergencias)
        )


def resumo_estatistico(df):
    """Estatísticas sobre as variáveis quantitativas e ordinal de cobrança."""

    colunas = [
        "call_failure",
        "subscription_length",
        "charge_amount",
        "seconds_of_use",
        "frequency_of_use",
        "frequency_of_sms",
        "distinct_called_numbers",
        "age",
        "customer_value",
    ]

    resumo = df[colunas].describe().T

    resumo = resumo.rename(columns={
        "count": "Quantidade",
        "mean": "Média",
        "std": "Desvio padrão",
        "min": "Mínimo",
        "25%": "Q1 (25%)",
        "50%": "Mediana",
        "75%": "Q3 (75%)",
        "max": "Máximo",
    })

    resumo.index = [
        ROTULOS.get(coluna, coluna)
        for coluna in resumo.index
    ]

    resumo.index.name = "Atributo"

    st.dataframe(
        resumo.round(2).reset_index(),
        hide_index=True,
        width="stretch",
    )

    st.caption(
        "Estatísticas descritivas calculadas sobre os 3.150 "
        "registros originais. A cobrança é ordinal: sua média "
        "descreve códigos de faixa, não dinheiro."
    )


def distribuicao_categorica(df):
    """Apresenta frequências das variáveis categóricas."""

    colunas = {
        "Reclamação": "complains",
        "Plano": "tariff_plan",
        "Situação da linha": "status",
        "Faixa etária": "age_group",
        "Churn": "churn",
    }

    selecionada = st.selectbox(
        "Escolha uma variável",
        options=list(colunas.keys()),
    )

    coluna = colunas[selecionada]

    if coluna == "complains":
        rotulos = SIM_NAO
    elif coluna == "tariff_plan":
        rotulos = PLANO
    elif coluna == "status":
        rotulos = SITUACAO
    elif coluna == "churn":
        rotulos = DESFECHO
    else:
        idades = df.groupby("age_group")["age"].unique()
        rotulos = {
            grupo: f"Grupo {grupo} ({', '.join(map(str, valores))} anos)"
            for grupo, valores in idades.items()
        }

    contagem = df[coluna].value_counts().sort_index()

    distribuicao = pd.DataFrame({
        "Categoria": [
            rotulos.get(codigo, str(codigo))
            for codigo in contagem.index
        ],
        "Registros": contagem.values,
        "Proporção (%)": (
            100 * contagem.values / len(df)
        ).round(2),
    })

    st.dataframe(
        distribuicao,
        hide_index=True,
        width="stretch",
    )

    st.bar_chart(
        distribuicao,
        x="Categoria",
        y="Registros",
    )


def qualidade_dados(df):
    """Auditoria de ausentes, unicidade e repetições."""

    ausentes = df.isna().sum()

    qualidade = pd.DataFrame({
        "Coluna": df.columns,
        "Valores ausentes": [
            int(ausentes[coluna])
            for coluna in df.columns
        ],
        "Ausentes (%)": [
            round(100 * ausentes[coluna] / len(df), 2)
            for coluna in df.columns
        ],
        "Valores distintos": [
            int(df[coluna].nunique(dropna=True))
            for coluna in df.columns
        ],
    })

    st.dataframe(
        qualidade,
        hide_index=True,
        width="stretch",
    )

    duplicatas_completas = int(df.duplicated().sum())
    linhas_repetidas = int(
        df.duplicated(keep=False).sum()
    )
    duplicatas_preditores = int(
        df.drop(columns="churn").duplicated().sum()
    )

    st.markdown("#### Registros repetidos")

    c1, c2, c3 = st.columns(3)

    c1.metric("Cópias completas excedentes", duplicatas_completas)
    c2.metric("Linhas em grupos repetidos", linhas_repetidas)
    c3.metric("Cópias dos preditores", duplicatas_preditores)

    st.info(
        "As 300 cópias completas são registros excedentes "
        "idênticos nas 14 colunas. Elas foram preservadas. "
        "Ao considerar apenas os 13 preditores, desconsiderando "
        "Churn, a contagem de cópias pode ser diferente."
    )

    st.caption(
        "A identificação de grupos de clientes com preditores "
        "idênticos é utilizada na separação treino/teste para "
        "evitar que perfis iguais apareçam nos dois conjuntos."
    )


def decisoes_preprocessamento():
    st.markdown(
        """
**1. Padronização dos nomes das colunas**

Os nomes originais foram convertidos para `snake_case`, com
remoção de espaços extras e uso de letras minúsculas.
Por exemplo, `Call  Failure` passou a `call_failure`.

**2. Tratamento de valores ausentes**

Não foi necessária imputação, pois o dataset não apresenta
valores ausentes.

**3. Preservação de duplicatas e valores extremos**

As 300 cópias completas foram preservadas porque registros
idênticos podem representar clientes diferentes. Valores
extremos plausíveis também foram mantidos, sem remoção,
truncamento ou winsorização.

**4. Seleção dos atributos**

O alvo é `churn`. Dos 13 preditores originais, `age_group`
foi excluído da entrada dos classificadores porque é
completamente determinado por `age` nesta base. Restaram
12 atributos de entrada.

**5. Codificação das variáveis binárias**

`complains` já utiliza 0 e 1. As variáveis `tariff_plan`
e `status`, originalmente codificadas com 1 e 2,
foram transformadas para 0 e 1 na entrada dos modelos:

- `tariff_plan`: 1 → 0; 2 → 1.
- `status`: 1 → 0; 2 → 1.

`charge_amount` foi mantido como atributo ordinal,
com valores de 0 a 10. Não representa cobrança monetária.

**6. Separação entre treino e teste**

Foi utilizado `StratifiedGroupKFold`, com semente 42
e agrupamento por combinações idênticas dos preditores
originais. O alvo não participou da construção dos grupos.
Assim, os perfis repetidos não foram compartilhados entre
treino e teste.

**7. Escalonamento e balanceamento**

Foram comparadas alternativas de escalonamento, como
`StandardScaler` e `MinMaxScaler`, conforme o algoritmo.
As transformações são ajustadas dentro dos pipelines de
treinamento, evitando vazamento do teste.

A comparação supervisionada incluiu ausência de
balanceamento, SMOTENC e subamostragem aleatória.
A reamostragem ocorreu somente nas partições de
treinamento dos folds.
"""
    )

    try:
        meta = metadados("model_metadata")
        st.caption(
            "Partição publicada: "
            f"{meta['n_treino']} registros de treino e "
            f"{meta['n_teste']} de teste. "
            f"Total: {meta['n_treino'] + meta['n_teste']}."
        )
    except (FileNotFoundError, KeyError):
        st.caption(
            "Os tamanhos da partição estão documentados "
            "nos metadados do modelo final."
        )


def main():
    st.title("Base de dados")

    st.markdown(
        "Documentação e análise de qualidade do "
        "**Iranian Churn Dataset**, utilizado no ChurnGuard. "
        "Esta página apresenta os dados originais e as decisões "
        "de preparação descritas na Seção 4 do artigo."
    )

    try:
        base = base_bruta()
    except (FileNotFoundError, ValueError) as erro:
        st.error(f"Erro ao carregar a base: {erro}")
        st.stop()

    # row_id é identificador auxiliar criado pelo carregador.
    # Não faz parte das 14 colunas originais.
    df = base.drop(columns=["row_id"])

    st.subheader("Visão geral da qualidade")
    mostrar_indicadores(df)

    st.divider()

    st.subheader("Dicionário de dados")
    st.caption(
        "Nomes originais preservados no dicionário; nomes "
        "normalizados usados internamente pelo projeto."
    )
    st.dataframe(
        dicionario_colunas(df),
        hide_index=True,
        width="stretch",
    )

    st.divider()

    st.subheader("Resumo estatístico")
    resumo_estatistico(df)

    st.divider()

    st.subheader("Distribuição das variáveis categóricas")
    distribuicao_categorica(df)

    st.divider()

    st.subheader("Ausentes, unicidade e duplicatas")
    qualidade_dados(df)

    st.divider()

    st.subheader("Decisões de pré-processamento")
    decisoes_preprocessamento()

    st.divider()

    st.caption(
        "Fonte: Iranian Churn Dataset (UCI). "
        "Os indicadores de qualidade são recalculados "
        "diretamente sobre o arquivo original."
    )


main()
