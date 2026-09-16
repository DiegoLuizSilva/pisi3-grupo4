# ChurnGuard — Machine Learning, Análise Exploratória e Dashboards

Este repositório reúne a parte de ciência de dados do projeto **ChurnGuard**: carregamento e validação dos dados, análise exploratória (EDA), pré-processamento e, futuramente, os notebooks/dashboards de clusterização, classificação e explicabilidade (SHAP) que alimentam o aplicativo mobile.

> Este repositório é a contraparte, para a disciplina de Machine Learning, do projeto ChurnGuard desenvolvido também na disciplina de Engenharia de Software. Aqui ficam apenas os artefatos de dados e modelagem — o app React Native e a API vivem no outro repositório.

---

## 📁 Estrutura de pastas

```
.
├── data/
│   └── raw/
│       └── iranian_churn.csv       # dataset original, sem alterações
├── notebooks/
│   ├── 01_carregamento.ipynb       # leitura e validação do CSV
│   └── 02_pre_processamento.ipynb  # limpeza, encoding e split treino/teste
├── src/ (ou scripts/)
│   └── eda_visual.py               # gera gráficos e tabelas de EDA em lote
├── reports/                        # gerado automaticamente pelo eda_visual.py
│   ├── figuras/                    # histogramas, boxplots, matriz de correlação...
│   ├── matriz_correlacao.csv
│   └── comparacao_medias.csv
├── dashboards/                     # (a anexar) visualizações interativas / apresentação dos resultados
└── requirements.txt
```

---

## 📊 Sobre o dataset

- **Fonte:** [Iranian Churn Dataset — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset)
- **Tamanho:** 3.150 linhas × 14 colunas (13 preditores + 1 alvo)
- **Alvo:** `Churn` — binário (`0` = permaneceu, `1` = cancelou)
- **Sem valores nulos**
- **Desbalanceamento:** 84,29% dos clientes não cancelaram vs. 15,71% que cancelaram (razão ≈ 5,4:1). Por isso, acurácia sozinha não é usada para escolher modelo — a avaliação prioriza precisão, recall e F1-score da classe `churn`.

| Coluna original | Descrição |
|---|---|
| Call Failure | Número de falhas de chamada |
| Complains | Se o cliente registrou reclamação (0/1) |
| Subscription Length | Tempo de assinatura (meses) |
| Charge Amount | Faixa ordinal de cobrança (0 a 10) |
| Seconds of Use | Total de segundos de uso |
| Frequency of use | Frequência de uso do serviço |
| Frequency of SMS | Frequência de uso de SMS |
| Distinct Called Numbers | Quantidade de números distintos chamados |
| Age Group | Faixa etária (redundante com `Age`, ver nota abaixo) |
| Tariff Plan | Tipo de plano (1 = pré-pago, 2 = contrato) |
| Status | Status da linha (1 = ativo, 2 = inativo) |
| Age | Idade do cliente |
| Customer Value | Valor estimado do cliente |
| Churn | Alvo: cancelou ou não |

---

## 📓 Notebooks

### `01_carregamento.ipynb`
Carrega o CSV de `data/raw/iranian_churn.csv` e valida o formato básico: dimensões (3.150 × 14), tipos de dado de cada coluna e uma prévia das primeiras linhas.

### `02_pre_processamento.ipynb`
Documenta e justifica todas as decisões de preparação dos dados antes da modelagem:

- **Nomes de colunas** normalizados para `snake_case` (ex.: `Call  Failure` → `call_failure`).
- **Sem valores nulos** — nenhuma imputação necessária.
- **Duplicatas:** 300 linhas completas se repetem, mas são preservadas, pois podem representar clientes distintos com o mesmo perfil de uso.
- **`age_group` removida** das variáveis do modelo por ser completamente determinada por `age` (cada idade pertence a uma única faixa) — não acrescenta informação.
- **Encoding:** `complains`, `tariff_plan` e `status` são binárias e foram mapeadas explicitamente para `0/1` (equivalente a label encoding de duas categorias); one-hot encoding foi descartado por ser redundante nesse caso. `charge_amount` foi mantida como variável ordinal.
- **Separação treino/teste:** feita com `StratifiedGroupKFold`, agrupando por perfil completo de preditores. Isso evita que clientes com o mesmo perfil (potenciais duplicatas) apareçam simultaneamente em treino e teste, prevenindo vazamento de dados. A proporção de churn é preservada entre os conjuntos (diferença < 1%).
- **Escalonamento:** `StandardScaler` e `MinMaxScaler` foram comparados lado a lado. A escolha final entre eles depende do algoritmo (SVM e KNN são sensíveis à escala; Random Forest não). Os transformadores são ajustados (`fit`) somente dentro do pipeline de treino, nunca antes do split.
- **Estratégias de balanceamento previstas para a próxima etapa:** baseline sem balanceamento, SMOTE e undersampling — a serem aplicadas apenas ao conjunto de treino.

---

## 📈 Script de EDA — `eda_visual.py`

Script standalone que gera uma bateria de análises exploratórias a partir do CSV bruto.

### Como rodar
```bash
python eda_visual.py data/raw/iranian_churn.csv
```
Se nenhum caminho for passado, usa `ml/data/raw/iranian_churn.csv` como padrão.

### O que ele gera
Salvo em `reports/` e `reports/figuras/`:

1. **Distribuição das variáveis numéricas** — histogramas com média e mediana (`histogramas.png`) e boxplots por classe de `Churn` (`boxplots.png`), além da assimetria (skewness) de cada variável.
2. **Matriz de correlação** (Spearman) entre preditores e o alvo, com heatmap (`matriz_correlacao.png`) e tabela (`matriz_correlacao.csv`), incluindo a identificação de pares colineares (correlação > 0,6).
3. **Comparação de médias entre quem cancelou e quem ficou** — diferença percentual, tamanho de efeito (Cohen's d) e teste de Mann-Whitney U por variável, com gráfico de barras (`tamanho_efeito.png`) e tabela (`comparacao_medias.csv`).
4. **Variáveis categóricas por taxa de churn** — taxa de cancelamento por categoria (`Complains`, `Tariff Plan`, `Status`, `Age Group`), com gráfico de barras (`churn_categoricas.png`) e teste qui-quadrado de independência (com V de Cramer) impresso no console.

---

## ⚙️ Como configurar o ambiente

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

### Dependências (`requirements.txt`)
```
pandas
scikit-learn
imbalanced-learn
matplotlib
seaborn
shap
plotly
jupyter
```

---

## 🗺️ Próximos passos

- [ ] Notebook de clusterização (KMeans, método do cotovelo, coeficiente de silhueta)
- [ ] Notebook de classificação (Random Forest, SVM, KNN, Regressão Logística)
- [ ] Balanceamento de classes (SMOTE / undersampling) aplicado dentro dos pipelines
- [ ] Explicabilidade com SHAP sobre o modelo final
- [ ] Dashboards com os resultados para apresentação/artigo

---

## 🔗 Relacionado

O aplicativo mobile (React Native/Expo) e a API que consomem o modelo produzido aqui estão no repositório principal do ChurnGuard, na disciplina de Engenharia de Software.