# ChurnGuard — Machine Learning, Análise de Dados e Dashboards

Projeto de análise de dados, segmentação de clientes, predição de cancelamento (*customer churn*) e explicabilidade de modelos de Machine Learning, desenvolvido pelo **Grupo 4 para a disciplina de PISI III**.

O ChurnGuard utiliza o **Iranian Churn Dataset**, disponibilizado pelo UCI Machine Learning Repository, para investigar padrões de comportamento associados ao cancelamento de serviços de telecomunicações.

O repositório contempla o pipeline completo de ciência de dados, desde a preparação do dataset até a disponibilização dos resultados em um painel interativo com Streamlit.

## 1. Visão geral

O projeto foi organizado em quatro frentes:

1. **Análise e preparação dos dados:** compreensão do dataset, auditoria de qualidade, análise exploratória, tratamento dos atributos e separação entre treino e teste.
2. **Segmentação de clientes:** aplicação do K-Means para identificar perfis de utilização dos serviços.
3. **Predição de churn:** comparação de classificadores supervisionados e seleção do modelo com melhor F1 da classe de cancelamento em validação cruzada.
4. **Visualização e explicabilidade:** geração de gráficos, relatórios, dashboards e interpretações SHAP para facilitar a compreensão das previsões.

Este repositório corresponde à parte de Machine Learning do ChurnGuard. O aplicativo React Native com Expo e a API FastAPI, desenvolvidos na disciplina de Engenharia de Software, são mantidos em outro repositório.

## 2. Tecnologias utilizadas

| Tecnologia | Finalidade |
|---|---|
| Python | Linguagem principal |
| Pandas e NumPy | Manipulação e análise de dados |
| SciPy | Análise estatística |
| Scikit-learn | Pré-processamento, clusterização, classificação e validação |
| Imbalanced-learn | Balanceamento de classes com SMOTENC e subamostragem |
| Matplotlib e Seaborn | Visualizações estatísticas |
| Plotly | Gráficos e dashboards interativos |
| SHAP | Explicabilidade das previsões |
| Streamlit | Interface web do painel de análise |
| JupyterLab | Desenvolvimento e execução interativa dos notebooks |
| Joblib | Persistência dos modelos treinados |

Os artefatos do modelo registram treinamento com **Python 3.12.3**. Recomenda-se utilizar Python 3.12 e as dependências definidas no projeto para reduzir incompatibilidades.

## 3. Estrutura do projeto

```text
pisi3-grupo4/
├── data/
│   └── raw/
│       └── iranian_churn.csv
│
├── notebooks/
│   ├── 01_carregamento.ipynb
│   ├── 02_pre_processamento.ipynb
│   ├── 03_clusterizacao.ipynb
│   ├── 04_classificacao.ipynb
│   ├── 05_explicabilidade_shap.ipynb
│   └── 07_perfis_complementar.ipynb
│
├── src/
│   ├── common.py
│   ├── revisao_k.py
│   ├── sensibilidade_clusters.py
│   ├── perfis_final.py
│   └── validar_inferencia.py
│
├── models/
│   ├── clusterizacao.joblib
│   └── modelo_final.joblib
│
├── reports/
│   ├── tabelas/
│   ├── figuras/


├── dashboard/
│   ├── app.py
│   ├── dados.py
│   ├── requirements.txt
│   └── paginas/
│       ├── inicio.py
│       ├── base_de_dados.py
│       ├── visao_geral.py
│       ├── exploratoria.py
│       ├── perfis.py
│       ├── modelo.py
│       └── explicabilidade.py
│
├── .streamlit/
│   └── config.toml
│
├── eda_visual.py
├── executar_projeto.py
├── requirements.txt
└── README.md
```

**Observação:** a árvore apresenta a estrutura planejada do painel completo. O sistema identifica automaticamente páginas disponíveis e apresenta um aviso de construção para as que ainda não foram implementadas. As páginas de análise exploratória e perfis seguem pendentes na versão examinada.

### Diretórios principais

- `data/raw/`: dataset original, preservado sem alterações.
- `notebooks/`: desenvolvimento experimental e documentação metodológica.
- `src/`: funções reutilizáveis, análises complementares, relatórios e validações.
- `models/`: modelos treinados e serializados com Joblib.
- `reports/tabelas/`: métricas, predições, metadados e resultados estruturados.
- `reports/figuras/`: gráficos gerados para análise e documentação acadêmica.
- `reports/dashboards/`: relatórios interativos independentes em HTML.
- `dashboard/`: painel web desenvolvido com Streamlit.

## 4. Dataset

### 4.1 Origem

**Iranian Churn Dataset — UCI Machine Learning Repository**

Fonte: https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset

O dataset contém informações de utilização de serviços de telecomunicações, características dos clientes e uma variável que indica a ocorrência de cancelamento.

### 4.2 Características gerais

| Indicador | Valor |
|---|---:|
| Registros | 3.150 |
| Colunas | 14 |
| Atributos explicativos originais | 13 |
| Variável-alvo | `Churn` |
| Clientes que permaneceram | 2.655 |
| Clientes que cancelaram | 495 |
| Taxa observada de churn | 15,71% |
| Valores ausentes | 0 |
| Cópias completas excedentes | 300 |
| Intervalo de `Charge Amount` | 0–10 |

A variável `Churn` é binária:

- `0`: cliente permaneceu.
- `1`: cliente cancelou.

O dataset apresenta desbalanceamento entre as classes, com aproximadamente 84,29% de permanência e 15,71% de cancelamento.

Por esse motivo, a avaliação supervisionada prioriza métricas como precisão, recall e F1-score da classe positiva, em vez de utilizar somente a acurácia.

### 4.3 Dicionário de dados

| Coluna original | Nome no código | Descrição |
|---|---|---|
| Call Failure | `call_failure` | Quantidade de falhas de chamadas |
| Complains | `complains` | Existência de reclamação |
| Subscription Length | `subscription_length` | Tempo de assinatura em meses |
| Charge Amount | `charge_amount` | Faixa ordinal de cobrança, de 0 a 10 |
| Seconds of Use | `seconds_of_use` | Tempo total de uso em segundos |
| Frequency of use | `frequency_of_use` | Frequência de utilização |
| Frequency of SMS | `frequency_of_sms` | Frequência de mensagens SMS |
| Distinct Called Numbers | `distinct_called_numbers` | Quantidade de números distintos chamados |
| Age Group | `age_group` | Categoria etária |
| Tariff Plan | `tariff_plan` | Tipo de plano |
| Status | `status` | Situação da linha |
| Age | `age` | Idade do cliente |
| Customer Value | `customer_value` | Indicador de valor do cliente |
| Churn | `churn` | Cancelamento observado |

**Importante:** `charge_amount` representa uma categoria ordinal de cobrança, não um valor monetário em reais. Da mesma forma, a unidade monetária de `customer_value` não está documentada como BRL.

A página **Base de dados** do Streamlit apresenta o dicionário completo, incluindo tipo de dado, unidade, domínio e estatísticas descritivas.

## 5. Pré-processamento

As decisões de pré-processamento estão documentadas no notebook `02_pre_processamento.ipynb` e implementadas nas rotinas compartilhadas do projeto.

### 5.1 Normalização das colunas

Os nomes originais foram convertidos para `snake_case`, removendo espaços excedentes e padronizando letras minúsculas.

Exemplo:

```text
Call  Failure → call_failure
Subscription  Length → subscription_length
```

### 5.2 Valores ausentes e duplicatas

Não foi necessária imputação, pois nenhuma das 14 colunas possui valores ausentes.

Foram identificadas 300 cópias completas excedentes, que foram preservadas. Registros iguais podem representar clientes diferentes com o mesmo perfil de uso.

As duplicatas também foram consideradas na separação dos conjuntos para evitar compartilhamento de perfis idênticos entre treino e teste.

### 5.3 Seleção e codificação de atributos

O atributo `age_group` foi excluído das entradas dos classificadores, pois é completamente determinado por `age` na base utilizada.

Dessa forma, os classificadores recebem 12 atributos.

As variáveis `tariff_plan` e `status` foram recodificadas:

| Variável | Original | Entrada do modelo |
|---|---|---|
| `tariff_plan` | 1 = Pré-pago | 0 |
| `tariff_plan` | 2 = Contrato | 1 |
| `status` | 1 = Ativa | 0 |
| `status` | 2 = Inativa | 1 |

`complains` já possui representação binária 0/1.

O atributo `charge_amount` foi mantido como ordinal.

### 5.4 Treino e teste

A separação utilizou `StratifiedGroupKFold`, com agrupamento de clientes que possuem combinações idênticas dos preditores originais.

| Conjunto | Registros |
|---|---:|
| Treinamento | 2.517 |
| Teste | 633 |
| Total | 3.150 |

A semente utilizada na separação foi 42.

A estratégia procura preservar a distribuição das classes e impedir que perfis idênticos sejam utilizados simultaneamente para treinamento e teste.

### 5.5 Escalonamento e balanceamento

Foram consideradas alternativas de escalonamento com `StandardScaler`, `MinMaxScaler` e ausência de escala, conforme o classificador.

Para tratar o desbalanceamento, foram comparadas três estratégias:

- Sem balanceamento.
- SMOTENC.
- Subamostragem aleatória (*undersampling*).

Os transformadores e as estratégias de reamostragem são aplicados dentro dos pipelines, respeitando os conjuntos de treinamento das etapas de validação.

Isso evita aplicar transformações ajustadas sobre o conjunto de teste.

## 6. Notebooks

| Notebook | Finalidade |
|---|---|
| `01_carregamento.ipynb` | Carregamento e validação inicial do dataset |
| `02_pre_processamento.ipynb` | Auditoria, preparação e separação dos dados |
| `03_clusterizacao.ipynb` | Segmentação de clientes com K-Means |
| `04_classificacao.ipynb` | Comparação e seleção dos modelos de churn |
| `05_explicabilidade_shap.ipynb` | Interpretação global e local com SHAP |
| `07_perfis_complementar.ipynb` | Análise complementar dos perfis de clientes |

Os notebooks 03 a 06 podem ser executados automaticamente pelo arquivo `executar_projeto.py`.

Os notebooks 01, 02 e 07 não fazem parte dessa execução automática e podem ser abertos individualmente no JupyterLab.

## 7. Clusterização de clientes

A segmentação foi realizada utilizando **K-Means**, sem utilizar `churn` como variável de agrupamento.

A seleção do número de clusters considerou principalmente o coeficiente de silhueta, com apoio do método do cotovelo e de análises complementares.

### Resultado

- Número de clusters: **3**
- Silhueta no treinamento: aproximadamente **0,306**
- PCA de duas componentes: aproximadamente **59,7% da variância**, utilizado para visualização.

Os perfis identificados foram:

| Perfil | Clientes | Taxa de churn |
|---|---:|---:|
| Intensivo em mensagens | 575 | 0,87% |
| Baixo engajamento | 1.875 | 25,92% |
| Intensivo em voz | 700 | 0,57% |

O grupo de baixo engajamento concentrou 486 dos 495 cancelamentos observados na base.

As taxas de churn foram analisadas após a definição dos clusters, não como critério de seleção dos agrupamentos.

Também foram realizadas análises de sensibilidade para verificar o comportamento da segmentação diante de diferentes sementes, escalas, atributos e algoritmos de agrupamento.

## 8. Classificação e predição de churn

### 8.1 Modelos avaliados

A etapa supervisionada comparou três classificadores:

- Random Forest.
- Support Vector Machine (SVM).
- K-Nearest Neighbors (KNN).

A busca considerou combinações de hiperparâmetros, alternativas de pré-processamento e três estratégias de balanceamento.

Foram avaliadas **60 configurações**, totalizando **300 ajustes em validação cruzada de cinco folds**.

A seleção do modelo foi realizada pelo F1 médio da classe churn na validação cruzada agrupada do treinamento, sem utilizar o conjunto de teste para decidir o vencedor.

### 8.2 Modelo selecionado

O modelo escolhido foi:

**Random Forest + SMOTENC**

| Métrica | Resultado |
|---|---:|
| F1 de churn em validação cruzada | 0,8699 |
| F1 de churn no teste | 0,8815 |
| Recall de churn no teste | 0,9300 |
| Precisão de churn no teste | 0,8378 |
| Acurácia no teste | 0,9605 |
| ROC-AUC no teste | 0,9905 |

No conjunto de teste, o modelo identificou 93 dos 100 cancelamentos observados.

O modelo final está disponível em:

```text
models/modelo_final.joblib
```

Os detalhes de treinamento, versões, atributos, parâmetros e hashes estão registrados em:

```text
reports/tabelas/model_metadata.json
```

As métricas representam os resultados experimentais do conjunto utilizado. Não constituem validação temporal ou externa em dados de uma operadora real.

### 8.3 Faixas de risco

Os escores do modelo são organizados nas faixas adotadas pelo projeto:

| Faixa | Regra |
|---|---|
| Risco baixo | Escore inferior a 0,30 |
| Risco médio | Escore entre 0,30 e 0,65, incluindo os limites aplicáveis |
| Risco alto | Escore superior a 0,65 |

Esses limites são utilizados na apresentação dos resultados e seguem o contrato de integração do ChurnGuard.

**Os escores não são probabilidades calibradas.** Além disso, os escores obtidos sobre registros de treinamento podem ser otimistas, pois o modelo já teve contato com esses dados.

As métricas de generalização devem ser interpretadas com base no conjunto de teste.

## 9. Explicabilidade com SHAP

A etapa de explicabilidade utiliza SHAP para interpretar as contribuições dos atributos às saídas do modelo selecionado.

As análises incluem:

- Importância global dos atributos.
- Distribuição das contribuições SHAP.
- Explicação de casos individuais.
- Comparação do escore previsto com o valor de referência.
- Exemplos de clientes com diferentes níveis de risco.

Os resultados experimentais estão disponíveis em:

```text
reports/tabelas/shap_importancia.csv
reports/tabelas/shap_valores.csv
reports/tabelas/shap_amostra.csv
reports/tabelas/shap_metadata.json
```

A implementação de inferência também utiliza explicações SHAP do estimador Random Forest para apoiar a integração com a API.

**Limitação:** SHAP explica como o modelo utiliza os atributos, mas não demonstra causalidade. Uma contribuição positiva não significa que modificar aquele atributo necessariamente reduzirá o churn.

## 10. Painel interativo — Streamlit

O ChurnGuard possui um painel de análise em Streamlit, organizado em páginas independentes.

A aplicação é iniciada por `dashboard/app.py`, enquanto `dashboard/dados.py` centraliza a leitura dos dados, tabelas, metadados e modelos.

As páginas utilizam os resultados já publicados pelo pipeline. A navegação normal pelo painel não executa novamente o treinamento.

### 10.1 Páginas do painel

| Página | Funcionalidades | Situação |
|---|---|---|
| Início | Contextualização do problema, objetivos e orientação de uso | Implementada |
| Base de dados | Dicionário, estatísticas, qualidade e pré-processamento | Implementada nas alterações recentes |
| Visão geral (BI) | Indicadores, filtros, matrizes e detalhamento da carteira | Implementada nas alterações recentes |
| Exploratória | Análises e visualizações exploratórias interativas | Pendente na versão examinada |
| Perfis | Visualização dos agrupamentos e características dos clientes | Pendente na versão examinada |
| Modelo | Comparação e avaliação dos classificadores | Implementada |
| Explicabilidade | Importância de atributos, SHAP e simulador de cliente | Implementada |

O arquivo `dashboard/app.py` verifica a existência das páginas configuradas. Quando um arquivo não existe, a navegação apresenta um aviso de que a página está em construção.

### 10.2 Base de dados

A página `dashboard/paginas/base_de_dados.py` apresenta:

- Indicadores de qualidade do dataset.
- Dicionário das 14 colunas originais.
- Nomes originais e normalizados.
- Tipos, unidades e domínios.
- Resumo estatístico.
- Distribuições de atributos categóricos.
- Valores ausentes e registros duplicados.
- Decisões de pré-processamento.

Os indicadores são calculados sobre os dados originais e confrontados com os valores documentados no artigo.

### 10.3 Visão geral (BI)

A página `dashboard/paginas/visao_geral.py` reúne quatro indicadores:

1. Quantidade de clientes.
2. Churn observado.
3. Clientes classificados em risco alto.
4. Valor em risco.

O painel possui filtros globais por situação da linha, plano, reclamação, faixa etária, perfil, faixa de risco e conjunto (treino ou teste).

Também apresenta:

- Distribuição de clientes por faixa de risco.
- Churn observado por perfil.
- Matriz perfil × faixa de risco.
- Tabela detalhada ordenada por escore de churn.
- Exportação da tabela filtrada.

Todas as visualizações e indicadores são recalculados quando os filtros são modificados.

O **valor em risco** é representado pela soma do atributo `customer_value` dos clientes classificados como risco alto. Trata-se de um indicador relativo do dataset, e não de uma previsão de perdas em reais.

A página também alerta sobre a interpretação otimista dos escores no conjunto de treinamento e recomenda o conjunto de teste para avaliações de generalização.

### 10.4 Modelo

A página de modelos apresenta:

- Comparação dos classificadores.
- Resultados da validação cruzada.
- Métricas de treinamento e teste.
- Matrizes de confusão.
- Curvas ROC e precisão-revocação.
- Análise do limiar de classificação.

### 10.5 Explicabilidade

A página de explicabilidade reúne as análises SHAP e um simulador que permite informar atributos de um cliente e obter seu escore de risco utilizando o modelo treinado.

O simulador realiza inferência sobre o modelo existente; não executa um novo treinamento.

## 11. Dashboards HTML e relatórios

Além do painel Streamlit, o projeto mantém três dashboards Plotly exportados em HTML:

| Arquivo | Conteúdo |
|---|---|
| `01_segmentacao.html` | Clusters, silhueta, PCA e características dos perfis |
| `02_classificacao.html` | Comparação dos classificadores e métricas do modelo |
| `03_explicabilidade.html` | Análises e visualizações SHAP |

Os arquivos ficam em:

```text
reports/resultados_artigo.md
```

O Streamlit e os HTMLs possuem finalidades complementares: o primeiro permite explorar os resultados em páginas com filtros e controles; os segundos disponibilizam relatórios independentes para consulta e apresentação.

## 12. Instalação e execução

Todos os comandos devem ser executados a partir da raiz do repositório, salvo indicação contrária.

### 12.1 Requisitos

- Python 3.12 recomendado.
- Pip.
- Ambiente virtual Python.
- Dependências declaradas nos arquivos de requisitos.

### 12.2 Criar o ambiente virtual

Linux/macOS:

```bash
python3.12 -m venv venv
source venv/bin/activate
```

Windows (PowerShell):

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
```

Se já existir um ambiente virtual criado com outra versão do Python, ativá-lo não altera sua versão. Nesse caso, crie um novo ambiente com Python 3.12.

### 12.3 Instalar as dependências

Para notebooks, treinamento e scripts:

```bash
python -m pip install -r requirements.txt
```

Para usar também o painel Streamlit:

```bash
python -m pip install -r dashboard/requirements.txt
```

O ambiente virtual deve permanecer ativado ao executar os comandos do projeto.

### 12.4 Abrir os notebooks

```bash
jupyter lab
```

O JupyterLab normalmente é disponibilizado em:

```text
http://localhost:8888
```

Pela interface, é possível abrir os arquivos da pasta `notebooks/` e executar ou inspecionar suas células.

### 12.5 Executar o pipeline automaticamente

```bash
python executar_projeto.py
```

Esse script executa sequencialmente os notebooks:

```text
03_clusterizacao.ipynb
04_classificacao.ipynb
05_explicabilidade_shap.ipynb
06_dashboards.ipynb
```

A execução pode ser demorada, especialmente nas etapas de validação cruzada e explicabilidade.

O script utiliza `nbclient` por padrão, executando as células por meio de um kernel Jupyter sem abrir a interface gráfica.

Como alternativa, existe o modo de execução no próprio processo:

```bash
python executar_projeto.py --in-process
```

Esse modo imprime o andamento por célula e pode ser útil em ambientes nos quais a inicialização do kernel Jupyter apresenta problemas.

**Atenção:** executar o pipeline novamente pode sobrescrever artefatos, resultados e saídas dos notebooks. Não é necessário repetir o treinamento somente para visualizar os resultados publicados.

### 12.6 Abrir o painel Streamlit

```bash
streamlit run dashboard/app.py
```

O painel normalmente fica disponível em:

```text
http://localhost:8501
```

Para apenas consultar os resultados já gerados, basta iniciar o Streamlit, sem executar `executar_projeto.py`.

### 12.7 Regenerar somente os relatórios

Para atualizar os dashboards HTML a partir das tabelas existentes:

```bash
python -c 'from src.dashboards import build_dashboards; build_dashboards()'
```

Para regenerar o relatório experimental:

```bash
python -c 'from src.report import build_report; build_report()'
```

Esses comandos utilizam os resultados já salvos, sem refazer a busca de hiperparâmetros.

## 13. Scripts auxiliares

Além dos notebooks, estão disponíveis os seguintes módulos:

| Script | Responsabilidade |
|---|---|
| `src/common.py` | Carregamento, divisão dos dados e pré-processadores compartilhados |
| `src/revisao_k.py` | Revisão dos critérios de escolha do número de clusters |
| `src/sensibilidade_clusters.py` | Sensibilidade dos agrupamentos a sementes, escalas, atributos e algoritmos |
| `src/perfis_final.py` | Geração de tabelas e figuras finais dos perfis |
| `src/dashboards.py` | Geração dos três dashboards HTML |
| `src/report.py` | Geração do relatório experimental textual |
| `src/validar_inferencia.py` | Validação do modelo e do contrato de integração |
| `eda_visual.py` | Script legado de análise exploratória em lote |

Os módulos auxiliares podem ser utilizados separadamente, conforme a etapa desejada.

Exemplos:

```bash
python -m src.revisao_k
python -m src.sensibilidade_clusters
python -m src.perfis_final
```

Para validar a inferência e gerar casos de referência:

```bash
python -m src.validar_inferencia
```

O validador também oferece opções para comparar os artefatos com o repositório da API e verificar respostas de uma API autenticada, conforme as opções descritas em seu código.

**Observação sobre `eda_visual.py`:** esse script ainda utiliza caminhos legados iniciados por `ml/`. Antes de executá-lo a partir da raiz deste repositório, esses caminhos precisam ser ajustados ou o script deve ser executado a partir de uma estrutura compatível. A pipeline principal de notebooks não depende dele.

## 14. Integração com o aplicativo ChurnGuard

O modelo supervisionado produzido por este repositório é utilizado pelo projeto de Engenharia de Software, composto por:

- Aplicativo React Native com Expo.
- API Python com FastAPI.
- Autenticação e persistência via Firebase.
- Inferência de risco de cancelamento.
- Apresentação dos resultados e histórico de avaliações.

A integração exige consistência entre os atributos, sua ordem, os mapeamentos categóricos, o modelo exportado e as faixas de risco.

O módulo `src/validar_inferencia.py` foi desenvolvido para conferir esses contratos, hashes de artefatos e casos de referência.

Este repositório não contém o código completo do aplicativo mobile ou da API.

## 15. Reprodutibilidade e limitações

O projeto registra sementes, versões das bibliotecas, configurações, métricas e hashes dos artefatos experimentais para facilitar a reprodução e auditoria.

Entretanto, algumas limitações devem ser consideradas:

- Os resultados foram obtidos sobre um único dataset de telecomunicações.
- A avaliação utiliza uma separação agrupada entre treino e teste, mas não representa validação temporal.
- A seleção supervisionada considera um conjunto finito de hiperparâmetros.
- Os escores do modelo não são calibrados.
- Os valores de desempenho no treinamento podem ser otimistas.
- A clusterização identifica agrupamentos descritivos e não relações causais.
- As explicações SHAP refletem associações aprendidas pelo modelo.
- Não há validação externa com dados independentes de uma operadora real.
- Não foi demonstrada experimentalmente a redução de churn decorrente de ações de retenção.
- O campo `customer_value` não deve ser interpretado automaticamente como receita monetária real.

O ChurnGuard fornece uma base experimental para identificação e análise de risco, mas os resultados não devem ser confundidos com garantias de desempenho ou de impacto financeiro em produção.

## 16. Estado atual do desenvolvimento

### Concluído

- [x] Carregamento e auditoria do dataset.
- [x] Análise exploratória inicial.
- [x] Pré-processamento e separação agrupada.
- [x] Clusterização K-Means.
- [x] Análise complementar dos perfis.
- [x] Comparação de Random Forest, SVM e KNN.
- [x] Seleção e exportação do modelo final.
- [x] Explicabilidade SHAP.
- [x] Geração de métricas, tabelas e figuras.
- [x] Dashboards Plotly em HTML.
- [x] Validação do contrato de inferência.
- [x] Estrutura de navegação do Streamlit.
- [x] Página inicial do Streamlit.
- [x] Página de avaliação do modelo.
- [x] Página de explicabilidade e simulador.
- [x] Página de base de dados, nas alterações recentes.
- [x] Página de visão geral (BI), nas alterações recentes.

### Pendente ou sujeito a validação

- [ ] Finalizar a página de análise exploratória interativa.
- [ ] Finalizar a página interativa dos perfis.
- [ ] Validar o funcionamento integrado de todas as páginas.
- [ ] Ajustar os caminhos legados de `eda_visual.py`, se o script continuar em uso.
- [ ] Consolidar um ambiente Nix reproduzível, se necessário.
- [ ] Realizar estudos futuros de calibração, validação externa e avaliação operacional.

## 17. Referências

**Dataset:** Iranian Churn Dataset — UCI Machine Learning Repository.

https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset

**Documentação das tecnologias:**

- Scikit-learn: https://scikit-learn.org/
- Imbalanced-learn: https://imbalanced-learn.org/
- SHAP: https://shap.readthedocs.io/
- Streamlit: https://docs.streamlit.io/
- Plotly: https://plotly.com/python/

---

**ChurnGuard — Grupo 4 · PISI III · Bacharelado em Sistemas de Informação · UFRPE**

Repositório dedicado à experimentação, avaliação, explicabilidade e visualização dos modelos de Machine Learning utilizados no projeto ChurnGuard.
