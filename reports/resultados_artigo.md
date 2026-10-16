# Resultados experimentais — ChurnGuard

Texto de apoio para atualizar as seções 7.1 e 7.2 do artigo. Valores derivados da execução sobre o CSV enviado, não das tabelas ilustrativas do PDF. Não é uma revisão integral do artigo.

## Desenho experimental

Foram preservados os 3.150 registros e os grupos de perfis idênticos. A partição reproduz o notebook 02: 2517 registros de treino e 633 de teste, sem perfis compartilhados. A seleção supervisionada utilizou cinco folds estratificados por grupo somente no treino. Foram avaliadas 60 configurações (300 ajustes em CV), cobrindo três classificadores (Random Forest, SVM e KNN), três estratégias de balanceamento e alternativas de escala. O critério de seleção foi o F1 da classe churn. A reamostragem ocorreu dentro dos pipelines, somente na parcela de treinamento de cada fold.

## PP1 — Perfis de clientes

O KMeans selecionou k=3 pela maior silhueta média no treino (0.3061) entre k=2 e k=10. A inércia apresenta ganhos progressivamente menores, especialmente depois das primeiras divisões; o cotovelo é evidência complementar, não um teste formal. A proximidade das silhuetas de k=2 e k=3 recomenda cautela ao tratar três grupos como solução única. A silhueta obtida indica separação parcial, não segmentos perfeitamente distintos. A projeção PCA explica 59.7% da variância do espaço preparado e é usada somente para visualização.

| conjunto | cluster | clientes | churns | taxa_churn |
| --- | --- | --- | --- | --- |
| teste | 0 | 116 | 2 | 0.0172 |
| teste | 1 | 382 | 97 | 0.2539 |
| teste | 2 | 135 | 1 | 0.0074 |
| treino | 0 | 459 | 3 | 0.0065 |
| treino | 1 | 1493 | 389 | 0.2605 |
| treino | 2 | 565 | 3 | 0.0053 |

As médias a seguir usam somente registros de treino:

| cluster | seconds_of_use | frequency_of_sms | customer_value | status | complains |
| --- | --- | --- | --- | --- | --- |
| 0 | 4274.8322 | 276.5708 | 1315.9094 | 0.0000 | 0.0174 |
| 1 | 2243.2371 | 14.9886 | 147.5760 | 0.4126 | 0.1085 |
| 2 | 10565.6496 | 61.6726 | 643.7997 | 0.0000 | 0.0248 |

Nesta execução, o grupo 0 apresenta maior frequência de SMS e maior valor médio de cliente; o grupo 1 concentra menor uso, menor valor de cliente e maior presença de status inativo; o grupo 2 apresenta maior tempo de uso de chamadas. Os identificadores são arbitrários. O grupo 1 concentra a maior taxa observada de churn tanto no treino quanto no teste. Essa relação é descritiva e inclui atributos como status e reclamações: não permite concluir que somente o consumo explica o cancelamento, nem que os segmentos respondem causalmente a ações de retenção. A presença de atributos correlacionados pode alterar o peso efetivo de cada dimensão na distância.

Figuras: `cluster_cotovelo_silhueta`, `cluster_pca`, `cluster_silhuetas`. Dashboard: `01_segmentacao.html`.

## PP2 — Comparação supervisionada

| experimento | cv_f1 | cv_f1_std | cv_precision | cv_recall |
| --- | --- | --- | --- | --- |
| Random Forest / SMOTENC | 0.8687 | 0.0310 | 0.8325 | 0.9090 |
| Random Forest / Sem balanceamento | 0.8437 | 0.0247 | 0.8821 | 0.8100 |
| KNN / Sem balanceamento | 0.8298 | 0.0160 | 0.8620 | 0.8025 |
| KNN / SMOTENC | 0.8267 | 0.0123 | 0.7565 | 0.9115 |
| SVM / SMOTENC | 0.8143 | 0.0277 | 0.7319 | 0.9191 |
| SVM / Sem balanceamento | 0.8097 | 0.0144 | 0.8733 | 0.7570 |
| SVM / Undersampling | 0.7647 | 0.0449 | 0.6402 | 0.9544 |
| KNN / Undersampling | 0.7624 | 0.0237 | 0.6479 | 0.9292 |
| Random Forest / Undersampling | 0.7424 | 0.0330 | 0.6125 | 0.9444 |

O modelo selecionado foi **Random Forest | SMOTENC**, com F1 médio de CV de **0.8687**. O desvio entre folds descreve variação entre partições e não é intervalo de confiança. A seleção entre configurações pode tornar a estimativa de CV otimista, sendo o holdout a avaliação externa.

| experimento | acuracia | precisao_churn | recall_churn | f1_churn | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- | --- |
| Random Forest / Sem balanceamento | 0.9668 | 0.9072 | 0.8800 | 0.8934 | 0.9945 | 0.9707 |
| Random Forest / SMOTENC | 0.9605 | 0.8378 | 0.9300 | 0.8815 | 0.9905 | 0.9433 |
| Random Forest / Undersampling | 0.9242 | 0.6781 | 0.9900 | 0.8049 | 0.9826 | 0.8571 |
| SVM / Sem balanceamento | 0.9463 | 0.8929 | 0.7500 | 0.8152 | 0.9818 | 0.9344 |
| SVM / SMOTENC | 0.9321 | 0.7176 | 0.9400 | 0.8139 | 0.9847 | 0.9333 |
| SVM / Undersampling | 0.9084 | 0.6382 | 0.9700 | 0.7698 | 0.9839 | 0.9308 |
| KNN / Sem balanceamento | 0.9431 | 0.8265 | 0.8100 | 0.8182 | 0.9806 | 0.9141 |
| KNN / SMOTENC | 0.9336 | 0.7231 | 0.9400 | 0.8174 | 0.9735 | 0.8332 |
| KNN / Undersampling | 0.9052 | 0.6316 | 0.9600 | 0.7619 | 0.9717 | 0.8055 |
| Baseline majoritária | 0.8420 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.1580 |

No teste, o vencedor obteve F1 de 0.8815, precisão de 83.78%, recall de 93.00%, ROC-AUC de 0.9905 e average precision de 0.9433. A baseline majoritária tem recall e F1 de churn iguais a zero, apesar da acurácia elevada. O F1 aparente do vencedor no treino foi 0.9714; sua queda para CV e teste sugere sobreajuste e impede usar a métrica de treino como expectativa operacional.

A Random Forest sem balanceamento obteve F1 de teste maior que o vencedor escolhido na CV, enquanto a versão com SMOTENC recuperou mais churns. Mantivemos a seleção anterior ao teste. Essa inversão e a pequena diferença de CV não demonstram superioridade estatística do SMOTENC. A escolha prática também depende do custo de falsos positivos e falsos negativos, que não foi fornecido; não se otimizou o limiar por custo.

As tabelas por classe, médias macro e ponderada e suporte estão em `relatorio_classes.csv`; a acurácia é apresentada separadamente em `metricas_treino_teste.csv`. Treino/teste usam seus registros originais, e o suporte de teste não muda com o balanceamento. Average precision é reportada como AP, não como área trapezoidal da curva PR.

Figuras: `classificacao_cv`, `classificacao_final`. Dashboard: `02_classificacao.html`.

## Explicabilidade

SHAP foi aplicado somente ao modelo final, usando 50 registros de treino como background e 80 registros aleatórios de teste para explicação. A saída explicada é: Score de probabilidade não calibrada. A soma da referência e das contribuições reconstruiu a saída com erro máximo de 7.77e-16.

| variavel | shap_absoluto_medio |
| --- | --- |
| status | 0.1054 |
| complains | 0.0629 |
| frequency_of_use | 0.0474 |
| seconds_of_use | 0.0329 |
| subscription_length | 0.0325 |
| call_failure | 0.0230 |
| customer_value | 0.0194 |
| distinct_called_numbers | 0.0148 |
| frequency_of_sms | 0.0144 |
| age | 0.0112 |
| charge_amount | 0.0069 |
| tariff_plan | 0.0008 |

Os três atributos de maior contribuição absoluta média nessa amostra são status, complains, frequency_of_use. O ranking descreve sensibilidade das previsões, não causalidade ou força de uma intervenção. A amostra é pequena; o ranking pode mudar com outra seleção de clientes. Atributos correlacionados podem compartilhar importância.

O waterfall representa o registro de índice 2127, de maior score entre os casos amostrados. A referência foi 0.1537 e a saída do modelo foi 1.0000. Esse score não representa certeza de cancelamento nem probabilidade calibrada.

Figuras: `shap_importancia`, `shap_beeswarm`, `shap_local`. Dashboard: `03_explicabilidade.html`.

## Ajustes necessários no artigo

- Nomear o balanceamento sintético como SMOTENC, variante do SMOTE para dados mistos.
- Substituir as tabelas fictícias de três classes por resultados binários reais (0 = não churn, 1 = churn).
- Distinguir treino aparente, CV usada para seleção e teste final. Não usar o teste para trocar o vencedor.
- Explicar que o critério operacional de k foi a maior silhueta e que o cotovelo foi complementar.
- Descrever SHAP como atribuição do modelo, sem afirmar que uma ação sobre o atributo reduzirá churn.
- Confirmar o momento de coleta de status, reclamações e demais atributos antes de alegar predição prospectiva. A avaliação atual não prova ausência de vazamento temporal ou adequação a uma operadora real.
- A seção 6 e a PP3 dependem de aplicativo React Native e avaliação com gestores, que não foram fornecidos. Estes dashboards apresentam resultados, mas não constituem evidência de usabilidade nem de efetividade de retenção.

## Reprodutibilidade e limites

Sementes, hash do dataset, versões e parâmetros estão em `model_metadata.json`. A execução usa um único holdout e busca finita. Não há validação temporal, calibração, estudo de custos, teste externo ou experimento causal de retenção. Duplicatas foram preservadas e cada registro tem peso igual nas métricas. Os artefatos registram essas escolhas para permitir revisão e novos experimentos.
