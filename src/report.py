"""Resumo factual regenerado a partir das tabelas da execução."""
from .common import ROOT
import pandas as pd
import json

def markdown_table(df):
    def fmt(v): return f'{v:.4f}' if isinstance(v,float) else str(v)
    rows=['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    rows += ['| '+' | '.join(fmt(v).replace('|','/') for v in row)+' |' for row in df.itertuples(index=False,name=None)]
    return '\n'.join(rows)

def build_report():
    T=ROOT/'reports/tabelas'
    metadata=json.loads((T/'model_metadata.json').read_text())
    cluster=json.loads((T/'cluster_metadata.json').read_text())
    shap=json.loads((T/'shap_metadata.json').read_text())
    metrics=pd.read_csv(T/'metricas_treino_teste.csv')
    ranking=pd.read_csv(T/'comparacao_cv.csv')
    importance=pd.read_csv(T/'shap_importancia.csv')
    summary=pd.read_csv(T/'cluster_resumo.csv')
    profiles=pd.read_csv(T/'clientes_clusters.csv')
    means=profiles[profiles.conjunto=='treino'].groupby('cluster')[['seconds_of_use','frequency_of_sms','customer_value','status','complains']].mean().reset_index()
    chosen=metrics[metrics.experimento==metadata['vencedor']].set_index('conjunto')
    test=chosen.loc['teste']
    text=f'''# Resultados experimentais — ChurnGuard

Texto de apoio para atualizar as seções 7.1 e 7.2 do artigo. Valores derivados da execução sobre o CSV enviado, não das tabelas ilustrativas do PDF. Não é uma revisão integral do artigo.

## Desenho experimental

Foram preservados os 3.150 registros e os grupos de perfis idênticos. A partição reproduz o notebook 02: {metadata['n_treino']} registros de treino e {metadata['n_teste']} de teste, sem perfis compartilhados. A seleção supervisionada utilizou cinco folds estratificados por grupo somente no treino. Foram avaliadas 78 configurações (390 ajustes em CV), cobrindo quatro classificadores, três estratégias de balanceamento e alternativas de escala. O critério de seleção foi o F1 da classe churn. A reamostragem ocorreu dentro dos pipelines, somente na parcela de treinamento de cada fold.

## PP1 — Perfis de clientes

O KMeans selecionou k={cluster['k']} pela maior silhueta média no treino ({cluster['silhueta_treino']:.4f}) entre k=2 e k=10. A inércia apresenta ganhos progressivamente menores, especialmente depois das primeiras divisões; o cotovelo é evidência complementar, não um teste formal. A proximidade das silhuetas de k=2 e k=3 recomenda cautela ao tratar três grupos como solução única. A silhueta obtida indica separação parcial, não segmentos perfeitamente distintos. A projeção PCA explica {cluster['pca_variancia']:.1%} da variância do espaço preparado e é usada somente para visualização.

{markdown_table(summary)}

As médias a seguir usam somente registros de treino:

{markdown_table(means)}

Nesta execução, o grupo 0 apresenta maior frequência de SMS e maior valor médio de cliente; o grupo 1 concentra menor uso, menor valor de cliente e maior presença de status inativo; o grupo 2 apresenta maior tempo de uso de chamadas. Os identificadores são arbitrários. O grupo 1 concentra a maior taxa observada de churn tanto no treino quanto no teste. Essa relação é descritiva e inclui atributos como status e reclamações: não permite concluir que somente o consumo explica o cancelamento, nem que os segmentos respondem causalmente a ações de retenção. A presença de atributos correlacionados pode alterar o peso efetivo de cada dimensão na distância.

Figuras: `cluster_cotovelo_silhueta`, `cluster_pca`, `cluster_silhuetas`. Dashboard: `01_segmentacao.html`.

## PP2 — Comparação supervisionada

{markdown_table(ranking[['experimento','cv_f1','cv_f1_std','cv_precision','cv_recall']])}

O modelo selecionado foi **{metadata['vencedor']}**, com F1 médio de CV de **{metadata['cv_f1']:.4f}**. O desvio entre folds descreve variação entre partições e não é intervalo de confiança. A seleção entre configurações pode tornar a estimativa de CV otimista, sendo o holdout a avaliação externa.

{markdown_table(metrics[metrics.conjunto=='teste'][['experimento','acuracia','precisao_churn','recall_churn','f1_churn','roc_auc','average_precision']])}

No teste, o vencedor obteve F1 de {test.f1_churn:.4f}, precisão de {test.precisao_churn:.2%}, recall de {test.recall_churn:.2%}, ROC-AUC de {test.roc_auc:.4f} e average precision de {test.average_precision:.4f}. A baseline majoritária tem recall e F1 de churn iguais a zero, apesar da acurácia elevada. O F1 aparente do vencedor no treino foi {chosen.loc['treino','f1_churn']:.4f}; sua queda para CV e teste sugere sobreajuste e impede usar a métrica de treino como expectativa operacional.

A Random Forest sem balanceamento obteve F1 de teste maior que o vencedor escolhido na CV, enquanto a versão com SMOTENC recuperou mais churns. Mantivemos a seleção anterior ao teste. Essa inversão e a pequena diferença de CV não demonstram superioridade estatística do SMOTENC. A escolha prática também depende do custo de falsos positivos e falsos negativos, que não foi fornecido; não se otimizou o limiar por custo.

As tabelas por classe, médias macro e ponderada e suporte estão em `relatorio_classes.csv`; a acurácia é apresentada separadamente em `metricas_treino_teste.csv`. Treino/teste usam seus registros originais, e o suporte de teste não muda com o balanceamento. Average precision é reportada como AP, não como área trapezoidal da curva PR.

Figuras: `classificacao_cv`, `classificacao_final`. Dashboard: `02_classificacao.html`.

## Explicabilidade

SHAP foi aplicado somente ao modelo final, usando {shap['n_background']} registros de treino como background e {shap['n_explicados']} registros aleatórios de teste para explicação. A saída explicada é: {shap['unidade']}. A soma da referência e das contribuições reconstruiu a saída com erro máximo de {shap['max_erro_aditividade']:.2e}.

{markdown_table(importance)}

Os três atributos de maior contribuição absoluta média nessa amostra são {', '.join(importance.variavel.head(3))}. O ranking descreve sensibilidade das previsões, não causalidade ou força de uma intervenção. A amostra é pequena; o ranking pode mudar com outra seleção de clientes. Atributos correlacionados podem compartilhar importância.

O waterfall representa o registro de índice {shap['row_id_local']}, de maior score entre os casos amostrados. A referência foi {shap['base_local']:.4f} e a saída do modelo foi {shap['score_local']:.4f}. Esse score não representa certeza de cancelamento nem probabilidade calibrada.

Figuras: `shap_importancia`, `shap_beeswarm`, `shap_local`. Dashboard: `03_explicabilidade.html`.

## Ajustes necessários no artigo

- Incluir Regressão Logística entre os métodos efetivamente comparados.
- Nomear o balanceamento sintético como SMOTENC, variante do SMOTE para dados mistos.
- Substituir as tabelas fictícias de três classes por resultados binários reais (0 = não churn, 1 = churn).
- Distinguir treino aparente, CV usada para seleção e teste final. Não usar o teste para trocar o vencedor.
- Explicar que o critério operacional de k foi a maior silhueta e que o cotovelo foi complementar.
- Descrever SHAP como atribuição do modelo, sem afirmar que uma ação sobre o atributo reduzirá churn.
- Confirmar o momento de coleta de status, reclamações e demais atributos antes de alegar predição prospectiva. A avaliação atual não prova ausência de vazamento temporal ou adequação a uma operadora real.
- A seção 6 e a PP3 dependem de aplicativo React Native e avaliação com gestores, que não foram fornecidos. Estes dashboards apresentam resultados, mas não constituem evidência de usabilidade nem de efetividade de retenção.

## Reprodutibilidade e limites

Sementes, hash do dataset, versões e parâmetros estão em `model_metadata.json`. A execução usa um único holdout e busca finita. Não há validação temporal, calibração, estudo de custos, teste externo ou experimento causal de retenção. Duplicatas foram preservadas e cada registro tem peso igual nas métricas. Os artefatos registram essas escolhas para permitir revisão e novos experimentos.
'''
    path=ROOT/'reports/resultados_artigo.md'
    path.write_text(text,encoding='utf-8')
    return path
