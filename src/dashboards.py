"""Dashboards offline gerados a partir dos resultados, sem retreinar modelos."""
from .common import ROOT
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from html import escape
T = ROOT/'reports/tabelas'
D = ROOT/'reports/dashboards'
COLORS=['#226B91','#008577','#DB6A35','#8156A5','#BA4670','#6A7A2E','#4783BC','#B67928','#5C6380','#C65546']

def read(name): return pd.read_csv(T/name)
def meta(name): return json.loads((T/name).read_text())
def export(fig,filename,title,subtitle,cards,notes,table=None):
    fig.update_layout(template='plotly_white',height=980,font={'family':'Arial','size':12,'color':'#253b4b'},
                      margin={'l':70,'r':45,'t':65,'b':70},paper_bgcolor='#ffffff',colorway=COLORS,
                      legend={'orientation':'h','y':-.12},hoverlabel={'font_size':13})
    chart=fig.to_html(full_html=False,include_plotlyjs=True,config={'responsive':True,'displaylogo':False,
                       'toImageButtonOptions':{'format':'svg','filename':filename}})
    blocks=''.join(f'<div class="card"><span>{escape(k)}</span><strong>{escape(str(v))}</strong></div>' for k,v in cards)
    table_html='' if table is None else '<div class="table">'+table.to_html(index=False, float_format=lambda x:f'{x:.3f}',border=0)+'</div>'
    html=f'''<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)}</title>
<style>body{{margin:0;background:#edf2f5;color:#253b4b;font-family:Arial,sans-serif}}main{{max-width:1320px;margin:auto;padding:32px}}.eyebrow{{letter-spacing:3px;color:#008577;font-size:12px;font-weight:bold}}h1{{font-size:36px;margin:12px 0}}p{{line-height:1.6;max-width:1100px}}.cards{{display:flex;gap:16px;flex-wrap:wrap;margin:26px 0}}.card{{background:white;padding:22px;border-radius:12px;flex:1;min-width:180px;border-top:4px solid #008577}}span{{font-size:13px;color:#526979}}strong{{display:block;font-size:26px;margin-top:12px}}.chart,.table{{background:white;border-radius:12px;padding:14px;margin:20px 0;overflow:auto}}.note{{border-left:4px solid #DB6A35;padding:10px 20px;background:white}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:12px;text-align:left;border-bottom:1px solid #e3eaf0}}footer{{font-size:12px;color:#526979;padding:20px 0}}@media(max-width:650px){{main{{padding:14px}}h1{{font-size:28px}}}}@media print{{body{{background:white}}main{{padding:0}}.chart{{break-inside:avoid}}}}</style></head><body><main><div class="eyebrow">CHURNGUARD / RESULTADOS EXPERIMENTAIS</div><h1>{escape(title)}</h1><p>{escape(subtitle)}</p><div class="cards">{blocks}</div><div class="chart">{chart}</div>{table_html}<p class="note">{escape(notes)}</p><footer>Iranian Churn · Semente 42 · Resultados reproduzíveis nos notebooks 03–06 · Hover, zoom e legenda interativos · HTML offline</footer></main></body></html>'''
    path=D/f'{filename}.html'
    path.write_text(html,encoding='utf-8')
    return path

def build_dashboards():
    D.mkdir(parents=True,exist_ok=True)
    quality, summary, points = read('cluster_qualidade.csv'),read('cluster_resumo.csv'),read('cluster_pca.csv')
    cm=meta('cluster_metadata.json')
    fig=make_subplots(rows=3,cols=2,subplot_titles=['Cotovelo · treino','Silhueta · treino','Perfis em PCA · treino','Tamanho dos segmentos','Taxa de churn por segmento','Perfil médio padronizado'],vertical_spacing=.12)
    fig.add_trace(go.Scatter(x=quality.k,y=quality.inercia,mode='lines+markers',name='Inércia',showlegend=False),1,1)
    fig.add_trace(go.Scatter(x=quality.k,y=quality.silhueta,mode='lines+markers',name='Silhueta',showlegend=False),1,2)
    for i,c in enumerate(sorted(points.cluster.unique())):
        part=points[points.cluster==c]
        fig.add_trace(go.Scattergl(x=part.PC1,y=part.PC2,mode='markers',marker={'size':4,'opacity':.5,'color':COLORS[i%len(COLORS)]},name=f'Cluster {c}'),2,1)
    for i,split in enumerate(['treino','teste']):
        part=summary[summary.conjunto==split]
        fig.add_trace(go.Bar(x=part.cluster.astype(str),y=part.clientes,name=split,marker_color=COLORS[i]),2,2)
        fig.add_trace(go.Bar(x=part.cluster.astype(str),y=part.taxa_churn,name=split,marker_color=COLORS[i],showlegend=False),3,1)
    profiles=read('clientes_clusters.csv')
    features=['seconds_of_use','frequency_of_sms','customer_value','complains','status']
    train=profiles[profiles.conjunto=='treino']
    standardized=(train.groupby('cluster')[features].mean()-train[features].mean())/train[features].std(ddof=0)
    fig.add_trace(go.Heatmap(z=standardized.T.values,x=standardized.index.astype(str),y=features,colorscale='RdBu',zmid=0,showscale=False),3,2)
    fig.update_yaxes(tickformat='.0%',row=3,col=1)
    fig.update_xaxes(title_text='k',row=1,col=1)
    fig.update_xaxes(title_text='k',row=1,col=2)
    fig.update_xaxes(title_text='PC1',row=2,col=1)
    fig.update_yaxes(title_text='PC2',row=2,col=1)
    path1=export(fig,'01_segmentacao','Perfis de clientes',
        'KMeans ajustado no treino; churn observado somente depois da escolha dos grupos.',
        [('Clientes',len(profiles)),('Grupos escolhidos',cm['k']),('Silhueta no treino',f"{cm['silhueta_treino']:.3f}"),('Variância PCA',f"{cm['pca_variancia']:.1%}")],
        'k maximiza a silhueta entre 2 e 10. O cotovelo é complementar e pode ser ambíguo. PCA é uma projeção, não o espaço usado no KMeans. Taxas de churn são associações descritivas; o teste não participou da escolha de k.',summary)
    ranking,metrics,pred = read('comparacao_cv.csv'),read('metricas_treino_teste.csv'),read('predicoes.csv')
    modelmeta=meta('model_metadata.json')
    winner=modelmeta['vencedor']
    final=metrics[(metrics.experimento==winner)&(metrics.conjunto=='teste')].iloc[0]
    test=pred[(pred.experimento==winner)&(pred.conjunto=='teste')]
    matrix=confusion_matrix(test.real,test.previsto,labels=[0,1])
    fig=make_subplots(rows=3,cols=2,subplot_titles=['F1 de churn · validação agrupada','F1 de churn · treino e teste','Matriz de confusão · vencedor no teste','Curva ROC · vencedor no teste','Precisão-recall · vencedor no teste','Recall e precisão · teste'],vertical_spacing=.16,horizontal_spacing=.2)
    short={r.experimento:r.modelo.replace('Regressão Logística','Logística').replace('Random Forest','RF')+' / '+r.balanceamento.replace('Sem balanceamento','Sem').replace('Undersampling','Under') for r in ranking.itertuples()}
    fig.add_trace(go.Bar(x=ranking.cv_f1,y=[short[k] for k in ranking.experimento],orientation='h',error_x={'type':'data','array':ranking.cv_f1_std},name='CV',marker_color='#226B91',showlegend=False),1,1)
    for i,split in enumerate(['treino','teste']):
        part=metrics[(metrics.conjunto==split)&(metrics.experimento.isin(short))].set_index('experimento').loc[list(short)]
        fig.add_trace(go.Bar(x=[short[k] for k in part.index],y=part.f1_churn,name=split,marker_color=COLORS[i]),1,2)
    fig.update_xaxes(tickangle=55,tickfont={'size':9},row=1,col=2)
    fig.update_yaxes(tickfont={'size':9},row=1,col=1)
    fig.add_trace(go.Heatmap(z=matrix,x=['Previsto: não churn','Previsto: churn'],y=['Real: não churn','Real: churn'],text=matrix,texttemplate='%{text}',colorscale='Blues',showscale=False),2,1)
    fpr,tpr,_=roc_curve(test.real,test.score)
    precision,recall,_=precision_recall_curve(test.real,test.score)
    fig.add_trace(go.Scatter(x=fpr,y=tpr,mode='lines',name='ROC',showlegend=False),2,2)
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',line={'dash':'dash','color':'gray'},showlegend=False),2,2)
    fig.add_trace(go.Scatter(x=recall,y=precision,mode='lines',name='PR',showlegend=False),3,1)
    fig.add_trace(go.Scatter(x=[0,1],y=[test.real.mean()]*2,mode='lines',line={'dash':'dash','color':'gray'},name='Prevalência',showlegend=False),3,1)
    part=metrics[(metrics.conjunto=='teste')&(metrics.experimento.isin(short))]
    fig.add_trace(go.Scatter(x=part.recall_churn,y=part.precisao_churn,mode='markers',text=part.experimento,hovertemplate='%{text}<br>Recall %{x:.3f}<br>Precisão %{y:.3f}<extra></extra>',marker={'size':12,'color':'#008577'},showlegend=False),3,2)
    fig.update_xaxes(title_text='Falsos positivos',row=2,col=2)
    fig.update_yaxes(title_text='Recall',row=2,col=2)
    for r,c in [(3,1),(3,2)]:
        fig.update_xaxes(title_text='Recall',range=[0,1.02],row=r,col=c)
        fig.update_yaxes(title_text='Precisão',range=[0,1.02],row=r,col=c)
    path2=export(fig,'02_classificacao','Detecção de churn',f'Modelo selecionado antes de avaliar o teste: {winner}.',
        [('F1 · teste',f'{final.f1_churn:.3f}'),('Recall · teste',f'{final.recall_churn:.1%}'),('Precisão · teste',f'{final.precisao_churn:.1%}'),('AP · teste',f'{final.average_precision:.3f}')],
        f"Holdout com {modelmeta['n_teste']} registros e sem perfis compartilhados com treino. Barras de CV: média ± desvio entre cinco folds, não intervalo de confiança. Comparações de teste são descritivas e não alteram o vencedor. Scores não calibrados; limiar padrão do estimador.",metrics[metrics.conjunto=='teste'].drop(columns=['conjunto']))
    imp,values,sample=read('shap_importancia.csv'),read('shap_valores.csv'),read('shap_amostra.csv')
    sm=meta('shap_metadata.json')
    local=values[values.row_id==sm['row_id_local']].iloc[0].drop('row_id')
    fig=make_subplots(rows=2,cols=2,subplot_titles=['Importância global · amostra do teste','Contribuições SHAP · principais atributos','Contribuições do caso local','Reconstrução do score local'],vertical_spacing=.16,horizontal_spacing=.22)
    fig.add_trace(go.Bar(x=imp.shap_absoluto_medio,y=imp.variavel,orientation='h',marker_color='#226B91',showlegend=False),1,1)
    for c in imp.variavel.head(5):
        fig.add_trace(go.Box(x=values[c],name=c,boxpoints='all',jitter=.3,pointpos=0,showlegend=False),1,2)
    local=local.sort_values()
    fig.add_trace(go.Bar(x=local.values,y=local.index,orientation='h',marker_color=['#DB6A35' if v>0 else '#226B91' for v in local],showlegend=False),2,1)
    ordered=local.reindex(local.abs().sort_values(ascending=False).index)
    fig.add_trace(go.Waterfall(x=['Base',*ordered.index,'Score'],y=[sm['base_local'],*ordered.values,0],measure=['absolute',*['relative']*len(ordered),'total'],showlegend=False),2,2)
    fig.update_xaxes(tickangle=55,tickfont={'size':9},row=2,col=2)
    fig.update_yaxes(tickfont={'size':10},row=1,col=1)
    fig.update_yaxes(tickfont={'size':10},row=2,col=1)
    path3=export(fig,'03_explicabilidade','Por que o modelo decide?',f"SHAP do modelo final: {sm['modelo']}. Unidade: {sm['unidade']}.",
        [('Registros explicados',sm['n_explicados']),('Background de treino',sm['n_background']),('Caso local · row_id',sm['row_id_local']),('Score do caso',f"{sm['score_local']:.3f}")],
        'Importância global calculada em amostra aleatória de teste. O caso local tem o maior score nessa amostra. Valores positivos aumentam a saída em relação à referência. SHAP explica associações aprendidas, não efeitos causais. Atributos correlacionados podem dividir importância.',imp)
    return [path1,path2,path3]
