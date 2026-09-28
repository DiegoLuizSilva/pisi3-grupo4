"""Revisão da escolha de k pelos quatro critérios combinados.

Critérios internos, avaliados sem olhar o churn:
    1. cotovelo       queda relativa da inércia e sua desaceleração
    2. silhueta       média e fração de observações mal alocadas (silhueta < 0)
    3. tamanho        menor grupo em valor absoluto e em percentual
    4. clareza        separação dos perfis medida pela distância entre centroides
                      e pelo número de atributos que distinguem cada grupo

O churn entra apenas no fim, como coluna descritiva, para registro — nunca
como critério de escolha.

Gera reports/tabelas/revisao_k.csv e revisao_k_perfis.csv
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import load_data, preprocessor, SEED
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

KS = range(2, 9)
SAIDA = ROOT / 'reports/tabelas'


def main():
    df, X, y, groups, tr, te = load_data()
    X_train, y_train = X.iloc[tr], y.iloc[tr]
    Z = preprocessor(X).fit_transform(X_train)
    n = len(Z)

    inercia = {}
    for k in range(1, 11):
        inercia[k] = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(Z).inertia_

    linhas = []
    perfis_linhas = []
    for k in KS:
        m = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(Z)
        rot = m.labels_
        sil_amostras = silhouette_samples(Z, rot)
        tamanhos = np.bincount(rot)

        # cotovelo: quanto a inércia ainda cai ao passar de k-1 para k,
        # e quanto essa queda desacelera em relação ao passo seguinte
        queda = inercia[k - 1] - inercia[k]
        queda_seguinte = inercia[k] - inercia[k + 1]
        desaceleracao = queda / queda_seguinte if queda_seguinte > 0 else np.nan

        # clareza: distância mínima entre centroides (espaço padronizado)
        cent = m.cluster_centers_
        dists = [np.linalg.norm(cent[i] - cent[j])
                 for i in range(k) for j in range(i + 1, k)]

        # clareza: atributos em que o grupo se afasta mais de 0,8 desvio da média geral
        perfis = pd.DataFrame(Z, columns=X.columns)
        perfis['cluster'] = rot
        medias = perfis.groupby('cluster')[list(X.columns)].mean()
        distintivos = (medias.abs() > 0.8).sum(axis=1)

        linhas.append({
            'k': k,
            'inercia': inercia[k],
            'queda_inercia_%': 100 * queda / inercia[k - 1],
            'desaceleracao': desaceleracao,
            'silhueta': silhouette_score(Z, rot),
            'mal_alocados_%': 100 * (sil_amostras < 0).mean(),
            'menor_grupo': int(tamanhos.min()),
            'menor_grupo_%': 100 * tamanhos.min() / n,
            'maior_grupo_%': 100 * tamanhos.max() / n,
            'dist_min_centroides': float(np.min(dists)),
            'atributos_distintivos_min': int(distintivos.min()),
            'grupos_sem_distintivo': int((distintivos == 0).sum()),
        })

        if k in (2, 3, 4):
            tmp = X_train.copy()
            tmp['cluster'] = rot
            tmp['churn'] = y_train.values
            res = tmp.groupby('cluster').agg(
                clientes=('churn', 'size'), taxa_churn=('churn', 'mean'))
            res['k'] = k
            res['silhueta_media'] = [sil_amostras[rot == c].mean() for c in res.index]
            perfis_linhas.append(res.reset_index())

    revisao = pd.DataFrame(linhas)
    revisao.to_csv(SAIDA / 'revisao_k.csv', index=False)
    perfis = pd.concat(perfis_linhas)[
        ['k', 'cluster', 'clientes', 'silhueta_media', 'taxa_churn']]
    perfis.to_csv(SAIDA / 'revisao_k_perfis.csv', index=False)

    pd.set_option('display.width', 220)
    print('=' * 100)
    print('REVISÃO DA ESCOLHA DE k · treino (n =', n, ')')
    print('=' * 100)
    print(revisao.round(4).to_string(index=False))
    print()
    print('desaceleracao > 2 indica cotovelo: a queda em k é mais que o dobro da seguinte')
    print('atributos_distintivos_min = 0 significa que algum grupo não tem característica própria')
    print()
    print('=' * 100)
    print('CANDIDATOS k = 2, 3, 4 · tamanho, coesão e churn (churn apenas descritivo)')
    print('=' * 100)
    print(perfis.round(4).to_string(index=False))


if __name__ == '__main__':
    main()