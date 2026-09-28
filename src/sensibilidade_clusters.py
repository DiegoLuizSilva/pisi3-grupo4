"""Sensibilidade da segmentação a semente, escala e conjunto de atributos.

Todas as decisões usam apenas critérios internos (silhueta, inércia, estabilidade
das partições). O churn não entra em nenhuma escolha: ele é consultado somente na
descrição dos grupos, depois que a configuração já está fixada.

Gera em reports/tabelas/:
    sens_sementes.csv      silhueta e estabilidade por k, variando a semente
    sens_escala.csv        silhueta por k para cada escala
    sens_atributos.csv     silhueta por k para cada subconjunto de atributos
    sens_algoritmos.csv    KMeans x Ward x GMM, silhueta e concordância
    sens_resumo.json       leitura consolidada
"""
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import load_data, preprocessor, SEED, BIN
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score

KS = range(2, 7)
SEMENTES = list(range(20))
SAIDA = ROOT / 'reports/tabelas'
SAIDA.mkdir(parents=True, exist_ok=True)


def preparar(Xs, escala='standard'):
    """Mesma lógica de src.common.preprocessor, mas tolerante a subconjuntos
    de colunas: só trata como binária a coluna que estiver presente."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    scaler = {'standard': StandardScaler(), 'minmax': MinMaxScaler(),
              'none': 'passthrough'}[escala]
    binarias = [c for c in BIN if c in Xs.columns]
    numericas = [c for c in Xs.columns if c not in binarias]
    return ColumnTransformer(
        [('numeric', scaler, numericas), ('binary', 'passthrough', binarias)],
        verbose_feature_names_out=False).fit_transform(Xs)


def estabilidade(particoes):
    """ARI médio entre todos os pares de partições."""
    pares = [adjusted_rand_score(a, b)
             for i, a in enumerate(particoes) for b in particoes[i + 1:]]
    return float(np.mean(pares)), float(np.min(pares))


def main():
    df, X, y, groups, tr, te = load_data()
    X_train = X.iloc[tr]
    Z = preprocessor(X).fit_transform(X_train)

    # ------------------------------------------------------------------
    # 1. Semente
    # ------------------------------------------------------------------
    linhas = []
    for n_init in [1, 20]:
        for k in KS:
            sils, inercias, particoes = [], [], []
            for s in SEMENTES:
                m = KMeans(n_clusters=k, n_init=n_init, random_state=s).fit(Z)
                sils.append(silhouette_score(Z, m.labels_))
                inercias.append(m.inertia_)
                particoes.append(m.labels_)
            ari_medio, ari_min = estabilidade(particoes)
            linhas.append({
                'n_init': n_init, 'k': k,
                'silhueta_media': np.mean(sils), 'silhueta_dp': np.std(sils),
                'silhueta_min': np.min(sils), 'silhueta_max': np.max(sils),
                'inercia_media': np.mean(inercias), 'inercia_dp': np.std(inercias),
                'ari_medio': ari_medio, 'ari_minimo': ari_min,
                'particoes_distintas': len({tuple(p) for p in particoes}),
            })
    sementes = pd.DataFrame(linhas)
    sementes.to_csv(SAIDA / 'sens_sementes.csv', index=False)

    # quantas vezes cada k venceria, se a semente fosse outra
    vitorias = {}
    for n_init in [1, 20]:
        contagem = {k: 0 for k in KS}
        for s in SEMENTES:
            melhor, melhor_sil = None, -np.inf
            for k in KS:
                m = KMeans(n_clusters=k, n_init=n_init, random_state=s).fit(Z)
                sil = silhouette_score(Z, m.labels_)
                if sil > melhor_sil:
                    melhor, melhor_sil = k, sil
            contagem[melhor] += 1
        vitorias[n_init] = contagem

    # ------------------------------------------------------------------
    # 2. Escala
    # ------------------------------------------------------------------
    linhas = []
    for escala in ['standard', 'minmax', 'none']:
        Ze = preprocessor(X, scale=escala).fit_transform(X_train)
        for k in KS:
            m = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(Ze)
            linhas.append({'escala': escala, 'k': k,
                           'silhueta': silhouette_score(Ze, m.labels_),
                           'inercia': m.inertia_,
                           'menor_grupo': int(np.bincount(m.labels_).min()),
                           'maior_grupo': int(np.bincount(m.labels_).max())})
    escalas = pd.DataFrame(linhas)
    escalas.to_csv(SAIDA / 'sens_escala.csv', index=False)

    # ------------------------------------------------------------------
    # 3. Conjunto de atributos
    # ------------------------------------------------------------------
    numericos = [c for c in X.columns if c not in BIN]
    uso = ['seconds_of_use', 'frequency_of_use', 'frequency_of_sms',
           'distinct_called_numbers']
    conjuntos = {
        'completo (12)': list(X.columns),
        'sem binarias (9)': numericos,
        'sem customer_value (11)': [c for c in X.columns if c != 'customer_value'],
        'somente uso (4)': uso,
    }
    linhas = []
    for nome, colunas in conjuntos.items():
        Xs = X_train[colunas]
        Zs = preparar(Xs)
        for k in KS:
            m = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(Zs)
            linhas.append({'conjunto': nome, 'n_atributos': len(colunas), 'k': k,
                           'silhueta': silhouette_score(Zs, m.labels_),
                           'menor_grupo': int(np.bincount(m.labels_).min())})
    atributos = pd.DataFrame(linhas)
    atributos.to_csv(SAIDA / 'sens_atributos.csv', index=False)

    # ------------------------------------------------------------------
    # 4. Outros agrupadores
    # ------------------------------------------------------------------
    linhas = []
    for k in KS:
        km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(Z)
        ward = AgglomerativeClustering(n_clusters=k, linkage='ward').fit(Z)
        gmm = GaussianMixture(n_components=k, n_init=5, random_state=SEED).fit(Z)
        rotulos_gmm = gmm.predict(Z)
        for nome, rot in [('KMeans', km.labels_), ('Ward', ward.labels_),
                          ('GMM', rotulos_gmm)]:
            linhas.append({
                'algoritmo': nome, 'k': k,
                'silhueta': silhouette_score(Z, rot),
                'menor_grupo': int(np.bincount(rot).min()),
                'maior_grupo': int(np.bincount(rot).max()),
                'ari_vs_kmeans': adjusted_rand_score(km.labels_, rot),
            })
    algoritmos = pd.DataFrame(linhas)
    algoritmos.to_csv(SAIDA / 'sens_algoritmos.csv', index=False)

    # ------------------------------------------------------------------
    resumo = {
        'k_vencedor_por_semente': {str(n): {str(k): v for k, v in c.items()}
                                   for n, c in vitorias.items()},
        'sementes_testadas': len(SEMENTES),
        'silhueta_k2_k3_referencia': {
            'k2': float(sementes.query('n_init == 20 and k == 2').silhueta_media.iloc[0]),
            'k3': float(sementes.query('n_init == 20 and k == 3').silhueta_media.iloc[0]),
        },
    }
    (SAIDA / 'sens_resumo.json').write_text(json.dumps(resumo, indent=2))

    pd.set_option('display.width', 200)
    print('=' * 72)
    print('1. SEMENTE')
    print('=' * 72)
    print(sementes.round(4).to_string(index=False))
    print('\nk vencedor por silhueta, em', len(SEMENTES), 'sementes:')
    for n_init, contagem in vitorias.items():
        print(f'  n_init={n_init}:', {k: v for k, v in contagem.items() if v})

    print('\n' + '=' * 72)
    print('2. ESCALA')
    print('=' * 72)
    print(escalas.round(4).to_string(index=False))

    print('\n' + '=' * 72)
    print('3. ATRIBUTOS')
    print('=' * 72)
    print(atributos.round(4).to_string(index=False))

    print('\n' + '=' * 72)
    print('4. ALGORITMOS')
    print('=' * 72)
    print(algoritmos.round(4).to_string(index=False))
    print('\ntabelas salvas em reports/tabelas/sens_*.csv')


if __name__ == '__main__':
    main()