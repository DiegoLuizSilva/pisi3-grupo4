"""Tabelas e figuras finais da segmentação, prontas para o artigo.

Usa o modelo já treinado em models/clusterizacao.joblib — não reajusta nada.
Os nomes dos perfis vêm das características observadas nos desvios padronizados
e ficam centralizados em PERFIS, para serem revisados num lugar só.

Gera em reports/tabelas/:
    perfil_tamanho.csv           tamanho e taxa de churn por grupo
    perfil_caracteristicas.csv   médias em unidades originais
    perfil_desvios.csv           desvios em relação à média geral, em desvios-padrão
em reports/figuras/:
    perfil_escolha_k             cotovelo e silhueta, com k escolhido destacado
    perfil_caracteristicas       desvios por atributo e grupo
    perfil_projecao              projeção bidimensional dos grupos
    perfil_tamanho_churn         tamanho do grupo e taxa de churn
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import load_data, SEED

# Nomes propostos a partir das características observadas.
# Revisar com a equipe antes de fechar o artigo.
PERFIS = {
    0: 'Intensivo em mensagens',
    1: 'Baixo engajamento',
    2: 'Intensivo em voz',
}

# Paleta categórica validada para daltonismo (deutan/protan/tritan).
CORES = {0: '#2a78d6', 1: '#eb6834', 2: '#1baf7a'}

TINTA = '#0b0b0b'
TINTA2 = '#52514e'
MUTED = '#898781'
GRADE = '#e1e0d9'

UNIDADES = {
    'call_failure': 'falhas de chamada',
    'complains': 'proporção com reclamação',
    'subscription_length': 'meses',
    'charge_amount': 'faixa ordinal (0–10)',
    'seconds_of_use': 'segundos',
    'frequency_of_use': 'chamadas',
    'frequency_of_sms': 'mensagens',
    'distinct_called_numbers': 'números distintos',
    'tariff_plan': 'proporção em contrato',
    'status': 'proporção inativa',
    'age': 'anos',
    'customer_value': 'unidades da operadora',
}

ROTULOS = {
    'call_failure': 'Falhas de chamada',
    'complains': 'Reclamação registrada',
    'subscription_length': 'Tempo de assinatura',
    'charge_amount': 'Faixa de cobrança',
    'seconds_of_use': 'Segundos de uso',
    'frequency_of_use': 'Chamadas realizadas',
    'frequency_of_sms': 'Mensagens enviadas',
    'distinct_called_numbers': 'Contatos distintos',
    'tariff_plan': 'Plano por contrato',
    'status': 'Linha inativa',
    'age': 'Idade',
    'customer_value': 'Valor do cliente',
}


def estilo(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(GRADE)
    ax.tick_params(colors=MUTED, labelsize=9)
    for rotulo in ax.get_xticklabels() + ax.get_yticklabels():
        rotulo.set_color(TINTA2)
    return ax


def salvar(fig, nome, ajustar=True):
    if ajustar:
        fig.tight_layout()
    for ext in ['png', 'svg']:
        fig.savefig(ROOT / f'reports/figuras/{nome}.{ext}', dpi=300,
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  figura:', f'reports/figuras/{nome}.png')


def main():
    df, X, y, groups, tr, te = load_data()
    artefatos = joblib.load(ROOT / 'models/clusterizacao.joblib')
    prep, km, pca = artefatos['preprocessor'], artefatos['kmeans'], artefatos['pca']
    rotulos = km.predict(prep.transform(X))
    k = km.n_clusters
    conjunto = np.where(np.isin(np.arange(len(X)), tr), 'treino', 'teste')

    if set(PERFIS) != set(range(k)):
        raise SystemExit(f'PERFIS precisa ter exatamente {k} nomes, '
                         f'um por grupo (0 a {k - 1}).')

    # ------------------------------------------------------------------
    # Tabela 1 · tamanho e churn
    # ------------------------------------------------------------------
    base = X.copy()
    base['grupo'] = rotulos
    base['churn'] = y.values
    base['conjunto'] = conjunto

    tamanho = base.groupby('grupo').agg(
        clientes=('churn', 'size'), cancelaram=('churn', 'sum'),
        taxa_churn=('churn', 'mean')).reset_index()
    tamanho['perfil'] = tamanho.grupo.map(PERFIS)
    tamanho['participacao'] = tamanho.clientes / len(base)
    for parte in ['treino', 'teste']:
        sub = base[base.conjunto == parte].groupby('grupo').churn.mean()
        tamanho[f'taxa_churn_{parte}'] = tamanho.grupo.map(sub)
    tamanho = tamanho[['grupo', 'perfil', 'clientes', 'participacao', 'cancelaram',
                       'taxa_churn', 'taxa_churn_treino', 'taxa_churn_teste']]
    tamanho.to_csv(ROOT / 'reports/tabelas/perfil_tamanho.csv', index=False)

    # ------------------------------------------------------------------
    # Tabela 2 · características em unidades originais
    # ------------------------------------------------------------------
    medias = base.groupby('grupo')[list(X.columns)].mean().T
    medias.columns = [PERFIS[c] for c in medias.columns]
    medias.insert(0, 'geral', X.mean())
    medias.insert(0, 'unidade', [UNIDADES[i] for i in medias.index])
    medias.index = [ROTULOS[i] for i in medias.index]
    medias.index.name = 'atributo'
    medias.round(2).to_csv(ROOT / 'reports/tabelas/perfil_caracteristicas.csv')

    # ------------------------------------------------------------------
    # Tabela 3 · desvios padronizados
    # ------------------------------------------------------------------
    Xz = (X - X.mean()) / X.std()
    Xz['grupo'] = rotulos
    desvios = Xz.groupby('grupo')[list(X.columns)].mean().T
    desvios.columns = [PERFIS[c] for c in desvios.columns]
    desvios.index = [ROTULOS[i] for i in desvios.index]
    desvios.index.name = 'atributo'
    desvios.round(2).to_csv(ROOT / 'reports/tabelas/perfil_desvios.csv')

    print('\nTAMANHO E CHURN POR GRUPO')
    print(tamanho.round(4).to_string(index=False))
    print('\nDESVIOS EM RELAÇÃO À MÉDIA GERAL (em desvios-padrão)')
    print(desvios.round(2).to_string())

    # ------------------------------------------------------------------
    # Figura 1 · escolha de k
    # ------------------------------------------------------------------
    qualidade = pd.read_csv(ROOT / 'reports/tabelas/cluster_qualidade.csv')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = estilo(axes[0])
    ax.plot(qualidade.k, qualidade.inercia / 1000, '-', color=MUTED, lw=2, zorder=1)
    ax.scatter(qualidade.k, qualidade.inercia / 1000, s=42, color=MUTED, zorder=2)
    sel = qualidade[qualidade.k == k]
    ax.scatter(sel.k, sel.inercia / 1000, s=110, color=CORES[0], zorder=3,
               edgecolor='white', linewidth=2)
    ax.annotate(f'k = {k}', (k, sel.inercia.iloc[0] / 1000), textcoords='offset points',
                xytext=(14, 12), fontsize=10, color=CORES[0], fontweight='bold')
    ax.set_xlabel('Número de grupos (k)', color=TINTA2, fontsize=10)
    ax.set_ylabel('Inércia (milhares)', color=TINTA2, fontsize=10)
    ax.set_title('Método do cotovelo', color=TINTA, fontsize=11, loc='left', pad=10)
    ax.grid(axis='y', color=GRADE, lw=0.8)
    ax.set_axisbelow(True)

    ax = estilo(axes[1])
    q = qualidade.dropna(subset=['silhueta'])
    ax.plot(q.k, q.silhueta, '-', color=MUTED, lw=2, zorder=1)
    ax.scatter(q.k, q.silhueta, s=42, color=MUTED, zorder=2)
    sel = q[q.k == k]
    ax.scatter(sel.k, sel.silhueta, s=110, color=CORES[0], zorder=3,
               edgecolor='white', linewidth=2)
    ax.annotate(f'k = {k} · silhueta {sel.silhueta.iloc[0]:.3f}',
                (k, sel.silhueta.iloc[0]), textcoords='offset points',
                xytext=(14, -4), fontsize=10, color=CORES[0], fontweight='bold',
                va='center', ha='left')
    ax.set_ylim(q.silhueta.min() - 0.012, q.silhueta.max() + 0.012)
    ax.set_xlabel('Número de grupos (k)', color=TINTA2, fontsize=10)
    ax.set_ylabel('Silhueta média', color=TINTA2, fontsize=10)
    ax.set_title('Coeficiente de silhueta', color=TINTA, fontsize=11, loc='left', pad=10)
    ax.grid(axis='y', color=GRADE, lw=0.8)
    ax.set_axisbelow(True)
    salvar(fig, 'perfil_escolha_k')

    # ------------------------------------------------------------------
    # Figura 2 · características por grupo
    # ------------------------------------------------------------------
    ordem = desvios.abs().max(axis=1).sort_values().index
    d = desvios.loc[ordem]
    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    estilo(ax)
    pos = np.arange(len(d))
    altura = 0.26
    for i, grupo in enumerate(range(k)):
        nome = PERFIS[grupo]
        ax.barh(pos + (i - 1) * altura, d[nome], height=altura * 0.88,
                color=CORES[grupo], label=nome, zorder=2)
    ax.axvline(0, color='#c3c2b7', lw=1.2, zorder=1)
    ax.set_yticks(pos)
    ax.set_yticklabels(d.index, fontsize=9.5)
    ax.set_xlabel('Distância da média geral, em desvios-padrão', color=TINTA2, fontsize=10)
    ax.set_title('Características de cada grupo', color=TINTA, fontsize=12,
                 loc='left', pad=12)
    ax.grid(axis='x', color=GRADE, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', frameon=False, fontsize=9.5)
    salvar(fig, 'perfil_caracteristicas')

    # ------------------------------------------------------------------
    # Figura 3 · projeção bidimensional
    # ------------------------------------------------------------------
    pontos = pca.transform(prep.transform(X))
    variancia = pca.explained_variance_ratio_.sum()
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    estilo(ax)
    for grupo in range(k):
        m = rotulos == grupo
        ax.scatter(pontos[m, 0], pontos[m, 1], s=13, alpha=0.45,
                   color=CORES[grupo], linewidths=0, zorder=2)
    for grupo in range(k):
        m = rotulos == grupo
        cx, cy = pontos[m, 0].mean(), pontos[m, 1].mean()
        ax.scatter([cx], [cy], s=200, color=CORES[grupo], edgecolor='white',
                   linewidth=2.5, zorder=4)
        ax.annotate(PERFIS[grupo], (cx, cy), textcoords='offset points',
                    xytext=(0, 20), ha='center', fontsize=11, fontweight='bold',
                    color=CORES[grupo], zorder=5,
                    bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                              edgecolor='none', alpha=0.85))
    ax.set_xlabel(f'Componente principal 1 ({pca.explained_variance_ratio_[0]:.0%} da variância)',
                  color=TINTA2, fontsize=10)
    ax.set_ylabel(f'Componente principal 2 ({pca.explained_variance_ratio_[1]:.0%} da variância)',
                  color=TINTA2, fontsize=10)
    ax.set_title('Projeção bidimensional dos grupos', color=TINTA, fontsize=12,
                 loc='left', pad=12)
    ax.grid(color=GRADE, lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.text(0.01, 0.015,
             f'As duas componentes retêm {variancia:.0%} da variância dos 12 atributos. '
             'A figura resume, mas não reproduz,\nas distâncias usadas no agrupamento: '
             'grupos que se tocam na projeção podem estar separados no espaço completo.',
             fontsize=9, color=MUTED, va='bottom', linespacing=1.5)
    salvar(fig, 'perfil_projecao', ajustar=False)

    # ------------------------------------------------------------------
    # Figura 4 · tamanho e churn
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    nomes = [PERFIS[g] for g in tamanho.grupo]
    cores = [CORES[g] for g in tamanho.grupo]

    ax = estilo(axes[0])
    barras = ax.bar(nomes, tamanho.clientes, color=cores, width=0.6, zorder=2)
    for barra, valor, pct in zip(barras, tamanho.clientes, tamanho.participacao):
        ax.annotate(f'{valor}\n({pct:.1%})', (barra.get_x() + barra.get_width() / 2, valor),
                    textcoords='offset points', xytext=(0, 5), ha='center',
                    fontsize=9.5, color=TINTA2)
    ax.set_ylabel('Clientes', color=TINTA2, fontsize=10)
    ax.set_title('Tamanho do grupo', color=TINTA, fontsize=11, loc='left', pad=10)
    ax.set_ylim(0, tamanho.clientes.max() * 1.22)
    ax.grid(axis='y', color=GRADE, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', labelsize=9)

    ax = estilo(axes[1])
    barras = ax.bar(nomes, tamanho.taxa_churn * 100, color=cores, width=0.6, zorder=2)
    for barra, valor in zip(barras, tamanho.taxa_churn * 100):
        ax.annotate(f'{valor:.1f}%', (barra.get_x() + barra.get_width() / 2, valor),
                    textcoords='offset points', xytext=(0, 5), ha='center',
                    fontsize=9.5, color=TINTA2)
    media = 100 * y.mean()
    ax.axhline(media, color=MUTED, ls='--', lw=1.3, zorder=3)
    ax.set_ylabel('Taxa de cancelamento (%)', color=TINTA2, fontsize=10)
    ax.set_title('Cancelamento observado por grupo', color=TINTA, fontsize=11,
                 loc='left', pad=10)
    topo = max(tamanho.taxa_churn.max() * 100, media) * 1.25
    ax.set_ylim(0, topo)
    ax.text(0.015, media / topo + 0.015, f'média geral: {media:.2f}%',
            transform=ax.transAxes, fontsize=9, color=MUTED, va='bottom',
            ha='left', zorder=5)
    ax.grid(axis='y', color=GRADE, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', labelsize=9)
    salvar(fig, 'perfil_tamanho_churn')

    print('\ntabelas salvas em reports/tabelas/perfil_*.csv')


if __name__ == '__main__':
    main()