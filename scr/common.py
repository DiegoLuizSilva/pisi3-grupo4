"""Preparação determinística equivalente ao notebook 02; nenhum fit global."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
ROOT = Path(__file__).resolve().parents[1]
SEED = 42
BIN = ['complains', 'tariff_plan', 'status']

def load_data():
    df = pd.read_csv(ROOT / 'data/raw/iranian_churn.csv')
    df.columns = df.columns.str.strip().str.lower().str.replace(r'\s+', '_', regex=True)
    assert df.shape == (3150, 14) and not df.isna().any().any()
    original = df.drop(columns='churn')
    groups, _ = pd.factorize(pd.MultiIndex.from_frame(original), sort=False)
    X = original.drop(columns='age_group').copy()
    for c in ['tariff_plan', 'status']:
        X[c] = X[c].map({1: 0, 2: 1})
    y = df.churn.copy()
    assert not X.isna().any().any() and set(y) == {0, 1}
    tr, te = next(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, groups))
    assert set(groups[tr]).isdisjoint(groups[te])
    assert not set(map(tuple, X.iloc[tr].values)) & set(map(tuple, X.iloc[te].values))
    return df, X, y, groups, tr, te

def preprocessor(X, scale='standard'):
    scaler = {'standard': StandardScaler(), 'minmax': MinMaxScaler(), 'none': 'passthrough'}[scale]
    return ColumnTransformer([('numeric', scaler, [c for c in X if c not in BIN]),
                              ('binary', 'passthrough', BIN)], verbose_feature_names_out=False)

def savefig(fig, name):
    fig.tight_layout()
    for ext in ['png', 'svg']:
        fig.savefig(ROOT / f'reports/figuras/{name}.{ext}', dpi=300, bbox_inches='tight')

def fingerprint():
    return hashlib.sha256((ROOT / 'data/raw/iranian_churn.csv').read_bytes()).hexdigest()
