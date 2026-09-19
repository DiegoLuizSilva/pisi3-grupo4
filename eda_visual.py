import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

CSV = sys.argv[1] if len(sys.argv) > 1 else "ml/data/raw/iranian_churn.csv"
OUT = "ml/reports"
FIG = f"{OUT}/figuras"
os.makedirs(FIG, exist_ok=True)

df = pd.read_csv(CSV)
df.columns = [" ".join(c.split()).strip() for c in df.columns]

numericas = ["Call Failure", "Subscription Length", "Charge Amount", "Seconds of Use",
             "Frequency of use", "Frequency of SMS", "Distinct Called Numbers",
             "Age Group", "Age", "Customer Value"]
categoricas = ["Complains", "Tariff Plan", "Status", "Age Group"]

print("=" * 70)
print("1. DISTRIBUICAO DAS VARIAVEIS NUMERICAS")
print("=" * 70)

fig, axes = plt.subplots(4, 3, figsize=(15, 16))
for ax, c in zip(axes.flat, numericas):
    ax.hist(df[c], bins=30, color="#4C72B0", edgecolor="white")
    ax.axvline(df[c].mean(), color="#C44E52", linestyle="--", linewidth=1.5, label="média")
    ax.axvline(df[c].median(), color="#55A868", linestyle="-", linewidth=1.5, label="mediana")
    ax.set_title(c, fontsize=10)
    ax.legend(fontsize=7)
for ax in axes.flat[len(numericas):]:
    ax.axis("off")
fig.suptitle("Distribuicao das variaveis numericas", fontsize=14)
fig.tight_layout()
fig.savefig(f"{FIG}/histogramas.png", dpi=120)
plt.close(fig)

fig, axes = plt.subplots(4, 3, figsize=(15, 16))
for ax, c in zip(axes.flat, numericas):
    dados = [df[df["Churn"] == 0][c], df[df["Churn"] == 1][c]]
    bp = ax.boxplot(dados, tick_labels=["ficou", "cancelou"], patch_artist=True, widths=0.6)
    for patch, cor in zip(bp["boxes"], ["#4C72B0", "#C44E52"]):
        patch.set_facecolor(cor)
        patch.set_alpha(0.7)
    ax.set_title(c, fontsize=10)
for ax in axes.flat[len(numericas):]:
    ax.axis("off")
fig.suptitle("Boxplots por classe de Churn", fontsize=14)
fig.tight_layout()
fig.savefig(f"{FIG}/boxplots.png", dpi=120)
plt.close(fig)

assim = df[numericas].skew().sort_values(ascending=False).round(2)
print("assimetria (skewness):")
print(assim.to_string())
print()
print("interpretacao: > 1 = cauda longa a direita, ~0 = simetrica")

print()
print("=" * 70)
print("2. MATRIZ DE CORRELACAO")
print("=" * 70)

corr = df[numericas + ["Churn"]].corr(method="spearman")

fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)))
ax.set_yticks(range(len(corr)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(corr.columns, fontsize=9)
for i in range(len(corr)):
    for j in range(len(corr)):
        v = corr.iloc[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                color="white" if abs(v) > 0.55 else "black")
fig.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Matriz de correlacao (Spearman)", fontsize=13)
fig.tight_layout()
fig.savefig(f"{FIG}/matriz_correlacao.png", dpi=120)
plt.close(fig)
corr.round(3).to_csv(f"{OUT}/matriz_correlacao.csv")

print("correlacao de cada variavel com Churn (Spearman):")
alvo = corr["Churn"].drop("Churn").sort_values(key=abs, ascending=False).round(3)
print(alvo.to_string())

print()
print("pares de variaveis com correlacao acima de 0.6 (colinearidade):")
pares = []
cols = numericas
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        v = corr.loc[cols[i], cols[j]]
        if abs(v) > 0.6:
            pares.append((cols[i], cols[j], round(v, 3)))
for a, b, v in sorted(pares, key=lambda x: -abs(x[2])):
    print(f"  {a}  x  {b}:  {v}")

print()
print("=" * 70)
print("3. COMPARACAO DE MEDIAS: CANCELOU x FICOU")
print("=" * 70)

linhas = []
for c in numericas:
    a = df[df["Churn"] == 0][c]
    b = df[df["Churn"] == 1][c]
    dp = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    d = (b.mean() - a.mean()) / dp
    u, p = stats.mannwhitneyu(a, b)
    linhas.append({
        "variavel": c,
        "media_ficou": round(a.mean(), 2),
        "media_cancelou": round(b.mean(), 2),
        "diferenca_%": round(100 * (b.mean() - a.mean()) / a.mean(), 1) if a.mean() != 0 else None,
        "cohen_d": round(d, 3),
        "p_valor": f"{p:.2e}",
    })
comp = pd.DataFrame(linhas).sort_values("cohen_d", key=abs, ascending=False)
print(comp.to_string(index=False))
comp.to_csv(f"{OUT}/comparacao_medias.csv", index=False)
print()
print("cohen_d: 0.2 pequeno | 0.5 medio | 0.8 grande. sinal negativo = menor em quem cancelou")

fig, ax = plt.subplots(figsize=(9, 6))
ordem = comp.sort_values("cohen_d")
cores = ["#C44E52" if v > 0 else "#4C72B0" for v in ordem["cohen_d"]]
ax.barh(ordem["variavel"], ordem["cohen_d"], color=cores)
ax.axvline(0, color="black", linewidth=0.8)
for x in [-0.8, -0.5, -0.2, 0.2, 0.5, 0.8]:
    ax.axvline(x, color="gray", linestyle=":", linewidth=0.7)
ax.set_xlabel("Cohen's d  (positivo = maior em quem cancelou)")
ax.set_title("Tamanho do efeito por variavel")
fig.tight_layout()
fig.savefig(f"{FIG}/tamanho_efeito.png", dpi=120)
plt.close(fig)

print()
print("=" * 70)
print("4. VARIAVEIS CATEGORICAS POR TAXA DE CHURN")
print("=" * 70)

rotulos = {
    "Complains": {0: "0 - nao reclamou", 1: "1 - reclamou"},
    "Tariff Plan": {1: "1 - pre-pago", 2: "2 - contrato"},
    "Status": {1: "1 - ativo", 2: "2 - inativo"},
    "Age Group": {1: "1 - mais jovem", 2: "2", 3: "3", 4: "4", 5: "5 - mais velho"},
}

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, c in zip(axes.flat, categoricas):
    tab = df.groupby(c)["Churn"].agg(["count", "mean"])
    tab.columns = ["clientes", "taxa_churn"]
    tab["taxa_churn_%"] = (100 * tab["taxa_churn"]).round(2)
    tab["cancelaram"] = (tab["clientes"] * tab["taxa_churn"]).round().astype(int)
    tab.index = [rotulos[c].get(i, str(i)) for i in tab.index]
    print(f"\n[{c}]")
    print(tab[["clientes", "cancelaram", "taxa_churn_%"]].to_string())

    ax.bar(range(len(tab)), tab["taxa_churn_%"], color="#C44E52", alpha=0.85)
    ax.axhline(100 * df["Churn"].mean(), color="black", linestyle="--",
               linewidth=1.2, label=f"media geral {100*df['Churn'].mean():.2f}%")
    ax.set_xticks(range(len(tab)))
    ax.set_xticklabels(tab.index, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("taxa de churn (%)")
    ax.set_title(c, fontsize=11)
    ax.legend(fontsize=8)
    for i, v in enumerate(tab["taxa_churn_%"]):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=8)
fig.suptitle("Taxa de churn por variavel categorica", fontsize=14)
fig.tight_layout()
fig.savefig(f"{FIG}/churn_categoricas.png", dpi=120)
plt.close(fig)

print()
print("teste qui-quadrado de independencia com Churn:")
for c in categoricas:
    tab = pd.crosstab(df[c], df["Churn"])
    chi2, p, gl, esp = stats.chi2_contingency(tab)
    v = np.sqrt(chi2 / (len(df) * (min(tab.shape) - 1)))
    print(f"  {c:<14} chi2={chi2:9.2f}  p={p:.2e}  V de Cramer={v:.3f}")

print()
print("figuras salvas em", FIG)
print("tabelas salvas em", OUT)