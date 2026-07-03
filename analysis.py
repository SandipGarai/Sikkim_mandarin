# %% [markdown]
# Modelling fruit yield of Sikkim mandarin (Citrus reticulata Blanco) from
# fruit-set and fruit-quality traits.
#
# The ANN hyper-parameters are optimised with the Canine Olfactory Optimization
# (COO) metaheuristic (Garai et al. 2026). Performance is reported on an
# independent train / validation / test partition and by leave-one-out
# cross-validation, and benchmarked against multiple linear regression (MLR).
#
# Run as an ipykernel script (cells delimited by `# %%`).

# %% Imports and global settings
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_predict, LeaveOneOut
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from coo_algorithm import COO

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

ROOT = Path(__file__).resolve(
).parent if "__file__" in globals() else Path(".")
DATA = ROOT / "data" / "data.csv"
FIGDIR = ROOT / "figures"
OUTDIR = ROOT / "outputs"
FIGDIR.mkdir(exist_ok=True)
OUTDIR.mkdir(exist_ok=True)

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "axes.linewidth": 0.8,
    "savefig.dpi": 600,
    "figure.dpi": 120,
})

# %% Load data and standardise column names
raw = pd.read_csv(DATA, index_col=0)
rename = {
    "Yield (kg)": "Yield", "PBFS (%)": "PBFS", "TSS ?B": "TSS",
    "TSS \u02daB": "TSS", "PH": "pH", "ACIDITY (%)": "Acidity",
    "FW (g)": "FW", "FD (cm)": "FD", "NS": "NS", "PHFS (%)": "PHFS",
}
df = raw.rename(columns=rename).apply(pd.to_numeric, errors="coerce")

TARGET = "Yield"
PREDICTORS = ["PBFS", "TSS", "pH", "Acidity", "FW", "FD", "NS", "PHFS"]
ALLVARS = [TARGET] + PREDICTORS
LABELS = {
    "Yield": "Yield (kg)", "PBFS": "PBFS (%)", "TSS": "TSS (\u00b0Brix)",
    "pH": "pH", "Acidity": "Acidity (%)", "FW": "FW (g)", "FD": "FD (cm)",
    "NS": "NS", "PHFS": "PHFS (%)",
}
print("Observations:", df.shape[0], "| Variables:", df.shape[1],
      "| Missing:", int(df.isna().sum().sum()))

# %% Table 1 - descriptive statistics
desc = pd.DataFrame({
    "Mean": df[ALLVARS].mean(), "SD": df[ALLVARS].std(ddof=1),
    "Min": df[ALLVARS].min(), "Max": df[ALLVARS].max(),
})
desc["CV (%)"] = 100 * desc["SD"] / desc["Mean"]
desc.insert(0, "Variable", [LABELS[v] for v in desc.index])
desc = desc.round(2).reset_index(drop=True)
desc.to_csv(OUTDIR / "table1_descriptive_statistics.csv", index=False)
print(desc.to_string(index=False))


# %% Correlation matrix with significance stars
def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


corr = df[ALLVARS].corr(method="pearson")
pmat = pd.DataFrame(
    [[stats.pearsonr(df[i], df[j])[1] for j in ALLVARS] for i in ALLVARS],
    index=ALLVARS, columns=ALLVARS)

# Table 2: r with stars (diagonal = 1)
t2 = pd.DataFrame(index=ALLVARS, columns=ALLVARS, dtype=object)
for i in ALLVARS:
    for j in ALLVARS:
        t2.loc[i,
               j] = "1" if i == j else f"{corr.loc[i, j]:.2f}{stars(pmat.loc[i, j])}"
t2_out = t2.copy()
t2_out.index = [LABELS[v] for v in t2_out.index]
t2_out.columns = [LABELS[v] for v in t2_out.columns]
t2_out.to_csv(OUTDIR / "table2_correlation_matrix.csv")
print(t2_out.to_string())

yc = corr[TARGET].drop(TARGET).sort_values(
    key=lambda s: s.abs(), ascending=False)
print("\nCorrelation with Yield:")
for v in yc.index:
    print(
        f"  {LABELS[v]:14s} r={corr.loc[TARGET, v]:+.3f} p={pmat.loc[TARGET, v]:.4f} {stars(pmat.loc[TARGET, v])}")

# %% Figure 1 - lower-triangular correlation heat map with significance stars
n = len(ALLVARS)
M = corr.values.copy()
# hide diagonal + upper triangle
mask = np.triu(np.ones_like(M, dtype=bool), k=0)
Mm = np.ma.masked_where(mask, M)

fig, ax = plt.subplots(figsize=(6.8, 6.0))
cmap = plt.cm.RdBu_r.copy()
cmap.set_bad("white")
im = ax.imshow(Mm, cmap=cmap, vmin=-1, vmax=1, aspect="equal")
labs = [LABELS[v] for v in ALLVARS]
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(labs, rotation=45, ha="right")
ax.set_yticklabels(labs)
for i in range(n):
    for j in range(n):
        if not mask[i, j]:
            s = stars(pmat.iloc[i, j])
            ax.text(j, i, s if s else "ns", ha="center", va="center",
                    color="white" if abs(M[i, j]) > 0.6 else "black", fontsize=9)
ax.set_xticks(np.arange(-.5, n, 1), minor=True)
ax.set_yticks(np.arange(-.5, n, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.0)
ax.tick_params(which="minor", length=0)
for spine in ax.spines.values():
    spine.set_visible(False)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Pearson's correlation coefficient (r)", rotation=90)
cbar.outline.set_linewidth(0.8)
ax.text(0.70, 0.40,
        "Significance\n*** p < 0.001\n**  p < 0.01\n*   p < 0.05\nns  not significant",
        transform=ax.transAxes, fontsize=10, va="center", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.6", lw=0.8))
fig.tight_layout()
fig.savefig(FIGDIR / "Fig1_correlation_heatmap.png", bbox_inches="tight")
fig.savefig(FIGDIR / "Fig1_correlation_heatmap.tiff", bbox_inches="tight")
plt.show()
print("Saved Figure 1")

# %% Train / validation / test partition (y-stratified, 20/5/5)
X = df[PREDICTORS].values.astype(float)
y = df[TARGET].values.astype(float)
order = np.argsort(y)
idx_tr, idx_va, idx_te = [], [], []
for rank, idx in enumerate(order):
    m = rank % 6
    (idx_va if m == 2 else idx_te if m == 5 else idx_tr).append(idx)
idx_tr, idx_va, idx_te = map(np.array, (idx_tr, idx_va, idx_te))
setname = np.empty(len(y), dtype=object)
setname[idx_tr], setname[idx_va], setname[idx_te] = "Training", "Validation", "Test"
Xtr, ytr = X[idx_tr], y[idx_tr]
Xva, yva = X[idx_va], y[idx_va]
Xte, yte = X[idx_te], y[idx_te]
print(
    f"Partition -> train {len(idx_tr)}, validation {len(idx_va)}, test {len(idx_te)}")


def metrics(yt, yp):
    return dict(R2=r2_score(yt, yp), RMSE=np.sqrt(mean_squared_error(yt, yp)),
                MAE=mean_absolute_error(yt, yp))


# %% COO-optimised ANN
ACT = ["tanh", "relu", "logistic"]


def decode(pos):
    h1 = max(1, int(round(pos[0])))
    h2 = int(round(pos[1]))
    alpha = float(10 ** pos[2])
    act = ACT[min(len(ACT) - 1, int(pos[3]))]
    hidden = (h1,) if h2 < 1 else (h1, h2)
    return hidden, alpha, act


def build_ann(pos):
    hidden, alpha, act = decode(pos)
    return Pipeline([
        ("sc", StandardScaler()),
        ("mlp", MLPRegressor(hidden_layer_sizes=hidden, activation=act,
                             solver="lbfgs", alpha=alpha, max_iter=4000,
                             random_state=RANDOM_STATE))])


_cvk = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def coo_objective(pos):
    """Maximise 5-fold cross-validated Q2 within the TRAINING set only.
    The validation and test sets are never seen during optimisation."""
    try:
        oof = cross_val_predict(build_ann(pos), Xtr, ytr, cv=_cvk)
        return float(r2_score(ytr, oof))
    except Exception:
        return -1e3


# search space: hidden1 (2-16), hidden2 (0-8; 0 = single hidden layer),
# log10(L2 alpha) in [-4, 2], activation index -> {tanh, relu, logistic}
bounds = [(2, 16), (0, 8), (-4.0, 2.0), (0, 2.999)]
optimizer = COO(bounds=bounds, n_packs=2, init_pack_size=8, max_iterations=20,
                surrogate_enabled=False, random_state=RANDOM_STATE, verbose=False)
best_pos, best_fit, conv_hist, diag, _ = optimizer.optimize(coo_objective)
best_hidden, best_alpha, best_act = decode(best_pos)
print(
    f"COO best: hidden={best_hidden}, alpha={best_alpha:.4g}, act={best_act}")
print(f"  training 5-fold Q2 (objective) = {best_fit:.3f} | "
      f"exact evals = {diag['exact_evals']}, iterations = {diag['iterations']}")

ann = build_ann(best_pos)
ann.fit(Xtr, ytr)
mlr = Pipeline([("sc", StandardScaler()), ("lr", LinearRegression())])
mlr.fit(Xtr, ytr)

# %% Performance table (train / validation / test / LOOCV) for ANN and MLR
loo = LeaveOneOut()
rows = {}
for name, model in [("ANN", ann), ("MLR", mlr)]:
    rows[(name, "Training")] = metrics(ytr, model.predict(Xtr))
    rows[(name, "Validation")] = metrics(yva, model.predict(Xva))
    rows[(name, "Test")] = metrics(yte, model.predict(Xte))
    proto = build_ann(best_pos) if name == "ANN" else Pipeline(
        [("sc", StandardScaler()), ("lr", LinearRegression())])
    oof_full = cross_val_predict(proto, X, y, cv=loo)
    rows[(name, "LOOCV (all data)")] = metrics(y, oof_full)

perf = pd.DataFrame(rows).T[["R2", "RMSE", "MAE"]].round(3)
perf.index = pd.MultiIndex.from_tuples(perf.index, names=["Model", "Subset"])
perf.to_csv(OUTDIR / "table3_model_performance.csv")
print(perf.to_string())

ann_pred_all = ann.predict(X)
mlr_pred_all = mlr.predict(X)

# %% Figure 2 - observed vs predicted, coloured by data subset
set_colors = {"Training": "#1f77b4",
              "Validation": "#ff7f0e", "Test": "#d62728"}
fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.9))
lo = min(y.min(), ann_pred_all.min(), mlr_pred_all.min()) - 1
hi = max(y.max(), ann_pred_all.max(), mlr_pred_all.max()) + 1
for ax, tag, pred, name in [(axes[0], "(a)", ann_pred_all, "ANN"),
                            (axes[1], "(b)", mlr_pred_all, "MLR")]:
    for s in ["Training", "Validation", "Test"]:
        m = setname == s
        ax.scatter(y[m], pred[m], s=48, c=set_colors[s], edgecolors="black",
                   linewidths=0.6, alpha=0.9, label=s, zorder=3)
    ax.plot([lo, hi], [lo, hi], "--", color="0.35", linewidth=1.0, zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Observed yield (kg)")
    ax.set_ylabel("Predicted yield (kg)")
    ax.text(0.04, 0.95, tag, transform=ax.transAxes,
            fontweight="bold", va="top")
    m_te = setname == "Test"                    # observed & predicted aligned
    # (same order) -> matches Table 3
    r2te = r2_score(y[m_te], pred[m_te])
    ax.text(0.96, 0.06, f"{name}\nTest $R^2$ = {r2te:.2f}", transform=ax.transAxes,
            ha="right", va="bottom")
    ax.set_aspect("equal", adjustable="box")
axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, 0.90), frameon=False,
               handletextpad=0.3, borderpad=0.2)
fig.tight_layout()
fig.savefig(FIGDIR / "Fig2_observed_vs_predicted.png", bbox_inches="tight")
fig.savefig(FIGDIR / "Fig2_observed_vs_predicted.tiff", bbox_inches="tight")
plt.show()
print("Saved Figure 2")

# %% Permutation importance from the best (COO-optimised) ANN, refit on all data
best_ann_full = build_ann(best_pos).fit(X, y)
perm = permutation_importance(best_ann_full, X, y, n_repeats=200,
                              random_state=RANDOM_STATE, scoring="r2")
imp = pd.DataFrame({"Predictor": [LABELS[p] for p in PREDICTORS],
                    "Importance": perm.importances_mean,
                    "SD": perm.importances_std}
                   ).sort_values("Importance", ascending=False).reset_index(drop=True)
pos_imp = imp["Importance"].clip(lower=0)
imp["Relative (%)"] = (100 * pos_imp / pos_imp.sum()).round(1)
imp.round(4).to_csv(OUTDIR / "table_permutation_importance.csv", index=False)
print(imp.to_string(index=False))

# %% Figure 3 - permutation importance, colour graded by importance
order_imp = imp.sort_values("Importance")
vals = order_imp["Importance"].values
norm = mpl.colors.Normalize(vmin=vals.min(), vmax=vals.max())
colors = plt.cm.viridis(norm(vals))
fig, ax = plt.subplots(figsize=(6.8, 4.5))
ax.barh(order_imp["Predictor"], vals, xerr=order_imp["SD"], color=colors,
        edgecolor="black", linewidth=0.6,
        error_kw=dict(ecolor="0.25", lw=0.8, capsize=3))
ax.set_xlabel("Permutation importance (mean decrease in $R^2$)")
ax.axvline(0, color="black", linewidth=0.8)
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])
cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.03)
cb.set_label("Importance", rotation=90)
cb.outline.set_linewidth(0.8)
fig.tight_layout()
fig.savefig(FIGDIR / "Fig3_variable_importance.png", bbox_inches="tight")
fig.savefig(FIGDIR / "Fig3_variable_importance.tiff", bbox_inches="tight")
plt.show()
print("Saved Figure 3")

# %% Save COO summary and print manuscript summary
coo_summary = {
    "hidden_layers": str(best_hidden), "alpha": round(best_alpha, 4),
    "activation": best_act, "train_cv_Q2": round(best_fit, 3),
    "exact_evals": diag["exact_evals"], "iterations": diag["iterations"],
}
pd.Series(coo_summary).to_csv(OUTDIR / "coo_selected_model.csv")

print("\n================ SUMMARY FOR MANUSCRIPT ================")
print(
    f"n = {df.shape[0]} (train {len(idx_tr)} / val {len(idx_va)} / test {len(idx_te)})")
print(f"COO-selected ANN: hidden={best_hidden}, activation={best_act}, "
      f"L2 alpha={best_alpha:.3g}")
for (mdl, sub), row in perf.iterrows():
    print(
        f"  {mdl:3s} {sub:16s} R2={row['R2']:.3f} RMSE={row['RMSE']:.3f} MAE={row['MAE']:.3f}")
print("Top predictors:", ", ".join(imp["Predictor"].head(3)))
print("=======================================================")
