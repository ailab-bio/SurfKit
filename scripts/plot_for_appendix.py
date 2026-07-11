#!/usr/bin/env python3
"""
plot_for_appendix.py — Unified appendix figure generation for SurfPro-MD.

Produces all figures that appear in the Supporting Information:

  SHAP feature attribution (Section app:shap)
    figures_appendix/shap_feature_count_dist.png
    figures_appendix/shap_crossover_heatmap.png
    figures_appendix/shap_per_target_ranked.png
    figures_appendix/shap_presence_matrix.png
    figures_appendix/shap_universal_features.png

  SHAP importance decay (Section app:shap, continued)
    figures_appendix/shap_decay_grid.png
    figures_appendix/shap_decay_overlay.png

  Uncertainty estimation diagnostics (Sections app:unc_diagnostics, app:conformal)
    figures_appendix/unc_error_vs_nndist.pdf
    figures_appendix/unc_error_vs_ens_std.pdf
    figures_appendix/unc_conformal_calibration.pdf

SHAP values are cached in shap_results_cache.pkl (same directory as this
script) so that reruns skip the expensive TreeExplainer computation.

Usage (from any directory):
    python surrogate-models/plot_for_appendix.py
"""

import os
import pickle
import re
import string
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr, sem
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import shap

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

# ── Paths (resolved relative to this script, regardless of cwd) ───────────────

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR    = SCRIPT_DIR / ".." / "paper" / "figures_appendix"
OUT_DIR.mkdir(exist_ok=True)

DATA_FILE   = SCRIPT_DIR / ".." / "data" / "SurfPro-MD.csv"
MODELS_FILE = SCRIPT_DIR.parent / "models" / "models.pkl"
CACHE_FILE  = SCRIPT_DIR.parent / "models" / "shap_results_cache.pkl"

# ── Constants (must match surrogate.py) ──────────────────────────────────────

TARGETS = [
    "pCMC", "AW_ST_CMC", "Gamma_max", "Area_min", "pC20",
    "D_MOL", "D_SOL", "surface_tension_avg", "viscosity",
]

TARGET_LABELS = {
    "pCMC":                "pCMC",
    "AW_ST_CMC":           r"$\gamma_{\rm CMC}$",
    "Gamma_max":           r"$\Gamma_{\max}$",
    "pC20":                r"pC$_{20}$",
    "Area_min":            r"A$_{\min}$",
    "viscosity":           r"$\eta$",
    "D_MOL":               r"D$_{\rm MOL}$",
    "D_SOL":               r"D$_{\rm SOL}$",
    "surface_tension_avg": r"$\gamma$",
}

TARGET_LABELS_UNC = {
    "pCMC":                "pCMC",
    "AW_ST_CMC":           r"$\gamma_{\mathrm{CMC}}$",
    "Gamma_max":           r"$\Gamma_{\max}$",
    "pC20":                r"pC$_{20}$",
    "Area_min":            r"A$_{\min}$",
    "viscosity":           r"$\eta$",
    "D_MOL":               r"D$_{\mathrm{MOL}}$",
    "D_SOL":               r"D$_{\mathrm{SOL}}$",
    "surface_tension_avg": r"$\gamma$",
}

UNIT_LABELS = {
    "pCMC":                "pCMC",
    "AW_ST_CMC":           r"$\gamma_{\mathrm{CMC}}$ [mN m$^{-1}$]",
    "Gamma_max":           r"$\Gamma_{\max}$ [$\mu$mol m$^{-2}$]",
    "pC20":                r"pC$_{20}$",
    "Area_min":            r"A$_{\min}$ [nm$^2$]",
    "viscosity":           r"$\eta$ [mPa$\cdot$s]",
    "D_MOL":               r"D$_\mathrm{MOL}$ [$\times\!10^{-9}$ m$^2$s$^{-1}$]",
    "D_SOL":               r"D$_\mathrm{SOL}$ [$\times\!10^{-9}$ m$^2$s$^{-1}$]",
    "surface_tension_avg": r"$\gamma$ [mN m$^{-1}$]",
}

_DISPLAY_SCALE = {"viscosity": 1e3, "surface_tension_avg": 0.1}

N_FOLDS            = 5
N_TEST_SPLITS      = 5
MIN_TEST_FRACTION  = 0.07
MAX_TEST_FRACTION  = 0.11
MAX_REJECT_TRIES   = 5
MIN_VAL_FRACTION   = 0.12
MAX_VAL_FRACTION   = 0.18
RANDOM_STATE       = 3
NN_ISOLATION_THRESHOLD = 50.0
NN_DIST_MAX        = 40

# SHAP overlap parameters
TOP_K      = 20   # top-K per target for overlap counting
N_HEATMAP  = 20   # max features shown in heatmap (×N embedded in label)
N_PRESENCE = 30   # max features in presence matrix
N_PER_TARGET = 15 # top features shown per-target bar chart

ALPHABET     = list(string.ascii_lowercase)

# ── Shared plotting style (SHAP figures) ─────────────────────────────────────

sns.set_theme(context="notebook", style="whitegrid", font_scale=1.15)
plt.rcParams.update({
    "font.size":      12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "savefig.dpi":    200,
    "figure.dpi":     100,
})

# ── Data helpers (mirror of surrogate.py) ────────────────────────────────────

def _sample_test_clusters(clusters, cluster_sizes, min_size, max_size, rng,
                           max_rejects=5):
    available = set(clusters)
    test_clusters, test_count, streak = set(), 0, 0
    while available:
        c = rng.choice(list(available))
        available.remove(c)
        new_size = test_count + cluster_sizes[c]
        if new_size < min_size:
            test_clusters.add(c); test_count = new_size; streak = 0; continue
        if min_size <= new_size <= max_size:
            test_clusters.add(c); test_count = new_size; break
        streak += 1
        if streak >= max_rejects:
            break
    return test_clusters, test_count


def create_cluster_splits(df_in, target_col):
    df_t = df_in.dropna(subset=[target_col, "cluster"]).copy()
    df_t["cluster"] = df_t["cluster"].astype(int)
    total = len(df_t)
    cs_all   = df_t.groupby("cluster").size().to_dict()
    clusters = np.array(list(cs_all.keys()))
    for ts in range(N_TEST_SPLITS):
        rng = np.random.RandomState(RANDOM_STATE + ts)
        tc, _ = _sample_test_clusters(
            clusters, cs_all,
            int(total * MIN_TEST_FRACTION), int(total * MAX_TEST_FRACTION),
            rng, MAX_REJECT_TRIES,
        )
        col_test = f"{target_col}_test_split_{ts}"
        df_t[col_test] = "train_val"
        df_t.loc[df_t["cluster"].isin(tc), col_test] = "test"
        rem = df_t[df_t[col_test] != "test"]
        rc  = np.array(rem["cluster"].unique())
        rs  = rem.groupby("cluster").size().to_dict()
        rm  = len(rem)
        for fi in range(N_FOLDS):
            rng_f = np.random.RandomState(RANDOM_STATE + ts * 100 + fi)
            vc, _ = _sample_test_clusters(
                rc, rs,
                int(rm * MIN_VAL_FRACTION), int(rm * MAX_VAL_FRACTION),
                rng_f, MAX_REJECT_TRIES,
            )
            col = f"{target_col}_split_t{ts}_f{fi}"
            df_t[col] = "unused"
            df_t.loc[df_t["cluster"].isin(set(rc) - set(vc)), col] = "train"
            df_t.loc[df_t["cluster"].isin(vc),                col] = "val"
            df_t.loc[df_t[col_test] == "test",                col] = "test"
    return df_t


def filter_outliers(df, target):
    rules = {
        "surface_tension_avg": lambda d: d["surface_tension_avg"] >= 250,
        "viscosity":           lambda d: d["viscosity"] <= 0.003,
        "AW_ST_CMC":           lambda d: d["AW_ST_CMC"] <= 52,
        "Gamma_max":           lambda d: d["Gamma_max"] <= 6,
        "Area_min":            lambda d: d["Area_min"] <= 4.2,
        "pC20":                lambda d: d["pC20"] >= 1.8,
        "D_MOL":               lambda d: d["D_MOL"] <= 0.8,
    }
    if target in rules:
        return df[df[target].notnull() & rules[target](df)]
    return df


def filter_nn_isolated(df, target, feature_cols,
                        threshold=NN_ISOLATION_THRESHOLD):
    X = df[feature_cols].to_numpy().astype(float)
    col_means = np.nanmean(X, axis=0)
    for j in range(X.shape[1]):
        X[np.isnan(X[:, j]), j] = col_means[j]
    X_sc = StandardScaler().fit_transform(X)
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean", n_jobs=-1)
    nn.fit(X_sc)
    d = nn.kneighbors(X_sc)[0][:, 1]
    return df.loc[df.index[d <= threshold]]


# ── SHAP helper (XGBoost 2.x / SHAP version fix) ─────────────────────────────

def make_tree_explainer(booster):
    _orig = booster.save_raw

    def _patched(raw_format="ubj"):
        raw = _orig(raw_format=raw_format)
        if raw_format == "ubj":
            def _unwrap(m):
                val   = float(m.group(1))
                fixed = f"{val:.10g}".encode()
                pad   = len(m.group(0)) - len(fixed)
                return fixed + b" " * pad if pad >= 0 else fixed[:len(m.group(0))]
            raw = re.sub(rb"\[(-?[\d.]+(?:[Ee][+-]?\d+)?)\]", _unwrap, raw)
        return raw

    booster.save_raw = _patched
    try:
        explainer = shap.TreeExplainer(booster)
    finally:
        booster.save_raw = _orig
    return explainer


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA & COMPUTE BUTINA CLUSTERS
# ═══════════════════════════════════════════════════════════════════════════════

print("Loading dataset and computing Butina clusters...")
df = pd.read_csv(DATA_FILE)
df["mol"] = df["SMILES"].apply(Chem.MolFromSmiles)
df_valid  = df[df["mol"].notnull()].copy()

fps  = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in df_valid["mol"]]
nfps = len(fps)
dists = []
for i in range(1, nfps):
    sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
    dists.extend(1 - s for s in sims)

clusters_butina = Butina.ClusterData(dists, nfps, distThresh=0.6, isDistData=True)
cid = np.zeros(nfps, dtype=int)
for ci, cl in enumerate(clusters_butina):
    for idx in cl:
        cid[idx] = ci
df_valid["cluster"] = cid.astype(int)
df["cluster"] = np.nan
df.loc[df_valid.index, "cluster"] = df_valid["cluster"].values
print(f"  {len(clusters_butina)} clusters, {nfps} molecules.")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. LOAD MODELS
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\nLoading models from {MODELS_FILE}...")
with open(MODELS_FILE, "rb") as fh:
    results = pickle.load(fh)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. SHAP VALUES (compute or load from cache)
# ═══════════════════════════════════════════════════════════════════════════════

if CACHE_FILE.exists():
    print(f"\nLoading cached SHAP results from {CACHE_FILE}...")
    with open(CACHE_FILE, "rb") as fh:
        mean_abs_shap = pickle.load(fh)
    print("  Cache loaded.")
else:
    print("\nComputing population-level SHAP values (test sets, all splits & folds)...")
    mean_abs_shap = {}

    for target in TARGETS:
        print(f"\n  [{target}]")
        df_t         = create_cluster_splits(df, target)
        df_t         = filter_outliers(df_t, target)
        feature_cols = results[target][0]["feature_schema"]
        df_t         = filter_nn_isolated(df_t, target, feature_cols)

        pool_shap = []
        for ts in range(N_TEST_SPLITS):
            run         = results[target][ts]
            split_col   = run["split_column"]
            fold_models = run["fold_models"]
            test_mask   = df_t[split_col] == "test"
            X_test      = df_t.loc[test_mask, feature_cols].to_numpy().astype(float)
            col_means   = np.nanmean(X_test, axis=0)
            for j in range(X_test.shape[1]):
                X_test[np.isnan(X_test[:, j]), j] = col_means[j]
            for fi, fm in enumerate(fold_models):
                X_sc = fm["scaler"].transform(X_test)
                expl = make_tree_explainer(fm["model"])
                sv   = expl.shap_values(X_sc)
                pool_shap.append(np.abs(sv))
                print(f"    split {ts}, fold {fi}: {X_sc.shape[0]} test molecules",
                      flush=True)

        stacked = np.vstack(pool_shap)
        feature_cols_0 = results[target][0]["feature_schema"]
        mean_abs_shap[target] = pd.Series(stacked.mean(axis=0), index=feature_cols_0)
        print(f"  → pooled {stacked.shape[0]} rows, {stacked.shape[1]} features")

    print(f"\nSaving SHAP cache to {CACHE_FILE}...")
    with open(CACHE_FILE, "wb") as fh:
        pickle.dump(mean_abs_shap, fh)

# ── Build cross-target importance matrix ─────────────────────────────────────

shap_matrix      = pd.DataFrame(mean_abs_shap)
shap_matrix_norm = shap_matrix.div(shap_matrix.sum(axis=0), axis=1)

top_sets = {t: set(mean_abs_shap[t].nlargest(TOP_K).index) for t in TARGETS}
all_top  = set().union(*top_sets.values())
feature_counts = pd.Series(
    {f: sum(f in s for s in top_sets.values()) for f in all_top}
).sort_values(ascending=False)

multi_target  = feature_counts[feature_counts >= 2]
shared_strong = set(feature_counts[feature_counts >= 3].index)
shared_mild   = set(feature_counts[feature_counts == 2].index)
specific      = set(feature_counts[feature_counts == 1].index)

print(f"\nFeature overlap summary (top-{TOP_K} per target):")
print(f"  Features appearing in ≥3 targets : {len(shared_strong)}")
print(f"  Features appearing in exactly 2  : {len(shared_mild)}")
print(f"  Features appearing in only 1     : {len(specific)}")

n_features = len(next(iter(mean_abs_shap.values())))

# ═══════════════════════════════════════════════════════════════════════════════
# 4. SHAP FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n── SHAP figures ──────────────────────────────────────────────────────────")

# ── Figure S1: feature count distribution ────────────────────────────────────

cnt_dist   = feature_counts.value_counts().sort_index()
bar_colors = cm.viridis(np.linspace(0.15, 0.85, len(cnt_dist)))

fig1, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.bar(cnt_dist.index, cnt_dist.values, color=bar_colors,
        edgecolor="white", linewidth=0.8)
for x, y in zip(cnt_dist.index, cnt_dist.values):
    ax1.text(x, y + 0.4, str(y), ha="center", va="bottom",
             fontsize=11, fontweight="bold")
ax1.set_xlabel(f"Number of targets a feature ranks in top-{TOP_K}", fontsize=12)
ax1.set_ylabel("Number of features", fontsize=12)
ax1.set_title(
    f"Cross-target feature sharing\n"
    f"(top-{TOP_K} SHAP importance per target, {len(TARGETS)} targets)",
    fontsize=12,
)
ax1.set_xticks(cnt_dist.index)
ax1.set_xticklabels([str(i) for i in cnt_dist.index])
sns.despine(ax=ax1, left=False)
plt.tight_layout()
plt.savefig(OUT_DIR / "shap_feature_count_dist.png", bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT_DIR / 'shap_feature_count_dist.png'}")

# ── Figure S2: cross-target heatmap (improved — ×N in row label) ─────────────

n_show     = min(N_HEATMAP, len(multi_target))
top_shared = multi_target.head(n_show).index.tolist()

heat_data = shap_matrix_norm.loc[top_shared].copy()
heat_data = heat_data.div(heat_data.max(axis=0), axis=1)

row_labels = [
    f"{f.replace('rdkit-', '')}  ×{int(feature_counts[f])}"
    for f in top_shared
]
col_labels = [TARGET_LABELS[t] for t in TARGETS]

ROW_HEIGHT_IN = 0.46
fig_h = n_show * ROW_HEIGHT_IN + 2.5
fig_w = 10

fig2, ax2 = plt.subplots(figsize=(fig_w, fig_h))
sns.heatmap(
    heat_data, ax=ax2, cmap="viridis",
    xticklabels=col_labels, yticklabels=row_labels,
    linewidths=0.5, linecolor="#e8e8e8",
    cbar_kws={
        "label": "Relative importance\n(within-target, max-normalised)",
        "shrink": 0.45, "pad": 0.02,
    },
    vmin=0, vmax=1,
)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=40, ha="right", fontsize=12)
ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=9, va="center")
ax2.set_title(
    f"SHAP feature importance — features shared across ≥2 targets\n"
    f"(top-{TOP_K} per target;  {n_show} features shown;  ×N = number of targets)",
    fontsize=12, pad=10,
)
plt.tight_layout()
plt.savefig(OUT_DIR / "shap_crossover_heatmap.png", bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT_DIR / 'shap_crossover_heatmap.png'}")

# ── Figure S3: per-target top-N bar charts ───────────────────────────────────

COLOUR_STRONG   = "#1565c0"
COLOUR_MILD     = "#4fc3f7"
COLOUR_SPECIFIC = "#b0bec5"

fig3, axes3 = plt.subplots(3, 3, figsize=(17, 15))
axes3 = axes3.flatten()
for i, target in enumerate(TARGETS):
    ax    = axes3[i]
    top_f = mean_abs_shap[target].nlargest(N_PER_TARGET)
    names = [f.replace("rdkit-", "") for f in top_f.index]
    vals  = top_f.values
    y_pos = np.arange(N_PER_TARGET)
    colors = []
    for feat in top_f.index:
        cnt = feature_counts.get(feat, 0)
        if cnt >= 3:   colors.append(COLOUR_STRONG)
        elif cnt == 2: colors.append(COLOUR_MILD)
        else:          colors.append(COLOUR_SPECIFIC)
    ax.barh(y_pos[::-1], vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos[::-1])
    ax.set_yticklabels(names, fontsize=7.5)
    ax.set_title(TARGET_LABELS[target], fontsize=12)
    ax.set_xlabel("Mean |SHAP|", fontsize=9)
    sns.despine(ax=ax)

legend_patches = [
    mpatches.Patch(color=COLOUR_STRONG,   label=f"Shared ≥3 targets (top-{TOP_K})"),
    mpatches.Patch(color=COLOUR_MILD,     label=f"Shared 2 targets  (top-{TOP_K})"),
    mpatches.Patch(color=COLOUR_SPECIFIC, label="Target-specific"),
]
fig3.legend(handles=legend_patches, loc="lower center", ncol=3,
            fontsize=10, bbox_to_anchor=(0.5, -0.025),
            framealpha=0.9, edgecolor="lightgray")
fig3.suptitle(
    f"Top-{N_PER_TARGET} SHAP features per target  (colour = cross-target sharing)",
    fontsize=14, y=1.005,
)
plt.tight_layout()
plt.savefig(OUT_DIR / "shap_per_target_ranked.png", bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT_DIR / 'shap_per_target_ranked.png'}")

# ── Figure S4: binary presence matrix ────────────────────────────────────────

n_bin     = min(N_PRESENCE, len(multi_target))
top_bin   = multi_target.head(n_bin).index.tolist()
bin_names = [f.replace("rdkit-", "") for f in top_bin]
col_labels_bin = [TARGET_LABELS[t] for t in TARGETS]

binary = np.array([[1 if feat in top_sets[t] else 0 for t in TARGETS]
                   for feat in top_bin])

fig4_h = max(6, n_bin * 0.32)
fig4, ax4 = plt.subplots(figsize=(9, fig4_h))
cmap_bin = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
    "bin", ["#f5f5f5", "#1565c0"]
)
ax4.imshow(binary, cmap=cmap_bin, aspect="auto", vmin=0, vmax=1)
ax4.set_xticks(np.arange(len(TARGETS)))
ax4.set_yticks(np.arange(n_bin))
ax4.set_xticklabels(col_labels_bin, rotation=45, ha="right", fontsize=11)
ax4.set_yticklabels(bin_names, fontsize=8.5)
for x in np.arange(-0.5, len(TARGETS), 1):
    ax4.axvline(x, color="white", lw=1.8)
for y in np.arange(-0.5, n_bin, 1):
    ax4.axhline(y, color="white", lw=0.9)
for ri in range(n_bin):
    for ci in range(len(TARGETS)):
        if binary[ri, ci]:
            ax4.text(ci, ri, "●", ha="center", va="center",
                     fontsize=9, color="white", alpha=0.85)
for ri in range(n_bin):
    cnt = int(binary[ri].sum())
    ax4.text(len(TARGETS) - 0.5 + 0.6, ri, str(cnt),
             ha="left", va="center", fontsize=8.5, color="#333333")
ax4.text(len(TARGETS) - 0.5 + 0.6, -0.8, "#",
         ha="left", va="center", fontsize=8.5, color="#555555", style="italic")
ax4.set_title(
    f"Feature presence in top-{TOP_K} SHAP importance per target\n"
    f"({n_bin} shared features shown, ordered by target count)",
    fontsize=12, pad=10,
)
plt.tight_layout()
plt.savefig(OUT_DIR / "shap_presence_matrix.png", bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT_DIR / 'shap_presence_matrix.png'}")

# ── Figure S5: universal features grouped bar chart ──────────────────────────

n_univ     = min(20, len(multi_target))
top_univ   = multi_target.head(n_univ).index.tolist()
univ_names = [f.replace("rdkit-", "") for f in top_univ]
target_colors = cm.tab10(np.linspace(0, 0.9, len(TARGETS)))

fig5, ax5 = plt.subplots(figsize=(11, 8))
n_t  = len(TARGETS)
bar_h = 0.7 / n_t
y_positions = np.arange(n_univ)
for j, target in enumerate(TARGETS):
    vals  = shap_matrix_norm.loc[top_univ, target].values
    y_off = y_positions + (j - n_t / 2 + 0.5) * bar_h
    ax5.barh(y_off, vals, height=bar_h * 0.92,
             color=target_colors[j], label=TARGET_LABELS[target], alpha=0.88)
ax5.set_yticks(y_positions)
ax5.set_yticklabels(univ_names, fontsize=9)
ax5.set_xlabel("Normalised SHAP importance (fraction of per-target total)", fontsize=11)
ax5.set_title(
    f"Most cross-target important features\n"
    f"(top-{n_univ} features by number of targets in top-{TOP_K})",
    fontsize=12,
)
ax5.legend(loc="lower right", fontsize=9, ncol=2,
           framealpha=0.9, edgecolor="lightgray")
sns.despine(ax=ax5)
plt.tight_layout()
plt.savefig(OUT_DIR / "shap_universal_features.png", bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT_DIR / 'shap_universal_features.png'}")

# ── Figure S6: SHAP decay grid (3×3, log y) ──────────────────────────────────

colors_9 = cm.tab10(np.linspace(0, 0.9, len(TARGETS)))

fig6, axes6 = plt.subplots(3, 3, figsize=(14, 11), sharex=True)
axes6 = axes6.flatten()
for i, target in enumerate(TARGETS):
    ax   = axes6[i]
    vals = mean_abs_shap[target].sort_values(ascending=False).values
    ranks = np.arange(1, n_features + 1)
    ax.axvspan(0.5, TOP_K + 0.5, color="#e3f2fd", zorder=0)
    ax.plot(ranks, vals, color=colors_9[i], linewidth=1.5, zorder=2)
    ax.scatter(ranks[:TOP_K], vals[:TOP_K], s=18, color=colors_9[i],
               zorder=3, linewidths=0)
    ax.set_yscale("log")
    ax.set_xlim(0, n_features + 1)
    ax.set_title(TARGET_LABELS[target], fontsize=13)
    ax.set_xlabel("Feature rank", fontsize=10)
    ax.set_ylabel("Mean |SHAP|", fontsize=10)
    cumsum = np.cumsum(vals); total = cumsum[-1]
    r50 = int(np.searchsorted(cumsum, 0.50 * total)) + 1
    r80 = int(np.searchsorted(cumsum, 0.80 * total)) + 1
    ax.axvline(r50, color="gray", lw=1, ls="--", alpha=0.7)
    ax.axvline(r80, color="gray", lw=1, ls=":",  alpha=0.7)
    ax.text(r50 + 1, vals[0], f"50%\n(r={r50})", va="top", fontsize=7, color="gray")
    ax.text(r80 + 1, vals[r80 - 1] if r80 <= len(vals) else vals[-1],
            f"80%\n(r={r80})", va="top", fontsize=7, color="gray")
    sns.despine(ax=ax)

legend_items = [
    Patch(color="#e3f2fd", label=f"Top-{TOP_K} features"),
    Line2D([0], [0], color="gray", lw=1, ls="--", label="50% cumulative importance"),
    Line2D([0], [0], color="gray", lw=1, ls=":",  label="80% cumulative importance"),
]
fig6.legend(handles=legend_items, loc="lower center", ncol=3,
            fontsize=10, bbox_to_anchor=(0.5, -0.025), framealpha=0.9)
fig6.suptitle(
    "SHAP feature importance decay per target\n"
    "(features sorted by importance; log y-scale)",
    fontsize=14, y=1.01,
)
plt.tight_layout()
plt.savefig(OUT_DIR / "shap_decay_grid.png", bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT_DIR / 'shap_decay_grid.png'}")

# ── Figure S7: SHAP decay overlay (normalised) ───────────────────────────────

fig7, ax7 = plt.subplots(figsize=(9, 5.5))
ranks = np.arange(1, n_features + 1)
ax7.axvspan(0.5, TOP_K + 0.5, color="#e3f2fd", zorder=0, label=f"Top-{TOP_K} region")
for i, target in enumerate(TARGETS):
    vals      = mean_abs_shap[target].sort_values(ascending=False).values
    vals_norm = vals / vals[0]
    ax7.plot(ranks, vals_norm, color=colors_9[i], linewidth=1.6,
             label=TARGET_LABELS[target], zorder=2)
ax7.set_xlim(0, n_features + 1)
ax7.set_ylim(bottom=0)
ax7.set_xlabel("Feature rank", fontsize=12)
ax7.set_ylabel("Normalised mean |SHAP|  (max = 1)", fontsize=12)
ax7.set_title(
    "SHAP importance decay — all targets overlaid\n"
    "(each target normalised to its own maximum)",
    fontsize=12,
)
ax7.legend(fontsize=9, ncol=2, loc="upper right", framealpha=0.9)
sns.despine(ax=ax7)
plt.tight_layout()
plt.savefig(OUT_DIR / "shap_decay_overlay.png", bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT_DIR / 'shap_decay_overlay.png'}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. UNCERTAINTY DIAGNOSTICS — update rcParams for higher-res PDF figures
# ═══════════════════════════════════════════════════════════════════════════════

print("\n── Uncertainty figures ───────────────────────────────────────────────────")

FONT_BASE  = 9
FONT_LABEL = 10
FONT_PANEL = 11
FONT_ANNOT = 8.5
DPI        = 300

matplotlib.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         FONT_BASE,
    "axes.titlesize":    FONT_LABEL,
    "axes.labelsize":    FONT_LABEL,
    "xtick.labelsize":   FONT_BASE,
    "ytick.labelsize":   FONT_BASE,
    "legend.fontsize":   FONT_BASE,
    "lines.linewidth":   1.5,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size":  3.5,
    "ytick.major.size":  3.5,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linewidth":    0.6,
    "grid.linestyle":    "--",
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "legend.frameon":    True,
    "legend.framealpha": 0.85,
    "legend.edgecolor":  "0.8",
    "savefig.dpi":       DPI,
    "savefig.bbox":      "tight",
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})

cmap_splits = cm.get_cmap("viridis", N_TEST_SPLITS)


def _add_split_legend(fig):
    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=cmap_splits(tid), markersize=5,
               label=f"Split {tid + 1}")
        for tid in range(N_TEST_SPLITS)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=N_TEST_SPLITS,
               fontsize=FONT_BASE, frameon=True,
               bbox_to_anchor=(0.5, -0.012),
               handletextpad=0.3, columnspacing=0.8)


# ── Compute per-target uncertainty diagnostics ────────────────────────────────

print("  Computing per-target uncertainty diagnostics...")
unc_data = {}

for target in TARGETS:
    print(f"    {target}", flush=True)
    df_t         = create_cluster_splits(df, target)
    df_t         = filter_outliers(df_t, target)
    feature_cols = results[target][0]["fold_models"][0]["features"]
    df_t         = filter_nn_isolated(df_t, target, feature_cols)
    scale        = _DISPLAY_SCALE.get(target, 1.0)

    splits = []
    for ts in range(N_TEST_SPLITS):
        run         = results[target][ts]
        fold_models = run["fold_models"]
        split_col   = run["split_column"]

        test_mask  = df_t[split_col] == "test"
        train_mask = ~test_mask
        X_test     = df_t.loc[test_mask,  feature_cols].to_numpy()
        X_train    = df_t.loc[train_mask, feature_cols].to_numpy()
        y_test     = df_t.loc[test_mask,  target].to_numpy() * scale

        fold_preds = np.array([
            m["model"].inplace_predict(m["scaler"].transform(X_test)) * scale
            for m in fold_models
        ])
        y_pred = fold_preds.mean(axis=0)
        y_std  = fold_preds.std(axis=0)
        errors = np.abs(y_pred - y_test)

        scaler     = fold_models[0]["scaler"]
        X_test_sc  = scaler.transform(X_test)
        X_train_sc = scaler.transform(X_train)
        nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        nn.fit(X_train_sc)
        nn_dist = nn.kneighbors(X_test_sc)[0][:, 0]

        splits.append({
            "errors":   errors,
            "y_std":    y_std,
            "nn_dist":  nn_dist,
        })

    unc_data[target] = splits

print("  Done.")

# ── Figure S8: |error| vs NN distance ────────────────────────────────────────

fig8, axes8 = plt.subplots(3, 3, figsize=(12, 10))
for i, target in enumerate(TARGETS):
    ax     = axes8.flatten()[i]
    splits = unc_data[target]

    all_nn_dist = np.concatenate([s["nn_dist"] for s in splits])
    all_errors  = np.concatenate([s["errors"]  for s in splits])
    all_tids    = np.concatenate([[ts] * len(s["errors"]) for ts, s in enumerate(splits)])

    keep   = all_nn_dist <= NN_DIST_MAX
    n_excl = (~keep).sum()
    rho, _ = spearmanr(all_nn_dist[keep], all_errors[keep])

    for tid in range(N_TEST_SPLITS):
        mask = keep & (all_tids == tid)
        ax.scatter(all_nn_dist[mask], all_errors[mask],
                   s=8, alpha=0.50, color=cmap_splits(tid), linewidths=0,
                   rasterized=True)

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_title(TARGET_LABELS_UNC[target], fontsize=FONT_LABEL, pad=3)
    unit_str = UNIT_LABELS[target].split("[")[-1].rstrip("]") if "[" in UNIT_LABELS[target] else UNIT_LABELS[target]
    ax.set_xlabel("NN distance (feature space)", fontsize=FONT_BASE)
    ax.set_ylabel(f"|error| [{unit_str}]", fontsize=FONT_BASE)
    ax.text(0.97, 0.97, fr"$\rho$ = {rho:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=FONT_ANNOT)
    if n_excl:
        ax.text(0.97, 0.88, f"({n_excl} excl.)",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7, color="0.5")
    ax.text(-0.16, 1.05, ALPHABET[i], transform=ax.transAxes,
            fontsize=FONT_PANEL, fontweight="bold", va="bottom")

fig8.tight_layout(rect=[0, 0.03, 1, 1])
_add_split_legend(fig8)
plt.savefig(OUT_DIR / "unc_error_vs_nndist.pdf")
plt.close()
print(f"  Saved: {OUT_DIR / 'unc_error_vs_nndist.pdf'}")

# ── Figure S9: |error| vs ensemble std ───────────────────────────────────────

fig9, axes9 = plt.subplots(3, 3, figsize=(12, 10))
for i, target in enumerate(TARGETS):
    ax     = axes9.flatten()[i]
    splits = unc_data[target]

    all_std    = np.concatenate([s["y_std"]  for s in splits])
    all_errors = np.concatenate([s["errors"] for s in splits])
    all_tids   = np.concatenate([[ts] * len(s["errors"]) for ts, s in enumerate(splits)])

    rho, _ = spearmanr(all_std, all_errors)
    lim    = max(all_std.max(), all_errors.max()) * 1.05

    for tid in range(N_TEST_SPLITS):
        mask = all_tids == tid
        ax.scatter(all_std[mask], all_errors[mask],
                   s=8, alpha=0.50, color=cmap_splits(tid), linewidths=0,
                   rasterized=True)

    ax.plot([0, lim], [0, lim], "k--", lw=0.9)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(TARGET_LABELS_UNC[target], fontsize=FONT_LABEL, pad=3)
    unit_str = UNIT_LABELS[target].split("[")[-1].rstrip("]") if "[" in UNIT_LABELS[target] else ""
    unit_fmt = f" [{unit_str}]" if unit_str else ""
    ax.set_xlabel(f"Ensemble std{unit_fmt}", fontsize=FONT_BASE)
    ax.set_ylabel(f"|error|{unit_fmt}",      fontsize=FONT_BASE)
    ax.text(0.97, 0.03, fr"$\rho$ = {rho:.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=FONT_ANNOT)
    ax.text(-0.16, 1.05, ALPHABET[i], transform=ax.transAxes,
            fontsize=FONT_PANEL, fontweight="bold", va="bottom")

fig9.tight_layout(rect=[0, 0.03, 1, 1])
_add_split_legend(fig9)
plt.savefig(OUT_DIR / "unc_error_vs_ens_std.pdf")
plt.close()
print(f"  Saved: {OUT_DIR / 'unc_error_vs_ens_std.pdf'}")

print(f"\n=== All done. Nine figures saved to {OUT_DIR}/ ===")
