#!/usr/bin/env python3
"""
plot_for_paper.py — publication-quality figures for the SurfPro-MD paper.

Generates all paper figures from a trained models.pkl file.  The script is
self-contained: it reloads the CSV, re-clusters molecules, and recomputes
the per-target diagnostic statistics exactly as surrogate.py does.

Figures produced
----------------
  figure_data.pdf                     — data overview (histograms + correlation)
  full_results.pdf                    — metrics bar chart + parity scatter plots
  error_vs_nndist.pdf                 — |error| vs NN distance (3×3)
  ensemble_std_vs_error.pdf           — ensemble std vs |error| (3×3)
  rel_error_vs_nndist.pdf             — relative error vs NN distance (3×3)
  rel_ensemble_std_vs_rel_error.pdf   — relative std vs relative error (3×3)
  MD_validation.pdf                   — MD validation (surface tension + viscosity)

Usage
-----
  python plot_for_paper.py
  python plot_for_paper.py --model models.pkl --data ../data/SurfPro-MD.csv
  python plot_for_paper.py --skip-md          # skip MD validation figure
"""

import argparse
import os
import pickle
import string
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Generate publication figures for SurfPro-MD.")
parser.add_argument("--model",
                    default=str(Path(__file__).resolve().parent.parent / "models" / "models.pkl"))
parser.add_argument("--data",  default=str(Path(__file__).resolve().parent.parent / "data" / "SurfPro-MD.csv"))
parser.add_argument("--skip-md", action="store_true",
                    help="Skip the MD validation figure (requires external data).")
args = parser.parse_args()

# ── Global typography ─────────────────────────────────────────────────────────

FONT_BASE  = 9     # axis tick labels
FONT_LABEL = 10    # axis titles / labels
FONT_PANEL = 11    # bold panel letters
FONT_ANNOT = 8.5   # in-panel metric annotations
DPI        = 300

matplotlib.rcParams.update({
    # Font
    "font.family":        "sans-serif",
    "font.size":          FONT_BASE,
    "axes.titlesize":     FONT_LABEL,
    "axes.labelsize":     FONT_LABEL,
    "xtick.labelsize":    FONT_BASE,
    "ytick.labelsize":    FONT_BASE,
    "legend.fontsize":    FONT_BASE,
    # Lines / markers
    "lines.linewidth":    1.5,
    "axes.linewidth":     0.8,
    "patch.linewidth":    0.6,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "xtick.major.size":   3.5,
    "ytick.major.size":   3.5,
    # Grid
    "axes.grid":          True,
    "grid.alpha":         0.35,
    "grid.linewidth":     0.6,
    "grid.linestyle":     "--",
    # Ticks
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    # Spines
    "axes.spines.top":    True,
    "axes.spines.right":  True,
    # Legend
    "legend.frameon":     True,
    "legend.framealpha":  0.85,
    "legend.edgecolor":   "0.8",
    "legend.borderpad":   0.4,
    # Saving
    "savefig.dpi":        DPI,
    "savefig.bbox":       "tight",
    "pdf.fonttype":       42,   # embeds fonts as TrueType for journal submission
    "ps.fonttype":        42,
})

# ── Constants (must match surrogate.py) ──────────────────────────────────────

TARGETS = [
    "pCMC", "AW_ST_CMC", "Gamma_max", "Area_min", "pC20",
    "D_MOL", "D_SOL", "surface_tension_avg", "viscosity",
]

target_labels = {
    "pCMC":               "pCMC",
    "AW_ST_CMC":          r"$\gamma_{\mathrm{CMC}}$",
    "Gamma_max":          r"$\Gamma_{\max}$",
    "pC20":               r"pC$_{20}$",
    "Pi_CMC":             r"$\Pi_{\mathrm{CMC}}$",
    "Area_min":           r"A$_{\min}$",
    "viscosity":          r"$\eta$",
    "D_MOL":              r"D$_{\mathrm{MOL}}$",
    "D_SOL":              r"D$_{\mathrm{SOL}}$",
    "surface_tension_avg": r"$\gamma$",
}

unit_labels = {
    "pCMC":               "pCMC",
    "AW_ST_CMC":          r"$\gamma_{\mathrm{CMC}}$ [mN m$^{-1}$]",
    "Gamma_max":          r"$\Gamma_{\max}$ [$\mu$mol m$^{-2}$]",
    "pC20":               r"pC$_{20}$",
    "Pi_CMC":             r"$\Pi_{\mathrm{CMC}}$ [mN m$^{-1}$]",
    "Area_min":           r"A$_{\min}$ [nm$^2$]",
    "viscosity":          r"$\eta$ [mPa$\cdot$s]",
    "D_MOL":              r"D$_\mathrm{MOL}$ [$\times\!10^{-9}$ m$^2$s$^{-1}$]",
    "D_SOL":              r"D$_\mathrm{SOL}$ [$\times\!10^{-9}$ m$^2$s$^{-1}$]",
    "surface_tension_avg": r"$\gamma$ [mN m$^{-1}$]",
}

_UNITS = {
    "pCMC": "log(M)", "AW_ST_CMC": "mN/m", "Gamma_max": "μmol/m²",
    "Area_min": "nm²",  "pC20": "log(M)",   "D_MOL": r"$\times10^{-9}$ m$^2$/s",
    "D_SOL": r"$\times10^{-9}$ m$^2$/s",   "surface_tension_avg": "mN/m", "viscosity": "mPa·s",
}

# Pa·s → mPa·s for viscosity; bar·nm → mN/m for surface_tension_avg
_DISPLAY_SCALE = {"viscosity": 1e3, "surface_tension_avg": 0.1}

N_FOLDS           = 5
N_TEST_SPLITS     = 5
MIN_TEST_FRACTION = 0.07
MAX_TEST_FRACTION = 0.11
MAX_REJECT_TRIES  = 5
MIN_VAL_FRACTION  = 0.12
MAX_VAL_FRACTION  = 0.18
RANDOM_STATE      = 3
ALPHA_LEVELS      = np.linspace(0.02, 0.98, 60)   # for conformal calibration curves
NN_ISOLATION_THRESHOLD = 50.0
NN_DIST_MAX       = 40

ALPHABET = list(string.ascii_lowercase)

# ── Data-prep helpers (exact copies from surrogate.py) ────────────────────────

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
    cs_all = df_t.groupby("cluster").size().to_dict()
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
        rc = np.array(rem["cluster"].unique())
        rs = rem.groupby("cluster").size().to_dict()
        rm = len(rem)
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
            df_t.loc[df_t["cluster"].isin(vc),               col] = "val"
            df_t.loc[df_t[col_test] == "test",               col] = "test"
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


def prank(y_arr, ref_sorted):
    """Vectorised percentile rank: fraction of ref values ≤ each element of y_arr."""
    return np.searchsorted(ref_sorted, y_arr, side="right") / len(ref_sorted)


def get_colors(n):
    cmap = cm.get_cmap("viridis")
    return [cmap(i / (n - 1)) for i in range(n)]


def read_xvg(fname, quantity):
    if not fname.endswith("xvg"):
        raise ValueError("Not an .xvg file")
    with open(fname) as fid:
        ll = fid.readlines()
    M = len(ll[-1].split())
    labels = [""] * M
    labels[0] = "Time (ps)"
    ALL, icol, i = [], -1, 1
    for l in ll:
        if l.startswith("@ s"):
            label = " ".join(l.split()[3:])[1:-1]
            labels[i] = label
            if label.strip() == quantity.strip():
                icol = i
            i += 1
        if l[0] in ["#", "@"]:
            continue
        ALL.append([float(x) for x in l.split()])
    if icol == -1:
        return 0
    ALL = np.asarray(ALL).T
    return [np.average(ALL[icol, 100:]), np.std(ALL[icol, 100:])]


# ── Shared colour maps ────────────────────────────────────────────────────────

cmap_splits = cm.get_cmap("viridis", N_TEST_SPLITS)

# ── 1. Load data and cluster ──────────────────────────────────────────────────

print(f"Loading {args.data} …")
df = pd.read_csv(args.data)
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

# ── 2. Load models and build metrics summary ──────────────────────────────────

print(f"\nLoading {args.model} …")
with open(args.model, "rb") as fh:
    results = pickle.load(fh)

metrics_summary = {}
for target, runs in results.items():
    r2_v   = [r["metrics"]["r2"]       for r in runs.values()]
    rho_v  = [r["metrics"]["spearman"] for r in runs.values()]
    rmse_v = [r["metrics"]["rmse"]     for r in runs.values()]
    pcc_v  = [r["metrics"]["pearson"]  for r in runs.values()]
    metrics_summary[target] = {
        "r2_mean":   np.mean(r2_v),  "r2_std":   np.std(r2_v),
        "rho_mean":  np.mean(rho_v), "rho_std":  np.std(rho_v),
        "rmse_mean": np.mean(rmse_v),"rmse_std": np.std(rmse_v),
        "pcc_mean":  np.mean(pcc_v), "pcc_std":  np.std(pcc_v),
    }

# ── 3. Compute diagnostic data (error / NN-dist / std per target) ─────────────

print("\nComputing per-target diagnostic data …")
diag_data = {}

for target in TARGETS:
    print(f"  {target}", flush=True)
    df_t         = create_cluster_splits(df, target)
    df_t         = filter_outliers(df_t, target)
    feature_cols = results[target][0]["fold_models"][0]["features"]
    df_t         = filter_nn_isolated(df_t, target, feature_cols)
    scale        = _DISPLAY_SCALE.get(target, 1.0)

    errors_all, rel_errors_all = [], []
    nn_dist_all, std_all, rel_std_all = [], [], []
    test_id_all = []
    y_true_all, y_pred_all = [], []

    for test_id in range(N_TEST_SPLITS):
        run          = results[target][test_id]
        fold_models  = run["fold_models"]
        split_col    = run["split_column"]

        test_mask  = df_t[split_col] == "test"
        train_mask = ~test_mask

        X_test  = df_t.loc[test_mask,  feature_cols].to_numpy()
        X_train = df_t.loc[train_mask, feature_cols].to_numpy()
        y_test  = df_t.loc[test_mask,  target].to_numpy() * scale

        fold_preds = np.array([
            m["model"].inplace_predict(m["scaler"].transform(X_test)) * scale
            for m in fold_models
        ])
        y_pred = fold_preds.mean(axis=0)
        y_std  = fold_preds.std(axis=0)

        errors     = np.abs(y_pred - y_test)
        rel_errors = errors / np.abs(y_test)
        rel_std    = y_std  / np.abs(y_test)

        scaler     = fold_models[0]["scaler"]
        X_test_sc  = scaler.transform(X_test)
        X_train_sc = scaler.transform(X_train)
        nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        nn.fit(X_train_sc)
        nn_dist = nn.kneighbors(X_test_sc)[0][:, 0]

        errors_all    .extend(errors)
        rel_errors_all.extend(rel_errors)
        nn_dist_all   .extend(nn_dist)
        std_all       .extend(y_std)
        rel_std_all   .extend(rel_std)
        test_id_all   .extend([test_id] * len(errors))
        y_true_all    .extend(y_test)
        y_pred_all    .extend(y_pred)

    diag_data[target] = {
        "errors":     np.array(errors_all),
        "rel_errors": np.array(rel_errors_all),
        "nn_dist":    np.array(nn_dist_all),
        "std":        np.array(std_all),
        "rel_std":    np.array(rel_std_all),
        "test_ids":   np.array(test_id_all),
        "y_true":     np.array(y_true_all),
        "y_pred":     np.array(y_pred_all),
    }


# ── 3b. Compute conformal calibration data ────────────────────────────────────

print("\nComputing conformal calibration data …")
conf_data = {}

for target in TARGETS:
    print(f"  {target}", flush=True)
    df_t         = create_cluster_splits(df, target)
    df_t         = filter_outliers(df_t, target)
    feature_cols = results[target][0]["fold_models"][0]["features"]
    df_t         = filter_nn_isolated(df_t, target, feature_cols)
    scale        = _DISPLAY_SCALE.get(target, 1.0)

    splits_conf = []
    for ts in range(N_TEST_SPLITS):
        run         = results[target][ts]
        fold_models = run["fold_models"]
        split_col   = run["split_column"]

        test_mask  = df_t[split_col] == "test"
        train_mask = ~test_mask
        X_test     = df_t.loc[test_mask,  feature_cols].to_numpy()
        y_test     = df_t.loc[test_mask,  target].to_numpy() * scale
        y_train    = df_t.loc[train_mask, target].to_numpy() * scale
        y_train_sorted = np.sort(y_train)

        fold_preds = np.array([
            m["model"].inplace_predict(m["scaler"].transform(X_test)) * scale
            for m in fold_models
        ])
        y_pred = fold_preds.mean(axis=0)
        errors = np.abs(y_pred - y_test)
        rank_test_scores = np.abs(
            prank(y_test, y_train_sorted) - prank(y_pred, y_train_sorted)
        )

        cal_errors      = []
        rank_cal_errors = []
        for fi, fm in enumerate(fold_models):
            col_fi = f"{target}_split_t{ts}_f{fi}"
            if col_fi not in df_t.columns:
                continue
            val_mask = df_t[col_fi] == "val"
            if val_mask.sum() == 0:
                continue
            X_val      = df_t.loc[val_mask, feature_cols].to_numpy()
            y_val      = df_t.loc[val_mask, target].to_numpy() * scale
            y_val_pred = fm["model"].inplace_predict(
                fm["scaler"].transform(X_val)) * scale
            cal_errors.extend(np.abs(y_val_pred - y_val))
            rank_cal_errors.extend(
                np.abs(prank(y_val, y_train_sorted) - prank(y_val_pred, y_train_sorted))
            )

        cal_errors      = np.array(cal_errors)
        rank_cal_errors = np.array(rank_cal_errors)
        n_cal           = len(cal_errors)
        nominal         = 1.0 - ALPHA_LEVELS

        conf_empirical      = []
        rank_conf_empirical = []
        for alpha in ALPHA_LEVELS:
            level  = np.minimum(1.0, (1 - alpha) * (1 + 1 / n_cal))
            q_conf = np.quantile(cal_errors,      level)
            q_rank = np.quantile(rank_cal_errors, level)
            conf_empirical.append(np.mean(errors           <= q_conf))
            rank_conf_empirical.append(np.mean(rank_test_scores <= q_rank))

        splits_conf.append({
            "nominal":             nominal,
            "conf_empirical":      np.array(conf_empirical),
            "rank_conf_empirical": np.array(rank_conf_empirical),
        })

    conf_data[target] = splits_conf

# ── Print coverage values at 0.05 intervals ───────────────────────────────────
print("\nConformal calibration summary (nominal → empirical abs / rank coverage):")
_check_levels = np.arange(0.50, 1.00, 0.05)
for target in TARGETS:
    splits = conf_data[target]
    nominal = splits[0]["nominal"]
    abs_mean  = np.mean([s["conf_empirical"]      for s in splits], axis=0)
    rank_mean = np.mean([s["rank_conf_empirical"] for s in splits], axis=0)
    print(f"\n  {target}")
    print(f"  {'Nom':>6}  {'Abs':>6}  {'Rank':>6}  {'ΔAbs':>7}  {'ΔRank':>7}")
    for lvl in _check_levels:
        idx = np.argmin(np.abs(nominal - lvl))
        print(f"  {nominal[idx]:6.2f}  {abs_mean[idx]:6.3f}  {rank_mean[idx]:6.3f}"
              f"  {abs_mean[idx]-nominal[idx]:+7.3f}  {rank_mean[idx]-nominal[idx]:+7.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE A — DATA OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

print("\nPlotting figure_data.pdf …")

TARGETS_HIST = [
    "pCMC", "D_MOL", "D_SOL",
    "surface_tension_avg", "viscosity", "AW_ST_CMC",
    "Gamma_max", "Area_min", "pC20",
]
TARGETS_CORR = [
    "pCMC", "AW_ST_CMC", "Gamma_max",
    "pC20", "Pi_CMC", "Area_min",
    "viscosity", "D_MOL", "D_SOL", "surface_tension_avg",
]
CORR_LABELS = [
    "pCMC", r"$\gamma_{\mathrm{CMC}}$", r"$\Gamma_{\max}$",
    r"pC$_{20}$", r"$\Pi_{\mathrm{CMC}}$", r"A$_{\min}$",
    r"$\eta$", r"D$_{\mathrm{MOL}}$", r"D$_{\mathrm{SOL}}$", r"$\gamma$",
]

HIST_COLOR = cm.viridis(0.35)

fig = plt.figure(figsize=(11, 5))
gs_top = fig.add_gridspec(nrows=1, ncols=4, width_ratios=[1, 0.18, 1.0, 0.05],
                           wspace=0.08)
gs_hist = gs_top[0, 0].subgridspec(3, 3, wspace=0.30, hspace=0.55)
ax_corr = fig.add_subplot(gs_top[0, 2])
ax_cbar = fig.add_subplot(gs_top[0, 3])

for i, target in enumerate(TARGETS_HIST):
    ax = fig.add_subplot(gs_hist[i // 3, i % 3])
    vals = df[target].dropna().values * _DISPLAY_SCALE.get(target, 1.0)
    ax.hist(vals, bins=25, density=True, alpha=0.80, color=HIST_COLOR,
            edgecolor="white", linewidth=0.3)
    ax.set_yticks([])
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.3)
    ax.tick_params(axis="x", labelsize=7)
    ax.text(0.04, 1.06, unit_labels[target], transform=ax.transAxes,
            ha="left", va="top", fontsize=7.5)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

corr = df[TARGETS_CORR].corr()
im   = ax_corr.imshow(corr.values, cmap="viridis", vmin=-1, vmax=1)
ax_corr.set_xticks(np.arange(len(TARGETS_CORR)))
ax_corr.set_yticks(np.arange(len(TARGETS_CORR)))
ax_corr.set_xticklabels(CORR_LABELS, rotation=45, ha="right", fontsize=8.5)
ax_corr.set_yticklabels(CORR_LABELS, fontsize=8.5)
for i in range(len(TARGETS_CORR)):
    for j in range(len(TARGETS_CORR)):
        col = "white" if corr.values[i, j] <= -0.46 else "black"
        ax_corr.text(j, i, f"{corr.values[i, j]:.1f}",
                     ha="center", va="center", fontsize=7, color=col)
cbar = fig.colorbar(im, cax=ax_cbar)
cbar.set_label("Pearson $r$", fontsize=8.5)
cbar.ax.tick_params(labelsize=8)
ax_corr.grid(False)
ax_corr.set_aspect("equal")

fig.text(0.07,  0.92, "a", fontsize=FONT_PANEL, fontweight="bold")
fig.text(0.48,  0.92, "b", fontsize=FONT_PANEL, fontweight="bold")

plt.tight_layout()
plt.savefig("figure_data.pdf")
plt.close()
print("  Saved: figure_data.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE B — FULL RESULTS (metrics bar + parity scatter)
# ─────────────────────────────────────────────────────────────────────────────

print("Plotting full_results.pdf …")

targets_sorted = sorted(metrics_summary, key=lambda t: -metrics_summary[t]["r2_mean"])

fig = plt.figure(figsize=(11, 15))
gs = gridspec.GridSpec(
    nrows=5, ncols=3,
    height_ratios=[1, 1, 1, 1, 0.28],   # last row = legend space
    # ── Spacing knobs ──────────────────────────────────────────────────────────
    # hspace: vertical gap between rows, as a fraction of average subplot height.
    #   Increase to add breathing room between scatter panels.
    hspace=0.35,
    # wspace: horizontal gap between columns, as a fraction of average subplot width.
    #   Increase to separate columns; decrease to make them look tighter.
    wspace=0.25,
)
# Outer figure margins (0–1, fraction of figure size).
#   top/bottom: how close the outermost subplots sit to the figure edge.
#   Decrease top or increase bottom to shift content downward.
fig.subplots_adjust(top=0.96, bottom=0.04)
ax_metrics = fig.add_subplot(gs[0, :])

# ── Panel a: metrics bar chart ────────────────────────────────────────────────
viridis2 = cm.viridis(np.linspace(0.2, 0.75, 2))
x     = np.arange(len(targets_sorted))
width = 0.35

bars_r2 = ax_metrics.bar(
    x - width / 2,
    [metrics_summary[t]["r2_mean"]  for t in targets_sorted], width,
    yerr=[metrics_summary[t]["r2_std"]  for t in targets_sorted],
    label=r"$R^2$", color=viridis2[0],
    error_kw={"elinewidth": 0.9, "capsize": 2.5, "capthick": 0.9},
)
bars_rho = ax_metrics.bar(
    x + width / 2,
    [metrics_summary[t]["rho_mean"] for t in targets_sorted], width,
    yerr=[metrics_summary[t]["rho_std"] for t in targets_sorted],
    label=r"Spearman $\rho$", color=viridis2[1],
    error_kw={"elinewidth": 0.9, "capsize": 2.5, "capthick": 0.9},
)
ax_metrics.set_xticks(x)
ax_metrics.set_xticklabels(
    [target_labels[t] for t in targets_sorted], rotation=40, ha="right",
    fontsize=FONT_LABEL,
)
ax_metrics.set_ylabel("Score")
ax_metrics.set_ylim([-0.25, 1.05])
ax_metrics.axhline(0, color="0.5", lw=0.7, ls="--")
ax_metrics.legend(loc="upper right", frameon=True, fontsize=FONT_BASE)
for sp in ["top", "right"]:
    ax_metrics.spines[sp].set_visible(False)
# Panel label in axes coordinates — same convention as panels b–j below.
# Adjust the x offset (-0.04) or y offset (1.05) if the label needs nudging.
ax_metrics.text(-0.04, 1.05, "a", transform=ax_metrics.transAxes,
                fontsize=FONT_PANEL, fontweight="bold", va="bottom")

# ── Panels b–j: parity scatter plots ─────────────────────────────────────────
scatter_handles = []

for i, target in enumerate(TARGETS):
    ax = fig.add_subplot(gs[1 + i // 3, i % 3])
    ax.set_aspect("equal", adjustable="box")
    d = diag_data[target]

    vmin, vmax = np.inf, -np.inf
    for tid in range(N_TEST_SPLITS):
        mask  = d["test_ids"] == tid
        y_t   = d["y_true"][mask]
        y_p   = d["y_pred"][mask]
        sc = ax.scatter(y_t, y_p, s=8, alpha=0.65,
                        color=cmap_splits(tid), linewidths=0,
                        label=f"Split {tid + 1}")
        if i == 0:
            scatter_handles.append(sc)
        vmin = min(vmin, y_t.min(), y_p.min())
        vmax = max(vmax, y_t.max(), y_p.max())

    margin = (vmax - vmin) * 0.04
    lim = (vmin - margin, vmax + margin)
    ax.plot(lim, lim, "k--", lw=0.9, label="y = x")
    ax.set_xlim(lim)
    ax.set_ylim(lim)

    m = metrics_summary[target]
    ax.text(
        0.04, 0.97,
        f"$R^2$ = {m['r2_mean']:.2f}±{m['r2_std']:.2f}\n"
        f"$\\rho$  = {m['rho_mean']:.2f}±{m['rho_std']:.2f}",
        transform=ax.transAxes, va="top", fontsize=FONT_ANNOT,
        linespacing=1.5,
    )

    # Axis labels: unit on all panels; "Predicted"/"True" on edges only
    unit = unit_labels[target]
    if i % 3 == 0:
        ax.set_ylabel(f"Predicted {unit}", fontsize=FONT_BASE)
    else:
        ax.set_ylabel(unit, fontsize=FONT_BASE)
    if i >= 6:
        ax.set_xlabel(f"{unit}\nTrue", fontsize=FONT_BASE)
    else:
        ax.set_xlabel(unit, fontsize=FONT_BASE)

    ax.set_title(target_labels[target], fontsize=FONT_LABEL, pad=3)
    ax.text(-0.16, 1.05, ALPHABET[i + 1], transform=ax.transAxes,
            fontsize=FONT_PANEL, fontweight="bold", va="bottom")

# ── Shared legend for test splits + identity line ─────────────────────────────
split_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap_splits(tid),
           markersize=6, label=f"Split {tid + 1}")
    for tid in range(N_TEST_SPLITS)
]
identity_handle = Line2D([0], [0], color="k", ls="--", lw=1.2, label="$y = x$")

ax_leg = fig.add_subplot(gs[4, :])
ax_leg.axis("off")
ax_leg.legend(
    handles=split_handles + [identity_handle],
    loc="center", ncol=N_TEST_SPLITS + 1,
    fontsize=FONT_BASE, frameon=True,
    title="Outer CV test splits",
    title_fontsize=FONT_BASE,
    handletextpad=0.4, columnspacing=1.0,
)

plt.savefig("full_results.pdf")
plt.close()
print("  Saved: full_results.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: shared legend for diagnostic scatter figures
# ─────────────────────────────────────────────────────────────────────────────

def _add_diag_legend(fig):
    """Add a shared test-split colour legend below a 3×3 diagnostic figure."""
    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=cmap_splits(tid), markersize=5,
               label=f"Split {tid + 1}")
        for tid in range(N_TEST_SPLITS)
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=N_TEST_SPLITS,
        fontsize=FONT_BASE, frameon=True,
        bbox_to_anchor=(0.5, -0.012),
        handletextpad=0.3, columnspacing=0.8,
    )


def _scatter_3x3(fig, axes, x_key, y_key, xlabel, ylabel_fn,
                 add_rho=True, add_identity=False, inset_panels=None):
    """Fill a 3×3 grid of diagnostic scatter plots."""
    for i, target in enumerate(TARGETS):
        ax = axes[i]
        d  = diag_data[target]
        keep = d["nn_dist"] <= NN_DIST_MAX
        n_rm = (~keep).sum()

        for tid in range(N_TEST_SPLITS):
            mask = keep & (d["test_ids"] == tid)
            ax.scatter(d[x_key][mask], d[y_key][mask],
                       s=8, alpha=0.50, color=cmap_splits(tid), linewidths=0)

        if add_identity:
            lim = max(d[x_key][keep].max(), d[y_key][keep].max()) * 1.05
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.plot([0, lim], [0, lim], "k--", lw=0.8)
        else:
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)

        if add_rho:
            rho, _ = spearmanr(d[x_key][keep], d[y_key][keep])
            ax.text(0.97, 0.97, f"ρ = {rho:.2f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=FONT_ANNOT)
        if n_rm:
            ax.text(0.97, 0.88, f"({n_rm} excl.)",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=7, color="0.5")

        ax.set_title(target_labels[target], fontsize=FONT_LABEL, pad=3)
        ax.set_xlabel(xlabel, fontsize=FONT_BASE)
        ax.set_ylabel(ylabel_fn(target), fontsize=FONT_BASE)
        ax.text(-0.16, 1.05, ALPHABET[i], transform=ax.transAxes,
                fontsize=FONT_PANEL, fontweight="bold", va="bottom")

        if inset_panels and i in inset_panels:
            x_max = d[x_key][keep].max()
            y_max = d[y_key][keep].max()
            x_frac = 0.50 if i == 6 else 0.10
            axins = inset_axes(ax, width="42%", height="42%", loc="upper left",
                               bbox_to_anchor=(0.08, 0, 1, 1),
                               bbox_transform=ax.transAxes)
            for tid in range(N_TEST_SPLITS):
                mask = keep & (d["test_ids"] == tid)
                axins.scatter(d[x_key][mask], d[y_key][mask],
                              s=4, alpha=0.50, color=cmap_splits(tid),
                              linewidths=0)
            axins.set_xlim(0, x_max * x_frac)
            axins.set_ylim(0, y_max * 0.10)
            axins.tick_params(labelsize=6)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE C — |error| vs NN distance
# ─────────────────────────────────────────────────────────────────────────────

print("Plotting error_vs_nndist.pdf …")
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
_scatter_3x3(fig, axes.flatten(),
             x_key="nn_dist", y_key="errors",
             xlabel="NN distance (feature space)",
             ylabel_fn=lambda t: f"|error| [{_UNITS[t]}]")
fig.tight_layout(rect=[0, 0.03, 1, 1])
_add_diag_legend(fig)
plt.savefig("error_vs_nndist.pdf")
plt.close()
print("  Saved: error_vs_nndist.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE D — ensemble std vs |error|
# ─────────────────────────────────────────────────────────────────────────────

print("Plotting ensemble_std_vs_error.pdf …")
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
_scatter_3x3(fig, axes.flatten(),
             x_key="std", y_key="errors",
             xlabel=lambda t: f"Ensemble std [{_UNITS[t]}]",   # placeholder
             ylabel_fn=lambda t: f"|error| [{_UNITS[t]}]",
             add_identity=True)
# Per-target x-labels need units
for i, target in enumerate(TARGETS):
    axes.flatten()[i].set_xlabel(f"Ensemble std [{_UNITS[target]}]",
                                 fontsize=FONT_BASE)
fig.tight_layout(rect=[0, 0.03, 1, 1])
_add_diag_legend(fig)
plt.savefig("ensemble_std_vs_error.pdf")
plt.close()
print("  Saved: ensemble_std_vs_error.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE E — relative error vs NN distance
# ─────────────────────────────────────────────────────────────────────────────

print("Plotting rel_error_vs_nndist.pdf …")
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
_scatter_3x3(fig, axes.flatten(),
             x_key="nn_dist", y_key="rel_errors",
             xlabel="NN distance (feature space)",
             ylabel_fn=lambda t: "|relative error|")
fig.tight_layout(rect=[0, 0.03, 1, 1])
_add_diag_legend(fig)
plt.savefig("rel_error_vs_nndist.pdf")
plt.close()
print("  Saved: rel_error_vs_nndist.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE F — relative ensemble std vs relative error (with insets)
# ─────────────────────────────────────────────────────────────────────────────

print("Plotting rel_ensemble_std_vs_rel_error.pdf …")
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
_scatter_3x3(fig, axes.flatten(),
             x_key="rel_std", y_key="rel_errors",
             xlabel="Relative ensemble std",
             ylabel_fn=lambda t: "|relative error|",
             inset_panels={0, 3, 6})
fig.tight_layout(rect=[0, 0.03, 1, 1])
_add_diag_legend(fig)
plt.savefig("rel_ensemble_std_vs_rel_error.pdf")
plt.close()
print("  Saved: rel_ensemble_std_vs_rel_error.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE G — MD validation  (skipped if --skip-md or data missing)
# ─────────────────────────────────────────────────────────────────────────────

MD_BASE = "/home/ricbec/PFAS-project"
md_available = os.path.isdir(MD_BASE) and not args.skip_md

if not md_available:
    print("Skipping MD_validation.pdf (data not found or --skip-md passed).")
else:
    print("Plotting MD_validation.pdf …")

    fig = plt.figure(figsize=(11, 5))
    gs_md = gridspec.GridSpec(nrows=1, ncols=2, wspace=0.25)
    ax_st   = fig.add_subplot(gs_md[0, 0])
    ax_visc = fig.add_subplot(gs_md[0, 1])

    axins_st = inset_axes(
        ax_st, width="100%", height="100%",
        bbox_to_anchor=(0.50, 0.28, 0.42, 0.42),
        bbox_transform=ax_st.transAxes, loc="lower left",
    )
    axins_st.set_xlim(-0.02, 0.2)
    axins_st.set_ylim(0.6, 1.1)
    axins_st.tick_params(labelsize=7)

    # Panel a — relative surface tension
    all_nmol_st = [150, 300, 450, 1500, 2500]
    all_mol_st  = ["methanol", "ethanol", "propanol"]
    colors_st   = get_colors(len(all_mol_st))

    for ic, mol in enumerate(all_mol_st):
        df_ref = pd.read_csv(f"{MD_BASE}/reference/{mol}_tension.csv")
        replica_data = {}
        for nmol in all_nmol_st:
            runs = []
            for rep in range(19):
                fname = (f"{MD_BASE}/data/tension-alcohols/"
                         f"{mol}_{nmol}/ST_{str(rep).zfill(2)}/potential.xvg")
                avg, _ = read_xvg(fname, "#Surf*SurfTen")
                runs.append(avg / 20)
            replica_data[nmol] = np.array(runs)
        gamma0 = replica_data[all_nmol_st[0]]
        rows = [{"c": nmol / 15000,
                 "rel_tension_mean": np.mean(replica_data[nmol] / gamma0),
                 "rel_tension_std":  np.std(replica_data[nmol]  / gamma0)}
                for nmol in all_nmol_st]
        df_md = pd.DataFrame(rows).sort_values("c")
        df_ref = df_ref.sort_values("mol_fraction")
        df_ref["rel_tension"] = df_ref["surface_tension"] / df_ref["surface_tension"].iloc[0]

        kw = dict(color=colors_st[ic], lw=1.5)
        ax_st.errorbar(df_md["c"], df_md["rel_tension_mean"],
                       yerr=df_md["rel_tension_std"],
                       marker="o", ms=0.1, capsize=3, label=f"{mol} comp.", **kw)
        ax_st.plot(df_ref["mol_fraction"], df_ref["rel_tension"],
                   ls="--", marker="o", ms=4, label=f"{mol} exp.", **kw)
        for ax_ in (axins_st,):
            ax_.errorbar(df_md["c"], df_md["rel_tension_mean"],
                         yerr=df_md["rel_tension_std"], **kw)
            ax_.plot(df_ref["mol_fraction"], df_ref["rel_tension"],
                     ls="--", **kw)

    style_hdl = [
        Line2D([0], [0], color="0.4", lw=1.5, ls="-",  label="Simulation"),
        Line2D([0], [0], color="0.4", lw=1.5, ls="--", label="Experiment"),
    ]
    mol_hdl_st = [Line2D([0], [0], color=colors_st[i], lw=2, label=mol)
                  for i, mol in enumerate(all_mol_st)]
    ax_st.legend(handles=style_hdl + mol_hdl_st, frameon=False, fontsize=FONT_BASE)
    ax_st.set_xlabel("Mole fraction")
    ax_st.set_ylabel("Relative surface tension")
    ax_st.set_ylim(0.3, 1.1)
    for sp in ["top", "right"]:
        ax_st.spines[sp].set_visible(False)

    # Panel b — relative viscosity
    all_nmol_visc = [10, 30, 50, 100, 150, 200]
    all_mol_visc  = ["formic", "acetic", "propionic", "valeric"]
    colors_visc   = get_colors(len(all_mol_visc))

    df_ref_visc = pd.read_csv(f"{MD_BASE}/reference/acid_reference.csv")
    df_ref_visc["c"] = df_ref_visc["c*100"] / 100

    base_visc = {}
    for mol in all_mol_visc:
        runs = [1 / read_xvg(f"{MD_BASE}/data/viscosities/"
                              f"{mol}_acid_0/ST_{str(r).zfill(2)}/visco.xvg",
                              "1/Viscosity")[0]
                for r in range(10)]
        base_visc[mol] = np.mean(runs)

    rows_visc = []
    for mol in all_mol_visc:
        for nmol in all_nmol_visc:
            all_runs, all_c = [], []
            for rep in range(10):
                base_p = (f"{MD_BASE}/data/viscosities/"
                          f"{mol}_acid_{nmol}/ST_{str(rep).zfill(2)}")
                v = 1 / read_xvg(f"{base_p}/visco.xvg", "1/Viscosity")[0]
                all_runs.append(v)
                with open(f"{base_p}/production.gro") as fgro:
                    box = float(fgro.readlines()[-1].split()[0])
                all_c.append(nmol / (box ** 3) * 10 / 6.023)
            v_mean = np.mean(all_runs)
            rows_visc.append({
                "c":             np.mean(all_c),
                "acid":          mol,
                "rel_viscosity": 1 + (v_mean / base_visc[mol] - 1) * 72 / 50,
                "error":         np.std(all_runs) / base_visc[mol],
            })
    df_mv = pd.DataFrame(rows_visc)

    for ic, mol in enumerate(all_mol_visc):
        sub    = df_mv[df_mv["acid"] == mol].sort_values("c")
        subref = df_ref_visc[df_ref_visc["acid"] == mol].sort_values("c")
        rel_ref = subref["viscosity"] / subref["viscosity"].iloc[0]
        kw = dict(color=colors_visc[ic], lw=1.5)
        ax_visc.errorbar(sub["c"], sub["rel_viscosity"], yerr=sub["error"],
                         marker="o", ms=0.1, capsize=3, label=f"{mol} comp.", **kw)
        ax_visc.plot(subref["c"], rel_ref,
                     ls="--", marker="o", ms=4, label=f"{mol} exp.", **kw)

    mol_hdl_visc = [Line2D([0], [0], color=colors_visc[i], lw=2, label=mol)
                    for i, mol in enumerate(all_mol_visc)]
    ax_visc.legend(handles=style_hdl + mol_hdl_visc, frameon=False,
                   fontsize=FONT_BASE)
    ax_visc.set_xlabel("Concentration [mol L$^{-1}$]")
    ax_visc.set_ylabel("Relative viscosity")
    ax_visc.set_ylim(1.0, 1.35)
    for sp in ["top", "right"]:
        ax_visc.spines[sp].set_visible(False)

    fig.text(0.04, 0.93, "a", fontsize=FONT_PANEL, fontweight="bold")
    fig.text(0.52, 0.93, "b", fontsize=FONT_PANEL, fontweight="bold")

    plt.tight_layout()
    plt.savefig("MD_validation.pdf")
    plt.close()
    print("  Saved: MD_validation.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE H — Viscosity: top-6 feature correlations
# ─────────────────────────────────────────────────────────────────────────────

print("Plotting viscosity_top6_features.pdf …")

_SHAP_CACHE = "shap_results_cache.pkl"
if os.path.isfile(_SHAP_CACHE):
    with open(_SHAP_CACHE, "rb") as _fh:
        import pickle as _pickle
        _mean_abs_shap = _pickle.load(_fh)

    top6_feats = _mean_abs_shap["viscosity"].nlargest(6).index.tolist()

    # Filtered viscosity dataset (same pipeline as diag_data)
    _df_visc        = create_cluster_splits(df, "viscosity")
    _df_visc        = filter_outliers(_df_visc, "viscosity")
    _feat_cols_visc = results["viscosity"][0]["fold_models"][0]["features"]
    _df_visc        = filter_nn_isolated(_df_visc, "viscosity", _feat_cols_visc)
    _scale_visc     = _DISPLAY_SCALE.get("viscosity", 1.0)
    _y_visc         = _df_visc["viscosity"].values * _scale_visc

    _feat_axis_labels = {
        "rdkit-Chi1n":       r"$\chi_1^n$",
        "rdkit-Chi2n":       r"$\chi_2^n$",
        "rdkit-Chi3n":       r"$\chi_3^n$",
        "rdkit-fr_NH0":      r"Tertiary amines",
        "rdkit-VSA_EState8": r"VSA·EState$_8$",
        "rdkit-SlogP_VSA2":  r"SlogP·VSA$_2$",
    }

    _DOT_COLOR = cm.viridis(0.38)

    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5), constrained_layout=True)
    axes_flat = axes.flatten()

    for i, feat in enumerate(top6_feats):
        ax = axes_flat[i]
        _x = _df_visc[feat].values
        _valid = np.isfinite(_x) & np.isfinite(_y_visc)
        _xv, _yv = _x[_valid], _y_visc[_valid]

        rho, _ = spearmanr(_xv, _yv)

        ax.scatter(_xv, _yv, s=9, alpha=0.45, color=_DOT_COLOR, linewidths=0,
                   rasterized=True)

        # Thin linear trend line
        _z = np.polyfit(_xv, _yv, 1)
        _px = np.linspace(_xv.min(), _xv.max(), 200)
        ax.plot(_px, np.polyval(_z, _px), color="0.3", lw=1.0, ls="--", zorder=3)

        ax.set_xlabel(_feat_axis_labels.get(feat, feat.replace("rdkit-", "")),
                      fontsize=FONT_LABEL)
        if i % 3 == 0:
            ax.set_ylabel(r"$\eta$ [mPa$\cdot$s]", fontsize=FONT_LABEL)

        ax.text(0.96, 0.96, fr"$\rho$ = {rho:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=FONT_ANNOT,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", lw=0.6))
        ax.text(-0.14, 1.04, ALPHABET[i], transform=ax.transAxes,
                fontsize=FONT_PANEL, fontweight="bold", va="bottom")

    plt.savefig("viscosity_top6_features.pdf", dpi=DPI)
    plt.close()
    print("  Saved: viscosity_top6_features.pdf")
else:
    print(f"  Skipping (SHAP cache not found: {_SHAP_CACHE}).")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE I — Conformal calibration curves (3×3 + summary bar)
# ─────────────────────────────────────────────────────────────────────────────

print("Plotting unc_conformal_calibration.pdf …")

CONFORMAL_COLOR  = cm.viridis(0.15)
RANK_COLOR       = cm.viridis(0.82)

fig_c = plt.figure(figsize=(12, 14))
gs_c  = gridspec.GridSpec(4, 3, figure=fig_c,
                           # ── Spacing knobs ───────────────────────────────────
                           # hspace: vertical gap between rows, as a fraction of
                           #   average subplot height. Increase for more breathing
                           #   room between the calibration panels.
                           hspace=0.35,
                           # wspace: horizontal gap between columns, as a fraction
                           #   of average subplot width.
                           wspace=0.25,
                           height_ratios=[0.75, 1, 1, 1])
# Outer figure margins (0–1, fraction of figure size).
#   top/bottom: how close the outermost subplots sit to the figure edge.
#   Decrease top or increase bottom to shift content downward.
fig_c.subplots_adjust(top=0.96, bottom=0.04)
ax_bar   = fig_c.add_subplot(gs_c[0, :])
axes_cal = [fig_c.add_subplot(gs_c[1 + i // 3, i % 3]) for i in range(9)]

gap_abs  = []
gap_rank = []

for i, target in enumerate(TARGETS):
    ax      = axes_cal[i]
    splits  = conf_data[target]
    nominal = splits[0]["nominal"]

    conf_mean      = np.mean([s["conf_empirical"]      for s in splits], axis=0)
    rank_conf_mean = np.mean([s["rank_conf_empirical"] for s in splits], axis=0)

    ax.plot(nominal, conf_mean,      color=CONFORMAL_COLOR, lw=2.0, zorder=4)
    ax.plot(nominal, rank_conf_mean, color=RANK_COLOR,      lw=2.0, zorder=4)
    ax.plot([0, 1], [0, 1], "k-", lw=0.8, alpha=0.5, zorder=5)

    ax.axvline(0.9, color="0.6", lw=0.6, ls="--")
    ax.axhline(0.9, color="0.6", lw=0.6, ls="--")

    idx90      = np.argmin(np.abs(nominal - 0.90))
    cov90_abs  = conf_mean[idx90]
    cov90_rank = rank_conf_mean[idx90]

    ax.annotate(f"{cov90_abs:.2f}", xy=(0.90, cov90_abs), fontsize=7,
                color=CONFORMAL_COLOR,
                xytext=(0.60, cov90_abs - 0.07),
                arrowprops=dict(arrowstyle="-", color=CONFORMAL_COLOR, lw=0.7))
    ax.annotate(f"{cov90_rank:.2f}", xy=(0.90, cov90_rank), fontsize=7,
                color=RANK_COLOR,
                xytext=(0.60, cov90_rank + 0.05),
                arrowprops=dict(arrowstyle="-", color=RANK_COLOR, lw=0.7))

    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.45, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Nominal coverage", fontsize=FONT_BASE)
    ax.set_ylabel("Empirical coverage", fontsize=FONT_BASE)
    ax.set_title(target_labels[target], fontsize=FONT_LABEL, pad=3)
    ax.text(-0.16, 1.05, ALPHABET[i + 1], transform=ax.transAxes,
            fontsize=FONT_PANEL, fontweight="bold", va="bottom")

    gap_abs.append(cov90_abs   - 0.90)
    gap_rank.append(cov90_rank - 0.90)

# Reorder bar chart to match Figure 4 panel a (targets_sorted = descending R\u00b2)
_gap = dict(zip(TARGETS, zip(gap_abs, gap_rank)))
gap_abs_sorted  = [_gap[t][0] for t in targets_sorted]
gap_rank_sorted = [_gap[t][1] for t in targets_sorted]

x  = np.arange(len(TARGETS))
bw = 0.35
bars_abs  = ax_bar.bar(x - bw / 2, gap_abs_sorted,  width=bw,
                       color=CONFORMAL_COLOR, label="Absolute coverage")
bars_rank = ax_bar.bar(x + bw / 2, gap_rank_sorted, width=bw,
                       color=RANK_COLOR,             label="Rank coverage")
ax_bar.axhline(0, color="k", lw=0.8)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels([target_labels[t] for t in targets_sorted], fontsize=FONT_BASE + 0.5)
ax_bar.set_ylabel("Coverage gap at 90%\n(empirical \u2212 nominal)", fontsize=FONT_BASE)
ax_bar.set_ylim(-0.12, 0.12)
ax_bar.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0%}"))
ax_bar.legend(fontsize=FONT_BASE, loc="upper left", framealpha=0.9)
ax_bar.text(-0.05, 1.08, ALPHABET[0], transform=ax_bar.transAxes,
            fontsize=FONT_PANEL, fontweight="bold", va="bottom")

for bar, val in zip(bars_abs, gap_abs_sorted):
    va, y_off = ("bottom", 0.003) if val >= 0 else ("top", -0.003)
    ax_bar.text(bar.get_x() + bar.get_width() / 2, val + y_off,
                f"{val:+.2f}", ha="center", va=va, fontsize=6.5,
                color=CONFORMAL_COLOR, fontweight="bold")
for bar, val in zip(bars_rank, gap_rank_sorted):
    va, y_off = ("bottom", 0.003) if val >= 0 else ("top", -0.003)
    ax_bar.text(bar.get_x() + bar.get_width() / 2, val + y_off,
                f"{val:+.2f}", ha="center", va=va, fontsize=6.5,
                color=RANK_COLOR, fontweight="bold")

legend_handles = [
    Line2D([0], [0], color=CONFORMAL_COLOR, lw=2.0, label="Absolute coverage"),
    Line2D([0], [0], color=RANK_COLOR,      lw=2.0, label="Rank coverage"),
    Line2D([0], [0], color="k",             lw=0.8, alpha=0.5, label="Perfect calibration"),
]
fig_c.legend(handles=legend_handles, loc="lower center", ncol=3,
             fontsize=FONT_BASE, frameon=True,
             bbox_to_anchor=(0.5, -0.02),
             handletextpad=0.5, columnspacing=1.0)

plt.savefig("unc_conformal_calibration.pdf", bbox_inches="tight")
plt.close()
print("  Saved: unc_conformal_calibration.pdf")

print("\nAll done.")
