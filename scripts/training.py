#!/usr/bin/env python3
"""
training.py — SurfKit surrogate model trainer

Trains XGBoost ensemble models for nine surfactant physicochemical properties
from RDKit molecular descriptors.  By default, only the production model set
(raw RDKit descriptors) is trained.  Pass --full-analysis to additionally
train the PCA-compressed variants and all Morgan fingerprint baselines.

Output files
------------
  models.pkl                        — raw RDKit descriptors (always)
  models_reduced.pkl                — two-stage PCA (--full-analysis)
  models_pca.pkl                    — global PCA 95% (--full-analysis)
  models_morgan_r{r}_n{n}.pkl       — Morgan fingerprints, r∈{2,3,4},
                                       n∈{512,1024,2048} (--full-analysis)

Usage
-----
  python training.py --data SurfPro-MD.csv
  python training.py --data SurfPro-MD.csv --quick
  python training.py --data SurfPro-MD.csv --full-analysis
  python training.py --data SurfPro-MD.csv --full-analysis --cpus 20
"""

import argparse
import datetime
import os
import pickle
import re
import sys
import threading
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from joblib import Parallel, delayed
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# CONSTANTS
# =============================================================================

TARGETS = [
    "pCMC", "AW_ST_CMC", "Gamma_max", "Area_min", "pC20",
    "D_MOL", "D_SOL", "surface_tension_avg", "viscosity",
]

N_FOLDS            = 5
N_TEST_SPLITS      = 5
MIN_TEST_FRACTION  = 0.07
MAX_TEST_FRACTION  = 0.11
MAX_REJECT_TRIES   = 5
MIN_VAL_FRACTION   = 0.12
MAX_VAL_FRACTION   = 0.18
RANDOM_STATE       = 3
NN_ISOLATION_THRESHOLD = 50.0
COMPRESSION_THRESHOLD  = 0.80   # within-group variance threshold for two-stage PCA

MORGAN_RADII  = [2, 3, 4]
MORGAN_NBITS  = [512, 1024, 2048]

# RDKit feature groups for two-stage PCA (keyed by name prefix / regex)
GROUP_RULES = {
    "Chi_n":      lambda n: bool(re.fullmatch(r"Chi\d+n", n)),
    "Chi_v":      lambda n: bool(re.fullmatch(r"Chi\d+v", n)),
    "Chi":        lambda n: bool(re.fullmatch(r"Chi\d+",  n)),
    "Kappa":      lambda n: (bool(re.fullmatch(r"Kappa\d+",  n)) or
                             bool(re.fullmatch(r"KappaM\d+", n)) or
                             n in ("Phi", "HallKierAlpha")),
    "PEOE_VSA":   lambda n: bool(re.fullmatch(r"PEOE_VSA\d+",  n)),
    "SMR_VSA":    lambda n: bool(re.fullmatch(r"SMR_VSA\d+",   n)),
    "SlogP_VSA":  lambda n: bool(re.fullmatch(r"SlogP_VSA\d+", n)),
    "EState_VSA": lambda n: bool(re.fullmatch(r"EState_VSA\d+",n)),
    "VSA_EState": lambda n: bool(re.fullmatch(r"VSA_EState\d+",n)),
    "MQN":        lambda n: bool(re.fullmatch(r"MQN\d+",       n)),
    "BCUT2D":     lambda n: n.startswith("BCUT2D_"),
    "MolWt":      lambda n: n in ("MolWt", "HeavyAtomMolWt", "ExactMolWt"),
    "RingCounts": lambda n: n in (
                      "RingCount", "NumAromaticRings", "NumAliphaticRings",
                      "NumSaturatedRings", "NumAromaticHeterocycles",
                      "NumAromaticCarbocycles", "NumSaturatedHeterocycles",
                      "NumSaturatedCarbocycles", "NumAliphaticHeterocycles",
                      "NumAliphaticCarbocycles"),
}

# =============================================================================
# CLI
# =============================================================================

parser = argparse.ArgumentParser(
    description="Train SurfKit surrogate models.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
examples:
  python training.py --data SurfPro-MD.csv
  python training.py --data SurfPro-MD.csv --quick
  python training.py --data SurfPro-MD.csv --full-analysis --cpus 20
    """,
)
parser.add_argument("--data",          default="../data/SurfPro-MD.csv",
                    help="Path to SurfPro-MD.csv (default: ../data/SurfPro-MD.csv).")
parser.add_argument("--output-dir",
                    default=str(Path(__file__).resolve().parent.parent / "models"),
                    help="Directory for output pkl files (default: ../models/ relative to this script).")
parser.add_argument("--trials",        type=int, default=30,
                    help="Optuna trials per fold (default: 30).")
parser.add_argument("--quick",         action="store_true",
                    help="Shorthand for --trials 10; useful for smoke-testing.")
parser.add_argument("--full-analysis", action="store_true",
                    help="Also train two-stage PCA, global PCA, and all Morgan variants.")
parser.add_argument("--cpus",          type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4)),
                    help="Total CPU budget (default: SLURM_CPUS_PER_TASK or cpu_count).")
args = parser.parse_args()

N_TRIALS      = 10 if args.quick else args.trials
N_FOLD_JOBS   = min(N_FOLDS, args.cpus)
N_OPTUNA_JOBS = max(1, args.cpus // N_FOLD_JOBS)
XGB_THREADS   = 1

os.makedirs(args.output_dir, exist_ok=True)

# =============================================================================
# LOGGING  — tee stdout to a timestamped log file
# =============================================================================

_ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
_log_path = os.path.join(args.output_dir, f"training_{_ts}.log")
_log_fh  = open(_log_path, "w")
_print_lock = threading.Lock()


class _Tee:
    def __init__(self, r, f): self._r, self._f = r, f
    def write(self, t):  self._r.write(t); self._f.write(t); self._f.flush()
    def flush(self): self._r.flush(); self._f.flush()
    def isatty(self): return False


sys.stdout = _Tee(sys.stdout, _log_fh)

print(f"Logging to {_log_path}\n")
print(f"CPUs: {args.cpus}  →  "
      f"{N_FOLD_JOBS} parallel folds × "
      f"{N_OPTUNA_JOBS} Optuna trials × "
      f"{XGB_THREADS} XGBoost thread(s)")
print(f"Optuna trials per fold: {N_TRIALS}")
print(f"Full analysis: {args.full_analysis}\n")

# =============================================================================
# DATA LOADING AND CLUSTERING
# =============================================================================

print(f"Loading {args.data} …")
df = pd.read_csv(args.data)

df["mol"]   = df["SMILES"].apply(Chem.MolFromSmiles)
df_valid    = df[df["mol"].notnull()].copy()
fps         = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in df_valid["mol"]]
nfps        = len(fps)

print(f"  {len(df)} rows, {nfps} valid molecules")
print("  Computing Butina clusters …")

dists = []
for i in range(1, nfps):
    sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
    dists.extend(1 - s for s in sims)

clusters_b = Butina.ClusterData(dists, nfps, distThresh=0.6, isDistData=True)
cid        = np.zeros(nfps, dtype=int)
for ci, cl in enumerate(clusters_b):
    for idx in cl:
        cid[idx] = ci

df_valid["cluster"] = cid.astype(int)
df["cluster"]       = np.nan
df.loc[df_valid.index, "cluster"] = df_valid["cluster"].values
print(f"  {len(clusters_b)} Butina clusters\n")

# Pull RDKit feature columns (added by the CSV generation pipeline)
rdkit_cols = sorted([c for c in df.columns if c.startswith("rdkit-")])
print(f"  {len(rdkit_cols)} RDKit descriptor columns\n")

# =============================================================================
# SHARED DATA-PREP HELPERS
# =============================================================================

def _sample_test_clusters(clusters, sizes, mn, mx, rng):
    available, selected, n, streak = set(clusters), set(), 0, 0
    while available:
        c = rng.choice(sorted(available))
        available.remove(c)
        new_n = n + sizes[c]
        if new_n < mn:
            selected.add(c); n = new_n; streak = 0; continue
        if mn <= new_n <= mx:
            selected.add(c); break
        streak += 1
        if streak >= MAX_REJECT_TRIES:
            with _print_lock:
                print("  WARN: could not fill set to target size")
            break
    return selected


def create_cluster_splits(df_input, target_col):
    df_t  = df_input.dropna(subset=[target_col, "cluster"]).copy()
    df_t["cluster"] = df_t["cluster"].astype(int)
    total = len(df_t)
    cs    = df_t.groupby("cluster").size().to_dict()
    clust = np.array(list(cs.keys()))
    for tid in range(N_TEST_SPLITS):
        rng = np.random.RandomState(RANDOM_STATE + tid)
        tc  = _sample_test_clusters(clust, cs,
                                    int(total * MIN_TEST_FRACTION),
                                    int(total * MAX_TEST_FRACTION), rng)
        colt = f"{target_col}_test_split_{tid}"
        df_t[colt] = "train_val"
        df_t.loc[df_t["cluster"].isin(tc), colt] = "test"
        rem = df_t[df_t[colt] != "test"]
        rc  = np.array(rem["cluster"].unique())
        rs  = rem.groupby("cluster").size().to_dict()
        rn  = len(rem)
        for fid in range(N_FOLDS):
            rng_f = np.random.RandomState(RANDOM_STATE + tid * 100 + fid)
            vc    = _sample_test_clusters(rc, rs,
                                          int(rn * MIN_VAL_FRACTION),
                                          int(rn * MAX_VAL_FRACTION), rng_f)
            colf  = f"{target_col}_split_t{tid}_f{fid}"
            df_t[colf] = "unused"
            df_t.loc[df_t["cluster"].isin(set(rc) - vc), colf] = "train"
            df_t.loc[df_t["cluster"].isin(vc),  colf] = "val"
            df_t.loc[df_t[colt] == "test",      colf] = "test"
    return df_t


def filter_outliers(df, target):
    n0 = len(df)
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
        df = df[df[target].notnull() & rules[target](df)]
    with _print_lock:
        print(f"  filter_outliers({target}): {n0} → {len(df)} (-{n0-len(df)})")
    return df


def filter_nn_isolated(df, feature_cols, threshold=NN_ISOLATION_THRESHOLD):
    X      = df[feature_cols].to_numpy().astype(float)
    means  = np.nanmean(X, axis=0)
    for j in range(X.shape[1]):
        X[np.isnan(X[:, j]), j] = means[j]
    X_sc   = StandardScaler().fit_transform(X)
    dists  = NearestNeighbors(n_neighbors=2, n_jobs=-1).fit(X_sc).kneighbors(X_sc)[0][:, 1]
    keep   = dists <= threshold
    n_rm   = int((~keep).sum())
    with _print_lock:
        print(f"  NN isolation filter: {len(df)} → {len(df)-n_rm} (-{n_rm})")
    return df.loc[df.index[keep]]


# =============================================================================
# HPO + TRAINING
# =============================================================================

def optimise_fold(df, feature_cols, target_col, split, fold_id):
    Xtr = df.loc[split == "train", feature_cols].reset_index(drop=True)
    Xva = df.loc[split == "val",   feature_cols].reset_index(drop=True)
    ytr = df.loc[split == "train", target_col].reset_index(drop=True)
    yva = df.loc[split == "val",   target_col].reset_index(drop=True)

    sc     = StandardScaler()
    Xtr_sc = sc.fit_transform(Xtr)
    Xva_sc = sc.transform(Xva)

    def objective(trial):
        p = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth":     trial.suggest_int("max_depth", 3, 10),
            "n_estimators":  trial.suggest_int("n_estimators", 100, 500),
        }
        m = xgb.XGBRegressor(**p, objective="reg:squarederror",
                              nthread=XGB_THREADS, verbosity=0)
        m.fit(Xtr_sc, ytr)
        return r2_score(yva, m.predict(Xva_sc))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_OPTUNA_JOBS, show_progress_bar=False)

    best = xgb.XGBRegressor(**study.best_params, objective="reg:squarederror",
                             nthread=XGB_THREADS, verbosity=0)
    best.fit(Xtr_sc, ytr)
    with _print_lock:
        print(f"    fold {fold_id} R²(val)={r2_score(yva, best.predict(Xva_sc)):.3f}  "
              f"{study.best_params}")
    return {"model": best.get_booster(), "scaler": sc,
            "features": feature_cols, "best_params": study.best_params}


def ensemble_predict(fold_models, X):
    return np.mean([m["model"].inplace_predict(m["scaler"].transform(X))
                    for m in fold_models], axis=0)


def print_summary(results, label):
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    for t in TARGETS:
        if t not in results:
            continue
        runs = {k: v for k, v in results[t].items() if isinstance(k, int)}
        r2s  = [r["metrics"]["r2"]       for r in runs.values()]
        rhos = [r["metrics"]["spearman"] for r in runs.values()]
        print(f"  {t:<22} R²={np.mean(r2s):.3f}±{np.std(r2s):.3f}  "
              f"ρ={np.mean(rhos):.3f}±{np.std(rhos):.3f}")


# =============================================================================
# FEATURE BUILDERS
# =============================================================================

def _impute_and_scale(df_t, cols):
    X     = df_t[cols].to_numpy().astype(float)
    means = np.nanmean(X, axis=0)
    for j in range(X.shape[1]):
        X[np.isnan(X[:, j]), j] = means[j]
    sc = StandardScaler()
    return sc.fit_transform(X), sc


def build_rdkit_features(df_t, cols):
    """Identity builder: just return the raw RDKit descriptor columns."""
    return df_t, list(cols), None


def build_global_pca_features(df_t, cols):
    """Global PCA on all RDKit descriptors, retaining 95% variance."""
    X_sc, raw_sc = _impute_and_scale(df_t, cols)
    global_pca   = PCA(random_state=RANDOM_STATE)
    X_pca        = global_pca.fit_transform(X_sc)
    n_pcs        = int(np.searchsorted(np.cumsum(global_pca.explained_variance_ratio_), 0.95)) + 1
    pc_cols      = [f"pc_{i}" for i in range(n_pcs)]
    df_t = df_t.copy()
    for i, col in enumerate(pc_cols):
        df_t[col] = X_pca[:, i]
    pca_transform = {
        "type": "global", "raw_feature_cols": cols, "raw_scaler": raw_sc,
        "global_pca": global_pca, "n_pcs": n_pcs, "feature_schema": pc_cols,
    }
    with _print_lock:
        print(f"  Global PCA: {len(cols)} raw → {n_pcs} PCs (95% var)")
    return df_t, pc_cols, pca_transform


def _assign_group(col):
    bare = col.replace("rdkit-", "")
    for label, rule in GROUP_RULES.items():
        if rule(bare):
            return label
    return None


def build_two_stage_pca_features(df_t, cols):
    """Two-stage PCA: well-compressed groups → single PC1; rest → pool PCA 95%."""
    X_sc, raw_sc = _impute_and_scale(df_t, cols)

    groups, ungrouped_idx = {}, []
    for ci, col in enumerate(cols):
        g = _assign_group(col)
        if g:
            groups.setdefault(g, []).append(ci)
        else:
            ungrouped_idx.append(ci)

    kept_cols, pooled_idx, group_pcas = [], list(ungrouped_idx), {}
    for label, idxs in groups.items():
        pca_g    = PCA(n_components=1, random_state=RANDOM_STATE)
        pc1      = pca_g.fit_transform(X_sc[:, idxs]).ravel()
        var_expl = float(pca_g.explained_variance_ratio_[0])
        if var_expl >= COMPRESSION_THRESHOLD:
            col_name = f"grp_{label}"
            df_t = df_t.copy(); df_t[col_name] = pc1
            kept_cols.append(col_name)
            group_pcas[label] = {"pca": pca_g, "col_indices": idxs}
        else:
            pooled_idx.extend(idxs)

    pool_sc    = StandardScaler()
    X_pool_sc  = pool_sc.fit_transform(X_sc[:, pooled_idx])
    pool_pca   = PCA(random_state=RANDOM_STATE)
    X_pool_pca = pool_pca.fit_transform(X_pool_sc)
    n_pool     = int(np.searchsorted(np.cumsum(pool_pca.explained_variance_ratio_), 0.95)) + 1
    pool_cols  = [f"pool_{i}" for i in range(n_pool)]
    df_t       = df_t.copy()
    for i, col in enumerate(pool_cols):
        df_t[col] = X_pool_pca[:, i]

    feature_schema = kept_cols + pool_cols
    pca_transform  = {
        "type": "two_stage", "raw_feature_cols": cols, "raw_scaler": raw_sc,
        "group_pcas": group_pcas, "pool_col_indices": pooled_idx,
        "pool_scaler": pool_sc, "pool_pca": pool_pca, "n_pool_pcs": n_pool,
        "feature_schema": feature_schema,
    }
    with _print_lock:
        print(f"  Two-stage PCA: {len(cols)} raw → "
              f"{len(kept_cols)} group PC1s + {n_pool} pool PCs = {len(feature_schema)} features")
    return df_t, feature_schema, pca_transform


def build_morgan_features(df_t, cols, radius, nbits):
    """Compute Morgan fingerprints for the molecules in df_t."""
    fp_cols = [f"morgan_{i}" for i in range(nbits)]
    fps_arr = np.zeros((len(df_t), nbits), dtype=np.float32)
    smiles  = df_t["SMILES"].tolist()
    for row_i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
            DataStructs.ConvertToNumpyArray(fp, fps_arr[row_i])
    df_t = df_t.copy()
    for j, col in enumerate(fp_cols):
        df_t[col] = fps_arr[:, j]
    return df_t, fp_cols, {"type": "morgan", "radius": radius, "nbits": nbits}


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def run_training(df_base, feature_cols, build_features_fn, out_file, label,
                 feature_fn_kwargs=None):
    """Train 5×5 ensemble for all targets and save to out_file.

    build_features_fn(df_t, feature_cols, **kwargs) → (df_t, schema, pca_transform)
    """
    if feature_fn_kwargs is None:
        feature_fn_kwargs = {}

    print(f"\n{'#'*60}\n# {label}\n# Output: {out_file}\n{'#'*60}")
    results = {}

    for target in TARGETS:
        print(f"\nWorking on {target}")

        df_t = create_cluster_splits(df_base, target)
        df_t = filter_outliers(df_t, target)
        df_t, feature_schema, pca_transform = build_features_fn(
            df_t, feature_cols, **feature_fn_kwargs)
        df_t = filter_nn_isolated(df_t, feature_schema)

        results[target] = {}
        if pca_transform is not None:
            results[target]["_pca_transform"] = pca_transform

        for tid in range(N_TEST_SPLITS):
            print(f"\n  ### TEST SPLIT {tid}  "
                  f"({N_FOLDS} folds × {N_TRIALS} Optuna trials, "
                  f"{N_FOLD_JOBS} parallel)")

            fold_models = Parallel(n_jobs=N_FOLD_JOBS)(
                delayed(optimise_fold)(
                    df_t, feature_schema, target,
                    df_t[f"{target}_split_t{tid}_f{fid}"], fid,
                )
                for fid in range(N_FOLDS)
            )

            split_col0 = f"{target}_test_split_{tid}"
            test_mask  = df_t[split_col0] == "test"
            X_test     = df_t.loc[test_mask, feature_schema].to_numpy()
            y_test     = df_t.loc[test_mask, target].to_numpy()
            y_pred     = ensemble_predict(fold_models, X_test)

            r2   = r2_score(y_test, y_pred)
            rho, _ = spearmanr(y_test, y_pred)
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            print(f"  Ensemble R² = {r2:.3f}  ρ = {rho:.3f}")

            results[target][tid] = {
                "fold_models":    fold_models,
                "test_idx":       df_t.index[test_mask].to_numpy(),
                "feature_schema": feature_schema,
                "y_test":         y_test.tolist(),
                "y_pred":         y_pred.tolist(),
                "metrics": {"r2": r2, "rmse": rmse, "spearman": rho},
            }

            with open(out_file, "wb") as f:
                pickle.dump(results, f)
            print(f"  Checkpoint → {out_file}")

    print_summary(results, label)
    return results


# =============================================================================
# RUN
# =============================================================================

# ── 1. Production model: raw RDKit descriptors ────────────────────────────────

run_training(
    df_base=df,
    feature_cols=rdkit_cols,
    build_features_fn=build_rdkit_features,
    out_file=os.path.join(args.output_dir, "models.pkl"),
    label="Raw RDKit descriptors (217 features)",
)

if args.full_analysis:

    # ── 2. Two-stage PCA ──────────────────────────────────────────────────────

    run_training(
        df_base=df,
        feature_cols=rdkit_cols,
        build_features_fn=build_two_stage_pca_features,
        out_file=os.path.join(args.output_dir, "models_reduced.pkl"),
        label="Two-stage PCA (group PC1s + pool 95%)",
    )

    # ── 3. Global PCA 95% ─────────────────────────────────────────────────────

    run_training(
        df_base=df,
        feature_cols=rdkit_cols,
        build_features_fn=build_global_pca_features,
        out_file=os.path.join(args.output_dir, "models_pca.pkl"),
        label="Global PCA (95% variance)",
    )

    # ── 4. Morgan fingerprints (9 configurations) ─────────────────────────────

    for radius in MORGAN_RADII:
        for nbits in MORGAN_NBITS:
            run_training(
                df_base=df,
                feature_cols=rdkit_cols,      # passed for signature compat; unused
                build_features_fn=build_morgan_features,
                out_file=os.path.join(args.output_dir,
                                      f"models_morgan_r{radius}_n{nbits}.pkl"),
                label=f"Morgan fingerprints r={radius}, n={nbits}",
                feature_fn_kwargs={"radius": radius, "nbits": nbits},
            )

print(f"\nAll done.  Log: {_log_path}")
