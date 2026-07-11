#!/usr/bin/env python3
"""
predict.py — SurfPro-MD surrogate model predictor

Predict surfactant properties from a SMILES string using the XGBoost ensemble
models trained in surrogate.py.  Every prediction comes with three complementary
uncertainty measures and SHAP-based feature attribution.

Predicted properties
--------------------
  pCMC             — log critical micelle concentration
  AW_ST_CMC        — air-water surface tension at the CMC
  Gamma_max        — maximum surface excess
  Area_min         — minimum area per molecule
  pC20             — efficiency of surface tension reduction
  D_MOL            — molecular self-diffusion coefficient  [×10⁻⁹ m²/s]
  D_SOL            — solvent self-diffusion coefficient    [×10⁻⁹ m²/s]
  surface_tension_avg — MD surface tension                [mN/m]
  viscosity        — dynamic viscosity                    [mPa·s]

Uncertainty measures
--------------------
  ensemble_std       — std of all 25 fold predictions; a relative model-confidence
                       signal (no guaranteed coverage).
  nn_dist            — Euclidean distance to the nearest training molecule in the
                       scaled RDKit feature space; a structural novelty indicator.
                       Requires fit_applicability_domain() to be called first.
  conformal_lower/upper — prediction interval with nominal coverage `coverage`
                       (default 90 %).  Guaranteed to contain the true value for
                       ≥ coverage fraction of molecules similar to the training
                       set (marginal coverage guarantee).
                       Requires calibrate() to be called first.

Usage (CLI)
-----------
  python predict.py "CCCCCCCCCC(=O)O"
  python predict.py "CCCCCCCCCC(=O)O" --training-csv ../data/SurfPro-MD.csv
  python predict.py "CCCCCCCCCC(=O)O" --coverage 0.95 --top-features 5
  python predict.py --input-file molecules.txt --training-csv ../data/SurfPro-MD.csv

Usage (Python API)
------------------
  from predict import SurfProPredictor

  p = SurfProPredictor()  # defaults to ../models/models.pkl

  # One-time calibration step (saves conformal_calibration.pkl alongside models)
  p.calibrate("../data/SurfPro-MD.csv")
  p.save_calibration()                    # writes conformal_calibration.pkl

  # Build applicability domain (for NN-distance measure)
  p.fit_applicability_domain("../data/SurfPro-MD.csv")

  # Predict with all uncertainty measures + top-5 SHAP features
  result = p.predict("CCCCCCCCCC(=O)O", coverage=0.90, top_features=5)
  print(result["prediction"])    # point estimates + units
  print(result["uncertainty"])   # ensemble_std / conformal interval / nn_dist
  print(result["top_features"])  # per-target top-N SHAP contributions

Dependencies
------------
  rdkit, numpy, pandas, xgboost, scikit-learn
  shap  (optional, only needed for top_features > 0)
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors


# ── Module-level descriptor calculator (built once at import time) ────────────

_DESCRIPTOR_NAMES = [name for name, _ in Descriptors._descList]
_CALCULATOR = MoleculeDescriptors.MolecularDescriptorCalculator(_DESCRIPTOR_NAMES)

# ── Display configuration ─────────────────────────────────────────────────────

DEFAULT_LABELS = {
    "pCMC":               "pCMC",
    "AW_ST_CMC":          "gamma_CMC",
    "Gamma_max":          "Gamma_max",
    "pC20":               "pC20",
    "Pi_CMC":             "Pi_CMC",
    "Area_min":           "A_min",
    "viscosity":          "eta",
    "D_MOL":              "D_MOL",
    "D_SOL":              "D_SOL",
    "surface_tension_avg": "gamma",
}

UNITS = {
    "pCMC":               "log(M)",
    "AW_ST_CMC":          "mN/m",
    "Gamma_max":          "umol/m2",
    "pC20":               "log(M)",
    "Pi_CMC":             "mN/m",
    "Area_min":           "nm2",
    "viscosity":          "mPa·s",
    "D_MOL":              "x1e-9 m2/s",   # stored in 1e-5 cm2/s = 1e-9 m2/s
    "D_SOL":              "x1e-9 m2/s",
    "surface_tension_avg": "mN/m",
}

# Multiply raw model output by this factor before reporting in UNITS above.
DISPLAY_SCALE = {
    "viscosity":          1e3,   # Pa·s  → mPa·s
    "surface_tension_avg": 0.1,  # bar·nm → mN/m
}


# ── Calibration pipeline constants (must match surrogate.py) ──────────────────

_N_FOLDS           = 5
_N_TEST_SPLITS     = 5
_RANDOM_STATE      = 3
_MIN_TEST_FRAC     = 0.07
_MAX_TEST_FRAC     = 0.11
_MIN_VAL_FRAC      = 0.12
_MAX_VAL_FRAC      = 0.18
_MAX_REJECT_TRIES  = 5
_NN_ISO_THRESHOLD  = 50.0


# ── SMILES utilities ──────────────────────────────────────────────────────────

def canonicalize_smiles(smiles: str, keep_largest: bool = True) -> str:
    """Return canonical SMILES, stripping counterions when keep_largest=True."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    if keep_largest and "." in smiles:
        fragments = Chem.GetMolFrags(mol, asMols=True)
        mol = max(fragments, key=lambda m: m.GetNumHeavyAtoms())
    return Chem.MolToSmiles(mol, canonical=True)


def compute_rdkit_descriptors(smiles: str) -> dict:
    """Return a dict of all RDKit descriptors keyed as 'rdkit-<Name>'."""
    mol = Chem.MolFromSmiles(smiles)
    values = _CALCULATOR.CalcDescriptors(mol)
    return {f"rdkit-{name}": val for name, val in zip(_DESCRIPTOR_NAMES, values)}


# ── SHAP / XGBoost compatibility helper ──────────────────────────────────────

def _make_shap_tree_explainer(booster):
    """Create shap.TreeExplainer with a workaround for the XGBoost 2.x base_score
    bracket-notation bug (SHAP cannot parse '[3.208E0]' stored in UBJSON)."""
    import re
    import shap

    _orig_save_raw = booster.save_raw

    def _fixed_save_raw(raw_format="ubj"):
        raw = _orig_save_raw(raw_format=raw_format)
        if raw_format == "ubj":
            def _unwrap(m):
                val = float(m.group(1))
                fixed = f"{val:.10g}".encode()
                if len(fixed) < len(m.group(0)):
                    fixed = fixed + b" " * (len(m.group(0)) - len(fixed))
                return fixed[: len(m.group(0))]
            raw = re.sub(rb"\[(-?[\d.]+(?:[Ee][+-]?\d+)?)\]", _unwrap, raw)
        return raw

    booster.save_raw = _fixed_save_raw
    try:
        explainer = shap.TreeExplainer(booster)
    finally:
        booster.save_raw = _orig_save_raw
    return explainer


# ── Calibration pipeline helpers (mirror of surrogate.py data prep) ───────────
# These are only called during calibrate().  All rdkit clustering imports are
# deferred to inside calibrate() to keep the normal prediction path lightweight.

def _sample_clusters(clusters, sizes, min_n, max_n, rng):
    available = set(clusters)
    chosen, count, streak = set(), 0, 0
    while available:
        c = rng.choice(list(available))
        available.remove(c)
        new = count + sizes[c]
        if new < min_n:
            chosen.add(c); count = new; streak = 0; continue
        if min_n <= new <= max_n:
            chosen.add(c); count = new; break
        streak += 1
        if streak >= _MAX_REJECT_TRIES:
            break
    return chosen


def _create_cluster_splits(df_in, target_col):
    df_t    = df_in.dropna(subset=[target_col, "cluster"]).copy()
    df_t["cluster"] = df_t["cluster"].astype(int)
    total   = len(df_t)
    cs_all  = df_t.groupby("cluster").size().to_dict()
    clusters = np.array(list(cs_all.keys()))
    for ts in range(_N_TEST_SPLITS):
        rng = np.random.RandomState(_RANDOM_STATE + ts)
        tc  = _sample_clusters(clusters, cs_all,
                                int(total * _MIN_TEST_FRAC),
                                int(total * _MAX_TEST_FRAC), rng)
        col_test = f"{target_col}_test_split_{ts}"
        df_t[col_test] = "train_val"
        df_t.loc[df_t["cluster"].isin(tc), col_test] = "test"
        rem  = df_t[df_t[col_test] != "test"]
        rc   = np.array(rem["cluster"].unique())
        rs   = rem.groupby("cluster").size().to_dict()
        rm   = len(rem)
        for fi in range(_N_FOLDS):
            rng_f = np.random.RandomState(_RANDOM_STATE + ts * 100 + fi)
            vc    = _sample_clusters(rc, rs,
                                     int(rm * _MIN_VAL_FRAC),
                                     int(rm * _MAX_VAL_FRAC), rng_f)
            col = f"{target_col}_split_t{ts}_f{fi}"
            df_t[col] = "unused"
            df_t.loc[df_t["cluster"].isin(set(rc) - vc), col] = "train"
            df_t.loc[df_t["cluster"].isin(vc),           col] = "val"
            df_t.loc[df_t[col_test] == "test",           col] = "test"
    return df_t


def _filter_outliers(df, target):
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


def _filter_nn_isolated(df, feature_cols):
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    X = df[feature_cols].to_numpy().astype(float)
    col_means = np.nanmean(X, axis=0)
    for j in range(X.shape[1]):
        X[np.isnan(X[:, j]), j] = col_means[j]
    X_sc = StandardScaler().fit_transform(X)
    nn   = NearestNeighbors(n_neighbors=2, metric="euclidean", n_jobs=-1).fit(X_sc)
    d    = nn.kneighbors(X_sc)[0][:, 1]
    return df.loc[df.index[d <= _NN_ISO_THRESHOLD]]


# ── Predictor ─────────────────────────────────────────────────────────────────

class SurfProPredictor:
    """Ensemble XGBoost predictor for nine surfactant properties.

    Quick start
    -----------
    >>> p = SurfProPredictor("models.pkl")
    >>> p.calibrate("../data/SurfPro-MD.csv")   # one-time; saves calibration file
    >>> p.fit_applicability_domain("../data/SurfPro-MD.csv")
    >>> result = p.predict("CCCCCCCCCC(=O)O")
    >>> print(result["prediction"])
    >>> print(result["uncertainty"])
    >>> print(result["top_features"]["eta"])   # top SHAP features for viscosity
    """

    _DEFAULT_MODEL = str(Path(__file__).resolve().parent.parent / "models" / "models.pkl")

    def __init__(
        self,
        model_file: str = None,
        cal_file: str = None,
        target_labels: dict = None,
    ):
        """
        Parameters
        ----------
        model_file : str
            Path to models.pkl produced by surrogate.py.
        cal_file : str, optional
            Path to conformal_calibration.pkl.  If None, looks for
            '<model_dir>/conformal_calibration.pkl' and loads it automatically.
        target_labels : dict, optional
            Override display names (keys = internal target names).
        """
        if model_file is None:
            model_file = self._DEFAULT_MODEL
        with open(model_file, "rb") as f:
            self._results = pickle.load(f)

        self.targets      = list(self._results.keys())
        self.target_labels = target_labels or DEFAULT_LABELS.copy()
        self._model_file   = model_file
        self._conformal_cal: dict = {}
        self._ad_data: dict = {}

        # Auto-load calibration file if it exists
        if cal_file is None:
            cal_file = os.path.join(os.path.dirname(os.path.abspath(model_file)),
                                    "conformal_calibration.pkl")
        if os.path.isfile(cal_file):
            self.load_calibration(cal_file)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_features(self, smiles: str, feature_schema: list) -> pd.DataFrame:
        canonical   = canonicalize_smiles(smiles)
        descriptors = compute_rdkit_descriptors(canonical)
        return pd.DataFrame([descriptors]).reindex(columns=feature_schema)

    def _label(self, target: str) -> str:
        return self.target_labels.get(target, target)

    def _all_fold_preds(self, smiles: str, target: str) -> np.ndarray:
        """Return all 25 raw (unscaled) fold predictions for one target."""
        first_info   = next(iter(self._results[target].values()))
        X            = self._build_features(smiles, first_info["feature_schema"])
        all_preds    = []
        for info in self._results[target].values():
            for fold in info["fold_models"]:
                all_preds.append(
                    fold["model"].inplace_predict(fold["scaler"].transform(X))[0]
                )
        return np.array(all_preds)

    # ── Conformal calibration ─────────────────────────────────────────────────

    def calibrate(self, csv_path: str, smiles_col: str = "SMILES",
                  verbose: bool = True) -> "SurfProPredictor":
        """Compute conformal calibration scores from the training CSV.

        For each target and each (test_split, fold) pair, runs the fold model
        on its validation set and pools the absolute residuals as calibration
        scores.  Also stores ensemble stds on the same val sets to support
        normalised conformal (adaptive per-molecule intervals).

        The calibration is stored on the predictor and used automatically by
        predict().  Call save_calibration() to persist it.

        Parameters
        ----------
        csv_path : str
            Path to the training CSV (same file used by surrogate.py).
        smiles_col : str
            SMILES column name in the CSV.
        verbose : bool
            Print progress.
        """
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
        from rdkit.ML.Cluster import Butina
        from sklearn.preprocessing import StandardScaler

        if verbose:
            print(f"Calibrating from {csv_path} …")

        df = pd.read_csv(csv_path)

        # Butina clustering (same seed / threshold as surrogate.py)
        df["mol"]   = df[smiles_col].apply(Chem.MolFromSmiles)
        df_valid    = df[df["mol"].notnull()].copy()
        fps         = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024)
                       for m in df_valid["mol"]]
        nfps        = len(fps)
        dists       = []
        for i in range(1, nfps):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
            dists.extend(1 - s for s in sims)
        clusters_b  = Butina.ClusterData(dists, nfps, distThresh=0.6, isDistData=True)
        cid         = np.zeros(nfps, dtype=int)
        for ci, cl in enumerate(clusters_b):
            for idx in cl:
                cid[idx] = ci
        df_valid["cluster"] = cid.astype(int)
        df["cluster"]       = np.nan
        df.loc[df_valid.index, "cluster"] = df_valid["cluster"].values

        self._conformal_cal = {}

        for target in self.targets:
            if verbose:
                print(f"  {target}", flush=True)

            df_t         = _create_cluster_splits(df, target)
            df_t         = _filter_outliers(df_t, target)
            feature_cols = self._results[target][0]["fold_models"][0]["features"]
            df_t         = _filter_nn_isolated(df_t, feature_cols)

            cal_errors, cal_stds = [], []

            for ts in range(_N_TEST_SPLITS):
                fold_models = self._results[target][ts]["fold_models"]
                for fi, fm in enumerate(fold_models):
                    col_fi   = f"{target}_split_t{ts}_f{fi}"
                    if col_fi not in df_t.columns:
                        continue
                    val_mask = df_t[col_fi] == "val"
                    if val_mask.sum() == 0:
                        continue

                    X_val = df_t.loc[val_mask, feature_cols].to_numpy()
                    y_val = df_t.loc[val_mask, target].to_numpy()

                    # Single-fold prediction (the model that withheld this val set)
                    y_val_pred = fm["model"].inplace_predict(
                        fm["scaler"].transform(X_val))
                    cal_errors.extend(np.abs(y_val_pred - y_val))

                    # Ensemble std on val (all 5 folds, same split ts)
                    fold_preds_val = np.array([
                        m["model"].inplace_predict(m["scaler"].transform(X_val))
                        for m in fold_models
                    ])
                    cal_stds.extend(fold_preds_val.std(axis=0))

            cal_errors = np.array(cal_errors)  # raw model units
            cal_stds   = np.array(cal_stds)

            self._conformal_cal[target] = {
                "cal_errors": cal_errors,
                "cal_stds":   cal_stds,
                "n_cal":      len(cal_errors),
            }

        if verbose:
            print("  Calibration complete.")
        return self

    def save_calibration(self, path: str = None) -> None:
        """Save calibration scores to a pickle file.

        Default path: '<model_dir>/conformal_calibration.pkl'
        (auto-loaded by __init__ on the next run).
        """
        if path is None:
            path = os.path.join(os.path.dirname(os.path.abspath(self._model_file)),
                                "conformal_calibration.pkl")
        with open(path, "wb") as f:
            pickle.dump(self._conformal_cal, f)
        print(f"Saved conformal calibration → {path}")

    def load_calibration(self, path: str) -> "SurfProPredictor":
        """Load calibration scores from a pickle file."""
        with open(path, "rb") as f:
            self._conformal_cal = pickle.load(f)
        return self

    # ── Applicability domain ──────────────────────────────────────────────────

    def fit_applicability_domain(
        self, csv_path: str, smiles_col: str = "SMILES", verbose: bool = True
    ) -> "SurfProPredictor":
        """Fit per-target nearest-neighbour applicability domain.

        For each target, indexes the scaled RDKit feature vectors of all
        training molecules that have a measured value for that target.
        predict() then reports the distance from a query molecule to its
        nearest training neighbour, and a normalised version
        (distance / mean within-training spacing).

        Parameters
        ----------
        csv_path : str
            Path to the training CSV.
        smiles_col : str
            SMILES column name.
        verbose : bool
            Print progress.
        """
        from sklearn.neighbors import NearestNeighbors
        from sklearn.metrics.pairwise import euclidean_distances

        if verbose:
            print(f"Fitting applicability domain from {csv_path} …")

        df = pd.read_csv(csv_path)
        ref_info     = self._results[self.targets[0]][0]
        feature_schema = ref_info["feature_schema"]
        scaler         = ref_info["fold_models"][0]["scaler"]

        self._ad_data = {}

        for target in self.targets:
            subset = df[df[target].notna()] if target in df.columns else df
            smiles_list = subset[smiles_col].dropna().tolist()

            feats = []
            for smi in smiles_list:
                try:
                    X = self._build_features(smi, feature_schema).values[0].astype(float)
                    X[np.isnan(X)] = scaler.mean_[np.isnan(X)]
                    feats.append(X)
                except Exception:
                    continue

            if not feats:
                continue

            X_train = scaler.transform(np.array(feats))
            D = euclidean_distances(X_train)
            np.fill_diagonal(D, np.inf)
            baseline = D.min(axis=1)

            self._ad_data[target] = {
                "X_train":        X_train,
                "baseline_mean":  float(baseline.mean()),
                "baseline_std":   float(baseline.std()),
                "n_training":     len(X_train),
            }

        if verbose:
            print("  Applicability domain ready.")
        return self

    def _query_ad(self, smiles: str) -> dict:
        """Return per-target NN-distance info for one query molecule."""
        if not self._ad_data:
            return {}
        ref_info = self._results[self.targets[0]][0]
        feature_schema = ref_info["feature_schema"]
        scaler         = ref_info["fold_models"][0]["scaler"]

        X = self._build_features(smiles, feature_schema).values[0].astype(float)
        X[np.isnan(X)] = scaler.mean_[np.isnan(X)]
        X_q = scaler.transform(X.reshape(1, -1))

        from sklearn.metrics.pairwise import euclidean_distances

        results = {}
        for target, data in self._ad_data.items():
            dists     = euclidean_distances(X_q, data["X_train"])[0]
            nn        = float(dists.min())
            bm        = data["baseline_mean"]
            threshold = bm + 2.0 * data["baseline_std"]
            results[target] = {
                "nn_dist":      nn,
                "nn_dist_norm": nn / threshold if threshold > 0 else float("inf"),
                "in_domain":    nn <= threshold,
            }
        return results

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(
        self,
        smiles: str,
        coverage: float = 0.90,
        top_features: int = 5,
        ad=None,          # kept for backward compat; prefer fit_applicability_domain()
    ) -> dict:
        """Predict all surfactant properties with full uncertainty quantification.

        Parameters
        ----------
        smiles : str
            Input molecule as a SMILES string.
        coverage : float
            Nominal coverage level for conformal prediction intervals (default 0.90).
            Requires calibrate() to have been called (or conformal_calibration.pkl
            to exist alongside models.pkl).
        top_features : int
            Number of top SHAP features to return per target.
            Set to 0 to skip SHAP (faster). Default: 5.
        ad : ApplicabilityDomain, optional
            Legacy applicability-domain object (from the old API).  Prefer calling
            fit_applicability_domain() on this predictor instead.

        Returns
        -------
        dict with keys:
            ``"prediction"``   — pd.DataFrame[prediction, unit]
            ``"uncertainty"``  — pd.DataFrame with columns:
                                   ensemble_std          (always)
                                   conformal_lower/upper (when calibrated)
                                   nn_dist, nn_dist_norm (when AD fitted)
            ``"top_features"`` — dict: label → pd.Series[feature → |SHAP|]
                                   (only when top_features > 0)
        """
        has_conformal = bool(self._conformal_cal)
        has_ad        = bool(self._ad_data) or (ad is not None)

        # NN-distance query (new-style AD takes priority over legacy object)
        ad_results = {}
        if self._ad_data:
            ad_results = self._query_ad(smiles)
        elif ad is not None:
            for target, info in ad.query(smiles).items():
                ad_results[target] = {
                    "nn_dist":      info.get("nn_distance", float("nan")),
                    "nn_dist_norm": info.get("nn_distance_normalized", float("nan")),
                    "in_domain":    info.get("in_domain", None),
                }

        pred_rows      = {}
        unc_rows       = {}
        in_domain_rows = {}

        for target in self.targets:
            label = self._label(target)
            scale = DISPLAY_SCALE.get(target, 1.0)
            eps   = 1e-10   # prevent division by zero in normalised conformal

            # All 25 raw predictions
            all_preds = self._all_fold_preds(smiles, target)
            pred_raw  = float(np.mean(all_preds))
            std_raw   = float(np.std(all_preds))    # std of all 25 fold predictions

            pred_rows[label] = {
                "prediction": pred_raw * scale,
                "unit":       UNITS.get(target, ""),
            }

            unc = {"ensemble_std": std_raw * scale}

            # ── Conformal prediction intervals ────────────────────────────────
            if has_conformal and target in self._conformal_cal:
                cal        = self._conformal_cal[target]
                cal_errors = cal["cal_errors"]   # raw model units
                cal_stds   = cal["cal_stds"]     # raw model units
                n_cal      = cal["n_cal"]

                level = min(1.0, coverage * (1.0 + 1.0 / n_cal))

                # Standard conformal: fixed-width interval
                q_conf = np.quantile(cal_errors, level)
                unc["conformal_lower"] = (pred_raw - q_conf) * scale
                unc["conformal_upper"] = (pred_raw + q_conf) * scale

                # Normalised conformal: interval scales with ensemble std
                cal_norm_scores = cal_errors / (cal_stds + eps)
                q_norm          = np.quantile(cal_norm_scores, level)
                hw_norm         = q_norm * (std_raw + eps)
                unc["conformal_norm_lower"] = (pred_raw - hw_norm) * scale
                unc["conformal_norm_upper"] = (pred_raw + hw_norm) * scale
            else:
                unc["conformal_lower"]      = float("nan")
                unc["conformal_upper"]      = float("nan")
                unc["conformal_norm_lower"] = float("nan")
                unc["conformal_norm_upper"] = float("nan")

            # ── Applicability domain ──────────────────────────────────────────
            if target in ad_results:
                unc["nn_dist"]      = ad_results[target]["nn_dist"]
                unc["nn_dist_norm"] = ad_results[target]["nn_dist_norm"]
            else:
                unc["nn_dist"]      = float("nan")
                unc["nn_dist_norm"] = float("nan")

            # in_domain is boolean/None — kept separate so the numeric DataFrame
            # stays float dtype throughout (mixed types break pandas dtype inference)
            in_domain_rows[label] = ad_results.get(target, {}).get("in_domain", None)

            unc_rows[label] = unc

        unc_df = pd.DataFrame(unc_rows).T.astype(float)
        # Attach in_domain as a string column to keep float dtype on all others
        unc_df["in_domain"] = pd.Series(in_domain_rows).map(
            {True: "yes", False: "NO", None: "—"}
        ).fillna("—")

        output = {
            "prediction": pd.DataFrame(pred_rows).T,
            "uncertainty": unc_df,
        }

        # ── SHAP feature attribution ──────────────────────────────────────────
        if top_features > 0:
            output["top_features"] = {}
            for target in self.targets:
                label    = self._label(target)
                mean_imp, _ = self.explain_target(
                    smiles, target, test_id=0, max_folds=1, return_std=True
                )
                output["top_features"][label] = mean_imp.head(top_features).rename(
                    lambda f: f.replace("rdkit-", "")
                )

        return output

    # ── Backward-compatible helpers ───────────────────────────────────────────

    def predict_target(self, smiles: str, target: str) -> tuple:
        """(mean, std) across all 25 fold predictions. Kept for backward compat."""
        all_preds = self._all_fold_preds(smiles, target)
        return float(np.mean(all_preds)), float(np.std(all_preds))

    def predict_with_error_bars(self, smiles: str) -> pd.DataFrame:
        """Per-split breakdown (mean, std, min, max). Kept for backward compat."""
        per_target = {}
        for target in self.targets:
            preds = []
            for info in self._results[target].values():
                first_info = next(iter(self._results[target].values()))
                X = self._build_features(smiles, first_info["feature_schema"])
                fold_preds = [
                    fold["model"].inplace_predict(fold["scaler"].transform(X))[0]
                    for fold in info["fold_models"]
                ]
                preds.append(np.mean(fold_preds))
            preds = np.array(preds)
            per_target[target] = {
                "mean": float(np.mean(preds)), "std":  float(np.std(preds)),
                "min":  float(np.min(preds)),  "max":  float(np.max(preds)),
            }
        return pd.DataFrame(per_target).T

    # ── SHAP explanations ─────────────────────────────────────────────────────

    def explain_target(
        self,
        smiles: str,
        target: str,
        test_id: int = 0,
        fold_ids: list = None,
        max_folds: int = 3,
        return_std: bool = True,
    ):
        """Mean absolute SHAP values for one target.

        Parameters
        ----------
        smiles : str
            Input molecule.
        target : str
            Internal target name.
        test_id : int
            Outer test split to use (0–4).
        fold_ids : list of int, optional
            Specific fold indices to evaluate.
        max_folds : int
            Cap on folds (fewer = faster; 1 is usually sufficient for attribution).
        return_std : bool
            If True return (mean_series, std_series), else just mean_series.
        """
        import shap

        info     = self._results[target][test_id]
        X        = self._build_features(smiles, info["feature_schema"])
        selected = (fold_ids or list(range(len(info["fold_models"]))))[:max_folds]

        shap_values_all = []
        for fold_id in selected:
            fold       = info["fold_models"][fold_id]
            X_scaled   = fold["scaler"].transform(X)
            explainer  = _make_shap_tree_explainer(fold["model"])
            shap_vals  = explainer.shap_values(X_scaled)[0]
            shap_values_all.append(np.abs(shap_vals))

        shap_arr = np.array(shap_values_all)
        mean_imp = (
            pd.Series(shap_arr.mean(axis=0), index=info["feature_schema"])
            .sort_values(ascending=False)
        )

        if return_std:
            std_imp = pd.Series(shap_arr.std(axis=0), index=info["feature_schema"]).loc[mean_imp.index]
            return mean_imp, std_imp
        return mean_imp

    def explain(self, smiles: str, max_folds: int = 1) -> dict:
        """Top-5 SHAP importances for all targets (uses split 0)."""
        results = {}
        for target in self.targets:
            mean_imp, std_imp = self.explain_target(
                smiles, target, test_id=0, max_folds=max_folds
            )
            results[target] = {
                "mean_importance": mean_imp.head(5),
                "std_importance":  std_imp.head(5),
            }
        return results


# ── Legacy ApplicabilityDomain class (kept for backward compat) ───────────────

class ApplicabilityDomain:
    """Backward-compatible AD class. Prefer SurfProPredictor.fit_applicability_domain()."""

    def __init__(self, predictor: SurfProPredictor):
        ref_info            = predictor._results[predictor.targets[0]][0]
        self._predictor     = predictor
        self._feature_schema = ref_info["feature_schema"]
        self._scaler        = ref_info["fold_models"][0]["scaler"]
        self._fill_values   = self._scaler.mean_
        self._per_target: dict = {}

    def _scale_smiles_list(self, smiles_list):
        features = []
        for smi in smiles_list:
            try:
                X = self._predictor._build_features(smi, self._feature_schema).values[0].astype(float)
                X[np.isnan(X)] = self._fill_values[np.isnan(X)]
                features.append(X)
            except Exception:
                continue
        if not features:
            return np.empty((0, len(self._feature_schema)))
        return self._scaler.transform(np.array(features))

    def fit(self, df, smiles_col="SMILES"):
        from sklearn.metrics.pairwise import euclidean_distances
        for target in self._predictor.targets:
            subset     = df[df[target].notna()] if target in df.columns else df
            X_train    = self._scale_smiles_list(subset[smiles_col].dropna().tolist())
            if len(X_train) == 0:
                continue
            D = euclidean_distances(X_train)
            np.fill_diagonal(D, np.inf)
            self._per_target[target] = {
                "training_scaled":   X_train,
                "baseline_nn_dists": D.min(axis=1),
                "n_training":        len(X_train),
            }
        return self

    def fit_from_csv(self, csv_path, smiles_col="SMILES"):
        return self.fit(pd.read_csv(csv_path), smiles_col=smiles_col)

    def query(self, smiles):
        from sklearn.metrics.pairwise import euclidean_distances
        X    = self._predictor._build_features(smiles, self._feature_schema).values[0].astype(float)
        X[np.isnan(X)] = self._fill_values[np.isnan(X)]
        X_q  = self._scaler.transform(X.reshape(1, -1))
        out  = {}
        for target, data in self._per_target.items():
            dists = euclidean_distances(X_q, data["training_scaled"])[0]
            nn    = float(dists.min())
            bm    = float(data["baseline_nn_dists"].mean())
            out[target] = {
                "nn_distance": nn, "nn_distance_normalized": nn / bm if bm > 0 else float("inf"),
                "baseline_mean": bm, "n_training": data["n_training"], "in_domain": nn <= bm,
            }
        return out

    @property
    def baseline_stats(self):
        rows = {}
        for target, data in self._per_target.items():
            d = data["baseline_nn_dists"]
            rows[target] = {"n_training": data["n_training"], "mean": float(d.mean()),
                            "std": float(d.std()), "min": float(d.min()),
                            "median": float(np.median(d)), "max": float(d.max())}
        return pd.DataFrame(rows).T


# ── CLI ───────────────────────────────────────────────────────────────────────

def _read_input_file(path: str) -> list:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts  = line.split(None, 1)
            smiles = parts[0]
            name   = parts[1].strip() if len(parts) > 1 else None
            entries.append((smiles, name))
    return entries


def _print_result(result: dict, name: str, args) -> None:
    width = 62
    print(f"\n{'='*width}")
    print(f"  {name}")
    print(f"{'='*width}")

    pred = result["prediction"]
    unc  = result["uncertainty"]

    # Merge into one display table
    display = pred[["prediction", "unit"]].copy()
    display["ensemble_std"] = unc["ensemble_std"]

    has_conformal = not unc["conformal_lower"].isna().all()
    if has_conformal:
        display[f"CI_lo ({int(args.coverage*100)}%)"] = unc["conformal_lower"]
        display[f"CI_hi ({int(args.coverage*100)}%)"] = unc["conformal_upper"]

    has_ad = not unc["nn_dist"].isna().all()
    if has_ad:
        display["nn_dist_norm"] = unc["nn_dist_norm"].round(2)
        display["in_domain"]    = unc["in_domain"]   # already a string column

    print(display.to_string(float_format=lambda x: f"{x:.4g}"))

    if not has_conformal:
        print("\n  [conformal intervals not available — "
              "run calibrate() and save_calibration() first]")
    if not has_ad:
        print("  [nn_dist not available — "
              "run fit_applicability_domain() or pass --training-csv]")

    if "top_features" in result:
        print(f"\n  Top {args.top_features} SHAP features per target:")
        for label, feat_series in result["top_features"].items():
            print(f"\n    {label}")
            for feat, val in feat_series.items():
                print(f"      {feat:40s}  {val:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Predict surfactant properties from a SMILES string.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python predict.py "CCCCCCCCCC(=O)O"
  python predict.py "CCCCCCCCCC(=O)O" --training-csv ../data/SurfPro-MD.csv --coverage 0.95
  python predict.py "CCCCCCCCCC(=O)O" --top-features 5
  python predict.py --input-file molecules.txt --training-csv ../data/SurfPro-MD.csv
        """,
    )
    parser.add_argument("smiles", nargs="?", default=None)
    parser.add_argument("--input-file", "-i", default=None, metavar="PATH")
    parser.add_argument("--model",
                        default=str(Path(__file__).resolve().parent.parent / "models" / "models.pkl"))
    parser.add_argument("--training-csv", default=None, metavar="PATH",
                        help="Training CSV; enables conformal calibration and AD distance.")
    parser.add_argument("--smiles-col",   default="SMILES", metavar="COL")
    parser.add_argument("--coverage",     type=float, default=0.90, metavar="FLOAT",
                        help="Nominal coverage for conformal intervals (default: 0.90).")
    parser.add_argument("--top-features", type=int,   default=0, metavar="N",
                        help="Number of top SHAP features to report (default: 0 = off).")
    args = parser.parse_args()

    if args.smiles is None and args.input_file is None:
        parser.error("Provide either a SMILES string or --input-file.")

    predictor = SurfProPredictor(args.model)

    if args.training_csv:
        if not predictor._conformal_cal:
            print("No calibration file found — running calibration (may take ~30 s)…")
            predictor.calibrate(args.training_csv, smiles_col=args.smiles_col)
            predictor.save_calibration()
        predictor.fit_applicability_domain(args.training_csv, smiles_col=args.smiles_col)

    molecules = (
        _read_input_file(args.input_file)
        if args.input_file
        else [(args.smiles, None)]
    )

    for idx, (smiles, name) in enumerate(molecules, start=1):
        label = name if name is not None else f"molecule {idx}"
        try:
            result = predictor.predict(
                smiles,
                coverage=args.coverage,
                top_features=args.top_features,
            )
            _print_result(result, label, args)
        except Exception as exc:
            print(f"\n{'='*62}\n  {label}  [ERROR]\n{'='*62}\n  {exc}")


if __name__ == "__main__":
    main()
