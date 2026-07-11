#!/usr/bin/env python3
import os
import glob
from io import StringIO
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

# === USER CONFIG ===
BASE_DIR = "/mimer/NOBACKUP/groups/naiss2023-6-290/richard/data/"
equi_dirname = os.path.join(BASE_DIR, "equilibrations")
bulk_dirname = os.path.join(BASE_DIR, "bulk-properties")
visc_dirname = os.path.join(BASE_DIR, "viscosity")
tens_dirname = os.path.join(BASE_DIR, "tension")

reference_csv = "../reference/surfpro_full.csv"
output_file = "../data/SurFPro-MD.csv"

MIN_INDEX = 1
MAX_INDEX = 1623 
EQ_STEPS = 100





# ---------------------------
# Utility: Standardize + Canonicalize SMILES
# ---------------------------

def standardize_smiles(s):
    """Standardize a SMILES string:
    - keep only the largest fragment
    - cleanup + neutralize
    - canonicalize
    """
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None
    try:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            print("MolFromSmiles failed for:", repr(s))
            return None

        # Step 1: keep largest fragment
        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) > 1:
            mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())

        # Step 2: cleanup
        mol = rdMolStandardize.Cleanup(mol)

        # Step 3: select parent fragment (modern RDKit)
        mol = rdMolStandardize.FragmentParent(mol, skipStandardize=True)

        # Step 4: uncharge
        mol = rdMolStandardize.Uncharger().uncharge(mol)

        # Step 5: canonicalize
        smiles = Chem.MolToSmiles(mol, canonical=True)
        return smiles

    except Exception as e:
        print("Exception in standardization:", e)
        return None



# ---------------------------
# Utility: read_xvg
# ---------------------------
def read_xvg(fname, quantity, eq_steps=EQ_STEPS):
    """Parse an .xvg file and compute mean/std for a given column label."""
    if not os.path.isfile(fname):
        print(f"problem in {fname}")
        return None, None
    with open(fname, "r") as fid:
        ll = fid.readlines()
    if len(ll) == 0:
        print(f"problem in {fname}")
        return None, None

    last_data_line = next((l for l in reversed(ll) if not l.startswith(("@", "#")) and l.strip()), None)
    if last_data_line is None:
        print(f"problem in {fname}")
        return None, None

    try:
        M = len(last_data_line.split())
    except Exception:
        print(f"problem in {fname}")
        return None, None

    labels = [""] * M
    labels[0] = "Time (ps)"
    ALL = []
    icol = -1
    i = 1
    for l in ll:
        if l.startswith("@ s") and "legend" in l:
            parts = l.split()
            label = ' '.join(parts[3:])[1:-1]
            if i < len(labels):
                labels[i] = label
                if label.strip() == quantity.strip():
                    icol = i
            i += 1
        if l.startswith("#") or l.startswith("@"):
            continue
        try:
            ALL.append([float(x) for x in l.split()])
        except Exception:
            print(f"problem in {fname}")
            continue

    if icol == -1 or len(ALL) == 0:
        return None, None

    ALL = np.asarray(ALL).T
    if icol >= ALL.shape[0]:
        return None, None

    data = ALL[icol, eq_steps:] if ALL.shape[1] > eq_steps else ALL[icol, :]
    return float(np.mean(data)), float(np.std(data))


# ---------------------------
# Helper: find surfpro dir
# ---------------------------
def find_surfpro_dir(parent_dir, index):
    for cand in [f"surfpro_{index}_150", f"surfpro_{index}_150_15000"]:
        p = os.path.join(parent_dir, cand)
        if os.path.isdir(p):
            return p
    return None


# ---------------------------
# Load reference CSV and standardize SMILES
# ---------------------------
ref_df = pd.read_csv(reference_csv)
if "index" not in ref_df.columns:
    ref_df = ref_df.reset_index().rename(columns={"index": "index"})
if "SMILES" not in ref_df.columns or "pCMC" not in ref_df.columns:
    raise ValueError("Reference CSV must contain 'SMILES' and 'pCMC' columns.")

ref_df["SMILES"] = ref_df["SMILES"].astype(str).str.strip()
ref_df["SMILES_std"] = ref_df["SMILES"].apply(standardize_smiles)  # ✅ standardized reference SMILES

results = []

# ---------------------------
# Main loop
# ---------------------------
failed_indices=[]
for idx in range(MIN_INDEX, MAX_INDEX + 1):
    try:
        equi_surf = find_surfpro_dir(equi_dirname, idx)
        bulk_surf = find_surfpro_dir(bulk_dirname, idx)
        tens_surf = find_surfpro_dir(tens_dirname, idx)
        visc_surf = find_surfpro_dir(visc_dirname, idx)

        #if not (bulk_surf and visc_surf and tens_surf):
        if not (bulk_surf and tens_surf):
            print(f"one of the directories missing for index {idx}")
            failed_indices.append(idx)
            continue

        # --- BULK ---
        longnpt_dir = os.path.join(bulk_surf, "long-NPT")
        diff_file = os.path.join(longnpt_dir, "diffusion.xvg")
        pot_file = os.path.join(longnpt_dir, "potential.xvg")
        if not (os.path.exists(diff_file) and os.path.exists(pot_file)):
            failed_indices.append(idx)
            continue

        D_MOL, D_SOL = None, None
        with open(diff_file, "r") as f:
            for line in f:
                if "MOL" in line:
                    try:
                        D_MOL = float(line.split("=")[1].split()[0])
                    except Exception:
                        pass
                elif "SOL" in line:
                    try:
                        D_SOL = float(line.split("=")[1].split()[0])
                    except Exception:
                        pass

        with open(pot_file, "r") as f:
            lines = f.readlines()
        headers = [line.split("legend")[-1].strip().strip('"') for line in lines if line.startswith("@ s")]
        numeric_lines = [l for l in lines if not l.startswith(("@", "#"))]
        if not numeric_lines:
            failed_indices.append(idx)
            continue

        df_pot = pd.read_csv(StringIO("".join(numeric_lines)), sep=r"\s+", header=None)
        colnames = ["Time"] + headers if len(df_pot.columns) == len(headers) + 1 else headers
        df_pot.columns = colnames
        averages = df_pot.mean().to_dict()
        fluctuations = df_pot.std().to_dict()

        # --- SMILES ---
        smiles_file = os.path.join(equi_surf, "setup", f"surfpro_{idx}.smi")
        raw_smiles = None
        if os.path.exists(smiles_file):
            with open(smiles_file, "r") as f:
                first_line = f.readline().strip()
                if first_line:
                    raw_smiles = first_line.split()[0].strip()  # ✅ only first token

        local_smiles = standardize_smiles(raw_smiles) if raw_smiles else None  # ✅ standardized local SMILES

        # --- Match pCMC and surf_type ---
        pCMC = None
        surf_type = None
        
        if local_smiles is not None:
            match = ref_df[ref_df["SMILES_std"] == local_smiles]
            if not match.empty:
                pCMC = match["pCMC"].iloc[0]
                if "type" in match.columns:
                    surf_type = match["type"].iloc[0]
                else:
                    surf_type="other"


        # --- VISCOSITY ---
        visc_files = sorted(glob.glob(os.path.join(visc_surf, "ST_??", "visco.xvg")))
        if len(visc_files) == 0:
            continue
        inv_visc = [read_xvg(f, "1/Viscosity")[0] for f in visc_files]
        inv_visc = [v for v in inv_visc if v is not None]
        if len(inv_visc) == 0:
            continue
        mean_inv, std_inv = np.mean(inv_visc), np.std(inv_visc)
        viscosity = 1.0 / mean_inv if mean_inv != 0 else None
        viscosity_std = std_inv / (mean_inv**2) if mean_inv != 0 else None

        # --- TENSION ---
        tens_files = sorted(glob.glob(os.path.join(tens_surf, "ST_??", "potential.xvg")))
        if len(tens_files) == 0:
            failed_indices.append(idx)
            continue
        tens_avgs = []
        for tf in tens_files:
            avg, std = read_xvg(tf, "#Surf*SurfTen")
            if avg is not None:
                tens_avgs.append(avg / 2)
        if len(tens_avgs) == 0:
            failed_indices.append(idx)
            continue
        tension_avg, tension_std = np.mean(tens_avgs), np.std(tens_avgs)

        # --- Assemble row ---
        row = {
            "index": idx,
            "SMILES_raw": raw_smiles,
            "SMILES": local_smiles,
            "surf_type": surf_type,
            "pCMC": pCMC,
            "D_MOL": D_MOL,
            "D_SOL": D_SOL,
            "viscosity": viscosity,
            "viscosity_std": viscosity_std,
            "surface_tension_avg": tension_avg,
            "surface_tension_std": tension_std,
        }
        for k, v in averages.items():
            row[f"{k}_avg"] = v
        for k, v in fluctuations.items():
            row[f"{k}_std"] = v

        results.append(row)

    except Exception as e:
        print("\n Failed indices:")
        print(failed_indices)
        continue


# ---------------------------
# Save and diagnostics
# ---------------------------
if len(results) == 0:
    print("❌ No complete entries found.")
else:
    df_out = pd.DataFrame(results)
    df_out = df_out.sort_values(by="index").reset_index(drop=True)
    df_out.to_csv(output_file, index=False)

    n_total = len(df_out)
    n_with_pCMC = df_out["pCMC"].notna().sum()
    print(f"✅ Done. Wrote {n_total} rows to {output_file}")
    print(f"ℹ️  {n_with_pCMC}/{n_total} rows ({100*n_with_pCMC/n_total:.1f}%) have pCMC values.")

    if n_with_pCMC < n_total:
        missing = df_out[df_out["pCMC"].isna()]
        print("\n⚠️ Example SMILES without pCMC match:")
        print(missing["SMILES"].head(10).to_list())

