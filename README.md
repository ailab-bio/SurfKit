# SurfKit

Surrogate models for predicting nine physicochemical surfactant properties directly from SMILES strings. Predictions are returned in milliseconds and come with ensemble uncertainty estimates, conformal prediction intervals, and optional SHAP feature attributions.

**Predicted properties**

| Property | Description | Units |
|---|---|---|
| pCMC | log critical micelle concentration | — |
| γ_CMC | air–water surface tension at the CMC | mN/m |
| Γ_max | maximum surface excess | μmol/m² |
| A_min | minimum area per molecule | nm² |
| pC20 | efficiency of surface tension reduction | — |
| D_MOL | molecular self-diffusion coefficient | ×10⁻⁹ m²/s |
| D_SOL | solvent self-diffusion coefficient | ×10⁻⁹ m²/s |
| γ (MD) | MD-derived surface tension | mN/m |
| η | dynamic viscosity | mPa·s |

---

## Quick start

### 1 — Build the container

All scripts run inside an [Apptainer](https://apptainer.org/) container that bundles Python, RDKit, XGBoost, and all other dependencies. Build it once from the definition file at the root of this repository:

```bash
apptainer build surfkit.sif surfkit.def
```

This takes a few minutes and produces `surfkit.sif`. Keep it next to the repository root.

### 2 — Train the models

Run training once to produce `models/models.pkl`. The script reads `data/SurfPro-MD.csv` and writes all output to `models/` by default.

```bash
apptainer exec surfkit.sif python scripts/training.py
```

For a quick smoke-test with only 10 Optuna trials per fold:

```bash
apptainer exec surfkit.sif python scripts/training.py --quick
```

Training with the default 30 trials per fold takes several hours on a multi-core workstation. Use `--cpus` to set the CPU budget explicitly:

```bash
apptainer exec surfkit.sif python scripts/training.py --cpus 16
```

Once `models/models.pkl` exists you are ready to predict.

---

## Predicting properties with `predict.py`

### Basic usage

Pass a SMILES string directly:

```bash
apptainer exec surfkit.sif python scripts/predict.py "CCCCCCCCCC(=O)O"
```

The script prints a table of point estimates for all nine targets.

### Conformal prediction intervals

Provide the training CSV to obtain calibrated prediction intervals with a guaranteed coverage level (default 90 %). The calibration step runs automatically on the first call and saves a file `models/conformal_calibration.pkl` so subsequent calls are instant.

```bash
apptainer exec surfkit.sif python scripts/predict.py "CCCCCCCCCC(=O)O" \
    --training-csv data/SurfPro-MD.csv
```

Change the coverage level with `--coverage`:

```bash
apptainer exec surfkit.sif python scripts/predict.py "CCCCCCCCCC(=O)O" \
    --training-csv data/SurfPro-MD.csv \
    --coverage 0.95
```

The output adds two columns per target — `lower` and `upper` — that define the prediction interval.

When `--training-csv` is provided the script also reports `nn_dist`: the Euclidean distance from the query molecule to the nearest training molecule in scaled RDKit feature space. Large values indicate the query is structurally dissimilar to the training set; predictions for such molecules should be interpreted with caution.

### SHAP feature attributions

Add `--top-features N` to see the N most influential RDKit descriptors for each target:

```bash
apptainer exec surfkit.sif python scripts/predict.py "CCCCCCCCCC(=O)O" \
    --top-features 5
```

### Batch predictions

Create a plain-text file with one SMILES per line (optionally a second column for names):

```
CCCCCCCCCC(=O)O  decanoic_acid
CCCCCCCCCCC(=O)O  undecanoic_acid
```

Then pass it with `--input-file`:

```bash
apptainer exec surfkit.sif python scripts/predict.py \
    --input-file molecules.txt \
    --training-csv data/SurfPro-MD.csv
```

### Using the Python API

```python
from scripts.predict import SurfProPredictor

p = SurfProPredictor()                                   # loads models/models.pkl
p.calibrate("data/SurfPro-MD.csv")
p.save_calibration()                                     # writes models/conformal_calibration.pkl
p.fit_applicability_domain("data/SurfPro-MD.csv")

result = p.predict("CCCCCCCCCC(=O)O", coverage=0.90, top_features=5)
print(result["prediction"])    # point estimates + units
print(result["uncertainty"])   # ensemble_std / conformal interval / nn_dist
print(result["top_features"])  # per-target top-N SHAP contributions
```

---

## Full analysis and plotting

### Running the ablation study

Pass `--full-analysis` to train all feature-representation variants in addition to the production model. This produces eleven additional pkl files:

```bash
apptainer exec surfkit.sif python scripts/training.py --full-analysis --cpus 32
```

Output files written to `models/`:

| File | Feature representation |
|---|---|
| `models.pkl` | Raw RDKit descriptors (217 features) |
| `models_reduced.pkl` | Two-stage PCA (~40–51 features) |
| `models_pca.pkl` | Global PCA at 95% variance (35–46 PCs) |
| `models_morgan_r2_n512.pkl` … `models_morgan_r4_n2048.pkl` | Morgan fingerprints, 9 radius/bit combinations |

### Plotting paper figures

Generate all publication figures from `models/models.pkl`:

```bash
apptainer exec surfkit.sif python scripts/plot_for_paper.py
```

Figures are written to the current working directory. To skip the MD validation panel (which requires additional reference data not included in the repository):

```bash
apptainer exec surfkit.sif python scripts/plot_for_paper.py --skip-md
```

Override the model or data paths if needed:

```bash
apptainer exec surfkit.sif python scripts/plot_for_paper.py \
    --model /path/to/models.pkl \
    --data /path/to/SurfPro-MD.csv
```

### Plotting appendix figures (SHAP analysis)

```bash
apptainer exec surfkit.sif python scripts/plot_for_appendix.py
```

SHAP values are computationally expensive. The script caches results in `models/shap_results_cache.pkl` and reuses them on subsequent runs. Delete the cache file to force recomputation.

---

## Running MD simulations from scratch

The `MD-simulations/` directory contains all scripts needed to generate a new dataset. The pipeline has five sequential stages. Each stage assumes the previous one has completed. All GROMACS steps use the provided Apptainer container; the parameterisation stage additionally requires AmberTools (loaded as a cluster module in the example scripts).

Before you begin, open each `Blank/` SLURM script and set the two path variables at the top:

```bash
PROJECT_ROOT=/path/to/this/repository     # root of the SurfKit repo
DATA_DIR=/path/to/your/data/directory     # where simulation directories will be created
```

Also update the `--account` and `-p` (partition) SBATCH directives to match your cluster.

### Stage 0 — Build the GROMACS container

```bash
cd MD-simulations
apptainer build gromacs.sif gromacs.def
```

### Stage 1 — Parameterise molecules

Each surfactant is parameterised independently using OpenBabel (3D coordinate generation), Antechamber with GAFF2 partial charges, and ACPYPE for conversion to GROMACS format.

Place one SMILES file per molecule in `MD-simulations/seedfiles/surfpro/` using the naming convention `surfpro_<N>.smi`, where `<N>` is a sequential integer. The file should contain a single SMILES string on the first line.

```bash
cd MD-simulations/parameterise
# edit start_runs.sh to set the molecule index range, then:
bash start_runs.sh
```

`start_runs.sh` instantiates `Blank/parameterise.slurm` for each molecule index and submits it to the SLURM queue. Each job writes its output to `$DATA_DIR/parameterise/surfpro_<N>/`. The parameterisation wall-time is typically under 2 hours per molecule.

### Stage 2 — Equilibration

This stage solvates each parameterised molecule in a TIP4P water box (150 surfactant molecules + up to 15 000 water molecules), adds counterions to neutralise charge, and runs a four-step protocol:

1. Energy minimisation (steepest descent, ≤5000 steps, force tolerance 1000 kJ/mol/nm)
2. Short NVT thermalisation (100 steps, v-rescale thermostat, 298.15 K)
3. NPT equilibration (Berendsen barostat, 1 bar)
4. Long NPT production run (21 sequential segments saved as `npt_00.gro` … `npt_20.gro`)

```bash
cd MD-simulations/equilibrations
# edit start_runs.sh to set id_start, id_end, nmol, watermax, then:
bash start_runs.sh
```

Each molecule is submitted as a separate GPU job. Wall-time budget: 72 hours. Output lands in `$DATA_DIR/equilibrations/surfpro_<N>_150/`.

### Stage 3 — Surface tension

Surface tension is computed from 21 independent NVT production runs, each starting from one of the long-NPT snapshots. The simulation box is extended in the z-direction to create an explicit air–water interface and surface tension is extracted from the pressure tensor.

```bash
cd MD-simulations/tension
# edit start_runs.sh to set the molecule index range, then:
bash start_runs.sh
```

Each job runs up to `MAX_PARALLEL=4` replicas concurrently on a single GPU node. Output lands in `$DATA_DIR/tension/surfpro_<N>_150/ST_00/` … `ST_20/`. The key output file per replica is `potential.xvg`.

### Stage 4 — Viscosity

Viscosity is computed using non-equilibrium MD (NEMD) with a periodic perturbation (shear rate 0.02 ps⁻¹). Each of the 21 long-NPT snapshots is used as a starting point for a separate replica, submitted as a SLURM array.

```bash
cd MD-simulations/viscosity
# edit start_runs.sh to set the molecule index range, then:
bash start_runs.sh
```

Output lands in `$DATA_DIR/viscosity/surfpro_<N>_150/ST_00/` … `ST_20/`. The key output file per replica is `visco.xvg`, from which `1/Viscosity` is extracted and inverted.

### Stage 5 — Bulk diffusion

Self-diffusion coefficients for both the surfactant (D_MOL) and water (D_SOL) are computed from the mean-squared displacement over a long NPT run that starts from the end of the equilibration.

```bash
cd MD-simulations/bulk-properties
# edit start_runs.sh to set the molecule index range, then:
bash start_runs.sh
```

Output lands in `$DATA_DIR/bulk-properties/surfpro_<N>_150/long-NPT/`. The key output files are `diffusion.xvg` (MSD slopes) and `potential.xvg` (energy/pressure averages).

### Stage 6 — Data extraction

Once all five simulation stages have finished, collect the results into a single CSV:

```bash
# edit the USER CONFIG section at the top of the script:
#   BASE_DIR      — root of $DATA_DIR
#   reference_csv — path to the experimental reference file (SurfPro.csv)
#   output_file   — destination CSV
#   MIN_INDEX / MAX_INDEX — molecule index range to collect

apptainer exec surfkit.sif python MD-simulations/extract_data_canonicalised.py
```

The script reads `diffusion.xvg`, `potential.xvg`, `visco.xvg`, and the equilibration SMILES files, standardises and canonicalises SMILES with RDKit, and joins MD-derived properties with experimental values from the reference CSV. The output file can then be used directly as input to `training.py`.

---

## Repository layout

```
SurfKit/
├── surfkit.def                   # Apptainer definition file (ML environment)
├── data/
│   ├── SurfPro-MD.csv            # full dataset (experimental + MD properties)
│   └── SurfPro.csv               # experimental-only reference data
├── models/                       # trained model pkl files (written by training.py)
├── scripts/
│   ├── training.py               # train surrogate models
│   ├── predict.py                # predict properties for new SMILES
│   ├── plot_for_paper.py         # generate main paper figures
│   └── plot_for_appendix.py      # generate appendix figures (SHAP)
└── MD-simulations/
    ├── gromacs.def               # Apptainer definition file (GROMACS)
    ├── seedfiles/                # per-molecule SMILES input files
    ├── parameterise/             # Stage 1: GAFF2 parameterisation
    ├── equilibrations/           # Stage 2: solvation + equilibration
    ├── tension/                  # Stage 3: surface tension NEMD
    ├── viscosity/                # Stage 4: viscosity NEMD
    ├── bulk-properties/          # Stage 5: diffusion from MSD
    └── extract_data_canonicalised.py  # Stage 6: collect outputs → CSV
```
