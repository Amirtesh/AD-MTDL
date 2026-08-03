# AD-MTDL: Multi-Target Drug Design Pipeline for Alzheimer's Disease

A reinforcement learning pipeline for multi-target molecular generation targeting four Alzheimer's disease-relevant enzymes: AChE, BuChE, BACE1, and MAO-B.

The core technical contribution is an HTTP bridge that resolves a fundamental dependency incompatibility between ChemProp v2.x (required for the trained classifier) and REINVENT4 4.x (which requires ChemProp v1.x). A Flask server running in a separate conda environment loads the ChemProp v2 checkpoint and serves scoring via POST requests to localhost:8765, while a REINVENT4 contrib plugin calls it per batch.

---

## Overview

The pipeline has three stages:

1. **Classification** — A multi-task D-MPNN trained on ChEMBL 36 data predicts activity probability for all four targets simultaneously.
2. **Generation** — REINVENT4 uses reinforcement learning (Mol2Mol, DAP strategy) to optimize molecules toward high classifier scores, subject to drug-likeness constraints (QED, SA score, molecular weight, custom structural alerts).
3. **Analysis** — Generated compounds are evaluated by classifier re-scoring and chemical space analysis (UMAP). Molecular docking (GNINA) was performed during the project but is archived as supplementary material and is not part of the reported pipeline scope.

---

## Repository Structure

```
.
├── AD_MTDL_best.ckpt          # Trained ChemProp v2.2.3 multi-task classifier
├── AD_MTDL_metadata.json      # Model metadata: targets, thresholds, training config
├── campaign_seeds.csv         # Seed SMILES actually used for the five reported campaigns
├── optimize.py                # Single campaign entry point
├── predict.py                 # Standalone inference on SMILES/CSV input
├── run_campaigns.sh           # Orchestrator for all campaigns
├── scoring_server.py          # Flask oracle server (ad_mtdl env)
├── select_seeds.py            # Seed selection from moderate activity zone
├── env_reinvent4.yml          # Conda environment: REINVENT4 + ChemProp 1.5.2
├── env_ad_mtdl.yml            # Conda environment: ChemProp 2.2.3 + Flask
├── env_analysis.yml           # Conda environment: UMAP + plotting (analysis only)
├── priors/
│   ├── mol2mol_medium_similarity.prior   # Mol2Mol prior (Zenodo: 10.5281/zenodo.15641297)
│   └── reinvent.prior                    # Default REINVENT prior
└── reinvent_plugins/
    └── components/
        └── comp_ad_mtdl.py    # REINVENT4 contrib plugin — calls scoring server per batch
```

---

## Targets and Model

| Target | ChEMBL ID | Role in AD | Threshold | Training n (Active + Inactive) |
|--------|-----------|------------|-----------|-------------------------------|
| AChE   | CHEMBL220 | Cholinergic deficit | 0.59 | 2,700 |
| BuChE  | CHEMBL1914 | ACh hydrolysis | 0.68 | 1,700 |
| BACE1  | CHEMBL4822 | Amyloid-beta production | 0.60 | 5,489 |
| MAO-B  | CHEMBL2039 | Neuroinflammation | 0.51 | 2,113 |

Activity labels: Active = pChEMBL >= 7.0, Inactive = pChEMBL < 5.0, Moderate (5.0-7.0) excluded from training and used as generation seeds. Model output probabilities are bounded to approximately [0.5, 0.731] — confirmed directly from raw pre-sigmoid model outputs, not inferred from probability range alone; thresholds are optimized within this range by MCC maximization on the validation set.

---

## Installation

Two separate conda environments are required. They must not be merged — the dependency conflict between ChemProp v1 and v2 is the reason this bridge exists.

**Environment 1: ad_mtdl (ChemProp v2, Flask server)**

```bash
conda env create -f env_ad_mtdl.yml
conda activate ad_mtdl
```

**Environment 2: reinvent4 (REINVENT4, ChemProp v1)**

```bash
conda env create -f env_reinvent4.yml
conda activate reinvent4
```

Register the contrib plugin with REINVENT4:

```bash
conda activate reinvent4
# Find the REINVENT4 contrib directory
python -c "import reinvent; import os; print(os.path.join(os.path.dirname(reinvent.__file__), 'scoring', 'components'))"
# Copy the plugin to that path
cp reinvent_plugins/components/comp_ad_mtdl.py <path_from_above>/
```

**Environment 3: ad_mtdl_analysis (UMAP, plots — optional)**

```bash
conda env create -f env_analysis.yml
conda activate ad_mtdl_analysis
```

**Prior file**

The Mol2Mol medium similarity prior is not included in this repository due to file size. Download from Zenodo:

```bash
# https://zenodo.org/records/15641297
wget -O priors/mol2mol_medium_similarity.prior https://zenodo.org/records/15641297/files/mol2mol_medium_similarity.prior?download=1
```

---

## Model Training Provenance

The `Chemprop-model-training/` directory contains the original scripts, curated per-target datasets, and notebook used to train the classifier and produce `AD_MTDL_best.ckpt`, `AD_MTDL_metadata.json`, and `AD_MTDL_combined_chembl36.csv`. These three files also appear at the repository root, where they are used directly by `predict.py`, `scoring_server.py`, and the generation pipeline; the copies in both locations are identical. `Chemprop-model-training/` is retained separately as the training-time record for reproducibility — it documents how the classifier was built, including ChEMBL curation (`retrieve.py`), external validation against clinical/reference inhibitors (`evaluate.py`, `AD_eval_for_prediction.csv`, `external_validation_corrected_summary.csv`), and the training notebook itself (`chemprop-alzheimer-targets.ipynb`).

---

## Usage

### 1. Standalone inference on new compounds

Run predictions on any SMILES file or CSV without starting the scoring server:

```bash
conda activate ad_mtdl

python predict.py \
    --input compounds.csv \
    --smiles_col smiles \
    --ckpt AD_MTDL_best.ckpt \
    --output predictions.csv
```

Output columns: `smiles`, `AChE_prob`, `AChE_pred`, `BuChE_prob`, `BuChE_pred`, `BACE1_prob`, `BACE1_pred`, `MAO_B_prob`, `MAO_B_pred`. Predicted class uses the MCC-optimized thresholds from `AD_MTDL_metadata.json`.

### 2. Start the scoring server

The server must be running in the `ad_mtdl` environment before launching any generation campaign. Open a terminal and leave it running:

```bash
conda activate ad_mtdl
python scoring_server.py --ckpt AD_MTDL_best.ckpt --port 8765
```

The model is loaded once at startup. All four classifier heads are served via POST to `http://localhost:8765/score`. Test with:

```bash
curl -X POST http://localhost:8765/score \
    -H "Content-Type: application/json" \
    -d '{"smiles": ["CC(=O)Oc1ccccc1C(=O)O"]}'
```

Or check server health directly:

```bash
curl http://localhost:8765/health
```

### 3. Run a single generation campaign

In a second terminal (reinvent4 environment), with the scoring server already running:

```bash
conda activate reinvent4

python optimize.py \
    --target AChE \
    --seed "<seed SMILES — see campaign_seeds.csv>" \
    --steps 10000 \
    --top_k 1000 \
    --output campaigns/AChE/
```

Supported targets: `AChE`, `BuChE`, `BACE1`, `MAO_B`. Dual-target campaigns take two targets — consult `--help` for the exact syntax (list vs. space-separated vs. repeated flag).

### 4. Run all campaigns

```bash
conda activate reinvent4

# Start scoring server first (in a separate terminal, ad_mtdl env)
bash run_campaigns.sh
```

`run_campaigns.sh` is idempotent — re-running skips completed campaigns. Output for each campaign: `results_1.csv` (all generated SMILES with per-step scores) and `top_compounds.csv` (top-k by composite score).

### 5. Seed selection

`select_seeds.py` has no command-line interface — it is a standalone script with hardcoded input/output paths (`CSV_PATH`, `OUT_PATH` constants at the top of the file). To regenerate seeds:

```bash
conda activate ad_mtdl

# Edit CSV_PATH in select_seeds.py if your input file differs from
# AD_MTDL_combined_chembl36.csv, then:
python select_seeds.py
```

This processes all four single-target campaigns and the dual AChE+BACE1 campaign in one run, selecting seeds from the moderate-activity zone (pChEMBL 5.0–7.0) by MaxMin diversity, and writes all five seeds to `campaign_seeds.csv`.

**Note on the seeds actually reported in the paper:** the AChE single-target seed and the dual AChE+BACE1 seed used in the reported campaigns were selected manually, following exploratory iteration, and do **not** correspond to this script's automated output for those two campaigns. Only the BuChE, BACE1, and MAO-B single-target seeds were produced by this script as-is. See Methods Section 2.6 and `campaign_seeds.csv` for the seeds actually used in each reported campaign; do not assume re-running this script reproduces the AChE or dual-campaign seed.

---

## Scoring Function

The composite score used during RL generation is a geometric mean of the following components:

| Component | Transform | Notes |
|-----------|-----------|-------|
| Target oracle | Sigmoid, centered at threshold | high = threshold + 0.20, low = threshold − 0.20, k = 0.5 |
| QED | None | Raw QED value |
| SA score | Reverse sigmoid | high = 4.5, low = 2.0, k = 0.5 |
| Molecular weight | Double sigmoid | 200–550 Da window |
| Structural alerts | Custom SMARTS-based component | Scored as a weighted component within the geometric mean, not a hard filter |

For dual-target campaigns, both oracle components are included in the geometric mean with equal weight. A scaffold-based diversity filter (`IdenticalMurckoScaffold`, bucket size 25, minimum similarity 0.4) additionally penalizes over-sampling of any single scaffold during generation.

---

## Architecture: The HTTP Bridge

The core engineering problem: REINVENT4 4.x imports ChemProp internally and requires v1.x. ChemProp v2.x has an incompatible API and cannot be installed in the same environment.

Solution: two isolated environments communicate over localhost.

```
[reinvent4 env]                          [ad_mtdl env]
REINVENT4 RL loop
  -> comp_ad_mtdl.py (contrib plugin)
       -> HTTP POST localhost:8765/score  -> scoring_server.py (Flask)
                                               -> ChemProp v2.2.3
                                               -> AD_MTDL_best.ckpt
       <- JSON scores per batch          <-
  -> geometric mean aggregation
  -> DAP policy gradient update
```

`comp_ad_mtdl.py` sends batches of SMILES as JSON and receives per-target probabilities in a single HTTP call per RL step, regardless of how many target endpoints are configured. Invalid SMILES return 0.5 (sigmoid midpoint, model uncertainty default). The server holds the model in memory for the duration of the campaign — no per-batch reload. Connection failures, timeouts, and unexpected server errors during a run are logged and fall back to default scores (0.5) for the affected batch rather than terminating the RL run.

---

## Reproducing the Paper Campaigns

The five campaigns reported in the paper used the seeds in `campaign_seeds.csv`. The `campaign` column identifies which seed belongs to which run. **Three of the five seeds (BuChE, BACE1, MAO-B) were produced by `select_seeds.py`; the AChE and dual AChE+BACE1 seeds were selected manually — see the note in Section 5 above.**

```bash
# Verify seed file
head -6 campaign_seeds.csv

# Run all five campaigns in sequence
bash run_campaigns.sh --all

# Or run a specific campaign
bash run_campaigns.sh --only AChE
```

All random seeds are logged in the campaign output JSON. REINVENT4 generation is stochastic; exact SMILES will differ across runs, but statistical properties (score distributions, multi-target hit rates) should be consistent.

---

## Citation

If you use this pipeline, please cite:

```
[citation to be added upon publication]
```

Please also cite REINVENT4, ChemProp, and RDKit as appropriate.

---

## License

MIT License. See LICENSE file.

The trained checkpoint (AD_MTDL_best.ckpt) is derived from ChEMBL data (CC BY-SA 3.0) and the Mol2Mol prior from Zenodo (see prior file license).
