# TA-HGMAE: Self-Supervised Heterogeneous Graph Learning for Procurement Anomaly Detection

This repository holds the Jupyter notebook for our paper:


## What's in here

A single end-to-end notebook, `TA_HGMAE_Verification_Maslking_version_v5.ipynb`, that reproduces every table and figure in the paper. The notebook is split into two phases:

- **Cells 1–6** build the canonical pipeline: install PyTorch Geometric, download the Colombian OCDS archive, parse it into a heterogeneous Buyer–Tender graph with injected fraud, define TA-HGMAE and the three baselines, train, evaluate, and run the explainability module.
- **Cells 7–16** are the verification phase: seeded multi-run experiments, the Tabular AE diagnostic, the dataset-size sweep, the temporal / masking / heterogeneity ablations, and the held-out control.

I developed and ran the notebook on Google Colab with a T4 GPU. The canonical 6-cell pipeline finishes in about 5 minutes; the full verification phase takes a few hours end-to-end depending on which cells are run.

## Running it

The notebook is self-contained. Open it in Colab (or Jupyter locally with a CUDA-capable GPU) and run cells top-to-bottom. Cell 1 installs `torch-scatter`, `torch-sparse`, and `torch-geometric` directly from the PyG wheels. Cell 2 downloads the Colombian 2022 OCDS archive from `https://data.open-contracting.org/en/publication/61/download?name=2022.jsonl.gz` into `data/raw/`. Everything after that runs against the downloaded JSONL file.

If running on Colab disconnects, the verification cells (10, 11, 13, 14, 15, 16) write incrementally to CSV in `results/`, so re-running a cell resumes from the last completed (config, seed) pair rather than starting over.

## Cell-by-cell mapping to paper results

| Notebook cell | What it does | Paper artifact |
|---|---|---|
| 1 | Install PyTorch Geometric and its dependencies | — |
| 2 | Download the Colombian 2022 OCDS archive | §2.1 |
| 3 | Parse OCDS records, build the bipartite Buyer–Tender graph, inject burst + round-amount fraud at 6.54% | §2.1, §2.2, §2.3 |
| 4 | Define TA-HGMAE (HGT encoder + temporal features + masked AE) and the three baselines (Isolation Forest, Tabular AE, GraphMAE-Homo) | §2.4, §2.5 |
| 5 | Train all four models on the canonical scale, evaluate, print Precision@K, AUC-ROC, Detection Rate | — |
| 6 | Explainability module — edge-cosine similarity + per-feature residual z-scores | §2.4 (Eq. 8–9), §3.6 |
| 7 | Markdown header for the verification phase | — |
| 8 | Infrastructure — seeded RNG control, GPU memory cleanup, record-parsing cache, ablation variants | — |
| 9 | Tabular AE diagnostic — investigates the AUC ≈ 0.013 inversion across seeds | §3.3 |
| 10 | **Three-seed verification at 53,500 records** | **Tables 1 row, 2 row, 7** |
| 11 | **Size sweep across 1k / 5k / 10k / 20k / 50k records × 3 seeds** | **Tables 1, 2, 3** |
| 12 | Reads the sweep CSV and renders the scale-dependence chart | **Figure 3** |
| 13 | **Temporal feature ablation and masking-rate sweep at 53,500** | **Tables 5, 6** |
| 14 | **Heterogeneity ablation — hetero vs homogenised graph at 53,500** | **Table 4** |
| 15 | **Tabular AE held-out control — train on legitimate-only, score full set** | **§3.3 control** |
| 16 | GraphMAE re-mask decoding ablation (priority 4 follow-up) | — |

The `results/*.csv` files produced by cells 10–16 contain per-seed values, which is what the paper's mean ± std summaries are computed from.

## Key hyperparameters

These are set inside the relevant cells (mostly cell 4 for architecture, cell 5 for training, cell 8 for the verification infrastructure):

| Parameter | Value | Paper reference |
|---|---|---|
| HGT layers | L = 2 | §2.4 |
| HGT attention heads | H = 2 | §2.4 |
| Hidden dimension | d = 32 | §2.4 |
| Trainable parameters | ~25,000 | §2.4 |
| Temporal feature dimension | 19 | §2.4 (Eq. 3) |
| Default masking rate | μ = 0.15 | §2.4 |
| Scale-optimal masking rate found | μ = 0.75 | §3.2 (Table 6) |
| Optimizer | Adam, lr = 0.001 | §2.5 |
| Training epochs | 100 | §2.5 |
| Score threshold | 95th percentile of training reconstruction errors | §2.4 |
| Random seeds | 42, 123, 7 | §2.5 |
| Injection rate | 6.54% across all scales | §2.3 |

## Synthetic fraud injection

Cell 3 injects two fraud patterns drawn from documented OECD procurement-fraud typologies:

**Burst fraud** (kickback arrangements, end-of-fiscal-year collusion). At each scale the buyer count is fixed at 1% of the base dataset size (500 buyers at the 50,000-record scale). Each selected buyer issues 5 high-value tenders between 100M and 500M COP within a 100-day window in 2022.

**Round-amount fraud** (invoice fabrication, bid-rigging payoffs). 2% of the base size (1,000 tenders at 50,000 records), with values exactly at 100M, 200M, 500M, or 1B COP, attached to randomly selected buyers.

The combined injected fraction stays at 6.54% at every scale so the base rate is held constant while the total volume varies.

## Headline numbers

At the 50,000-record (53,500-tender) scale, the notebook reproduces:

| Metric | TA-HGMAE | Source |
|---|---|---|
| AUC-ROC | 0.979 ± 0.007 | Table 1 |
| Precision@100 | 0.437 | Table 2 |
| ΔP@100 vs GraphMAE-Homo | +0.294 | Table 3 |
| Heterogeneity ablation ΔP@50 | +0.040 | Table 4 |
| Masking rate optimum | μ = 0.75 | Table 6 |
| Flag-level precision (95th pct) | 77.1% | §3.5 |
| Explanation Hit Rate@1 | 0.849 | Table 8 |
| Seed-to-seed AUC variance vs GraphMAE-Homo | ~26× lower | §3.4 |

Per-seed numbers will vary slightly depending on the hardware and PyG/CUDA version. The qualitative patterns — TA-HGMAE crossing GraphMAE-Homo at 10,000 records, the scale-dependent ablation reversals on temporal and heterogeneity, the Tabular AE 2-of-3 seed inversion at every scale — are what the verification cells confirm and what the paper's conclusions rest on.

## Reproducibility notes

All experiments use seeds `[42, 123, 7]`. The scale-sweep cell varies both the fraud-injection randomness (which buyers, which values, which timestamps) and the model initialisation. The 53,500-record ablation cells (13, 14) vary only model initialisation against a fixed graph, so each metric aligns with the question the cell is built to answer.

Cell 8 provides `set_all_seeds(seed)` which deterministically seeds Python `random`, NumPy, and PyTorch (including CUDA). If a verification cell is re-run out of order, re-running cell 8 first restores deterministic state.

## Citation

If you use this code or build on the work, please cite:

```bibtex
@article{muchapirei2026tahgmae,
  title   = {A Scale-Dependent Evaluation of Self-Supervised Heterogeneous Graph Learning for Procurement Anomaly Detection},
  author  = {Muchapirei, Linval and Magadza, Tirivangani},
  journal = {Computer Science and Information Technologies},
  year    = {2026},
  note    = {Under review}
}
```

## Contact

**Linval Muchapirei** — h240755m@hit.ac.zw  
School of Information Sciences and Technology, Harare Institute of Technology, Belvedere, Harare, Zimbabwe.