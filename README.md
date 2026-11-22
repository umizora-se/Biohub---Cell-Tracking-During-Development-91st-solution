# Biohub – Cell Tracking During Development: 91st place (silver) solution

This is my solution to the Kaggle competition [Biohub – Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development). It placed **91st of 3947 teams** (private LB 0.929, silver medal).

Detection, linking and most of the post-processing come from a public notebook and are used unchanged. On top of that, I added a **learned division model**: a 3D-CNN cell-state classifier combined with LightGBM and ExtraTrees. It recovers cell divisions the base pipeline misses and prunes forks it does not believe.

The write-up, with figures, is in [writeup/solution.md](writeup/solution.md) (English) and [writeup/solution_ja.md](writeup/solution_ja.md) (Japanese).

## Submissions

| Notebook | Description | Public | Private |
|---|---|---|---|
| [notebooks/final_et07](notebooks/final_et07) | **Final submission.** 3 optical-flow iterations in motion relinking; division classifier 0.3 × LightGBM + 0.7 × ExtraTrees; add a fork if the score ≥ 0.3, prune a fork if < 0.05 | 0.970 | 0.929 |
| [notebooks/second_et05](notebooks/second_et05) | Second final pick. Same, with LightGBM / ExtraTrees 50:50 | 0.970 | 0.928 |
| [notebooks/best_private_readd_mix](notebooks/best_private_readd_mix) | Best private score (not selected). Cell-state ILP + ILP forks re-added after motion relinking; the classifier averages the base-graph model and a model retrained on this graph | 0.966 | **0.935** |

The notebooks are exactly as submitted. `sub/build_notebooks.py` rebuilds them byte-for-byte from the public base notebook and the cells in `sub/cells_final/`.

## Folder structure

```
.
├── README.md
├── LICENSE                       Apache License 2.0
├── NOTICE                        attribution, modifications, third-party material
├── requirements.txt              pinned Python packages
├── notebooks/                    the three submitted Kaggle notebooks (+ kernel-metadata.json)
│   ├── final_et07/
│   ├── second_et05/
│   └── best_private_readd_mix/
├── weights/                      division-model weights used by the submissions (~3 MB)
│   ├── cnn_all.pt                  3D-CNN cell-state model (continue / parent / daughter)
│   ├── divclf3_s{0,1,2}.txt        LightGBM division classifier, 3 seeds
│   ├── divclf_et.npz               ExtraTrees division classifier as numpy arrays
│   └── divclf3r_s{0,1,2}.txt       LightGBM retrained on the readd graph (best_private_readd_mix only)
├── repro/                        step-by-step scripts
│   ├── 00_setup.sh                 environment (.venv) + official metric repository
│   ├── 01_download.sh              competition data + public weights of the base notebook
│   ├── 02_base.sh                  base notebook on 98 training movies
│   ├── 03_cnn.sh                   3D-CNN cell-state models
│   ├── 04_divclf.sh                division classifier (LightGBM, ExtraTrees)
│   ├── 05_readd.sh                 cell-state ILP + readd + retrained classifier
│   ├── 06_build.sh                 collect trained weights, build notebooks
│   └── 07_infer_local.sh           run a notebook locally
├── det/                          3D-CNN cell-state model
│   ├── mitosis_data.py             3-frame patches from ground-truth tracks
│   ├── mitosis_cnn.py              network, training, inference
│   ├── cand_patches.py             patches for division candidates
│   └── folds/fold{0-4}.txt         movie-level CV folds (98 movies)
├── pipe/                         local pipeline and division classifier
│   ├── make_exp.py, run_cells.py   run a Kaggle notebook locally (Kaggle input layout under pipe/exps/<name>/kaggle)
│   ├── score_csv.py                official metric on training ground truth
│   ├── eval98.txt                  the 98 training movies used for validation
│   ├── divclf.py, divclf2.py, divclf3.py   candidates and features (also embedded in the notebooks)
│   ├── div_cands.py, divclf_oof.py         ground-truth matching, candidate table, LightGBM parameters
│   ├── all_node_probs_lib.py       CNN probabilities for graph nodes
│   ├── et_export.py                ExtraTrees training and numpy export
│   ├── ilp_rerun2.py               cell-state ILP re-solve
│   └── divres/                     training scripts (extend2, cv3, final34, coords_probs, cv_readd, final_readd, v3oof_on_readd)
├── sub/                          notebook builder
│   ├── build_notebooks.py          base notebook + cells -> the three submission notebooks
│   ├── cells_final/                cells inserted into the base notebook
│   └── replaces/csilp_pp.py        post-processing patch for readd
├── nb/biohub-0-953-lb-original/  public base notebook (input to the builder)
└── writeup/                      write-up (EN / JA), figures, submission history
```

Created by the scripts and not tracked by git: `.venv/`, `ref_royerlab/` (metric), `data/` (competition data), `inputs/` (public weights), `pipe/exps/` (local notebook runs), `pipe/divres/` outputs (tables, OOF predictions, trained classifiers), `det/runs/` and patch files (CNN), and `build/` (collected weights and built notebooks).

## Requirements

- Linux with an NVIDIA GPU. Developed on Ubuntu (WSL2) with an RTX 5080 (16 GB), CUDA 12.8.
- [uv](https://docs.astral.sh/uv/) and Python 3.12.
- Kaggle API credentials (`~/.config/kaggle/kaggle.json`), with the competition rules accepted.
- About 90 GB of free disk space for the data, plus room for intermediate files.

## Quick start

```bash
git clone <this-repository-url> biohub-solution
cd biohub-solution
repro/00_setup.sh        # .venv with pinned packages (requirements.txt) + official metric repository
repro/01_download.sh     # competition data -> data/ (~83 GB), public base-notebook weights -> inputs/
```

Then follow either:

- **Inference only:** use the submitted weights in `weights/`. See [Inference](#inference).
- **Full reproduction:** retrain all weights, then run inference with them. See [Training](#training).

The competition data is **not** included in this repository; `01_download.sh` fetches it from Kaggle.

## Inference

### On Kaggle (as submitted)

1. **Upload the weights as a Kaggle dataset.** Use `weights/` (submitted weights) or `build/weights/` (your own, after [Training](#training)):
   ```bash
   .venv/bin/kaggle datasets init -p weights
   # edit weights/dataset-metadata.json: "id": "<kaggle-user>/biohub-division-weights", "title": "biohub-division-weights"
   .venv/bin/kaggle datasets create -p weights
   ```
   The cells find the weights by searching for `cnn_all.pt` under `/kaggle/input`, so any dataset name works.
2. **Build the notebooks** with your Kaggle user name and dataset:
   ```bash
   .venv/bin/python sub/build_notebooks.py --owner <kaggle-user> --weights-dataset <kaggle-user>/biohub-division-weights
   ```
   This writes `build/notebooks/{final_et07,second_et05,best_private_readd_mix}/`. Their code is identical to `notebooks/`; check this with `--check`.
3. **Push a notebook** (it runs on a T4 GPU without internet; the public datasets of the base pipeline are attached automatically from its `kernel-metadata.json`):
   ```bash
   .venv/bin/kaggle kernels push -p build/notebooks/final_et07 --accelerator NvidiaTeslaT4
   ```
4. **Submit:** on the notebook's Kaggle page, use *Submit to Competition*. The competition has ended, so this counts as a late submission.

### Locally

```bash
repro/07_infer_local.sh final_et07                    # submitted weights (weights/)
repro/07_infer_local.sh final_et07 build/weights      # weights you trained yourself
```

This runs the notebook end to end on training movies mounted as the test set (default: the 98 movies in `pipe/eval98.txt`). It writes `pipe/exps/infer_final_et07/kaggle/working/submission.csv` and scores it against the ground truth. An optional third argument gives a different movie list. For example, the four movies of the public test set (`44b6_0113de3b 44b6_0b24845f 6bba_05b6850b 6bba_05db0fb1`, copies of training movies) take about 5 minutes, and the default 98 movies about 100 minutes on an RTX 5080.

All three notebooks were run this way with the weights in `weights/`. The score on training movies is optimistic, because the division classifier was trained on them (final_et07 on the 98 movies: 0.9519, vs. 0.9231 for the base pipeline). Use it to check that the pipeline works, not as a validation score.

## Training

After the [Quick start](#quick-start), run the steps in order:

```bash
repro/02_base.sh     # base notebook on 98 training movies
repro/03_cnn.sh      # 3D-CNN cell-state models
repro/04_divclf.sh   # division classifier (LightGBM, ExtraTrees)
repro/05_readd.sh    # cell-state ILP + readd + retrained classifier
repro/06_build.sh    # collect weights into build/weights/, build notebooks into build/notebooks/
```

Steps can be re-run; expensive finished parts (base run, CNN models, patches, ILP re-solve) are skipped. The steps write into the working folders (`det/`, `pipe/exps/`, `pipe/divres/`), which are not tracked by git. Each step prints its key metric; the table lists the values obtained during the competition.

| Step | What it does | Main outputs | Reference |
|---|---|---|---|
| `repro/02_base.sh` | Runs the public base notebook locally on 98 training movies; also writes the ILP inputs (edge cache) | `pipe/exps/r_base98/` | score 0.9231 |
| `repro/03_cnn.sh` | Extracts 3-frame patches from the ground-truth tracks of all 199 movies; trains 5 fold models (for out-of-fold features) and 1 model on all data | `det/runs/cnn_f{0-4}.pt`, `det/runs/cnn_all.pt` | 24,056 patches |
| `repro/04_divclf.sh` | Builds division candidates with geometry, intensity, temporal and CNN features; grouped 5-fold CV; trains the final LightGBM (3 seeds) and ExtraTrees (exported to numpy) | `pipe/divres/divclf3_s{0,1,2}.txt`, `pipe/divres/divclf_et.npz` | candidate AUC 0.965; end-to-end OOF 0.9396 |
| `repro/05_readd.sh` | Cell-state ILP (CNN-adjusted costs) on the edge cache, post-processing with ILP forks re-added, classifier retrained on this graph | `pipe/divres/divclf3r_s{0,1,2}.txt` | readd 0.9411; readd + mix OOF 0.9483 |
| `repro/06_build.sh` | Collects the trained weights and builds the three notebooks | `build/weights/`, `build/notebooks/` | – |

All steps need the GPU except `06_build.sh`. On an RTX 5080 the steps took about 95 min (`02`), 15 min (`03`), 15 min (`04`) and 40–60 min (`05`).

**Verification.** All steps were run from a clean copy of this repository:

- **Everything except the CNN training is deterministic.**
  - Step `02` reproduces the reference score exactly (0.9231).
  - With the same CNN models, steps `04`–`05` reproduce the reference values exactly (0.965 / 0.9396 / 0.9411 / 0.9483).
  - All LightGBM and ExtraTrees files they produce are byte-identical to `weights/`.
- **CNN training (`03`) is not bit-reproducible** (non-deterministic GPU training). The retrained models perform comparably, but the downstream values vary slightly. In the verification run they were candidate AUC 0.955, end-to-end OOF 0.9406, readd 0.9406 and readd + mix 0.9471.

After training, run inference with `repro/07_infer_local.sh <notebook> build/weights`. To submit, upload `build/weights/` as the Kaggle dataset instead of `weights/`.

## Credits and license

- The base pipeline is the public notebook [`anvithpothula/biohub-0-953-lb-original`](https://www.kaggle.com/code/anvithpothula/biohub-0-953-lb-original), released on Kaggle under the Apache 2.0 open source license and copied unchanged into `nb/`. It builds on [Pilkwang Kim](https://www.kaggle.com/pilkwang)'s public work and weights. Most of each submitted notebook (detection, edge scoring, ILP, post-processing) is their code; the modifications are listed in [NOTICE](NOTICE).
- The official metric comes from the organizers' repository [royerlab/kaggle-cell-tracking-competition](https://github.com/royerlab/kaggle-cell-tracking-competition) (BSD-3-Clause), which `00_setup.sh` clones.
- The code and weights in this repository are released under the [Apache License 2.0](LICENSE); see [NOTICE](NOTICE) for third-party material. The weights were trained on the competition data (CC0: Public Domain).
