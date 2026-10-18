# Autoencoder for Sensor Data Analysis

**A reconstruction-based investigation of structural sensor signals from the Hell Bridge Test Arena (HBTA).**

This repository contains Python scripts and selected numerical results from a Master's research project investigating whether an autoencoder trained on an undamaged acceleration recording can identify differences in other HBTA recordings. It compares eight damage-state configurations (DS1–DS8), tests generalisation to additional undamaged recordings, and examines feature-wise versus global standardisation.

> **Research limitation:** A high reconstruction error is an anomaly relative to the model's training data, **not proof of structural damage**. This is an experimental analysis pipeline, not a validated real-time bridge-monitoring or safety system.

## 1. Project objectives

- Inspect HBTA HDF5 datasets, sensor channels, configuration metadata and data quality.
- Extract and preprocess a selected acceleration signal into fixed-length windows.
- Train a fully connected autoencoder using undamaged reference data.
- Calculate window-level reconstruction mean squared error (MSE) and a validation-derived threshold.
- Evaluate DS1–DS8 and additional undamaged recordings.
- Compare feature-wise and global standardisation and consolidate experimental results.

## 2. Data and experimental scope

| Item | Configuration |
|---|---|
| Dataset | Hell Bridge Test Arena (HBTA) |
| Local dataset path | `data/data_100Hz.h5` |
| Sampling frequency | 100 Hz |
| Selected channel | `acceleration/AG01/y` |
| Undamaged reference | `MVS_P2_UDS_SM_Y_01` |
| Main damage states | `MVS_P2_DS1_SM_Y_01` through `MVS_P2_DS8_SM_Y_01` |
| Window size | 100 samples (1 second), non-overlapping |
| Reference split | Chronological 80% training / 20% validation |
| Baseline scaling | Feature-wise standardisation using training statistics |
| Comparison scaling | Global standardisation |
| Reconstruction metric | Window-level MSE |
| Detection threshold | 95th percentile of reference validation-window errors |

The selected reference signal contains 62,386 samples, producing 623 complete non-overlapping windows: 498 training and 125 validation windows. The evaluation is restricted to one acceleration channel; it does not perform multisensor fusion, damage localisation, severity estimation, or real-time deployment.

### Dataset availability

**The raw HDF5 dataset is not included in this GitHub repository.** The local file `data/data_100Hz.h5` is approximately 2.3 GB and is excluded through `.gitignore`. Obtain the dataset from an authorised HBTA source and place it at the path above, subject to its access and redistribution conditions. The repository does not itself provide a download link or licence grant.

## 3. Model and training

The model is a fully connected autoencoder with the following layer widths:

```text
Input (100) → 64 → 32 → 64 → Output (100)
```

The model is trained to reconstruct undamaged windows using mean squared error. The reported experimental configuration uses the Adam optimiser (learning rate `0.001`), batch size `32`, a maximum of `100` epochs, early-stopping patience `10`, and random seed `42`. Refer to `src/autoencoder.py` and `src/train_autoencoder.py` for implementation details, including activation functions and checkpoint handling. The global-standardisation experiment has its own preprocessing and training scripts.

## 4. Repository layout

```text
.
├── .gitignore
├── README.md
├── requirements.txt
├── requirements_report_snapshot.txt
├── src/
│   ├── autoencoder.py
│   ├── explore_h5.py
│   ├── dataset_inventory.py
│   ├── data_quality.py
│   ├── experiment_selection.py
│   ├── preprocess_signal.py
│   ├── preprocess_signal_global.py
│   ├── train_autoencoder.py
│   ├── train_autoencoder_global.py
│   ├── reconstruct_signal.py
│   ├── analyse_reconstruction_errors.py
│   ├── evaluate_damage_configs.py
│   ├── evaluate_damage_configs_global.py
│   ├── evaluate_uds_generalization.py
│   ├── evaluate_all_uds_configurations.py
│   ├── evaluate_detection_performance.py
│   └── consolidate_final_results.py
├── data/
│   └── data_100Hz.h5                  # local only; ignored by Git
├── models/                            # local checkpoints; ignored by Git
└── results/                           # selected CSV summaries may be tracked
```

This is an abbreviated layout; the `src/` directory also includes inventory, inspection, plotting, statistics, verification and configuration-summary scripts. Generated directories and outputs depend on which stages have been run.

## 5. Installation

### Prerequisites

- Python with a version compatible with the pinned packages and your operating system.
- Sufficient local storage and memory for the HBTA HDF5 dataset and intermediate arrays.
- Access to the raw HBTA dataset for end-to-end reproduction.

The original development environment was macOS with a project-specific Python virtual environment. Exact package versions and platform compatibility should be verified before reinstalling; some pinned versions may not have wheels for every Python or macOS release.

### Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, the activation command differs (for example, `.venv\\Scripts\\activate` in PowerShell).

### Dependency files

- **`requirements.txt`**: intended installation dependencies for the project; should include PyTorch, NumPy, pandas, scikit-learn, SciPy, h5py, Matplotlib and Seaborn, using versions tested with the project.
- **`requirements_report_snapshot.txt`**: full recorded development-environment package snapshot, including transitive dependencies and notebook tools. It is retained for auditability rather than guaranteed cross-platform installation.

The recorded snapshot lists `torch==2.14.1`, `numpy==2.5.3`, `pandas==3.0.6`, `scikit-learn==1.9.1`, `scipy==1.18.1`, `h5py==3.16.0`, `matplotlib==3.11.2`, and `seaborn==0.13.2`. These are **recorded environment values**, not independently verified installation instructions for every machine.

Check the active environment with:

```bash
python --version
python -m pip check
python -m pip show torch scikit-learn numpy h5py
```

## 6. Reproducing the experiments

The repository separates data inspection, preprocessing, training, evaluation and consolidation into scripts. A typical workflow is:

| Stage | Relevant script(s) |
|---|---|
| Inspect dataset | `src/explore_h5.py`, `src/dataset_inventory.py`, `src/data_quality.py` |
| Inspect/select configurations | `src/experiment_selection.py`, `src/inspect_uds_configurations.py` |
| Prepare reference windows | `src/preprocess_signal.py` |
| Train baseline autoencoder | `src/train_autoencoder.py` |
| Reconstruct and analyse validation windows | `src/reconstruct_signal.py`, `src/analyse_reconstruction_errors.py` |
| Evaluate damage states | `src/evaluate_damage_configs.py`, `src/evaluate_detection_performance.py` |
| Evaluate other undamaged recordings | `src/evaluate_uds_generalization.py`, `src/evaluate_all_uds_configurations.py` |
| Run global-standardisation comparison | `src/preprocess_signal_global.py`, `src/train_autoencoder_global.py`, `src/evaluate_damage_configs_global.py` |
| Consolidate results | `src/consolidate_final_results.py` |

**Execution note:** This table describes the intended sequence, not a tested one-command pipeline. Script arguments, working-directory assumptions, input filenames and output paths must be checked in the source before execution. In particular, some stages depend on artifacts created by earlier stages. Do not assume every script supports `--help` or can be run independently.

## 7. Selected experimental results

The reported reference validation metrics were:

| Metric | Feature-wise baseline | Global standardisation |
|---|---:|---:|
| Mean validation MSE | 0.089526 | 0.084491 |
| Median validation MSE | 0.008966 | 0.008844 |
| 95th-percentile error threshold | 0.465729 | 0.463218 |

For the feature-wise baseline, 7 of 125 reference validation windows (5.60%) exceeded the selected threshold. Across DS1–DS8, reported ROC-AUC values were approximately 0.519–0.610, indicating limited separation under the selected evaluation design. Although global standardisation reduced the mean reference validation MSE by approximately 5.62%, it did not yield a consistent damage-detection improvement.

These figures describe the project's recorded experiments, **not a guarantee of reproducibility on a different platform or proof of reliable physical damage detection**.

### Selected output files

- `results/final_consolidated_damage_results.csv` — consolidated damage-state results.
- `results/final_uds_generalization_results.csv` — additional undamaged-recording results.
- `results/damage_evaluation/damage_state_summary.csv` — baseline damage-state summary.
- `results/damage_evaluation/detection_performance_metrics.csv` — detection metrics.
- `results/damage_evaluation_global/global_damage_state_summary.csv` — global-scaling summary.
- `results/uds_all_configurations/uds_all_configurations_summary.csv` — summary across undamaged configurations.

Large per-window arrays and some intermediate CSV files are intentionally omitted from version control. Output files can differ depending on which scripts were executed.

## 8. Version control and large-file policy

The `.gitignore` excludes local virtual environments, the large HDF5 dataset, processed arrays (`.npy`, `.npz`), trained model checkpoints (`.pth`, `.pt`), caches, and selected large intermediate outputs. Lightweight source code, documentation, and summary CSVs are suitable for Git tracking.

```bash
git status
git status --ignored
```

**Note:** `.gitignore` does not remove a file from previous commits. The repository history was cleaned to remove the large HDF5 dataset; future contributors should avoid committing it again. Keep an authorised local dataset copy outside version control.

## 9. Limitations and future work

- The model uses a single acceleration channel and a single reference undamaged recording for training.
- Validation windows originate from the same recording as training windows, not an independent acquisition session.
- Changes in operating conditions, excitation and measurement characteristics may increase reconstruction errors even without damage.
- Threshold exceedance rates are window-level statistics, not probabilities of structural damage.
- The study does not establish damage localisation, severity prediction, operational calibration or real-time reliability.

Possible extensions include multiple sensor channels, independent-session validation, more robust calibration of false-alarm rates, alternative architectures, and tests under varied excitation conditions.

## 10. Research context and citation

This code accompanies the Curtin University Master's research project **_Autoencoder for Sensor Data Analysis_**. When using the underlying HBTA measurements, cite the original dataset publication or official source in accordance with its terms. A formal software citation, DOI, and redistribution licence have not been specified in this repository documentation.

**Repository:** https://github.com/lucky-banjara/Autoencoder-for-Sensor-Data-Analysis

---

**Reproducibility note:** To reproduce the full experiments, the reader needs the authorised HBTA dataset, a compatible Python environment, the repository source code, and any locally generated preprocessing artifacts and model checkpoints required by downstream scripts. These large artifacts are not bundled with the GitHub repository.
