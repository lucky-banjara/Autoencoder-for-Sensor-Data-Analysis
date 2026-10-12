
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    balanced_accuracy_score,
)


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

BASELINE_PATH = (
    "results/reconstruction/validation_window_mse.npy"
)

DAMAGE_DIR = "results/damage_evaluation"
OUTPUT_DIR = "results/damage_evaluation"
PLOT_DIR = "results/plots"

THRESHOLD_PERCENTILE = 95

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# --------------------------------------------------
# 2. Load reconstruction errors
# --------------------------------------------------

def load_errors(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    errors = np.asarray(np.load(path), dtype=np.float64)

    if errors.ndim != 1 or len(errors) == 0:
        raise ValueError(f"Expected a non-empty 1D array: {path}")

    if not np.isfinite(errors).all():
        raise ValueError(f"NaN or Inf found in: {path}")

    return errors


uds_errors = load_errors(BASELINE_PATH)

# Fixed threshold based only on the UDS reference.
threshold = float(
    np.percentile(uds_errors, THRESHOLD_PERCENTILE)
)

uds_predictions = uds_errors > threshold
specificity = float(np.mean(~uds_predictions))

print("=" * 72)
print("HBTA PRELIMINARY DETECTION-PERFORMANCE EVALUATION")
print("=" * 72)
print(f"UDS windows: {len(uds_errors)}")
print(f"Threshold ({THRESHOLD_PERCENTILE}th UDS percentile): {threshold:.6f}")
print(f"Observed UDS specificity: {specificity * 100:.2f}%")

rows = []
roc_data = []

# --------------------------------------------------
# 3. Evaluate each damage state
# --------------------------------------------------

for number in range(1, 9):
    state = f"DS{number}"

    path = os.path.join(
        DAMAGE_DIR,
        f"DS{number}_window_mse.npy"
    )

    damage_errors = load_errors(path)

    # Positive = damage-state window; negative = UDS window.
    y_true = np.concatenate([
        np.zeros(len(uds_errors), dtype=int),
        np.ones(len(damage_errors), dtype=int),
    ])

    scores = np.concatenate([
        uds_errors,
        damage_errors,
    ])

    y_pred = (scores > threshold).astype(int)

    sensitivity = float(
        np.mean(damage_errors > threshold)
    )

    balanced_accuracy = float(
        balanced_accuracy_score(y_true, y_pred)
    )

    auc = float(
        roc_auc_score(y_true, scores)
    )

    fpr, tpr, _ = roc_curve(y_true, scores)

    roc_data.append((state, fpr, tpr, auc))

    rows.append({
        "damage_state": state,
        "uds_windows": len(uds_errors),
        "damage_windows": len(damage_errors),
        "threshold": threshold,
        "sensitivity_percent": sensitivity * 100,
        "specificity_percent": specificity * 100,
        "balanced_accuracy_percent": balanced_accuracy * 100,
        "roc_auc": auc,
    })

    print(
        f"{state}: "
        f"Sensitivity={sensitivity * 100:.2f}%, "
        f"Specificity={specificity * 100:.2f}%, "
        f"Balanced accuracy={balanced_accuracy * 100:.2f}%, "
        f"ROC-AUC={auc:.4f}"
    )


# --------------------------------------------------
# 4. Save metrics
# --------------------------------------------------

results = pd.DataFrame(rows)

csv_path = os.path.join(
    OUTPUT_DIR,
    "detection_performance_metrics.csv"
)

results.to_csv(csv_path, index=False)

print("\nDetailed results:")
print(results.to_string(index=False, formatters={
    "threshold": "{:.6f}".format,
    "sensitivity_percent": "{:.2f}".format,
    "specificity_percent": "{:.2f}".format,
    "balanced_accuracy_percent": "{:.2f}".format,
    "roc_auc": "{:.4f}".format,
}))

print("\nSaved metrics:", csv_path)


# --------------------------------------------------
# 5. Plot ROC curves
# --------------------------------------------------

plt.figure(figsize=(10, 8))

for state, fpr, tpr, auc in roc_data:
    plt.plot(
        fpr,
        tpr,
        label=f"{state} (AUC={auc:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random ranking"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("HBTA Autoencoder: ROC Curves by Damage State")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plot_path = os.path.join(
    PLOT_DIR,
    "detection_roc_curves.png"
)

plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.show()

print("Saved ROC plot:", plot_path)
print("\nEvaluation complete.")
print(
    "Caution: these are preliminary window-level comparisons, "
    "not independent-recording validation."
)