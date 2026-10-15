
from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_DIR = Path("results")
DAMAGE_DIR = RESULTS_DIR / "damage_evaluation"
GLOBAL_DIR = RESULTS_DIR / "damage_evaluation_global"
UDS_GENERALIZATION_DIR = RESULTS_DIR / "uds_generalization"
UDS_ALL_DIR = RESULTS_DIR / "uds_all_configurations"


def load_csv(path):
    """Load a CSV file and verify that it exists."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    df = pd.read_csv(path)
    print(f"Loaded: {path} ({len(df)} rows)")
    return df


def check_unique(df, column, name):
    """Prevent accidental duplicate rows during merging."""
    if df[column].isna().any():
        raise ValueError(f"{name} contains missing {column} values.")

    if df[column].duplicated().any():
        raise ValueError(f"{name} contains duplicate {column} values.")


def consolidate_damage_results():
    print("\n" + "=" * 70)
    print("CONSOLIDATING DAMAGE EVALUATION RESULTS")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load original experiment results
    # --------------------------------------------------------

    original = load_csv(
        DAMAGE_DIR / "damage_state_summary.csv"
    )

    threshold = load_csv(
        DAMAGE_DIR / "detection_threshold_summary.csv"
    )

    metrics = load_csv(
        DAMAGE_DIR / "detection_performance_metrics.csv"
    )

    distribution = load_csv(
        DAMAGE_DIR / "error_distribution_comparison.csv"
    )

    global_results = load_csv(
        GLOBAL_DIR / "global_damage_state_summary.csv"
    )

    # --------------------------------------------------------
    # 2. Validate original data
    # --------------------------------------------------------

    for name, df in [
        ("Original", original),
        ("Threshold", threshold),
        ("Metrics", metrics),
    ]:
        check_unique(df, "damage_state", name)

    check_unique(distribution, "state", "Distribution")
    check_unique(global_results, "configuration", "Global")

    expected_states = {"UDS"} | {
        f"DS{i}" for i in range(1, 9)
    }

    if set(original["damage_state"]) != expected_states:
        raise ValueError(
            "Original summary must contain UDS and DS1-DS8."
        )

    # --------------------------------------------------------
    # 3. Select threshold-specific columns
    # --------------------------------------------------------

    threshold_selected = threshold[
        [
            "damage_state",
            "threshold_mse",
            "windows_above_threshold",
            "windows_at_or_below_threshold",
            "exceedance_rate_percent",
        ]
    ].rename(columns={
        "threshold_mse": "original_threshold_mse",
        "windows_above_threshold":
            "original_windows_above_threshold",
        "windows_at_or_below_threshold":
            "original_windows_at_or_below_threshold",
        "exceedance_rate_percent":
            "original_exceedance_percent",
    })

    # --------------------------------------------------------
    # 4. Select detection performance columns
    # --------------------------------------------------------

    metrics_selected = metrics[
        [
            "damage_state",
            "sensitivity_percent",
            "specificity_percent",
            "balanced_accuracy_percent",
            "roc_auc",
        ]
    ].rename(columns={
        "sensitivity_percent":
            "original_sensitivity_percent",
        "specificity_percent":
            "original_specificity_percent",
        "balanced_accuracy_percent":
            "original_balanced_accuracy_percent",
        "roc_auc": "original_roc_auc",
    })

    # --------------------------------------------------------
    # 5. Select distribution-specific columns
    # --------------------------------------------------------

    distribution_selected = distribution[
        [
            "state",
            "p90_mse",
            "median_ratio_vs_UDS",
        ]
    ].rename(columns={
        "state": "damage_state",
        "p90_mse": "original_p90_mse",
    })

    # --------------------------------------------------------
    # 6. Merge original experiment results
    # --------------------------------------------------------

    final = original.copy()

    final = final.merge(
        threshold_selected,
        on="damage_state",
        how="left",
        validate="one_to_one",
    )

    final = final.merge(
        metrics_selected,
        on="damage_state",
        how="left",
        validate="one_to_one",
    )

    final = final.merge(
        distribution_selected,
        on="damage_state",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # 7. Prepare global-scaling experiment results
    # --------------------------------------------------------

    global_results = global_results.copy()

    extracted_states = global_results[
        "configuration"
    ].str.extract(r"_DS([1-8])_")[0]

    if extracted_states.isna().any():
        raise ValueError(
            "Unable to identify a damage state in global results."
        )

    global_results["damage_state"] = (
        "DS" + extracted_states
    )

    check_unique(
        global_results,
        "damage_state",
        "Global damage states",
    )

    global_selected = global_results[
        [
            "damage_state",
            "window_count",
            "mean_mse",
            "median_mse",
            "p95_mse",
            "max_mse",
            "threshold",
            "exceedance_count",
            "exceedance_percent",
        ]
    ].rename(columns={
        "window_count": "global_window_count",
        "mean_mse": "global_mean_mse",
        "median_mse": "global_median_mse",
        "p95_mse": "global_p95_mse",
        "max_mse": "global_max_mse",
        "threshold": "global_threshold_mse",
        "exceedance_count":
            "global_windows_above_threshold",
        "exceedance_percent":
            "global_exceedance_percent",
    })

    # --------------------------------------------------------
    # 8. Merge global results with original results
    # --------------------------------------------------------

    final = final.merge(
        global_selected,
        on="damage_state",
        how="left",
        validate="one_to_one",
    )

    # The global model was evaluated only on DS1-DS8.
    # Therefore, global metrics for UDS remain NaN.

    # --------------------------------------------------------
    # 9. Calculate percentage-point differences
    # --------------------------------------------------------

    final["global_minus_original_exceedance_pp"] = (
        final["global_exceedance_percent"]
        - final["original_exceedance_percent"]
    )

    # --------------------------------------------------------
    # 10. Validate the consolidated table
    # --------------------------------------------------------

    if len(final) != 9:
        raise ValueError(
            f"Expected 9 rows, but obtained {len(final)}."
        )

    damage_rows = final[
        final["damage_state"] != "UDS"
    ]

    required_columns = [
        "original_exceedance_percent",
        "original_roc_auc",
        "global_exceedance_percent",
    ]

    if damage_rows[required_columns].isna().any().any():
        raise ValueError(
            "Missing detection metrics for one or more damage states."
        )

    # --------------------------------------------------------
    # 11. Sort UDS followed by DS1-DS8
    # --------------------------------------------------------

    order = {"UDS": 0}

    for i in range(1, 9):
        order[f"DS{i}"] = i

    final["_sort_order"] = final[
        "damage_state"
    ].map(order)

    final = (
        final.sort_values("_sort_order")
        .drop(columns="_sort_order")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # 12. Save consolidated results
    # --------------------------------------------------------

    output_path = (
        RESULTS_DIR / "final_consolidated_damage_results.csv"
    )

    final.to_csv(output_path, index=False)

    print("\n" + "=" * 70)
    print("CONSOLIDATED DAMAGE RESULTS")
    print("=" * 70)

    print(f"Saved: {output_path}")
    print(f"Rows: {len(final)}")

    display_columns = [
        "damage_state",
        "number_of_windows",
        "mean_mse",
        "median_mse",
        "original_exceedance_percent",
        "original_roc_auc",
        "global_exceedance_percent",
        "global_minus_original_exceedance_pp",
    ]

    print("\nSelected results:")
    print(
        final[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    return final


def consolidate_uds_results():
    print("\n" + "=" * 70)
    print("CONSOLIDATING UNDAMAGED CONFIGURATION RESULTS")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load UDS generalization results
    # --------------------------------------------------------

    generalization = load_csv(
        UDS_GENERALIZATION_DIR
        / "uds_generalization_summary.csv"
    )

    remaining_uds = load_csv(
        UDS_ALL_DIR
        / "uds_all_configurations_summary.csv"
    )

    # --------------------------------------------------------
    # 2. Add condition group to SM-Y configurations
    # --------------------------------------------------------

    generalization = generalization.copy()
    remaining_uds = remaining_uds.copy()

    generalization["condition_group"] = "SM-Y"

    # --------------------------------------------------------
    # 3. Select common columns
    # --------------------------------------------------------

    uds_columns = [
        "configuration",
        "condition_group",
        "raw_samples",
        "number_of_windows",
        "discarded_samples",
        "mean_mse",
        "median_mse",
        "p95_mse",
        "windows_above_threshold",
        "exceedance_rate_percent",
    ]

    # --------------------------------------------------------
    # 4. Combine UDS configurations
    # --------------------------------------------------------

    uds_combined = pd.concat(
        [
            generalization[uds_columns],
            remaining_uds[uds_columns],
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # 5. Validate combined UDS results
    # --------------------------------------------------------

    check_unique(
        uds_combined,
        "configuration",
        "Combined UDS results",
    )

    if len(uds_combined) != 9:
        raise ValueError(
            f"Expected 9 additional UDS configurations, "
            f"but obtained {len(uds_combined)}."
        )

    # --------------------------------------------------------
    # 6. Save combined UDS results
    # --------------------------------------------------------

    output_path = (
        RESULTS_DIR / "final_uds_generalization_results.csv"
    )

    uds_combined.to_csv(output_path, index=False)

    print("\n" + "=" * 70)
    print("UNDAMAGED CONFIGURATION RESULTS")
    print("=" * 70)

    print(f"Saved: {output_path}")
    print(f"Rows: {len(uds_combined)}")

    display_columns = [
        "configuration",
        "condition_group",
        "number_of_windows",
        "median_mse",
        "exceedance_rate_percent",
    ]

    print("\nSelected results:")
    print(
        uds_combined[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    return uds_combined


def main():
    print("=" * 70)
    print("HBTA AUTOENCODER - FINAL RESULTS CONSOLIDATION")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    consolidate_damage_results()
    consolidate_uds_results()

    print("\n" + "=" * 70)
    print("FINAL RESULTS CONSOLIDATION COMPLETE")
    print("=" * 70)

    print("\nGenerated files:")
    print("1. results/final_consolidated_damage_results.csv")
    print("2. results/final_uds_generalization_results.csv")


if __name__ == "__main__":
    main()
