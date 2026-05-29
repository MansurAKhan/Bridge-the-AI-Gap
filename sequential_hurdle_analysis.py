#!/usr/bin/env python3
"""
Two-stage sequential hurdle analysis for AI legislation outcomes
(self-contained final version with bootstrap inference + train/test accuracy).

Stage 1 (engagement hurdle): Inaction (0) vs. Any Action (1 or 2)
Stage 2 (advancement hurdle): Processed/Stalled (1) vs. Advanced (2), on the
                              already-engaged subset.

For each of N_BOOTSTRAPS resamples (drawn with replacement, seed = 42) we fit a
ridge (L2) logistic regression for each stage and record:
    * the coefficients (summarized across trials -> mean, std, z, p, FDR-q, 95% CI),
    * the in-sample TRAIN accuracy (predicting the resample it was fit on), and
    * the out-of-bag TEST accuracy (predicting the rows NOT drawn in that resample).

Num_Sponsors is standardized (StandardScaler) fit on the train resample and applied
to the out-of-bag rows, so the test accuracy has no information leakage.

Run:  python sequential_hurdle_analysis.py
Outputs (next to this script):
    sequential_hurdle_results.csv   <- coefficient summary + train/test accuracy
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "data_ordinal.csv"
RESULTS_PATH = SCRIPT_DIR / "sequential_hurdle_results.csv"

N_BOOTSTRAPS = 1000
BOOTSTRAP_RANDOM_STATE = 42

MODEL_KWARGS = {
    "penalty": "l2",
    "C": 1.0,
    "class_weight": "balanced",
    "max_iter": 2000,
    "solver": "lbfgs",
    "random_state": 42,
}

OUTCOME_LABELS = {
    0: "Inaction (Expired)",
    1: "Processed (Stalled in Committee)",
    2: "Advanced (Calendar + Resolved)",
}


# --------------------------------------------------------------------------- #
# Status / subfield parsing
# --------------------------------------------------------------------------- #
def normalize_status_text(status):
    return " ".join(str(status).strip().split()).lower()


def map_status_to_ordinal(status):
    status_parts = [part.strip() for part in str(status).split(";") if part.strip()]
    latest_status = status_parts[-1] if status_parts else str(status)
    normalized = normalize_status_text(latest_status)

    if (
        "amendment passed" in normalized
        or normalized == "passed"
        or "declined" in normalized
        or "calendar inaction" in normalized
        or "passed house" in normalized
        or "passed senate" in normalized
    ):
        return 2
    if (
        "stalled in committee" in normalized
        or "committee stall" in normalized
        or normalized.endswith("stall")
    ):
        return 1
    if "expired without action" in normalized:
        return 0
    raise ValueError(f"Unmapped legislative outcome: {status!r}")


def split_subfields(subfields):
    if pd.isna(subfields):
        return []
    return [part.strip() for part in str(subfields).split(";") if part.strip()]


def canonicalize_subfield(subfield):
    mapping = {
        "General Ethical Usage": "General Ethical Usage",
        "GEU": "General Ethical Usage",
        "Policy Advisory": "Policy Advisory",
        "Data Usage": "Data Usage",
        "AI in Government": "AI in Government + Military",
        "AI in Government + Military": "AI in Government + Military",
        "Push for AI Research": "Push for AI Research",
        "Deepfake": "Deepfake",
        "Job Security": "Job Security",
        "AGI": "AGI",
        "LLM": "LLM",
        "Large Language Models": "LLM",
        "Autonomous Driving": "Autonomous Driving",
    }
    return mapping.get(subfield, subfield)


# --------------------------------------------------------------------------- #
# Data loading / feature engineering
# --------------------------------------------------------------------------- #
def load_and_prepare_data():
    print("=" * 60)
    print("LOADING AND PREPARING HURDLE DATA")
    print("=" * 60)
    df = pd.read_csv(DATA_PATH)
    df["Legislative_Outcome_Ordinal"] = df["Failure Reason/ or Passed"].apply(map_status_to_ordinal)
    df["Legislative_Outcome_Label"] = df["Legislative_Outcome_Ordinal"].map(OUTCOME_LABELS)
    print(f"Dataset shape: {df.shape}")
    return df


def create_feature_dataframe(df):
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    df = df.rename(
        columns={"Bipartisan?": "Bipartisan", "# of Sponsors": "Num_Sponsors"}
    ).copy()

    before_party_filter = len(df)
    df = df[df["Sponsor Party"].fillna("") != "Independent"].copy()
    print(f"Removed Independent-sponsored bills: {before_party_filter - len(df)} rows removed")

    df["Sponsor Party"] = df["Sponsor Party"].fillna("Unknown")
    bipartisan_text = df["Bipartisan"].astype(str).str.strip().str.upper()
    df["Bipartisan"] = bipartisan_text.map({"TRUE": 1, "FALSE": 0}).fillna(0).astype(int)
    df["Num_Sponsors"] = pd.to_numeric(df["Num_Sponsors"], errors="coerce").fillna(0)
    df["Sponsor_Party_Binary"] = (df["Sponsor Party"] == "Democrat").astype(int)
    df["Chamber_Binary"] = (df["Chamber"] == "House").astype(int)

    canonical_subfield_lists = [
        [canonicalize_subfield(s) for s in split_subfields(subfields)]
        for subfields in df["Subfields"]
    ]

    base_subfields = [
        "General Ethical Usage", "Policy Advisory", "Data Usage",
        "AI in Government + Military", "Push for AI Research", "Deepfake",
        "Job Security", "AGI", "LLM", "Autonomous Driving",
    ]
    for subfield in base_subfields:
        df[subfield] = [int(subfield in subfields) for subfields in canonical_subfield_lists]

    df["Advanced AI"] = (
        (df["AGI"] == 1) | (df["LLM"] == 1) | (df["Autonomous Driving"] == 1)
    ).astype(int)

    feature_columns = [
        "Num_Sponsors", "Chamber_Binary", "Sponsor_Party_Binary",
        "General Ethical Usage", "Policy Advisory", "Data Usage",
        "AI in Government + Military", "Push for AI Research", "Deepfake",
        "Job Security", "Advanced AI", "Bipartisan",
    ]

    df["y"] = df["Legislative_Outcome_Ordinal"].astype(int)
    df["y_hurdle_1"] = (df["y"] > 0).astype(int)

    print(f"Stage 1 sample size (full dataset): n={len(df)}")
    print(f"Stage 2 sample size (engaged subset): n={(df['y'].isin([1, 2])).sum()}")
    return df.copy(), feature_columns


# --------------------------------------------------------------------------- #
# Bootstrap with train + out-of-bag test accuracy
# --------------------------------------------------------------------------- #
def normalize_num_sponsors(train_df, test_df):
    """StandardScaler on Num_Sponsors, fit on train, applied to both (no leakage)."""
    train_df, test_df = train_df.copy(), test_df.copy()
    scaler = StandardScaler()
    train_df["Num_Sponsors"] = scaler.fit_transform(train_df[["Num_Sponsors"]].astype(float))
    if len(test_df):
        test_df["Num_Sponsors"] = scaler.transform(test_df[["Num_Sponsors"]].astype(float))
    return train_df, test_df


def run_bootstrap_trials(modeling_df_raw, feature_columns,
                         n_bootstraps=N_BOOTSTRAPS, random_state=BOOTSTRAP_RANDOM_STATE):
    print("\n" + "=" * 60)
    print("BOOTSTRAP TRIAL ANALYSIS (train + out-of-bag test)")
    print("=" * 60)

    n = len(modeling_df_raw)
    n_features = len(feature_columns)
    all_idx = np.arange(n)
    rng = np.random.default_rng(random_state)

    s1_coef = np.full((n_bootstraps, n_features), np.nan)
    s2_coef = np.full((n_bootstraps, n_features), np.nan)
    s1_train = np.full(n_bootstraps, np.nan)
    s1_test = np.full(n_bootstraps, np.nan)
    s2_train = np.full(n_bootstraps, np.nan)
    s2_test = np.full(n_bootstraps, np.nan)

    for t in range(n_bootstraps):
        if t == 0 or (t + 1) % 100 == 0:
            print(f"Running bootstrap trial {t + 1}/{n_bootstraps}")

        boot_idx = rng.choice(n, n, replace=True)
        oob_idx = np.setdiff1d(all_idx, boot_idx)

        train_raw = modeling_df_raw.iloc[boot_idx].reset_index(drop=True)
        test_raw = modeling_df_raw.iloc[oob_idx].reset_index(drop=True)
        train_df, test_df = normalize_num_sponsors(train_raw, test_raw)

        # Stage 1
        try:
            Xtr, ytr = train_df[feature_columns], train_df["y_hurdle_1"].astype(int)
            m1 = LogisticRegression(**MODEL_KWARGS).fit(Xtr, ytr)
            s1_coef[t] = m1.coef_[0]
            s1_train[t] = accuracy_score(ytr, m1.predict(Xtr))
            yte = test_df["y_hurdle_1"].astype(int)
            if len(yte):
                s1_test[t] = accuracy_score(yte, m1.predict(test_df[feature_columns]))
        except Exception:
            pass

        # Stage 2 (engaged subset)
        try:
            tr2 = train_df[train_df["y"].isin([1, 2])].copy()
            tr2["y_hurdle_2"] = (tr2["y"] == 2).astype(int)
            Xtr2, ytr2 = tr2[feature_columns], tr2["y_hurdle_2"].astype(int)
            m2 = LogisticRegression(**MODEL_KWARGS).fit(Xtr2, ytr2)
            s2_coef[t] = m2.coef_[0]
            s2_train[t] = accuracy_score(ytr2, m2.predict(Xtr2))
            te2 = test_df[test_df["y"].isin([1, 2])].copy()
            te2["y_hurdle_2"] = (te2["y"] == 2).astype(int)
            if len(te2):
                s2_test[t] = accuracy_score(
                    te2["y_hurdle_2"].astype(int), m2.predict(te2[feature_columns])
                )
        except Exception:
            pass

    return {
        "feature_names": list(feature_columns),
        "stage_1_coef": s1_coef, "stage_2_coef": s2_coef,
        "stage_1_train_accuracy": s1_train, "stage_1_test_accuracy": s1_test,
        "stage_2_train_accuracy": s2_train, "stage_2_test_accuracy": s2_test,
    }


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def compute_q_values_with_nans(p_values):
    p_values = np.asarray(p_values, dtype=float)
    q_values = np.full_like(p_values, np.nan)
    valid_mask = np.isfinite(p_values)
    if valid_mask.any():
        _, valid_q, _, _ = multipletests(p_values[valid_mask], method="fdr_bh")
        q_values[valid_mask] = valid_q
    return q_values


def _stage_stats(coef):
    mean = np.nanmean(coef, axis=0)
    std = np.nanstd(coef, axis=0)
    count = np.sum(np.isfinite(coef), axis=0)
    se = np.divide(std, np.sqrt(count), out=np.full_like(std, np.nan), where=count > 0)
    safe_std = np.where(std == 0, np.nan, std)
    z = mean / safe_std
    p = 2 * (1 - norm.cdf(np.abs(np.nan_to_num(z, nan=0.0))))
    ci_lo = np.nanpercentile(coef, 2.5, axis=0)
    ci_hi = np.nanpercentile(coef, 97.5, axis=0)
    return mean, std, se, z, p, ci_lo, ci_hi, count


def build_bootstrap_aggregate_results(bootstrap_results):
    feats = bootstrap_results["feature_names"]
    s1 = _stage_stats(bootstrap_results["stage_1_coef"])
    s2 = _stage_stats(bootstrap_results["stage_2_coef"])
    combined_q = compute_q_values_with_nans(np.concatenate([s1[4], s2[4]]))
    q1, q2 = combined_q[: len(feats)], combined_q[len(feats):]

    return pd.DataFrame({
        "Variable Name": feats,
        "Hurdle 1 Coeff Mean": s1[0], "Hurdle 1 Std_Deviation": s1[1],
        "Hurdle 1 Std_Error_of_Mean": s1[2], "Hurdle 1 Z_score": s1[3],
        "Hurdle 1 P_value": s1[4], "Hurdle 1 FDR_Q": q1,
        "Hurdle 1 CI_95_Lower": s1[5], "Hurdle 1 CI_95_Upper": s1[6],
        "Hurdle 1 Successful_Trials": s1[7],
        "Hurdle 2 Coeff Mean": s2[0], "Hurdle 2 Std_Deviation": s2[1],
        "Hurdle 2 Std_Error_of_Mean": s2[2], "Hurdle 2 Z_score": s2[3],
        "Hurdle 2 P_value": s2[4], "Hurdle 2 FDR_Q": q2,
        "Hurdle 2 CI_95_Lower": s2[5], "Hurdle 2 CI_95_Upper": s2[6],
        "Hurdle 2 Successful_Trials": s2[7],
    })


def append_accuracy_rows(summary_df, bootstrap_results):
    """Append clearly-labeled train/test accuracy rows (mean across trials).

    Accuracy is per-stage, so it is stored in the 'Coeff Mean' columns (the mean
    accuracy) and the 'Std_Deviation' columns (its std) with all other cells blank.
    """
    def mean_std(arr):
        v = arr[np.isfinite(arr)]
        return float(np.mean(v)), float(np.std(v)), int(len(v))

    rows = []
    for label, key1, key2 in [
        ("ACCURACY: Train (mean over trials)", "stage_1_train_accuracy", "stage_2_train_accuracy"),
        ("ACCURACY: Test OOB (mean over trials)", "stage_1_test_accuracy", "stage_2_test_accuracy"),
    ]:
        m1, sd1, n1 = mean_std(bootstrap_results[key1])
        m2, sd2, n2 = mean_std(bootstrap_results[key2])
        row = {c: "" for c in summary_df.columns}
        row.update({
            "Variable Name": label,
            "Hurdle 1 Coeff Mean": m1, "Hurdle 1 Std_Deviation": sd1,
            "Hurdle 1 Successful_Trials": n1,
            "Hurdle 2 Coeff Mean": m2, "Hurdle 2 Std_Deviation": sd2,
            "Hurdle 2 Successful_Trials": n2,
        })
        rows.append(row)
    return pd.concat([summary_df, pd.DataFrame(rows)], ignore_index=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    print("SEQUENTIAL HURDLE ANALYSIS (train + test)")
    print("=" * 80)

    df = load_and_prepare_data()
    modeling_df_raw, feature_columns = create_feature_dataframe(df)

    bootstrap_results = run_bootstrap_trials(modeling_df_raw, feature_columns)
    summary_df = build_bootstrap_aggregate_results(bootstrap_results)
    results_df = append_accuracy_rows(summary_df, bootstrap_results)
    results_df.to_csv(RESULTS_PATH, index=False)

    def report(name, arr):
        v = arr[np.isfinite(arr)]
        print(f"  {name:<28} mean={np.mean(v):.4f}  std={np.std(v):.4f}  (n={len(v)})")

    print("\n" + "=" * 60)
    print("ACCURACY SUMMARY (1000 bootstrap trials)")
    print("=" * 60)
    report("Stage 1 TRAIN", bootstrap_results["stage_1_train_accuracy"])
    report("Stage 1 TEST  (out-of-bag)", bootstrap_results["stage_1_test_accuracy"])
    report("Stage 2 TRAIN", bootstrap_results["stage_2_train_accuracy"])
    report("Stage 2 TEST  (out-of-bag)", bootstrap_results["stage_2_test_accuracy"])

    print(f"\nResults written to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
