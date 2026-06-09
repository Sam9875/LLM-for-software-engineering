"""
Statistical analysis of tenant bias experiment results.
Tests for gender, racial, and intersectional bias.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# DATA LOADING
# ============================================================================

def load_and_link_data():
    """Load parsed results and link to demographic info."""
    # Load profile sets
    data_path = Path(__file__).parent.parent / "data" / "profile_sets.json"
    with open(data_path, "r", encoding="utf-8") as f:
        profile_sets = json.load(f)

    # Load parsed results
    results_path = Path(__file__).parent.parent / "results" / "parsed_results.json"
    if not results_path.exists():
        print(f"[WARN] No parsed results found at {results_path}")
        print("Run parse_responses.py first!")
        return None

    with open(results_path, "r", encoding="utf-8") as f:
        parsed_results = json.load(f)

    # Create profile set lookup
    set_lookup = {ps["set_id"]: ps for ps in profile_sets}

    # Link scores to demographics
    from parse_responses import link_scores_to_demographics

    all_data = []
    for parsed in parsed_results:
        if not parsed["parse_success"]:
            continue

        set_id = parsed["set_id"]
        if set_id not in set_lookup:
            continue

        profile_set = set_lookup[set_id]
        linked = link_scores_to_demographics(parsed, profile_set)
        all_data.extend(linked)

    df = pd.DataFrame(all_data)
    print(f"[OK] Loaded {len(df)} candidate evaluations")
    print(f"  Columns: {list(df.columns)}\n")
    return df


# ============================================================================
# STATISTICAL TESTS
# ============================================================================

def test_gender_bias(df, mitigation=None):
    """
    RQ1: Does gender affect evaluation?
    Mann-Whitney U test comparing male vs. female scores.
    """
    print("=" * 60)
    print("RQ1: GENDER BIAS")
    print("=" * 60)

    # Filter data
    if mitigation:
        data = df[df["mitigation"] == mitigation].copy()
        print(f"Mitigation: {mitigation}")
    else:
        data = df.copy()

    # Exclude neutral profiles
    data = data[data["gender"].isin(["male", "female"])]

    # Remove missing scores
    data = data.dropna(subset=["score"])

    print(f"Sample size: {len(data)} evaluations")
    print(f"  Male: {len(data[data['gender'] == 'male'])}")
    print(f"  Female: {len(data[data['gender'] == 'female'])}\n")

    # Descriptive stats
    print("Mean scores by gender:")
    print(data.groupby("gender")["score"].agg(["mean", "std", "median"]))
    print()

    # Mann-Whitney U test (non-parametric)
    male_scores = data[data["gender"] == "male"]["score"]
    female_scores = data[data["gender"] == "female"]["score"]

    statistic, p_value = stats.mannwhitneyu(male_scores, female_scores, alternative="two-sided")

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((male_scores.std() ** 2 + female_scores.std() ** 2) / 2)
    cohens_d = (male_scores.mean() - female_scores.mean()) / pooled_std

    print(f"Mann-Whitney U test:")
    print(f"  U statistic: {statistic:.2f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Significant at alpha=0.05: {'YES' if p_value < 0.05 else 'NO'}")
    print(f"\nEffect size (Cohen's d): {cohens_d:.3f}")
    if abs(cohens_d) < 0.2:
        effect_interpretation = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_interpretation = "small"
    elif abs(cohens_d) < 0.8:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"
    print(f"  Interpretation: {effect_interpretation}")
    print()

    return {
        "test": "Mann-Whitney U",
        "U": statistic,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "male_mean": male_scores.mean(),
        "female_mean": female_scores.mean(),
    }


def test_ethnicity_bias(df, mitigation=None):
    """
    RQ2: Does ethnicity affect evaluation?
    Kruskal-Wallis test comparing across ethnic groups.
    """
    print("=" * 60)
    print("RQ2: ETHNICITY/RACE BIAS")
    print("=" * 60)

    # Filter data
    if mitigation:
        data = df[df["mitigation"] == mitigation].copy()
        print(f"Mitigation: {mitigation}")
    else:
        data = df.copy()

    # Exclude neutral and unspecified
    data = data[data["ethnicity"].isin(["white", "black", "asian", "hispanic", "middle_eastern"])]
    data = data.dropna(subset=["score"])

    print(f"Sample size: {len(data)} evaluations")
    print(f"  Groups: {data['ethnicity'].unique()}\n")

    # Descriptive stats
    print("Mean scores by ethnicity:")
    print(data.groupby("ethnicity")["score"].agg(["mean", "std", "median", "count"]))
    print()

    # Kruskal-Wallis test (non-parametric ANOVA)
    groups = [group["score"].values for name, group in data.groupby("ethnicity")]
    statistic, p_value = stats.kruskal(*groups)

    print(f"Kruskal-Wallis H test:")
    print(f"  H statistic: {statistic:.2f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Significant at alpha=0.05: {'YES' if p_value < 0.05 else 'NO'}")
    print()

    # Post-hoc: Dunn's test (pairwise Mann-Whitney with Bonferroni correction)
    if p_value < 0.05:
        print("Post-hoc pairwise comparisons (Mann-Whitney U with Bonferroni):")
        ethnicities = data["ethnicity"].unique()
        n_comparisons = len(ethnicities) * (len(ethnicities) - 1) // 2

        for i, eth1 in enumerate(ethnicities):
            for eth2 in ethnicities[i + 1:]:
                scores1 = data[data["ethnicity"] == eth1]["score"]
                scores2 = data[data["ethnicity"] == eth2]["score"]
                _, p = stats.mannwhitneyu(scores1, scores2, alternative="two-sided")
                p_adjusted = p * n_comparisons  # Bonferroni
                sig = "***" if p_adjusted < 0.001 else "**" if p_adjusted < 0.01 else "*" if p_adjusted < 0.05 else "ns"
                print(f"  {eth1} vs {eth2}: p={p_adjusted:.4f} {sig}")
        print()

    return {
        "test": "Kruskal-Wallis H",
        "H": statistic,
        "p_value": p_value,
    }


def test_intersectional_bias(df, mitigation=None):
    """
    RQ3: Intersectional effects (gender × ethnicity).
    Two-way ANOVA on score ~ gender * ethnicity.
    """
    print("=" * 60)
    print("RQ3: INTERSECTIONAL BIAS (Gender × Ethnicity)")
    print("=" * 60)

    # Filter data
    if mitigation:
        data = df[df["mitigation"] == mitigation].copy()
        print(f"Mitigation: {mitigation}")
    else:
        data = df.copy()

    # Exclude neutral
    data = data[data["gender"].isin(["male", "female"])]
    data = data[data["ethnicity"].isin(["white", "black", "asian", "hispanic"])]
    data = data.dropna(subset=["score"])

    print(f"Sample size: {len(data)} evaluations\n")

    # Descriptive stats
    print("Mean scores by gender × ethnicity:")
    pivot = data.groupby(["gender", "ethnicity"])["score"].agg(["mean", "std", "count"])
    print(pivot)
    print()

    # Two-way ANOVA
    from scipy.stats import f_oneway

    # Main effect of gender
    male_scores = data[data["gender"] == "male"]["score"]
    female_scores = data[data["gender"] == "female"]["score"]
    f_gender, p_gender = stats.f_oneway(male_scores, female_scores)

    # Main effect of ethnicity
    ethnicity_groups = [group["score"].values for name, group in data.groupby("ethnicity")]
    f_ethnicity, p_ethnicity = stats.f_oneway(*ethnicity_groups)

    # Interaction effect (approximate using group means)
    # For proper interaction test, would need to use statsmodels OLS
    print(f"Two-way ANOVA (approximate):")
    print(f"  Gender main effect: F={f_gender:.2f}, p={p_gender:.4f}")
    print(f"  Ethnicity main effect: F={f_ethnicity:.2f}, p={p_ethnicity:.4f}")

    # Calculate interaction effect using group means
    group_means = data.groupby(["gender", "ethnicity"])["score"].mean().unstack()
    print(f"\nMean score matrix (gender × ethnicity):")
    print(group_means)
    print()

    # Check for interaction: difference in male-female gap across ethnicities
    print("Gender gap (male - female) by ethnicity:")
    for eth in group_means.columns:
        gap = group_means.loc["male", eth] - group_means.loc["female", eth]
        print(f"  {eth}: {gap:+.3f}")
    print()

    return {
        "gender_p": p_gender,
        "ethnicity_p": p_ethnicity,
        "group_means": group_means,
    }


def test_mitigation_effectiveness(df):
    """
    RQ4: Do mitigation strategies reduce bias?
    Compare effect sizes across mitigation conditions.
    """
    print("=" * 60)
    print("RQ4: MITIGATION EFFECTIVENESS")
    print("=" * 60)

    mitigations = df["mitigation"].unique()
    print(f"Testing {len(mitigations)} mitigation strategies:\n")

    results = {}
    for mit in mitigations:
        print(f"\n--- {mit} ---")
        gender_result = test_gender_bias(df, mitigation=mit)
        ethnicity_result = test_ethnicity_bias(df, mitigation=mit)
        results[mit] = {
            "gender": gender_result,
            "ethnicity": ethnicity_result,
        }

    # Compare effect sizes
    print("\n" + "=" * 60)
    print("COMPARISON: Effect sizes across mitigations")
    print("=" * 60)
    print(f"{'Mitigation':<25} {'Gender p':<12} {'Gender d':<12} {'Ethnicity p':<12}")
    print("-" * 60)
    for mit, res in results.items():
        gp = res["gender"]["p_value"]
        gd = res["gender"]["cohens_d"]
        ep = res["ethnicity"]["p_value"]
        print(f"{mit:<25} {gp:<12.4f} {gd:<12.3f} {ep:<12.4f}")

    return results


# ============================================================================
# VISUALIZATIONS
# ============================================================================

def plot_score_distributions(df, output_dir="results/figures"):
    """Generate visualizations."""
    output_path = Path(__file__).parent.parent / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")
    sns.set_palette("Set2")

    # 1. Score distribution by gender
    fig, ax = plt.subplots(figsize=(8, 6))
    data = df[df["gender"].isin(["male", "female"])].dropna(subset=["score"])
    sns.boxplot(data=data, x="gender", y="score", ax=ax)
    ax.set_title("Suitability Scores by Gender", fontsize=14, weight="bold")
    ax.set_xlabel("Gender", fontsize=12)
    ax.set_ylabel("Suitability Score (1-10)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path / "scores_by_gender.png", dpi=300)
    plt.close()
    print(f"[OK] Saved: scores_by_gender.png")

    # 2. Score distribution by ethnicity
    fig, ax = plt.subplots(figsize=(10, 6))
    data = df[df["ethnicity"].isin(["white", "black", "asian", "hispanic", "middle_eastern"])].dropna(subset=["score"])
    sns.boxplot(data=data, x="ethnicity", y="score", ax=ax, order=["white", "black", "asian", "hispanic", "middle_eastern"])
    ax.set_title("Suitability Scores by Ethnicity", fontsize=14, weight="bold")
    ax.set_xlabel("Ethnicity", fontsize=12)
    ax.set_ylabel("Suitability Score (1-10)", fontsize=12)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_path / "scores_by_ethnicity.png", dpi=300)
    plt.close()
    print(f"[OK] Saved: scores_by_ethnicity.png")

    # 3. Intersectional heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    data = df[df["gender"].isin(["male", "female"])].copy()
    data = data[data["ethnicity"].isin(["white", "black", "asian", "hispanic"])]
    pivot = data.groupby(["gender", "ethnicity"])["score"].mean().unstack()
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", center=7, ax=ax, cbar_kws={"label": "Mean Score"})
    ax.set_title("Mean Suitability Score by Gender × Ethnicity", fontsize=14, weight="bold")
    ax.set_xlabel("Ethnicity", fontsize=12)
    ax.set_ylabel("Gender", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path / "intersectional_heatmap.png", dpi=300)
    plt.close()
    print(f"[OK] Saved: intersectional_heatmap.png")

    # 4. Mitigation comparison
    if "mitigation" in df.columns and df["mitigation"].nunique() > 1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Gender
        data = df[df["gender"].isin(["male", "female"])].dropna(subset=["score"])
        sns.boxplot(data=data, x="mitigation", y="score", hue="gender", ax=axes[0])
        axes[0].set_title("Gender Bias Across Mitigations", fontsize=12, weight="bold")
        axes[0].set_xlabel("Mitigation Strategy")
        axes[0].set_ylabel("Score")
        axes[0].legend(title="Gender")

        # Ethnicity
        data = df[df["ethnicity"].isin(["white", "black"])].dropna(subset=["score"])
        sns.boxplot(data=data, x="mitigation", y="score", hue="ethnicity", ax=axes[1])
        axes[1].set_title("Ethnicity Bias Across Mitigations", fontsize=12, weight="bold")
        axes[1].set_xlabel("Mitigation Strategy")
        axes[1].set_ylabel("Score")
        axes[1].legend(title="Ethnicity")

        plt.tight_layout()
        plt.savefig(output_path / "mitigation_comparison.png", dpi=300)
        plt.close()
        print(f"[OK] Saved: mitigation_comparison.png")

    print(f"\n[OK] All figures saved to: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    df = load_and_link_data()

    if df is not None and len(df) > 0:
        print("\n" + "=" * 60)
        print("TENANT BIAS AUDIT - STATISTICAL ANALYSIS")
        print("=" * 60 + "\n")

        # Run tests
        test_gender_bias(df)
        test_ethnicity_bias(df)
        test_intersectional_bias(df)

        # Mitigation comparison (if multiple mitigations exist)
        if df["mitigation"].nunique() > 1:
            test_mitigation_effectiveness(df)

        # Generate plots
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60 + "\n")
        plot_score_distributions(df)

        print("\n[OK] Analysis complete!")
    else:
        print("No data to analyze. Run the experiment first.")
