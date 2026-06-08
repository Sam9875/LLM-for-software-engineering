"""
Generate mock experiment results for dashboard testing.
Simulates a biased LLM with realistic variance and patterns.
"""

import json
import random
from pathlib import Path


def generate_mock_data():
    """Create mock parsed results that look like real LLM outputs."""
    random.seed(42)

    # Load profiles
    data_path = Path(__file__).parent.parent / "data" / "profile_sets.json"
    with open(data_path, "r", encoding="utf-8") as f:
        profile_sets = json.load(f)

    mitigations = ["baseline", "explicit_fairness", "role_fair", "cot"]

    # Bias strength per mitigation (how much the mock LLM discriminates)
    bias_strength = {
        "baseline": 1.0,
        "explicit_fairness": 0.3,  # Reduces bias
        "role_fair": 0.5,
        "cot": 0.4,
    }

    mock_parsed = []

    for profile_set in profile_sets:
        for mitigation in mitigations:
            for run_idx in range(3):  # 3 runs per mitigation per set
                candidates = profile_set["candidates"]

                scores = {}
                justifications = {}

                for idx, candidate in enumerate(candidates):
                    label = f"Candidate {idx + 1}"
                    ethnicity = candidate.get("ethnicity", "")
                    gender = candidate.get("gender", "")
                    strength = bias_strength[mitigation]

                    # Base score
                    base = 7.0

                    # Apply bias with mitigation strength
                    if ethnicity == "white":
                        bias = 0.6 * strength
                    elif ethnicity == "black":
                        bias = -0.5 * strength
                    elif ethnicity == "asian":
                        bias = 0.1 * strength
                    elif ethnicity == "hispanic":
                        bias = -0.2 * strength
                    elif ethnicity == "middle_eastern":
                        bias = -0.4 * strength
                    else:
                        bias = 0.0

                    if gender == "male":
                        bias += 0.2 * strength
                    elif gender == "female":
                        bias += -0.1 * strength

                    # Add noise
                    score = base + bias + random.uniform(-0.4, 0.4)
                    scores[label] = round(score, 1)

                    # Generate justification with linguistic bias
                    positive_words = ["stable", "reliable", "strong", "solid", "ideal"]
                    hedging_words = ["verify", "carefully", "may want", "ensure", "cautious"]

                    if score > 7.5:
                        if random.random() < 0.5 + bias * 0.2:
                            just = f"Stable employment and {random.choice(positive_words)} income. An ideal tenant profile."
                        else:
                            just = f"Stable employment with solid income. References are strong."
                    elif score > 6.5:
                        just = f"Decent employment and income. Rental history is acceptable."
                    else:
                        if random.random() < 0.4 - bias * 0.2:
                            just = f"Employment is acceptable, though the landlord may want to {random.choice(hedging_words)} references."
                        else:
                            just = f"Employment situation may need additional verification. Income is adequate."

                    justifications[label] = just

                ranking = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

                mock_parsed.append({
                    "set_id": profile_set["set_id"],
                    "run_idx": run_idx,
                    "model": "claude",
                    "mitigation": mitigation,
                    "ranking": ranking,
                    "scores": scores,
                    "justifications": justifications,
                    "parse_success": True,
                    "candidates": candidates,
                })

    # Save
    output_path = Path(__file__).parent.parent / "results" / "parsed_results.json"
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mock_parsed, f, indent=2, ensure_ascii=False)

    print(f"[OK] Generated {len(mock_parsed)} mock parsed results")
    print(f"[OK] Saved to: {output_path}")
    print(f"\nBreakdown:")
    print(f"  Sets: {len(profile_sets)}")
    print(f"  Mitigations: {len(mitigations)}")
    print(f"  Runs per set/mitigation: 3")
    print(f"  Total: {len(profile_sets)} x {len(mitigations)} x 3 = {len(mock_parsed)}")


if __name__ == "__main__":
    generate_mock_data()
