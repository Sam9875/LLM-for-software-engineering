"""
Test the full pipeline using mock LLM responses.
Validates: profile generation, response parsing, analysis pipeline.
"""

import json
import random
from pathlib import Path

# Import modules
from generate_profiles import generate_all_sets
from parse_responses import parse_all_results, link_scores_to_demographics


def generate_mock_responses(profile_sets, model="mock", mitigation="baseline", bias_seed=42):
    """
    Generate mock LLM responses with controlled bias.
    This simulates what a biased LLM might output.
    """
    random.seed(bias_seed)
    results = []

    for profile_set in profile_sets:
        for run_idx in range(2):  # 2 runs per set
            candidates = profile_set["candidates"]

            # Simulate bias: white male gets higher scores, black female gets lower
            scores = {}
            for idx, candidate in enumerate(candidates):
                base_score = 7.0

                # Add demographic bias
                ethnicity = candidate.get("ethnicity", "")
                gender = candidate.get("gender", "")

                if ethnicity == "white":
                    base_score += 0.5
                elif ethnicity == "black":
                    base_score -= 0.5
                elif ethnicity == "middle_eastern":
                    base_score -= 0.3

                if gender == "male":
                    base_score += 0.2

                # Add noise
                score = base_score + random.uniform(-0.3, 0.3)
                scores[f"Candidate {idx + 1}"] = round(score, 1)

            # Create ranking based on scores
            ranking = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

            # Create justifications
            justifications = {}
            for idx, candidate in enumerate(candidates):
                label = f"Candidate {idx + 1}"
                justifications[label] = f"Profile {idx + 1} has a stable employment history and solid income."

            # Format as mock LLM response
            response_text = json.dumps({
                "ranking": ranking,
                "scores": scores,
                "justifications": justifications,
            }, indent=2)

            results.append({
                "set_id": profile_set["set_id"],
                "run_idx": run_idx,
                "model": model,
                "mitigation": mitigation,
                "prompt": "MOCK PROMPT",
                "response": response_text,
                "candidates": candidates,
            })

    return results


def test_full_pipeline():
    """Test the complete pipeline with mock data."""
    print("=" * 60)
    print("TESTING FULL PIPELINE WITH MOCK DATA")
    print("=" * 60 + "\n")

    # Step 1: Generate profiles
    print("Step 1: Generating profiles...")
    profile_sets = generate_all_sets(num_sets=20, candidates_per_set=5)
    print(f"  [OK] Generated {len(profile_sets)} sets\n")

    # Step 2: Generate mock responses
    print("Step 2: Generating mock LLM responses...")
    mock_results = generate_mock_responses(profile_sets, bias_seed=42)
    print(f"  [OK] Generated {len(mock_results)} mock responses\n")

    # Step 3: Parse responses
    print("Step 3: Parsing responses...")
    parsed = parse_all_results(mock_results)
    print(f"  [OK] Parsed {sum(1 for p in parsed if p['parse_success'])}/{len(parsed)} successfully\n")

    # Step 4: Link to demographics
    print("Step 4: Linking scores to demographics...")
    set_lookup = {ps["set_id"]: ps for ps in profile_sets}
    all_data = []
    for p in parsed:
        if p["parse_success"] and p["set_id"] in set_lookup:
            linked = link_scores_to_demographics(p, set_lookup[p["set_id"]])
            all_data.extend(linked)
    print(f"  [OK] Linked {len(all_data)} candidate evaluations\n")

    # Step 5: Quick analysis
    print("Step 5: Quick analysis...")
    import pandas as pd
    df = pd.DataFrame(all_data)

    print("\nMean score by gender:")
    print(df.groupby("gender")["score"].mean())

    print("\nMean score by ethnicity:")
    print(df.groupby("ethnicity")["score"].mean())

    print("\nMean score by gender x ethnicity:")
    print(df.groupby(["gender", "ethnicity"])["score"].mean().unstack())

    print("\n" + "=" * 60)
    print("PIPELINE TEST SUCCESSFUL!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Add your API key to .env file")
    print("2. Run: python scripts/run_experiment.py")
    print("3. Run: python scripts/parse_responses.py")
    print("4. Run: python scripts/analyze_results.py")


if __name__ == "__main__":
    test_full_pipeline()
