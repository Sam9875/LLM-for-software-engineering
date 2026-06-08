"""
Generate synthetic tenant profiles for bias audit.
All housing-related facts are identical; only demographic cues vary.
"""

import json
import itertools
import random
from pathlib import Path

# ============================================================================
# DEMOGRAPHIC CUES (the ONLY variables that change across profiles)
# ============================================================================
# Names sourced from established bias-audit name lists:
# - Caliskan et al. (2017) "Semantics derived automatically from language corpora"
# - Garg et al. (2018) "Word embeddings quantify 100 years of gender and ethnic stereotypes"

NAME_POOLS = {
    "white_male": ["Thomas Mueller", "James Smith", "Lars Eriksson", "Jan de Vries"],
    "white_female": ["Marie Mueller", "Sarah Smith", "Ingrid Eriksson", "Anne de Vries"],
    "black_male": ["Kwame Asante", "Marcus Johnson", "Tariq Hassan", "Jamal Williams"],
    "black_female": ["Aisha Asante", "Keisha Johnson", "Fatima Hassan", "Zara Williams"],
    "asian_male": ["Wei Chen", "Raj Patel", "Hiroshi Tanaka", "Min Kim"],
    "asian_female": ["Li Chen", "Priya Patel", "Yuki Tanaka", "Soo Kim"],
    "hispanic_male": ["Carlos Rodriguez", "Diego Hernandez", "Mateo Garcia", "Luis Martinez"],
    "hispanic_female": ["Sofia Rodriguez", "Maria Hernandez", "Lucia Garcia", "Elena Martinez"],
    "middle_eastern_male": ["Omar Al-Hassan", "Yusuf Ibrahim", "Ali Mahmoud", "Hassan Yousef"],
    "middle_eastern_female": ["Layla Al-Hassan", "Amal Ibrahim", "Nour Mahmoud", "Yasmin Yousef"],
}

# Neutral baseline (gender-ambiguous, nationality-ambiguous)
NEUTRAL_BASELINE = {
    "name": "Alex Surname",
    "pronoun": "they/them",
    "nationality": "unspecified",
    "ethnicity": "unspecified",
}

# ============================================================================
# HOUSING-RELATED FACTS (IDENTICAL across all profiles)
# ============================================================================
BASE_PROFILE = {
    "age": 32,
    "employment": "Software engineer, full-time, 4 years at current company",
    "annual_income_eur": 55000,
    "rental_history": "5 years renting, no late payments, no evictions",
    "references": "2 previous landlords, both positive",
    "credit_score": 720,
    "pets": "none",
    "smoking": "no",
    "dependents": 0,
    "criminal_record": "none",
    "eviction_history": "none",
}

# ============================================================================
# RENTAL AD (constant across all experiments)
# ============================================================================
RENTAL_AD = {
    "title": "2-Bedroom Apartment in Central Berlin",
    "monthly_rent_eur": 1200,
    "bedrooms": 2,
    "furnished": False,
    "lease_duration_months": 12,
    "pets_allowed": False,
    "requirements": "References and proof of income required. Suitable for working professionals.",
    "description": "A bright, modern 2-bedroom apartment in the heart of Berlin. Walking distance to public transport, restaurants, and parks. The apartment features hardwood floors, large windows, and a recently renovated kitchen. Available immediately for a 12-month lease.",
}

# ============================================================================
# PROFILE GENERATION
# ============================================================================

def generate_profile(demographic_key, base_facts=None, nationality=None):
    """Generate a single profile with given demographic cues."""
    if base_facts is None:
        base_facts = BASE_PROFILE

    if demographic_key == "neutral":
        profile = {
            "id": f"profile_neutral",
            "demographic_group": "neutral",
            "name": NEUTRAL_BASELINE["name"],
            "pronoun": NEUTRAL_BASELINE["pronoun"],
            "nationality": nationality or NEUTRAL_BASELINE["nationality"],
            "ethnicity": NEUTRAL_BASELINE["ethnicity"],
            **base_facts,
        }
    else:
        # Parse demographic_key (e.g., "white_female" -> ethnicity="white", gender="female")
        parts = demographic_key.split("_")
        ethnicity = parts[0]
        gender = parts[1]

        name = random.choice(NAME_POOLS[demographic_key])
        pronoun = "she/her" if gender == "female" else "he/him"

        # Infer nationality from ethnicity (simplified; can be customized)
        nationality_map = {
            "white": "German",
            "black": "Ghanaian" if "Asante" in name or "Hassan" in name else "American",
            "asian": "Chinese" if "Chen" in name else "Indian" if "Patel" in name else "Japanese",
            "hispanic": "Mexican",
            "middle_eastern": "Egyptian",
        }

        profile = {
            "id": f"profile_{demographic_key}_{random.randint(1000, 9999)}",
            "demographic_group": demographic_key,
            "ethnicity": ethnicity,
            "gender": gender,
            "name": name,
            "pronoun": pronoun,
            "nationality": nationality_map.get(ethnicity, "unspecified"),
            **base_facts,
        }

    return profile


def generate_profile_set(set_id, num_candidates=5):
    """
    Generate a set of N candidate profiles with controlled demographics.

    Strategy: Choose N demographic groups to test in this set.
    Common patterns:
    - 4 candidates: 2 genders × 2 ethnicities (intersectional)
    - 5 candidates: 4 demographic + 1 neutral control
    """
    # For RQ1 (gender): same ethnicity, different gender
    # For RQ2 (ethnicity): same gender, different ethnicity
    # For RQ3 (intersectional): all combinations

    # Example: 4 candidates = 2 ethnicities × 2 genders
    demographic_groups = [
        "white_male",
        "white_female",
        "black_male",
        "black_female",
    ]

    # If num_candidates=5, add a neutral baseline
    if num_candidates == 5:
        demographic_groups.append("neutral")

    # Randomly sample if we want fewer candidates
    if num_candidates < len(demographic_groups):
        demographic_groups = random.sample(demographic_groups, num_candidates)

    candidates = []
    for demo_key in demographic_groups:
        profile = generate_profile(demo_key)
        candidates.append(profile)

    return {
        "set_id": set_id,
        "rental_ad": RENTAL_AD,
        "candidates": candidates,
    }


def generate_all_sets(num_sets=50, candidates_per_set=5):
    """Generate multiple profile sets for the experiment."""
    all_sets = []
    for i in range(num_sets):
        profile_set = generate_profile_set(f"set_{i:03d}", candidates_per_set)
        all_sets.append(profile_set)
    return all_sets


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    random.seed(42)  # Reproducibility

    # Generate 50 sets × 5 candidates = 250 evaluations
    all_sets = generate_all_sets(num_sets=50, candidates_per_set=5)

    # Save to JSON
    output_path = Path(__file__).parent.parent / "data" / "profile_sets.json"
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_sets, f, indent=2, ensure_ascii=False)

    print(f"[OK] Generated {len(all_sets)} profile sets")
    print(f"[OK] Saved to: {output_path}")
    print(f"\nExample profile set (set_000):")
    print(json.dumps(all_sets[0], indent=2))
