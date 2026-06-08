# Paper Summaries

Detailed summaries of the most important papers informing this project. Each entry includes: research question, methodology, key findings (with statistics), limitations, and how it relates to our work.

---

## 1. Wilson & Caliskan (2024) — LLM Resume Screening Bias

**Title:** Gender, Race, and Intersectional Bias in Resume Screening via Language Model Retrieval

**Authors:** Kyra Wilson, Aylin Caliskan

**Venue:** Proceedings of the 2024 AAAI/ACM Conference on AI, Ethics, and Society (AIES '24)

**Link:** [arXiv:2407.20371](https://arxiv.org/abs/2407.20371) | [Brookings summary](https://www.brookings.edu/articles/gender-race-and-intersectional-bias-in-ai-resume-screening-via-language-model-retrieval/)

### Research Question
Do large language models used in resume screening disadvantage candidates based on protected demographic attributes (gender, race), and do these biases compound intersectionally?

### Methodology
- **Framework:** Document retrieval simulation of job candidate selection
- **Resumes:** 554 publicly available resumes, augmented with 80 names signaling 4 intersectional groups (Black women, Black men, white women, white men); held last name constant as "Williams" to isolate the first-name effect
- **Job descriptions:** 571 postings across 9 occupations
- **Models tested:** 3 Massive Text Embedding (MTE) models: E5-Mistral-7b-Instruct, GritLM-7B, SFR-Embedding-Mistral
- **Approach:** Zero-shot dense retrieval. Resumes ranked by cosine similarity to job descriptions. Top 10% of resumes selected per posting.
- **Statistical tests:** Chi-square tests on 27 comparisons per model/occupation (3 models × 9 occupations)
- **Scale:** ~40,000 resume-job description comparisons per model

### Key Findings
| Axis | Statistic | Interpretation |
|------|-----------|----------------|
| **Gender** | Male-associated names preferred in 51.9% of tests; female-associated in only 11.1%; equal in 37% | Strong male preference |
| **Race** | White-associated names preferred in 85.1% of tests; Black-associated in 8.6%; equal in 6.3% | Very strong white preference |
| **Intersectional** | Black men selected only 14.8% of the time when compared to Black women (equal in 18.5%); 0% selection rate when compared to white men (equal in 0%) | Black men most disadvantaged |
| **Document length** | Both resume length and corpus frequency of names significantly impacted selection | Confounds but doesn't explain away bias |
| **Intersectionality validated** | Confirmed three intersectionality hypotheses — bias compounds, not just adds | Supports intersectional theory empirically |

### Limitations
- Relies on publicly available resumes (may not represent all occupations)
- Focuses on embedding-based retrieval, not all LLM screening methods
- Limited to 3 embedding models (no GPT, Claude, or chat-based models)
- Synthetic name augmentation (not actual diverse candidate data)

### Relation to Our Project
- **Closest methodological precedent.** Same approach: vary demographic cues, hold qualifications constant, measure differential outcomes.
- We extend this to **housing** (not hiring) and use **Claude/GPT** (not just embedding models).
- We test **mitigation strategies** (which Wilson & Caliskan do not).
- Provides baseline bias magnitudes we can compare against.

---

## 2. An et al. (2025) — Cross-Model LLM Hiring Bias

**Title:** LLMs Reproduce Human Hiring Biases: Demonstrating Disparate Impact with Resume Names

**Authors:** J. An et al.

**Venue:** PNAS Nexus, 4(3), pgaf089 (2025)

**Link:** [academic.oup.com/pnasnexus](https://academic.oup.com/pnasnexus/article/4/3/pgaf089/8071848)

### Research Question
Do state-of-the-art LLMs exhibit the same hiring biases as human recruiters, and do these vary across models?

### Methodology
- **Sample:** ~361,000 fictitious resumes with randomized work experience, education, and skill sets
- **Demographic cue:** Each resume assigned a gender- and race-distinctive name
- **Models tested:** 5 LLMs
  - GPT-3.5 Turbo
  - GPT-4o
  - Gemini 1.5 Flash
  - **Claude 3.5 Sonnet** (same model family as ours)
  - Llama 3-70b
- **Task:** LLM scores each resume 0-100
- **Analysis:** Regression with controls for resume characteristics, position, state, and title fixed effects

### Key Findings (GPT-3.5 Turbo, baseline = white male)
| Group | Score Difference | Significance |
|-------|------------------|--------------|
| Female candidates | +0.452 | p < 0.001 |
| Black candidates | -0.074 | p < 0.1 (marginal) |
| Black females | +0.379 | — |
| White females | +0.223 | — |
| Black males | **-0.303** | p < 0.001 |
| **Practical impact** | Translates to 1.4-1.7 pp difference in hiring probability at 80-point threshold | — |

### Intersectional Effects
- **All 5 models** favored both Black and white female candidates
- **4 of 5 models** (all except Llama 3-70b) penalized Black males
- Pattern aligns with intersectionality theory: racial gap differs by gender (-0.303 for males, +0.156 for females)
- Consistent across positions, states (Dem/Rep), resume quality, and 500 resamples

### Limitations
- Resume scoring is just one decision stage (not end-to-end hiring)
- "Fairness through awareness" vs. "fairness through blindness" not deeply explored
- Names carry multiple confounds (class, age, region)

### Relation to Our Project
- **Most directly comparable to our work** — same model family (Claude) tested in similar setting
- Provides expected bias direction (Black males penalized, females slightly favored) we can compare with our tenant screening results
- Differences between housing and hiring will be informative — does Claude show the same bias in tenant context?
- We add: mitigation strategies, more intersectional combinations, justification analysis

---

## 3. SafeRent Settlement (2024) — Real-World Housing AI Discrimination

**Title:** CFPB and DOJ Settlement with SafeRent Solutions for Algorithmic Tenant Screening Discrimination

**Parties:** U.S. Consumer Financial Protection Bureau, U.S. Department of Justice, SafeRent Solutions

**Date:** 2024

**Settlement amount:** $2.275 million

**Links:**
- [CFPB press release](https://www.consumerfinance.gov/about-us/newsroom/cfpb-and-doj-settle-with-saferent-solutions-for-algorithmic-tenant-screening-discrimination/)
- [DOJ press release](https://www.justice.gov/opa/pr/justice-and-cfpb-settle-saferent-solutions-algorithmic-tenant-screening-discrimination)
- [ACLU coverage](https://www.aclu.org/news/civil-liberties/landmark-settlement-ai-tenant-screening-company-saferent)

### What Happened
SafeRent provided tenant screening scores used by landlords across the US. Their algorithm:
- Assigned "risk scores" to rental applicants
- **Systematically assigned higher risk scores to Black applicants** compared to white applicants with identical rental histories
- Used a "criminal record proxy" that effectively encoded racial disparities
- Result: Black applicants were denied housing at higher rates than equally-qualified white applicants

### The Evidence
- Internal testing by SafeRent itself revealed the disparate impact on Black applicants
- The algorithm was "secretly" deployed despite internal concerns
- Affected thousands of applicants

### Settlement Terms
- **$2.275M total** ($750K to CFPB civil penalty, $1.525M to affected consumers as restitution)
- Required algorithmic audit and deletion of the biased model
- Independent oversight of future AI tools
- First federal enforcement action against AI tenant screening under the Fair Housing Act

### Legal Significance
- First case establishing that algorithmic tenant screening can violate the Fair Housing Act even without explicit discriminatory intent ("disparate impact" doctrine)
- Sets precedent for landlord liability when using biased third-party AI tools
- Demonstrates the gap between vendor claims of "objective AI" and actual discriminatory outcomes

### Relation to Our Project
- **Establishes the stakes.** This is not theoretical — biased housing AI has already caused real harm at scale.
- Provides legal context (Fair Housing Act) for why bias detection in tenant screening is urgent.
- Demonstrates the type of bias we should be looking for (race-based score differences with no legitimate basis).
- Shows that "vendor says it's objective" is not a defense — actual disparate impact is what matters.

---

## 4. HUD AI & Fair Housing Guidance (2024)

**Title:** Artificial Intelligence and the Fair Housing Act

**Issuer:** U.S. Department of Housing and Urban Development (HUD)

**Date:** 2024 (formal guidance)

**Link:** [hud.gov](https://www.hud.gov/press/press_releases_media_advisories/hud_no_24-237)

### Key Points
- HUD formally stated that AI/algorithmic tools used in housing decisions are subject to the Fair Housing Act
- **Discriminatory effects — even if unintentional — violate the law**
- Covers advertising, tenant screening, underwriting, appraisals, and all housing-related AI use
- "Algorithmic tenant screening" specifically called out as high-risk

### What It Requires
- Landlords and housing providers must ensure AI tools don't have discriminatory effects, even if the AI claims to be "objective"
- Vendors of AI tools used in housing must comply with the Fair Housing Act
- Disparate impact analysis is required when using AI for housing decisions

### Why It Matters for This Project
- Establishes the **regulatory urgency** of bias detection in tenant screening LLMs
- LLM-based tenant screening (our focus) would be subject to the same scrutiny
- Provides motivation for the project: we need to know *whether* current LLMs comply before deployment

### Relation to Our Project
- Frames our work as proactive compliance / risk assessment
- The SafeRent case is the enforcement mechanism; HUD guidance is the rule book
- Our project can serve as a template for how organizations might audit LLM tools before deployment

---

## 5. Caliskan, Bryson & Narayanan (2017) — Foundation of Name-Based Bias

**Title:** Semantics Derived Automatically from Language Corpora Contain Human-Like Biases

**Authors:** Aylin Caliskan, Joanna J. Bryson, Arvind Narayanan

**Venue:** Science, 356(6334), 183-186 (2017)

**Link:** [science.org](https://www.science.org/doi/10.1126/science.org)

### Research Question
Do word embeddings (the basis of modern NLP) inherit human-like biases from training data?

### Methodology
- Applied Implicit Association Test (IAT) methodology to GloVe word embeddings
- Used standard name sets: European-American names vs. African-American names; female vs. male names
- Measured bias via cosine similarity in embedding space
- Compared against human IAT results

### Key Findings
- Word embeddings contain **human-like implicit biases** measurable via standard IAT methodology
- African-American names associated with unpleasant words more than European-American names
- Female names associated with family more than career; male names the reverse
- **Bias magnitude correlates with historical social biases** documented in IAT literature

### Significance
- First rigorous demonstration that AI systems inherit societal biases from training data
- Established the methodology of using names as demographic cues in bias audits
- Every paper after this (including ours) uses name-based audit methods derived from this work

### Relation to Our Project
- **Methodological foundation.** Our name pools come from the same tradition established here.
- Validates our experimental design: if bias exists in embeddings, it likely exists in the LLMs that use them.
- Provides the conceptual basis for why LLM bias in tenant screening is expected (not random noise).

---

## 6. Garg et al. (2018) — Quantifying Historical Bias

**Title:** Word Embeddings Quantify 100 Years of Gender and Ethnic Stereotypes

**Authors:** Nikhil Garg, Londa Schiebinger, Dan Jurafsky, James Zou

**Venue:** PNAS, 115(16), E3635-E3644 (2018)

**Link:** [pnas.org](https://www.pnas.org/doi/10.1073/pnas.1720347115)

### Research Question
Can historical biases be quantitatively tracked over time using word embeddings?

### Methodology
- Used Google Books corpus (1900-2000) to track changes in gender and ethnic stereotypes in language
- Validated ethnic name pools for bias audits
- Applied regression to embedding shifts over time

### Key Findings
- Historical biases **change measurably** over time in word embeddings
- Women's names became more associated with career words over the 20th century
- Ethnic stereotypes show documented patterns
- Validated a standard set of names for use in NLP bias audits

### Relation to Our Project
- Provides the **ethnic name pools** we use (validated for bias audit use)
- Validates the temporal assumption: biases in LLMs reflect documented historical patterns
- Methodological reference for systematic, reproducible name selection in bias audits

---

## Summary Table

| Paper | Year | Domain | Models | Key contribution |
|-------|------|--------|--------|------------------|
| Wilson & Caliskan | 2024 | Hiring | MTE embeddings | Closest precedent; methodology validation |
| An et al. | 2025 | Hiring | 5 LLMs (incl. Claude) | Cross-model comparison; Claude-specific results |
| SafeRent case | 2024 | Housing | Algorithmic screening | Real-world legal precedent |
| HUD guidance | 2024 | Housing | All AI | Regulatory framework |
| Caliskan et al. | 2017 | General | Word embeddings | Foundational methodology |
| Garg et al. | 2018 | General | Word embeddings | Validated name pools; temporal analysis |

---

## How These Papers Inform Our Project

**Methodologically:**
- We adopt Caliskan/Garg's name-based audit approach
- We replicate Wilson & Caliskan's controlled profile design
- We extend An et al.'s multi-model comparison to housing context

**Substantively:**
- SafeRent + HUD establish that housing AI bias has real consequences
- An et al. (Claude results) give us a prior for what to expect
- Wilson & Caliskan's intersectional findings motivate our RQ3

**Gaps our project addresses:**
- Limited work on **LLM bias in housing** specifically (most bias research focuses on hiring)
- Most studies don't test **mitigation strategies** (we do, for RQ4)
- Most studies don't analyze **justification language** (we do)
