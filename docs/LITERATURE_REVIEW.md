# Literature Review: LLM Bias in Tenant and Employment Screening

**Author:** [Your Name]
**Date:** June 2026
**Word count:** ~1,800

---

## 1. Introduction

Large language models (LLMs) are increasingly deployed in high-stakes decision-support systems across socially sensitive domains: hiring, lending, insurance, criminal justice, and housing. While these systems promise efficiency and objectivity, mounting evidence suggests they reproduce and sometimes amplify societal biases. This literature review surveys the empirical literature on LLM bias in screening decisions, with a focus on the methodologies used to detect bias, the specific bias patterns that have been documented, and the gap in the literature that the present project addresses.

## 2. Foundational Evidence: Bias in Language Representations

The study of bias in language technologies predates the current wave of LLMs. **Caliskan, Bryson, and Narayanan (2017)** demonstrated that GloVe word embeddings — the distributional representations underlying most modern NLP — contain measurable human-like biases when subjected to the Implicit Association Test. Names associated with African Americans were correlated with unpleasant concepts more than European American names; female names were correlated with family more than career. This work established the foundational methodology for subsequent bias audits: using name lists as demographic cues, with all other variables held constant, to measure differential treatment in language models.

**Garg, Schiebinger, Jurafsky, and Zou (2018)** extended this line of work by quantifying how biases in word embeddings have changed over the 20th century using the Google Books corpus. They validated standardized sets of names for use in bias audits and provided evidence that historical biases are systematically encoded in language statistics. The present project adopts these validated name pools for demographic cue variation.

These foundational studies establish three claims that motivate subsequent work: (1) language models encode measurable biases; (2) the biases reflect documented historical and social patterns; (3) the biases can be measured with controlled, reproducible audit methods.

## 3. Bias in LLM-Based Hiring Decisions

The most direct precedents for the present project are studies of bias in LLM-based resume and hiring systems.

**Wilson and Caliskan (2024)** conducted a resume audit study simulating LLM-mediated candidate selection. They used 554 resumes augmented with 80 names signaling four intersectional groups (Black women, Black men, white women, white men) and 571 job descriptions across 9 occupations. Testing 3 Massive Text Embedding models (E5-Mistral-7b-Instruct, GritLM-7B, SFR-Embedding-Mistral), they found stark disparities: white-associated names were preferred 85.1% of the time, while Black-associated names were preferred only 8.6%. Critically, they documented intersectional effects: Black men were selected less often than Black women (14.8% selection rate; equal in only 18.5% of tests), and 0% of the time when compared to white men. The study validated three intersectionality hypotheses, providing empirical support for the theoretical claim that bias compounds across demographic dimensions.

**An et al. (2025)** extended this work to state-of-the-art chat-based LLMs, testing 5 models (GPT-3.5 Turbo, GPT-4o, Gemini 1.5 Flash, Claude 3.5 Sonnet, and Llama 3-70b) on approximately 361,000 synthetic resumes. All 5 models favored female candidates overall, while 4 of 5 (all except Llama 3-70b) penalized Black male candidates. The racial gap differed by gender: -0.303 points for Black males vs. white males, but +0.156 points for Black females vs. white females — direct evidence of intersectional effects. An et al. found that the biases were consistent across job positions, US states (Democrat- and Republican-leaning), resume quality levels, and 500 bootstrap resamples, suggesting the patterns are robust.

These two studies establish that (a) LLM bias in employment contexts is empirically robust across model families, (b) intersectional effects exist beyond what single-axis analyses would predict, and (c) the bias patterns align with documented real-world employment disparities, particularly the disadvantaging of Black men.

## 4. The Gap: Tenant Screening

A striking gap in the literature is the relative scarcity of work on LLM bias in **housing and tenant screening**, despite this being a domain where automated decision-making is already widespread and where discrimination has legal consequences.

The most prominent real-world case is the **2024 SafeRent settlement** (CFPB and DOJ v. SafeRent Solutions), in which a tenant screening AI company agreed to pay $2.275 million for algorithmic discrimination. SafeRent's system assigned higher risk scores to Black applicants than to white applicants with identical rental histories, leading to disproportionate housing denials. The settlement was the first federal enforcement action against AI tenant screening under the Fair Housing Act, and required algorithmic audit, deletion of the biased model, and ongoing independent oversight.

The U.S. Department of Housing and Urban Development responded in 2024 with **formal guidance** stating that AI and algorithmic tools used in housing decisions are subject to the Fair Housing Act, and that discriminatory effects — even if unintentional — violate the law. The guidance specifically calls out tenant screening as a high-risk application.

Yet despite the regulatory urgency and the existence of real harm, the empirical literature on LLM bias in tenant screening is sparse. The present project addresses this gap directly.

## 5. Mitigation Strategies

A separate strand of the literature examines methods to reduce bias in LLM outputs. The most common interventions are prompt-based:

- **Explicit fairness instructions:** Prepending prompts with statements like "Do not consider race, gender, or national origin."
- **Role prompts:** Instructing the model to adopt a persona committed to fairness, e.g., "You are a fair, unbiased landlord."
- **Chain-of-thought reasoning:** Asking the model to "think step by step" before making a decision.
- **Few-shot examples:** Providing examples of fair evaluations in the prompt.
- **System-prompt debiasing:** Placing bias-reduction instructions in the system message rather than the user message.

**Wilson and Caliskan (2024)** and **An et al. (2025)** did not systematically test mitigation strategies, leaving open the question of how effective these interventions are. The present project addresses this gap with a comparative test of 4 mitigation strategies for RQ4.

## 6. Methodological Considerations

The literature has converged on a common methodology for bias audits: generate synthetic profiles with controlled demographic cues, hold all other variables constant, present the profile set to the model, and measure differential outputs (rankings, scores, language). This is the methodology adopted here.

Key methodological decisions include:
- **Sample size:** Wilson & Caliskan used ~40,000 comparisons; An et al. used ~361,000. The present project uses 3,000 evaluations — sufficient for statistical inference on the targeted effects while remaining tractable.
- **Statistical tests:** Non-parametric tests (Mann-Whitney U, Kruskal-Wallis) are appropriate for ordinal scores; ANOVA for interaction effects. The present project uses these standard tests plus effect-size reporting.
- **Name pools:** Validated sets from Caliskan (2017) and Garg (2018) are used to ensure demographic cues are realistic and reproducible.
- **Temperature:** Using temperature > 0 (we use 0.7) allows estimation of variance from multiple runs per profile set.

## 7. Linguistic Bias: A Missing Dimension

Most bias audits focus on numerical outputs (scores, rankings). A less-explored dimension is the **language** the model uses to justify its decisions. Even when scores are statistically similar, the model may use different adjectives, hedging language, or risk framings across demographic groups — a form of "linguistic bias" or "micro-inequity" that has been documented in human decision-making.

**An et al. (2025)** noted but did not systematically analyze justification language. The present project addresses this with a lexicon-based analysis of positive, hedging, and negative words across demographic groups (RQ4 in spirit, plus the justification analysis page in the dashboard).

## 8. Research Questions Addressed by the Present Project

Synthesizing the literature, the present project addresses the following:

- **RQ1:** Does gender bias exist in LLM tenant evaluation? (Extends Wilson & Caliskan, An et al. to housing)
- **RQ2:** Does racial/ethnic bias exist? (Same, with broader ethnic categories)
- **RQ3:** Are there intersectional effects? (Validates intersectionality hypotheses in a new domain)
- **RQ4:** Which mitigation strategies are effective? (Addresses gap left by Wilson & Caliskan and An et al.)

## 9. Conclusion

The literature on LLM bias in employment screening is mature and documents robust, intersectional disparities across model families. The literature on tenant screening is sparse despite the regulatory urgency and real-world harm demonstrated by the SafeRent case. The present project fills this gap by adapting established employment-screening audit methodologies to the housing domain, testing 4 mitigation strategies, and analyzing the language of LLM justifications in addition to numerical scores.

---

## References

1. An, J., et al. (2025). LLMs reproduce human hiring biases: Demonstrating disparate impact with resume names. *PNAS Nexus*, 4(3), pgaf089. https://academic.oup.com/pnasnexus/article/4/3/pgaf089/8071848

2. Bolukbasi, T., Chang, K.-W., Zou, J., Saligrama, V., & Kalai, A. (2016). Man is to computer programmer as woman is to homemaker? Debiasing word embeddings. *NeurIPS*. https://arxiv.org/abs/1607.06520

3. Caliskan, A., Bryson, J. J., & Narayanan, A. (2017). Semantics derived automatically from language corpora contain human-like biases. *Science*, 356(6334), 183-186. https://www.science.org/doi/10.1126/science.aal4230

4. Consumer Financial Protection Bureau & U.S. Department of Justice. (2024). CFPB and DOJ settle with SafeRent Solutions for algorithmic tenant screening discrimination. https://www.consumerfinance.gov/about-us/newsroom/cfpb-and-doj-settle-with-saferent-solutions-for-algorithmic-tenant-screening-discrimination/

5. Garg, N., Schiebinger, L., Jurafsky, D., & Zou, J. (2018). Word embeddings quantify 100 years of gender and ethnic stereotypes. *PNAS*, 115(16), E3635-E3644. https://www.pnas.org/doi/10.1073/pnas.1720347115

6. U.S. Department of Housing and Urban Development. (2024). HUD issues new guidance on artificial intelligence and the Fair Housing Act. https://www.hud.gov/press/press_releases_media_advisories/hud_no_24-237

7. Wilson, K., & Caliskan, A. (2024). Gender, race, and intersectional bias in resume screening via language model retrieval. *Proceedings of the 2024 AAAI/ACM Conference on AI, Ethics, and Society*. https://arxiv.org/abs/2407.20371

8. TechEquity Collaborative. (2025). Screened out of housing: How AI tenant screening harms renters. https://techequity.us/

9. ACLU. (2024). Landmark settlement with AI tenant screening company SafeRent. https://www.aclu.org/news/civil-liberties/landmark-settlement-ai-tenant-screening-company-saferent

10. Brennan Center for Justice. (2024). AI in rental housing: Emerging regulatory and enforcement actions. https://www.brennancenter.org/
