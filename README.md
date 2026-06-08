# Tenant Bias Audit: LLM Fairness Evaluation

## Project Goal
Investigate whether LLMs (Claude, GPT-4, etc.) show gender, racial, or intersectional bias when evaluating rental tenant applicants with otherwise identical qualifications.

## Research Questions
- **RQ1:** Does gender affect tenant evaluation?
- **RQ2:** Does race/ethnicity/nationality affect evaluation?
- **RQ3:** Are there intersectional effects (gender × race)?
- **RQ4:** Which mitigation strategies reduce bias?

## Project Structure
```
tenant_bias_project/
├── data/                    # Generated profiles
├── results/                 # LLM outputs (parsed)
├── scripts/                 # Pipeline code
│   ├── generate_profiles.py
│   ├── run_experiment.py
│   ├── parse_responses.py
│   ├── analyze_results.py
│   ├── generate_mock_data.py
│   └── test_pipeline.py
├── dashboard/               # Interactive Streamlit dashboard
│   └── app.py
├── analysis/
├── requirements.txt
├── .env.example
└── README.md
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up API key
```bash
copy .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 3. Run the pipeline

**Option A: With real LLM API (requires API key)**
```bash
python scripts/generate_profiles.py    # Generate 50 profile sets
python scripts/run_experiment.py        # Call Claude API
python scripts/parse_responses.py       # Parse responses
python scripts/analyze_results.py       # Statistical analysis
```

**Option B: With mock data (for testing, no API key needed)**
```bash
python scripts/generate_profiles.py
python scripts/generate_mock_data.py    # Generates mock LLM responses
python scripts/analyze_results.py
```

### 4. Launch the dashboard
```bash
streamlit run dashboard/app.py
```
Opens at http://localhost:8501

## Dashboard Pages
- **Overview** - Project summary, key metrics
- **Methodology** - Experimental design details
- **Results Dashboard** - Interactive charts (gender/ethnicity/intersectional)
- **Statistical Tests** - Mann-Whitney U, Kruskal-Wallis, ANOVA outputs
- **Justification Analysis** - Detects linguistic bias in LLM explanations
- **About** - References and tech stack

## Requirements
- Python 3.8+
- ~$10-40 in API credits (real experiment) or $0 (mock mode)
- No GPU needed
