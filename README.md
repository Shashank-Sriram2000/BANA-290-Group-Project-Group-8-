# BANA 290 Group 8 — Stack Overflow Data Pipeline
## Research: ChatGPT's Causal Effect on SO Usage (ITS + DiD)

---

## Quick Start

```bash
# 1. Install dependencies
pip install requests pandas tqdm python-dotenv statsmodels matplotlib seaborn

# 2. (Strongly recommended) Register a free app at https://stackapps.com
#    → copy your "Key" and paste it below
echo "SE_API_KEY=your_key_here" > .env

# 3. Collect data
python so_data_collector.py

# 4. Run analysis
python so_analysis.py
```

---

## File Overview

| File | Purpose |
|------|---------|
| `so_data_collector.py` | Pulls data from Stack Exchange API v2.3, saves `data/so_panel.csv` |
| `so_analysis.py` | EDA plots + ITS and DiD regressions |
| `data/so_panel.csv` | Main panel dataset (one row per tag × time window) |
| `data/survey_ai_adoption.csv` | SO Developer Survey AI adoption scaffold (fill manually) |
| `cache/` | Per-window JSON cache — re-running skips already-fetched windows |
| `plots/` | Auto-generated PNG figures |

---

## Dataset Schema (`so_panel.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `tag` | str | Stack Overflow tag |
| `date` | date | Window start date |
| `question_count` | int | New questions in window |
| `answered_rate` | float | Share with ≥1 answer |
| `accepted_answer_rate` | float | Share with accepted answer |
| `avg_score` | float | Mean question score |
| `avg_view_count` | float | Mean view count |
| `group` | str | `"treatment"` or `"control"` |
| `time_index` | int | t — linear trend counter |
| `post_intervention` | int | D = 1 if date ≥ 2022-11-30 |
| `time_since_intervention` | int | (t − T₀) for slope change |
| `is_treatment` | int | 1 for AI-adjacent tags |
| `did_interaction` | int | post × is_treatment |

---

## Tags

### Treatment (AI-adjacent) — expected high substitution
`python`, `machine-learning`, `deep-learning`, `nlp`, `pytorch`,
`tensorflow`, `keras`, `scikit-learn`, `openai-api`, `langchain`

### Control (AI-insulated) — expected low substitution
`sql`, `excel`, `vba`, `bash`, `r`, `powershell`, `sap`, `oracle`,
`ms-access`, `tableau`

---

## API Quota

| Access level | Daily quota | Est. days to collect (weekly granularity) |
|---|---|---|
| Anonymous | 300 req/day | ~7 days |
| **Registered app key** | **10,000 req/day** | **< 1 day** ← recommended |

Register at: https://stackapps.com/apps/oauth/register  
(No OAuth needed — just register to get a key.)

---

## ITS Model

```
log(y_t) = β0 + β1·t + β2·D_t + β3·(t−T₀)·D_t + ε_t

β1 = pre-intervention trend
β2 = immediate level shift at intervention
β3 = change in slope after intervention
```

Standard errors: Newey-West HAC (maxlags = 4)

## DiD Model

```
log(q_it) = α + β1·Treatment_i + β2·Post_t + β3·(Treatment×Post) + γ_i + ε_it

β3 = DiD estimator (ATT)
γ_i = tag fixed effects
```

Standard errors: clustered by tag

---

## Supplementary Data Sources

| Source | Use |
|--------|-----|
| [SO Survey 2022](https://survey.stackoverflow.co/2022/#technology) | Pre-intervention AI tool adoption baseline |
| [SO Survey 2023](https://survey.stackoverflow.co/2023/#technology) | First post-launch survey |
| [SO Survey 2024](https://survey.stackoverflow.co/2024/technology) | Year 2 post-launch |
| [SO Survey 2025](https://survey.stackoverflow.co/2025/technology) | Most recent; check for new AI section |


---

## Notes for the Paper

- **Parallel trends assumption**: Plot pre-period trends for treatment vs control tags
  to show they moved together before Nov 2022. This is your key identifying assumption.
- **Placebo test**: Re-run ITS with a fake intervention date (e.g., Nov 2021) and
  confirm β2/β3 are not significant.
- **Heterogeneity**: Run tag-level regressions to show AI-adjacent tags had larger
  declines than SQL/Excel tags.
- **answer_acceptance_rate** is arguably more interesting than question count for
  "long-term viability" — if fewer answers are being accepted, community quality
  is degrading even if volume looks ok.
