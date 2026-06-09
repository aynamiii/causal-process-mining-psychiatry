# Causal Process Mining for Psychiatric Readmission
### AI-Augmented Root Cause Analysis for Deviations in Psychiatric Care Processes

Implementation of **Bozorgi et al. (ICPM 2020)** applied to OMOP CDM data for depression readmission analysis.

---

## Overview

This pipeline identifies which treatments causally reduce 30-day readmission in depression patients. It combines:
- **Action rule mining** — finds which treatment changes are associated with improved outcomes under specific patient conditions
- **Uplift modelling**  — estimates the average treatment effect (ATE) of each candidate intervention after controlling for confounders

All processing runs locally. No patient data leaves the machine.

---

## Input Data

Seven OMOP CDM tables are required as CSV files placed in the `data/` folder.

### Tables and columns used

| Table | Columns used | Purpose in pipeline |
|---|---|---|
| `person` | `person_id`, `year_of_birth`, `gender_concept_id` | Patient age and gender |
| `visit_occurrence` | `visit_occurrence_id`, `person_id`, `visit_start_date`, `visit_end_date`, `visit_concept_id` | Hospital visits — unit of analysis |
| `condition_occurrence` | `person_id`, `condition_concept_id`, `condition_start_date`, `condition_source_value` | Depression diagnosis (cohort selection) and comorbidities (CCI) |
| `drug_exposure` | `person_id`, `drug_concept_id`, `drug_exposure_start_date`, `visit_occurrence_id` | Antidepressant prescriptions (cohort filter + drug features) |
| `procedure_occurrence` | `person_id`, `procedure_concept_id`, `procedure_date`, `visit_occurrence_id` | Therapy procedures (CBT, IPT, group therapy) |
| `concept` | `concept_id`, `concept_name` | Translates concept IDs to readable names at runtime |
| `concept_ancestor` | `ancestor_concept_id`, `descendant_concept_id` | Expands SNOMED hierarchies to capture all subtypes |

---

## Cohort Definition

The analysis is restricted to **incident depression cases** who are **treatment-naive**. The following filters are applied in order:

### Filter 1 — Depression diagnosis
`condition_occurrence.condition_concept_id` is matched against 22 depression SNOMED ancestor concept IDs (configured in `config.py`). The `concept_ancestor` table expands each ancestor to all its SNOMED descendants, capturing every specific subtype of depression.

Each patient's **earliest** depression diagnosis date is taken as their index date.

### Filter 2 — 365-day lookback
The patient's first ever record in the database (earliest of any `visit_start_date` or `condition_start_date`) must be at least **365 days before** their first depression diagnosis. This confirms the diagnosis is genuinely incident — not a carry-over from before the observation window.

### Filter 3 — Treatment-naive
Antidepressant concept IDs (SSRIs, SNRIs, Tricyclics, MAOIs, Atypicals, Augmentation agents — configured in `config.py`) are expanded via `concept_ancestor`. Patients who have any antidepressant recorded in `drug_exposure` **more than 30 days before** their depression diagnosis are excluded. Patients starting antidepressants within 30 days of diagnosis are kept, as this represents the same treatment episode.

### Filter 4 — Post-diagnosis visits only
Only visits on or after each patient's first depression diagnosis date are retained for feature construction and outcome labelling.

---

## Feature Construction

One row per visit. Features split into two categories used by the action rule algorithm.

### Stable attributes — cannot be changed by a clinician
These describe the patient's state at the time of the visit and are used as fixed conditions in rules.

| Feature | OMOP source | Construction |
|---|---|---|
| `age` | `person.year_of_birth` + `visit_start_date` | Visit year minus birth year |
| `gender` | `person.gender_concept_id` | OMOP standard codes: 8507 = M, 8532 = F |
| `visit_type` | `visit_occurrence.visit_concept_id` | Name looked up from `concept.concept_name` at runtime |
| `los_days` | `visit_start_date`, `visit_end_date` | Length of stay in days (clipped at 0) |
| `cci_score` | `condition_occurrence.condition_source_value` | Charlson Comorbidity Index computed from raw ICD-10 source codes using Quan et al. (2011) weights — covers 17 condition categories weighted 1–6 |

For action rule mining, continuous variables are binned:
- `age` → `age_lt40 / age_40_54 / age_55_64 / age_65_74 / age_gte75`
- `cci_score` → `none (0) / mild (1–2) / moderate (3–4) / severe (5+)`
- `los_days` → `same_day / short (1–3d) / medium (4–7d) / long (7d+)`

### Flexible attributes — the treatment candidates
Binary (0 or 1) per visit. These are the interventions the pipeline evaluates causally.

| Feature | OMOP source | Construction |
|---|---|---|
| `drug_<name>` × 10 | `drug_exposure.drug_concept_id` | Top-10 most frequently prescribed drugs across all cohort visits. Drug name resolved from `concept.concept_name`. One binary column per drug (1 = prescribed during visit). |
| `therapy_cbt` | `procedure_occurrence.procedure_concept_id` | 1 if any procedure in the visit descends from SNOMED 228557008 (Cognitive behavioural therapy) via `concept_ancestor` |
| `therapy_ipt` | `procedure_occurrence.procedure_concept_id` | 1 if any procedure descends from SNOMED 443730003 (Interpersonal psychotherapy) |
| `therapy_group` | `procedure_occurrence.procedure_concept_id` | 1 if any procedure descends from SNOMED 76168009 (Group psychotherapy) |

---

## Outcome

**30-day readmission** — binary label per visit.

For each visit, the pipeline looks at the same patient's next visit. If that next visit starts within 30 days of the current visit's end date, `readmitted = 1`, otherwise `0`. The last visit per patient is always labelled `0`.

---

## Pipeline Steps

```
1. Load data           7 OMOP CSV tables loaded into memory
2. Build cohort        Incident depression cases, treatment-naive, 365-day lookback
3. Build features      Demographics, CCI, top-10 drug indicators, therapy indicators
4. Label outcome       30-day readmission flag per visit
5. Action rule mining  Finds rules: (stable conditions, treatment change) -> outcome change
6. Extract treatments  Top-5 treatment candidates ranked by rule frequency
7. Uplift modelling    Estimates causal ATE per treatment (controls for age, gender, LOS, CCI, visit type)
8. Causal rules        Joins action rules with ATE scores, retains rules where ATE > 0
9. Evaluate            Observed readmission rates: treated vs control per treatment
```

---

## Outputs

All saved to `outputs/`.

| File | Contents |
|---|---|
| `action_rules.txt` | All action rules with support and confidence values |
| `uplift_summary.csv` | ATE per treatment, sorted best to worst |
| `causal_rules.csv` | Final recommendations — rules where ATE > 0 |
| `rule_evaluation.csv` | Raw observed readmission rates: treated vs control |
| `uplift_scores.png` | Bar chart of ATE values |

---

## Configuration

All parameters in `config.py`.

| Parameter | Default | Meaning |
|---|---|---|
| `COHORT_LOOKBACK_DAYS` | 365 | Minimum history before depression diagnosis |
| `TREATMENT_NAIVE_GRACE_DAYS` | 30 | Antidepressants within this window of diagnosis are allowed |
| `READMISSION_DAYS` | 30 | Readmission window in days |
| `TOP_N_DRUGS` | 10 | Number of most frequent drugs to include as features |
| `AR_MIN_SUPPORT` | 10 | Minimum visits a rule must cover |
| `AR_MIN_CONFIDENCE` | 0.6 | Minimum confidence threshold for action rules |
| `UPLIFT_MAX_DEPTH` | 4 | Maximum depth of uplift decision tree |
| `UPLIFT_MIN_LEAF` | 10 | Minimum visits per leaf node in uplift tree |

---

## How to Run

```powershell
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python main.py
```

## Reference

```bibtex
  author    = {Bozorgi, Zahra Dasht; Teinemaa, Irene; Dumas, Marlon;
               La Rosa, Marcello; Polyvyanyy, Artem},
  title     = {Process Mining Meets Causal Machine Learning:
               Discovering Causal Rules from Event Logs},
  year      = {2020},
  doi       = {10.1109/ICPM49681.2020.00028}
```
