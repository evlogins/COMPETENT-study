# CLAUDE.md — COMPETENT Study

## Project Overview

This is the **COMPETENT** study: a retrospective single-center cohort comparing **sevoflurane** (inhalation) vs **etomidate** (IV) for anaesthesia induction in high-risk patients undergoing interventional mitral valve repair.

- **Primary outcome**: Vasopressor requirement (cafedrine-theodrenaline use)
- **n=57** patients: 32 sevoflurane, 25 etomidate
- **Key finding**: Etomidate group required significantly more vasopressors (OR ~10.5, P < 0.001)

## Repository Structure

```
COMPETENT-study/
├── data/                    # Raw data (Excel)
│   └── VIMAMC_MH.xlsx      # Main dataset (57 patients, 84 columns)
├── manuscripts/             # Manuscript drafts
│   └── 2024_04_13_COMPETENT_final.docx
├── reviews/                 # Journal rejection letters & reviewer feedback
│   ├── Acta_Anaesthesiologica_Scandinavica.txt
│   └── JCVA.txt
├── analysis/                # Statistical analysis scripts
│   ├── COMPETENT_Analysis_v3.py         # Local version (run this)
│   └── COMPETENT_Analysis_v3_colab.py   # Original Google Colab version
├── figures/                 # Generated publication-ready figures
├── requirements.txt
├── CLAUDE.md                # This file
└── README.md
```

## Data Details

**Excel file**: `data/VIMAMC_MH.xlsx`, sheet `Endpunkttabelle`

### Key columns:
- `Narkose`: 1 = Sevoflurane, 0 = Etomidate
- `Alter`, `Gewicht`, `Größe`, `BMI`: Demographics
- `LVF`: Left ventricular function (1=<30%, 2=30-44%, 3=45-54%, 4=≥55%)
- `NYHA:`: NYHA class
- `Niereninsuff.`: Renal insufficiency stage
- `Z.n. kard Dekomp`: History of cardiac decompensation
- `VHF`: Atrial fibrillation

### Hemodynamic timepoints (T0–T4, every 5 min):
- **Known column naming issue**: `TO_MAP`/`TO_Hf` (letter O, not zero) and `T!_MAP`/`T!_Hf` (exclamation, not one)
- Systolic: `T0_syst`, `T1_syst`, `T2_sys`, `T3_syst`, `T4_syst`
- Diastolic: `T0_diast`, `T1_diast`, `T2_diast`, `T3_diast`, `T4_diast`
- MAP: `TO_MAP`, `T!_MAP`, `T2_MAP`, `T3_MAP`, `T4_MAP`
- HR: `TO_Hf`, `T!_Hf`, `T2_Hf`, `T3_Hf`, `T4_Hf`

### Medications:
- `Akrinor`: Cafedrine-theodrenaline dose (ml)
- `Noradrenalin`: Noradrenaline dose (µg)
- `Dobutamin`: Dobutamine dose (µg)
- `Remifentanil`: Remifentanil dose (µg)
- `Etomidat`, `Sevofluran`: Induction agent doses

## Running the Analysis

```bash
pip install -r requirements.txt
python analysis/COMPETENT_Analysis_v3.py
```

Figures are saved to `figures/`.

## Reviewer Concerns to Address

### From Acta Anaesthesiologica Scandinavica:
1. Title misleading (drugs vs drug classes)
2. Retrospective/uncontrolled design
3. Concurrent opioid administration (remifentanil confounding)
4. Blood pressure comparison choices not well motivated
5. Need for prospective RCT

### From JCVA:
1. Small sample size
2. Mitral insufficiency not truly "high risk"
3. Selection bias in retrospective design
4. Clinical relevance of vasopressor difference questioned
5. Missing details (patient size, sleep apnea, induction duration)
6. Etomidate dosing not standardized
7. Sevoflurane < 1 MAC
8. Hypotension may be from remifentanil, not induction agent

## Target Journals (PubMed indexed, IF > 4.0)

Priority targets identified:
1. **Anaesthesia** (IF ~6.9) — Best fit, fast turnaround
2. **British Journal of Anaesthesia** (IF 9.2) — Competitive but excellent
3. **European Journal of Anaesthesiology** — Good European option

## Key Analysis Features (v3)

- Median (IQR) for zero-inflated outcomes
- Rank-biserial r effect sizes (not Cohen's d)
- NNH with Newcombe 95% CI
- 5 logistic regression models (unadjusted through fully adjusted)
- IPTW propensity score analysis with 7 covariates
- Bootstrap CI for weighted risk difference
- Subgroup analyses with interaction tests (LVEF, age)
- VIF diagnostics, ROC-AUC
- Hierarchical multiple testing framework
