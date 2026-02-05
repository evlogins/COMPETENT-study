# COMPETENT Study

**Comparison between inhalation and intravenous induction of anaesthesia during interventional mitral valve repair**

## Overview

Retrospective single-center cohort study comparing sevoflurane (inhalation) vs etomidate (IV) induction in 57 ASA IV patients undergoing interventional mitral valve repair at Heidelberg University Hospital (2019–2020).

**Trial Registration**: [NCT04865614](https://clinicaltrials.gov/study/NCT04865614)

## Quick Start

```bash
pip install -r requirements.txt
python analysis/COMPETENT_Analysis_v3.py
```

## Key Results

- Etomidate group required significantly more vasopressor support (OR 10.5, 95% CI 2.5–44.6, P < 0.001)
- Blood pressure and heart rate did not differ significantly between groups
- NNH = 2.1 (one additional patient needs vasopressors for every ~2 induced with etomidate vs sevoflurane)
- Effect remained robust after adjusting for remifentanil, age, BMI, and LV function

## Project Structure

```
├── data/           # VIMAMC_MH.xlsx (n=57, 84 variables)
├── manuscripts/    # COMPETENT manuscript (docx)
├── reviews/        # Journal reviewer feedback
├── analysis/       # Statistical analysis scripts (Python)
├── figures/        # Publication-ready figures (PNG, 300 dpi)
└── CLAUDE.md       # Context for Claude Code
```

## Ethics

Approved by Ethics Committee of Heidelberg University Medical Faculty (S-109/2021).
