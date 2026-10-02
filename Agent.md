# CycleInsight Project Rules

## Project Goal

CycleInsight is a menstrual intelligence system focused on:

1. Exploratory Data Analysis
2. User behavior understanding
3. Menstrual cycle understanding
4. Future prediction models
5. Educational insights for new users

Current phase:
EDA only.
Do not build ML models unless explicitly requested.

---

## Code Style

Write code like a good university student.

Avoid:
- overly clever code
- excessive one-liners
- unnecessary abstractions
- enterprise-style complexity

Prefer:
- readable variable names
- clear comments
- logical notebook sections
- moderate function usage

---

## Notebook Structure

Every notebook should contain:

1. Imports
2. Configuration
3. Data Loading
4. Helper Functions
5. Analysis
6. Visualizations
7. Findings

Avoid duplicate imports and duplicate data loading.

---

## Visualizations

All figures must be saved to:

reports/figures/

Every figure must have:
- title
- axis labels
- readable size

---

## Documentation

Every EDA phase must end with:

- Key Findings
- Data Quality Notes
- Recommended Next Step

---

## Analysis Principles

Do not make medical claims.

Do not make causal claims.

Describe observations only.

Clearly separate:
- facts from data
- interpretations
- assumptions

---

## Repository Structure

datasets/
notebooks/
reports/
reports/figures/
docs/
docs/audits/
src/

Keep files organized.