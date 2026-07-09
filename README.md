# AI-Assisted Literature Review Pipeline for Carbon and Fuel Tax Acceptability

## Overview

This repository contains an AI-assisted literature review pipeline designed to identify, screen, extract, verify, and harmonize evidence on the determinants of public acceptability, support, opposition, preferences, vote choice, willingness to pay, and attitudes toward carbon-tax and fuel-tax instruments.

The pipeline is deliberately strict in scope. It retains only academic papers that directly analyze public responses to a carbon tax, CO2 tax, fossil-fuel tax, gasoline tax, diesel tax, fuel excise tax, transport fuel tax, TICPE-like fiscal instrument, progressive carbon tax, or carbon tax combined with revenue recycling. Papers that discuss carbon pricing only in general terms, or that focus exclusively on emissions trading systems, cap-and-trade mechanisms, subsidies, standards, green spending, pro-environmental behavior, or general climate-policy support are excluded unless they also provide tax-specific evidence.

The objective is to construct a transparent, auditable, and reproducible evidence base for a literature review on the political economy and public acceptability of carbon and fuel taxation.

---

## Research Scope

### Included policy instruments

The review focuses on tax instruments that directly affect the price of carbon-intensive or fossil-fuel-based consumption and production. Included instruments are:

- carbon taxes;
- CO2 taxes;
- fossil-fuel taxes;
- gasoline, petrol, or diesel taxes;
- transport fuel excise taxes;
- energy taxes explicitly based on carbon content or fossil-fuel consumption;
- TICPE-like fiscal instruments;
- progressive carbon taxes;
- carbon taxes with revenue recycling, lump-sum transfers, dividends, earmarking, or compensatory mechanisms.

### Included outcomes

The pipeline extracts determinants associated with public responses to these instruments, including:

- acceptability;
- acceptance;
- support;
- opposition;
- vote choice;
- willingness to pay;
- stated policy preferences;
- tax-specific attitudes.

### Excluded policy domains

The pipeline excludes papers that analyze only:

- emissions trading systems;
- cap-and-trade schemes;
- carbon markets;
- emission permits;
- carbon pricing in general without a tax-specific result;
- subsidies or incentives;
- environmental regulations or standards;
- green public spending;
- willingness to donate or contribute without a tax instrument;
- pro-environmental behavior without tax acceptability;
- general climate-policy support without a separate carbon-tax or fuel-tax item.

If a paper studies both carbon taxes and other climate-policy instruments, the paper may be included, but the extraction stage retains only the tax-specific evidence.

---

## Pipeline Architecture

The pipeline is structured into four main stages.

### Step 0: PDF identification and upload

Academic PDFs are stored locally in the literature directory and uploaded to the OpenAI API when needed. A local file cache avoids re-uploading the same PDF across runs. Files are identified using their SHA-256 hash.

### Step 1: Screening

Each PDF is screened to determine whether it contains direct evidence on the public acceptability, support, opposition, preferences, vote choice, willingness to pay, or attitudes toward a carbon-tax or fuel-tax instrument.

The screening stage produces a structured JSON file containing:

- paper metadata;
- tax mentions;
- excluded policy mentions;
- inclusion or exclusion decision;
- confidence level;
- concise inclusion or exclusion rationale.

### Step 2: Extraction

For papers that pass screening, the pipeline extracts determinants directly associated with tax acceptability or support.

Each extracted determinant includes:

- the raw determinant name used in the paper;
- a standardized candidate label;
- a determinant family;
- a determinant role;
- the direction of the relationship with tax acceptability;
- a short definition;
- the implied mechanism;
- the evidence type;
- the tax instrument analyzed;
- the outcome variable;
- the acceptability dimension;
- a supporting quote or evidence excerpt;
- page or section information;
- statistical or result details when available;
- confidence level;
- inclusion status for the main review table.

### Step 3: Verification

The verification stage performs a lightweight audit of the extraction. It does not rewrite the full extraction JSON. Instead, it checks whether each determinant is supported by the paper and whether the paper should remain in the strict tax corpus.

The verification output includes:

- an overall verdict;
- whether the paper should remain in the corpus;
- metadata checks;
- determinant-level checks;
- potentially missing tax-specific determinants;
- a concise final recommendation.

This design avoids excessively long model outputs and reduces the risk of truncated JSON responses.

### Step 4: Table construction and harmonization

After all papers have been processed, the pipeline reloads all completed JSON files from disk and builds several review tables:

- paper-level table;
- long determinant-level table;
- screening decision table;
- verification audit table;
- pipeline status table.

The pipeline then uses a final harmonization step to group semantically similar determinant labels while preserving relevant conceptual distinctions, such as the difference between perceived effectiveness, trust, fairness, economic self-interest, ideology, and revenue recycling.

---

## Repository Structure

A recommended repository structure is:

```text
AI_Assisted_Literature_Revue/
│
├── code/
│   └── ai_literature_review_pipeline.py
│
├── data/
│   └── papers/
│       └── *.pdf
│
├── outputs/
│   ├── json_outputs/
│   │   ├── 00_screening/
│   │   ├── 01_extraction/
│   │   ├── 02_verification/
│   │   ├── 03_final/
│   │   └── 99_status/
│   │
│   └── tables/
│       ├── screening_decisions.csv
│       ├── screening_decisions.xlsx
│       ├── table_1_paper_level_tax_only.csv
│       ├── table_1_paper_level_tax_only.xlsx
│       ├── determinants_long_format_tax_only.csv
│       ├── determinants_long_format_tax_only.xlsx
│       ├── verification_audit_report.csv
│       ├── verification_audit_report.xlsx
│       ├── pipeline_status.csv
│       ├── pipeline_status.xlsx
│       ├── harmonization_dictionary_tax_only.csv
│       ├── harmonization_dictionary_tax_only.xlsx
│       ├── determinants_long_format_tax_only_harmonized.csv
│       ├── determinants_long_format_tax_only_harmonized.xlsx
│       ├── table_2_determinant_level_tax_only.csv
│       └── table_2_determinant_level_tax_only.xlsx
│
├── requirements.txt
├── .gitignore
├── .gitattributes
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/MateoDib/AI_Assisted_Literature_Revue.git
cd AI_Assisted_Literature_Revue
```

### 2. Create a virtual environment

Using `venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Or using `conda`:

```bash
conda create -n ai_lit_review python=3.11
conda activate ai_lit_review
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

A minimal `requirements.txt` should include:

```text
openai
pandas
openpyxl
pydantic
tqdm
python-dotenv
```

---

## Environment Variables

The pipeline requires an OpenAI API key. It should be provided as an environment variable rather than written directly in the code.

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Optionally, the literature directory can also be defined externally:

```bash
export LITERATURE_DIR="/path/to/your/pdf/folder"
```

A portable version of the script can define the literature directory as follows:

```python
import os
from pathlib import Path

LITERATURE_DIR = Path(
    os.getenv("LITERATURE_DIR", Path(__file__).resolve().parents[1] / "data" / "papers")
)
```

This allows the pipeline to run either from the repository folder or from an external local literature directory.

---

## Running the Pipeline

From the repository root:

```bash
python code/ai_literature_review_pipeline.py
```

The script will:

1. list all PDFs found in the literature directory;
2. upload PDFs to OpenAI when necessary;
3. screen each paper;
4. extract tax-specific determinants;
5. verify the extraction;
6. save final JSON outputs;
7. build paper-level and determinant-level tables;
8. harmonize determinant labels;
9. export CSV and Excel tables.

---

## Parallel Processing

The pipeline supports parallel processing at the paper level. Each paper is processed sequentially internally, but several papers can be processed at the same time.

Relevant parameters are:

```python
PARALLEL_PROCESSING = True
MAX_WORKERS = 6
MAX_CONCURRENT_API_CALLS = 8
```

The recommended starting point is `MAX_WORKERS = 6` or `MAX_WORKERS = 8`, depending on API rate limits and document size.

---

## Robustness and Resume Logic

The pipeline is designed to be restartable. It creates one status file per paper and skips papers that have already been successfully completed.

It can handle:

- interrupted runs;
- failed papers;
- stale running statuses;
- corrupt JSON files;
- invalid intermediate outputs;
- partial results;
- JSON parsing failures;
- incomplete API responses;
- output truncation.

If an output JSON file is corrupt, it is moved aside rather than silently overwritten. If a response appears truncated, the pipeline retries the call with a larger output-token budget.

---

## Main Output Tables

### 1. `screening_decisions`

This table documents inclusion and exclusion decisions for all screened papers.

Main fields include:

- paper ID;
- title;
- authors;
- DOI;
- publication year;
- paper type;
- screening decision;
- tax mentions;
- excluded policy mentions;
- confidence;
- inclusion or exclusion reason.

### 2. `table_1_paper_level_tax_only`

This paper-level table summarizes included studies.

Main fields include:

- paper metadata;
- geographic scope;
- empirical scope or sample;
- tax instruments analyzed;
- exact tax terms used;
- number of extracted tax determinants;
- number of main tax determinants;
- compact determinant summary.

### 3. `determinants_long_format_tax_only`

This is the main long-format extraction table. Each row corresponds to one determinant-paper observation.

Main fields include:

- raw determinant;
- standardized candidate label;
- determinant family;
- determinant role;
- direction;
- mechanism;
- evidence type;
- tax instrument type;
- outcome variable;
- acceptability dimension;
- evidence quote;
- page or section;
- statistical detail;
- confidence;
- inclusion in main table.

### 4. `verification_audit_report`

This table summarizes the verification stage.

Main fields include:

- overall verdict;
- whether the paper remains in the corpus;
- number of determinant checks;
- number of missing tax determinants;
- status counts;
- metadata comments;
- final recommendation.

### 5. `table_2_determinant_level_tax_only`

This harmonized determinant-level table aggregates determinant labels across papers.

Main fields include:

- canonical determinant;
- recommended main-table label;
- determinant family;
- canonical definition;
- number of papers;
- papers;
- DOIs;
- raw terms;
- tax instruments;
- directions observed;
- evidence types;
- outcome variables.

---

## Determinant Families

The pipeline classifies determinants into the following families:

- policy design;
- revenue recycling;
- perceived effectiveness;
- fairness and distribution;
- economic self-interest or cost exposure;
- trust and institutions;
- ideology, partisanship, and values;
- social norms and perceived support;
- climate beliefs, risk perception, and concern;
- emotions and affect;
- knowledge, information, and misperceptions;
- policy experience and learning;
- demographics and socioeconomic status;
- geography and contextual exposure;
- sectoral or occupational exposure;
- other;
- unclear.

These families are intended to support systematic synthesis while preserving conceptual distinctions relevant to environmental economics and political economy.

---

## Methodological Notes

This pipeline should be understood as a semi-automated review assistant rather than a substitute for expert reading.

The model is used to:

- screen documents according to predefined inclusion criteria;
- extract tax-specific determinants;
- classify determinants into closed vocabularies;
- provide supporting evidence excerpts;
- flag uncertain or unsupported entries;
- harmonize determinant labels.

However, final interpretation should remain under researcher supervision. The verification table and evidence quotes are included to facilitate manual auditing.

For publication-quality use, the researcher should manually inspect:

- all included papers;
- all papers excluded with low or medium confidence;
- determinants marked as uncertain, mixed, unsupported, or not tax-specific;
- harmonized labels that merge several raw determinants;
- evidence quotes and page references;
- the direction assigned to each determinant.

---

## Reproducibility

The pipeline improves reproducibility through:

- closed vocabularies;
- Pydantic schemas;
- strict JSON outputs;
- paper-level status files;
- atomic JSON writing;
- cached PDF uploads;
- separation between screening, extraction, verification, and harmonization;
- exported audit tables.

Nevertheless, because the pipeline relies on large language models, exact outputs may vary across model versions, API updates, and inference settings. For full reproducibility, researchers should document:

- the model names used;
- the date of execution;
- the prompts;
- the JSON schemas;
- the list of PDFs;
- the raw and harmonized outputs;
- any manual corrections made after automated extraction.

---

## Suggested Citation

```text
Dib, M. (2026). AI-Assisted Literature Review Pipeline for Carbon and Fuel Tax Acceptability. GitHub repository.
```

---

## Disclaimer

This repository provides a computational assistant for structuring and auditing a literature review. It does not replace expert judgment, manual validation, or formal systematic-review protocols. All extracted results should be checked against the original papers before being used in academic publications.
