# Clinical Decision Support System (CDSS) — Machine Learning Project

A machine learning project for clinical outcome prediction using structured and unstructured patient data sourced from the **MIMIC-IV** database. The system is designed to assist clinicians by predicting key outcomes at the time of hospital admission.

## Project Overview

This project builds predictive models for two binary clinical outcomes:

| Label | Outcome | Derivation | Type |
|-------|---------|------------|------|
| **Y1** | In-Hospital Mortality | `admissions.hospital_expire_flag` | Binary (0/1) |
| **Y2** | 30-Day Readmission | Derived from `admissions.admittime` and `admissions.dischtime` — flag is 1 if the patient is readmitted within 30 days of discharge | Binary (0/1) |

## Features

| ID | Feature | Source (MIMIC-IV) | Type | Representation |
|----|---------|-------------------|------|----------------|
| F1 | Demographics | `patients`, `admissions` | Numeric | Age, gender, height, weight, BMI |
| F2 | Diagnosis History *(prior visits)* | `diagnoses_icd` | Coded text | ICD codes converted to text and embedded via BERT |
| F3 | Discharge Summary History *(prior visits)* | `note` (discharge type) | Free text | Embedded via BERT |
| F4 | Triage Data *(current visit)* | `triage`, early `chartevents` | Structured | Converted to natural language template and embedded via BERT |
| F5 | Chief Complaint *(current visit)* | `triage.chiefcomplaint` | Free text | Embedded via BERT |
| F6–F23 | Lab Events *(current visit, early window)* | `labevents` | Numeric | Grouped by clinical domain; aggregation strategy TBD |
| F24 | Radiology Note *(current visit)* | `note` (radiology type) | Free text | Embedded via BERT |
| Future | Microbiology Events | `microbiologyevents` | Mixed | Representation TBD |

> **Note on data leakage prevention:**
> - **F2** (Diagnosis History) and **F3** (Discharge Summary History) use only data from **prior admissions** to prevent target leakage into the current visit.
> - **F4** (Triage Data), **F5** (Chief Complaint), and **F24** (Radiology Note) are restricted to the **first 24–48 hours** of the current admission to avoid using information that would not yet be available at prediction time.
