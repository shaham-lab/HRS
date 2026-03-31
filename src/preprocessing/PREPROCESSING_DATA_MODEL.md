# CDSS-ML Preprocessing — Data Model

## Table of Contents

1. [Conventions](#1-conventions)
2. [Source Tables (MIMIC-IV)](#2-source-tables-mimic-iv)
   - 2.1 [admissions](#21-admissions)
   - 2.2 [patients](#22-patients)
   - 2.3 [omr](#23-omr)
   - 2.4 [chartevents](#24-chartevents)
   - 2.5 [labevents](#25-labevents)
   - 2.6 [d_labitems](#26-d_labitems)
   - 2.7 [microbiologyevents](#27-microbiologyevents)
   - 2.8 [diagnoses_icd](#28-diagnoses_icd)
   - 2.9 [d_icd_diagnoses](#29-d_icd_diagnoses)
   - 2.10 [note](#210-note)
   - 2.11 [triage](#211-triage)
   - 2.12 [edstays](#212-edstays)
3. [Pipeline Artefacts](#3-pipeline-artefacts)
   - 3.1 [data_splits.parquet](#31-data_splitsparquet)
   - 3.2 [demographics_features.parquet](#32-demographics_featuresparquet)
   - 3.3 [diag_history_features.parquet](#33-diag_history_featuresparquet)
   - 3.4 [discharge_history_features.parquet](#34-discharge_history_featuresparquet)
   - 3.5 [reduced_cdss_dataset.parquet](#35-reduced_cdss_datasetparquet)
   - 3.6 [triage_features.parquet](#36-triage_featuresparquet)
   - 3.7 [chief_complaint_features.parquet](#37-chief_complaint_featuresparquet)
   - 3.8 [labs_features.parquet](#38-labs_featuresparquet)
   - 3.9 [micro\_\<panel\>.parquet × 37](#39-micro_panelparquet--37)
   - 3.10 [radiology_features.parquet](#310-radiology_featuresparquet)
   - 3.11 [y_labels.parquet](#311-y_labelsparquet)
   - 3.12 [Embedding parquets × 55](#312-embedding-parquets--55)
   - 3.13 [final_cdss_dataset.parquet](#313-final_cdss_datasetparquet)

---

## 1. Conventions

**Nullable column:** A column marked `Yes` in the Nullable column may contain null/NaN values in production data. A column marked `No` must never be null — any null values are treated as a data quality error and logged before the row is dropped or flagged.

**Type notation:**

| Symbol | Meaning |
|--------|---------|
| `int8` / `int16` / `int32` / `int64` | Signed integer at specified bit width |
| `float32` / `float64` | IEEE 754 floating point |
| `varchar` | Variable-length UTF-8 string |
| `timestamp` | Datetime with microsecond precision, no timezone |
| `date` | Calendar date (no time component) |
| `float32[N]` | Fixed-length array of N float32 values (embedding vector) |
| `string` | UTF-8 string (used for text feature parquets) |

**Primary key:** The combination of columns that uniquely identifies a row. Enforced by assertion in the producing module and verified by `combine_dataset.py`.

**Source:** MIMIC-IV v3.1 tables are located at the configured `MIMIC_BASE_PATH`. Only the columns actually consumed by the pipeline are listed — full table schemas are available in the MIMIC-IV documentation.

---

## 2. Source Tables (MIMIC-IV)

Only columns consumed by the pipeline are listed per table.

### 2.1 admissions

**Purpose:** Hospital admission records. Primary source for admission/discharge times and labels Y1 and Y2.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `subject_id` | int64 | No | Patient identifier |
| `hadm_id` | int64 | No | Hospital admission identifier |
| `admittime` | timestamp | No | Admission datetime — used for time window filtering, age computation, and Y2 |
| `dischtime` | timestamp | Yes | Discharge datetime — used for full_admission window and Y2 computation; surviving patients with null dischtime are excluded from the dataset |
| `hospital_expire_flag` | int8 | No | Y1 source: 1 = died during admission, 0 = survived |

**Key:** `hadm_id` (unique per row)

---

### 2.2 patients

**Purpose:** Patient-level demographic data. Used for age computation and gender.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `subject_id` | int64 | No | Patient identifier |
| `gender` | varchar | No | Patient sex (M / F) — binary-encoded for F1 |
| `anchor_age` | int16 | No | Patient age at `anchor_year` — used to compute age at admission |
| `anchor_year` | int16 | No | Reference year for age computation |

**Key:** `subject_id` (unique per row)

---

### 2.3 omr

**Purpose:** Outpatient measurement records. Primary source for height, weight, and BMI in demographics feature (F1).

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `subject_id` | int64 | No | Patient identifier |
| `chartdate` | date | No | Date of measurement |
| `seq_num` | int16 | No | Sequence number for same-date records |
| `result_name` | varchar | No | Measurement name — filtered to BMI, Weight (Lbs), Weight (Kg), Height (Inches), Height (Cm) |
| `result_value` | varchar | No | Measurement value as string — parsed to float by the pipeline |

**Key:** (`subject_id`, `chartdate`, `seq_num`)

---

### 2.4 chartevents

**Purpose:** ICU flowsheet measurements. Fallback source for height and weight in demographics (F1) when omr records are absent. Also used for early vital signs in triage feature (F4).

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `subject_id` | int64 | No | Patient identifier |
| `hadm_id` | int64 | Yes | Hospital admission identifier — null for some ICU-only records |
| `stay_id` | int64 | No | ICU stay identifier |
| `itemid` | int64 | No | Chart item identifier — links to d_items |
| `charttime` | timestamp | No | Time of charting |
| `value` | varchar | Yes | Measurement value as string |
| `valuenum` | float64 | Yes | Numeric measurement value |
| `valueuom` | varchar | Yes | Unit of measure |

**Key:** No single-column key; (`stay_id`, `itemid`, `charttime`) is effectively unique. Streamed in 500k-row chunks — not fully loaded into memory.

---

### 2.5 labevents

**Purpose:** Laboratory measurement events. Source for all 13 lab group features (F6–F18).

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `labevent_id` | int64 | No | Unique lab event identifier |
| `subject_id` | int64 | No | Patient identifier |
| `hadm_id` | int64 | Yes | Hospital admission identifier — null for ~15% of records |
| `itemid` | int64 | No | Lab item identifier — links to d_labitems |
| `charttime` | timestamp | Yes | Time of specimen collection/charting — used as event timestamp |
| `value` | varchar | Yes | Result value as string — used when valuenum is null |
| `valuenum` | float64 | Yes | Numeric result value — preferred over value when present |
| `valueuom` | varchar | Yes | Unit of measure — included in text representation |
| `ref_range_lower` | float64 | Yes | Lower bound of reference range — used for ABNORMAL flag derivation |
| `ref_range_upper` | float64 | Yes | Upper bound of reference range — used for ABNORMAL flag derivation |
| `flag` | varchar | Yes | Abnormality flag — takes priority over ref_range comparison when present |

**Key:** `labevent_id` (unique per row). Streamed in 500k-row chunks — not fully loaded into memory.

---

### 2.6 d_labitems

**Purpose:** Lab item dictionary. Used to derive the 13 lab group assignments and to look up human-readable labels.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `itemid` | int64 | No | Lab item identifier |
| `label` | varchar | No | Human-readable lab test name — used in text construction |
| `fluid` | varchar | No | Specimen fluid type — used for group assignment |
| `category` | varchar | No | Lab category (Chemistry, Hematology, Blood Gas, etc.) — used for group assignment |

**Key:** `itemid` (unique per row). Fully loaded into memory (small reference table).

---

### 2.7 microbiologyevents

**Purpose:** Microbiology test events. Source for all 37 microbiology panel features (F20–F56).

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `microevent_id` | int64 | No | Unique microbiology event identifier |
| `subject_id` | int64 | No | Patient identifier |
| `hadm_id` | int64 | Yes | Hospital admission identifier — null for ~56% of records (outpatient population) |
| `charttime` | timestamp | Yes | Time of specimen collection/charting — used as event timestamp |
| `spec_type_desc` | varchar | No | Specimen type description — part of panel assignment key |
| `test_name` | varchar | No | Test name — part of panel assignment key |
| `org_name` | varchar | Yes | Organism identified — null when no growth or qualitative test |
| `ab_name` | varchar | Yes | Antibiotic name — null when no susceptibility testing performed |
| `interpretation` | varchar | Yes | Susceptibility interpretation: S (susceptible), I (intermediate), R (resistant) |
| `comments` | varchar | Yes | Qualitative result text — primary result field for serology, molecular, antigen, and gram stain panels |

**Key:** `microevent_id` (unique per row). Panel assignment uses (`test_name`, `spec_type_desc`) pair — see `micro_panel_config.yaml`.

**Note on `comments`:** This column carries the clinical result for ~61% of all rows. For qualitative tests (serology, NAAT, antigen) it is the only result field — `org_name` and `ab_name` are null. The column undergoes a multi-step cleaning pipeline (described in `microbiology_comments_cleaning_spec.md`) before use in text construction.

---

### 2.8 diagnoses_icd

**Purpose:** ICD diagnosis codes per admission. Source for diagnosis history feature (F2).

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `subject_id` | int64 | No | Patient identifier |
| `hadm_id` | int64 | No | Hospital admission identifier |
| `seq_num` | int16 | No | Diagnosis sequence number (1 = primary diagnosis) |
| `icd_code` | varchar | No | ICD-9 or ICD-10 diagnosis code |
| `icd_version` | int8 | No | ICD version (9 or 10) |

**Key:** (`hadm_id`, `seq_num`)

---

### 2.9 d_icd_diagnoses

**Purpose:** ICD code dictionary. Used to look up human-readable diagnosis descriptions for F2 text construction.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `icd_code` | varchar | No | ICD-9 or ICD-10 code |
| `icd_version` | int8 | No | ICD version (9 or 10) |
| `long_title` | varchar | No | Human-readable diagnosis description — used in text construction |

**Key:** (`icd_code`, `icd_version`)

---

### 2.10 note

**Purpose:** Clinical notes. Source for discharge summary history (F3) and radiology note (F19).

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `note_id` | varchar | No | Unique note identifier |
| `subject_id` | int64 | No | Patient identifier |
| `hadm_id` | int64 | Yes | Hospital admission identifier |
| `note_type` | varchar | No | Note type — filtered to `discharge` (F3) and `radiology` (F19) |
| `note_seq` | int16 | No | Sequence number for same-admission same-type notes — used to select most recent radiology note |
| `charttime` | timestamp | Yes | Time note was charted — used for chronological ordering in F3 |
| `text` | varchar | No | Full note text — cleaned before embedding |

**Key:** `note_id` (unique per row)

---

### 2.11 triage

**Purpose:** ED triage measurements and chief complaint. Source for triage feature (F4) and chief complaint feature (F5).

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `subject_id` | int64 | No | Patient identifier |
| `stay_id` | int64 | No | ED stay identifier — links to edstays for hadm_id resolution |
| `temperature` | float64 | Yes | Temperature at triage |
| `heartrate` | float64 | Yes | Heart rate at triage |
| `resprate` | float64 | Yes | Respiratory rate at triage |
| `o2sat` | float64 | Yes | Oxygen saturation at triage |
| `sbp` | float64 | Yes | Systolic blood pressure at triage |
| `dbp` | float64 | Yes | Diastolic blood pressure at triage |
| `pain` | varchar | Yes | Pain score at triage |
| `acuity` | float64 | Yes | Triage acuity level |
| `chiefcomplaint` | varchar | Yes | Chief complaint free text |

**Key:** `stay_id` (unique per row)

---

### 2.12 edstays

**Purpose:** ED stay records. Used to link `stay_id` (triage) to `hadm_id` (admission) for F4 and F5.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `subject_id` | int64 | No | Patient identifier |
| `hadm_id` | int64 | Yes | Hospital admission identifier — null if ED visit did not result in inpatient admission |
| `stay_id` | int64 | No | ED stay identifier |

**Key:** `stay_id` (unique per row)

---

## 3. Pipeline Artefacts

All parquet files use snappy compression unless noted otherwise. All artefacts are keyed on `hadm_id` (admission-level) except `labs_features.parquet` and `micro_<panel>.parquet` files which are event-level before aggregation.

---

### 3.1 data_splits.parquet

**Produced by:** `create_splits.py` (Step 1)  
**Row definition:** One row per hospital admission  

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `subject_id` | int64 | No | Patient identifier |
| `hadm_id` | int64 | No | Hospital admission identifier |
| `split` | varchar | No | Split assignment: `train`, `dev`, or `test` |

**Primary key:** `hadm_id`  
**Notes:** Splitting is patient-level — all admissions of a given `subject_id` are assigned to the same split. Stratified by Y1 (mortality rate). Random seed 42.

---

### 3.2 demographics_features.parquet

**Produced by:** `extract_demographics.py` (Step 2)  
**Row definition:** One row per hospital admission  

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `subject_id` | int64 | No | Patient identifier |
| `hadm_id` | int64 | No | Hospital admission identifier |
| `age` | float32 | No | Age at admission in years |
| `gender` | float32 | No | Binary-encoded sex: 1.0 = M, 0.0 = F |
| `height` | float32 | No | Height in cm — imputed when missing |
| `weight` | float32 | No | Weight in kg — imputed when missing |
| `bmi` | float32 | No | Body mass index — derived from height/weight if not directly available |
| `height_missing` | float32 | No | Missingness indicator: 1.0 = height was missing before imputation, 0.0 = present |
| `weight_missing` | float32 | No | Missingness indicator: 1.0 = weight was missing before imputation, 0.0 = present |
| `bmi_missing` | float32 | No | Missingness indicator: 1.0 = BMI was missing before imputation, 0.0 = present |

**Primary key:** `hadm_id`  
**Notes:** No null values after extraction — imputation is applied before writing. Imputation statistics (stratum means and standard deviations) are persisted in `imputation_stats.json` and applied identically to dev and test sets.

---

### 3.3 diag_history_features.parquet

**Produced by:** `extract_diag_history.py` (Step 3)  
**Row definition:** One row per hospital admission  

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `subject_id` | int64 | No | Patient identifier |
| `hadm_id` | int64 | No | Hospital admission identifier |
| `text` | string | No | Concatenated ICD long-title descriptions from all prior admissions, chronologically ordered. Empty string if no prior admissions exist |

**Primary key:** `hadm_id`

---

### 3.4 discharge_history_features.parquet

**Produced by:** `extract_discharge_history.py` (Step 4)  
**Row definition:** One row per hospital admission  

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `subject_id` | int64 | No | Patient identifier |
| `hadm_id` | int64 | No | Hospital admission identifier |
| `text` | string | No | Concatenated cleaned discharge notes from all prior admissions, chronologically ordered. Empty string if no prior discharge notes exist |

**Primary key:** `hadm_id`  
**Notes:** Each note is cleaned before inclusion — all text above the "Allergies:" section header is removed.

---

### 3.5 reduced_cdss_dataset.parquet

**Produced by:** `reduce_dataset.py` (Step 12, optional, runs after `combine_dataset.py`)  
**Row definition:** One row per hospital admission  

**Schema:** Identical to `final_cdss_dataset.parquet`, including canonical column order and data types. All metadata, label, structured, and embedding columns are present; only the embedding vector lengths differ.

**Embedding dimensionality:** Each `*_embedding` column stores `float32[128]` by default (configurable via `REDUCED_EMBEDDING_DIM`). With 55 embedding columns, the total feature vector per admission shrinks from 42,248 floats (8 + 55 × 768) to 7,048 floats (8 + 55 × 128).

**Artefacts:** Saved alongside the parquet are (1) the fitted reduction transform objects per embedding column for inference-time application and (2) explained variance statistics (JSON/txt) for auditability.

**Primary key:** `hadm_id`  
**Notes:** The transform for each column is fitted on `is_train == True` rows only to avoid data leakage, then applied to dev/test.

---

### 3.6 triage_features.parquet

**Produced by:** `extract_triage_and_complaint.py` (Step 5)  
**Row definition:** One row per hospital admission  

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `subject_id` | int64 | No | Patient identifier |
| `hadm_id` | int64 | No | Hospital admission identifier |
| `text` | string | No | Natural language template constructed from triage measurements. Empty string if no triage record linked to admission |

**Primary key:** `hadm_id`

---

### 3.7 chief_complaint_features.parquet

**Produced by:** `extract_triage_and_complaint.py` (Step 5)  
**Row definition:** One row per hospital admission  

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `subject_id` | int64 | No | Patient identifier |
| `hadm_id` | int64 | No | Hospital admission identifier |
| `text` | string | No | Chief complaint raw text prefixed with "Chief Complaint:\n". Empty string if absent |

**Primary key:** `hadm_id`

---

### 3.8 labs_features.parquet

**Produced by:** `extract_labs.py` (Step 6)  
**Row definition:** One row per lab event (long format — multiple rows per admission)  

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `subject_id` | int64 | No | Patient identifier |
| `hadm_id` | int64 | No | Hospital admission identifier — null rows dropped or linked before writing |
| `itemid` | int64 | No | Lab item identifier |
| `label` | varchar | No | Human-readable lab test name from d_labitems |
| `fluid` | varchar | No | Specimen fluid type from d_labitems |
| `category` | varchar | No | Lab category from d_labitems |
| `lab_group` | varchar | No | Assigned lab group (one of 13 canonical groups) |
| `charttime` | timestamp | No | Time of specimen collection/charting |
| `valuenum` | float64 | Yes | Numeric result value — null when only string value available |
| `value` | varchar | Yes | String result value — used when valuenum is null |
| `valueuom` | varchar | Yes | Unit of measure |
| `flag` | varchar | Yes | Abnormality flag from source |
| `ref_range_lower` | float64 | Yes | Lower reference range bound |
| `ref_range_upper` | float64 | Yes | Upper reference range bound |
| `is_abnormal` | int8 | No | Derived abnormality flag: 1 = abnormal, 0 = normal or unknown. Derived from `flag` first, then ref_range comparison |
| `elapsed_hours` | float32 | No | Hours since `admittime` — used for text timestamp format [HH:MM] |

**Primary key:** No single-column key — (`hadm_id`, `itemid`, `charttime`) is effectively unique per event.  
**Notes:** This is the only pipeline artefact in long format. `embed_features.py` groups by (`hadm_id`, `lab_group`) to construct per-admission per-group text blocks. Events are included from (`admittime` - `ED_LOOKBACK_HOURS`) to (`admittime` + `LAB_ADMISSION_WINDOW`), defaulting to a 24-hour lookback before `admittime` (`ED_LOOKBACK_HOURS`: 24) and a 24-hour forward window (`LAB_ADMISSION_WINDOW`: 24). The lookback captures pre-admission ED labs drawn before formal inpatient `admittime`. Null `hadm_id` events are resolved via `pd.merge_asof()` time-window linkage when `HADM_LINKAGE_STRATEGY` is `link`. Numeric values are formatted with Python's `{:g}` specifier to preserve clinically significant precision (e.g. Troponin 0.004 renders as "0.004", not "0.00").

---

### 3.9 micro\_\<panel\>.parquet × 37

**Produced by:** `extract_microbiology.py` (Step 7)  
**Files:** One parquet file per microbiology panel, named `micro_<panel_name>.parquet`  
**Row definition:** One row per hospital admission that has at least one event in the panel within the time window  

All 37 files share the same schema:

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `subject_id` | int64 | No | Patient identifier |
| `hadm_id` | int64 | No | Hospital admission identifier |
| `text` | string | No | Aggregated text representation of all panel events for this admission within the time window. Empty string if no events |

**Primary key:** `hadm_id` (unique per file — one row per admission per panel)

**Panel file names:**

| File | Feature ID |
|------|-----------|
| `micro_blood_culture_routine.parquet` | F20 |
| `micro_blood_bottle_gram_stain.parquet` | F21 |
| `micro_urine_culture.parquet` | F22 |
| `micro_urine_viral.parquet` | F23 |
| `micro_urinary_antigens.parquet` | F24 |
| `micro_respiratory_non_invasive.parquet` | F25 |
| `micro_respiratory_invasive.parquet` | F26 |
| `micro_respiratory_afb.parquet` | F27 |
| `micro_respiratory_viral.parquet` | F28 |
| `micro_respiratory_pcp_legionella.parquet` | F29 |
| `micro_gram_stain_respiratory.parquet` | F30 |
| `micro_gram_stain_wound_tissue.parquet` | F31 |
| `micro_gram_stain_csf.parquet` | F32 |
| `micro_wound_culture.parquet` | F33 |
| `micro_hardware_and_lines_culture.parquet` | F34 |
| `micro_pleural_culture.parquet` | F35 |
| `micro_peritoneal_culture.parquet` | F36 |
| `micro_joint_fluid_culture.parquet` | F37 |
| `micro_fluid_culture.parquet` | F38 |
| `micro_bone_marrow_culture.parquet` | F39 |
| `micro_csf_culture.parquet` | F40 |
| `micro_fungal_tissue_wound.parquet` | F41 |
| `micro_fungal_respiratory.parquet` | F42 |
| `micro_fungal_fluid.parquet` | F43 |
| `micro_mrsa_staph_screen.parquet` | F44 |
| `micro_resistance_screen.parquet` | F45 |
| `micro_cdiff.parquet` | F46 |
| `micro_stool_bacterial.parquet` | F47 |
| `micro_stool_parasitology.parquet` | F48 |
| `micro_herpesvirus_serology.parquet` | F49 |
| `micro_hepatitis_hiv.parquet` | F50 |
| `micro_syphilis_serology.parquet` | F51 |
| `micro_misc_serology.parquet` | F52 |
| `micro_herpesvirus_culture_antigen.parquet` | F53 |
| `micro_gc_chlamydia_sti.parquet` | F54 |
| `micro_vaginal_genital_flora.parquet` | F55 |
| `micro_throat_strep.parquet` | F56 |

**Notes:** Admissions with no events in a panel within the time window receive an empty string in the `text` column. The empty string propagates to `embed_features.py` and produces a zero vector at embedding time. Text construction follows the three-case format (Case A: organism present, Case B: comment only, Case C: pending) defined in the feature preprocessing specification. Events are included from (`admittime` - `ED_LOOKBACK_HOURS`) to (`admittime` + `MICRO_WINDOW_HOURS`), defaulting to a 24-hour lookback before `admittime` (`ED_LOOKBACK_HOURS`: 24) and a 72-hour forward window (`MICRO_WINDOW_HOURS`: 72). The lookback captures pre-admission ED cultures drawn before formal inpatient `admittime`. `"full_admission"` includes all events from (`admittime` - `ED_LOOKBACK_HOURS`) to `dischtime`. Panel definitions are read from `config/micro_panel_config.yaml` via `MICRO_PANEL_CONFIG_PATH` — this file is version-controlled in `config/` and is no longer generated by `build_micro_panel_config.py` (deleted).

---

### 3.10 radiology_features.parquet

**Produced by:** `extract_radiology.py` (Step 8)  
**Row definition:** One row per hospital admission  

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `subject_id` | int64 | No | Patient identifier |
| `hadm_id` | int64 | No | Hospital admission identifier |
| `text` | string | No | Most recent radiology note for the admission, cleaned to remove text above "EXAMINATION:" header. Empty string if no radiology note exists |

**Primary key:** `hadm_id`

---

### 3.11 y_labels.parquet

**Produced by:** `extract_y_data.py` (Step 9)  
**Row definition:** One row per hospital admission  

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `subject_id` | int64 | No | Patient identifier |
| `hadm_id` | int64 | No | Hospital admission identifier |
| `y1_mortality` | int8 | No | In-hospital mortality: 1 = died, 0 = survived. Sourced from `admissions.hospital_expire_flag`. Never null |
| `y2_readmission` | float32 | Yes | 30-day readmission: 1.0 = readmitted within 30 days, 0.0 = not readmitted, NaN = deceased (Y1=1). NaN assignment for deceased patients is mandatory — see notes |

**Primary key:** `hadm_id`

**Notes on Y2 NaN assignment:** After computing the readmission label, the following assignment is applied unconditionally:

```python
df.loc[df['y1_mortality'] == 1, 'y2_readmission'] = np.nan
```

This is mandatory before any downstream use. The following assertions must pass before the file is written:

```python
assert df.loc[df['y1_mortality'] == 1, 'y2_readmission'].isna().all()
assert df.loc[df['y1_mortality'] == 0, 'y2_readmission'].notna().all()
```

Surviving admissions with null or corrupt `dischtime` in `admissions` are excluded from the dataset entirely and logged — they are not assigned NaN.

---

### 3.12 Embedding parquets × 55

**Produced by:** `embed_features.py` (Step 10)  
**Files:** One parquet file per embedded feature  
**Row definition:** One row per hospital admission  

All 55 embedding parquet files share the same schema:

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `subject_id` | int64 | No | Patient identifier |
| `hadm_id` | int64 | No | Hospital admission identifier |
| `embedding` | float32[768] | No | 768-dimensional embedding vector produced by Clinical_ModernBERT mean pooling. Zero vector for admissions with empty text input |

**Primary key:** `hadm_id` (unique per file)

**Embedding file names and feature IDs:**

| File | Feature ID | Source text file |
|------|-----------|-----------------|
| `diag_history_embedding.parquet` | F2 | `diag_history_features.parquet` |
| `discharge_history_embedding.parquet` | F3 | `discharge_history_features.parquet` |
| `triage_embedding.parquet` | F4 | `triage_features.parquet` |
| `chief_complaint_embedding.parquet` | F5 | `chief_complaint_features.parquet` |
| `lab_blood_gas_embedding.parquet` | F6 | `labs_features.parquet` (group: blood_gas) |
| `lab_blood_chemistry_embedding.parquet` | F7 | `labs_features.parquet` (group: blood_chemistry) |
| `lab_blood_hematology_embedding.parquet` | F8 | `labs_features.parquet` (group: blood_hematology) |
| `lab_urine_chemistry_embedding.parquet` | F9 | `labs_features.parquet` (group: urine_chemistry) |
| `lab_urine_hematology_embedding.parquet` | F10 | `labs_features.parquet` (group: urine_hematology) |
| `lab_other_body_fluid_chemistry_embedding.parquet` | F11 | `labs_features.parquet` (group: other_body_fluid_chemistry) |
| `lab_other_body_fluid_hematology_embedding.parquet` | F12 | `labs_features.parquet` (group: other_body_fluid_hematology) |
| `lab_ascites_embedding.parquet` | F13 | `labs_features.parquet` (group: ascites) |
| `lab_pleural_embedding.parquet` | F14 | `labs_features.parquet` (group: pleural) |
| `lab_csf_embedding.parquet` | F15 | `labs_features.parquet` (group: csf) |
| `lab_bone_marrow_embedding.parquet` | F16 | `labs_features.parquet` (group: bone_marrow) |
| `lab_joint_fluid_embedding.parquet` | F17 | `labs_features.parquet` (group: joint_fluid) |
| `lab_stool_embedding.parquet` | F18 | `labs_features.parquet` (group: stool) |
| `radiology_embedding.parquet` | F19 | `radiology_features.parquet` |
| `micro_blood_culture_routine_embedding.parquet` | F20 | `micro_blood_culture_routine.parquet` |
| `micro_blood_bottle_gram_stain_embedding.parquet` | F21 | `micro_blood_bottle_gram_stain.parquet` |
| `micro_urine_culture_embedding.parquet` | F22 | `micro_urine_culture.parquet` |
| `micro_urine_viral_embedding.parquet` | F23 | `micro_urine_viral.parquet` |
| `micro_urinary_antigens_embedding.parquet` | F24 | `micro_urinary_antigens.parquet` |
| `micro_respiratory_non_invasive_embedding.parquet` | F25 | `micro_respiratory_non_invasive.parquet` |
| `micro_respiratory_invasive_embedding.parquet` | F26 | `micro_respiratory_invasive.parquet` |
| `micro_respiratory_afb_embedding.parquet` | F27 | `micro_respiratory_afb.parquet` |
| `micro_respiratory_viral_embedding.parquet` | F28 | `micro_respiratory_viral.parquet` |
| `micro_respiratory_pcp_legionella_embedding.parquet` | F29 | `micro_respiratory_pcp_legionella.parquet` |
| `micro_gram_stain_respiratory_embedding.parquet` | F30 | `micro_gram_stain_respiratory.parquet` |
| `micro_gram_stain_wound_tissue_embedding.parquet` | F31 | `micro_gram_stain_wound_tissue.parquet` |
| `micro_gram_stain_csf_embedding.parquet` | F32 | `micro_gram_stain_csf.parquet` |
| `micro_wound_culture_embedding.parquet` | F33 | `micro_wound_culture.parquet` |
| `micro_hardware_and_lines_culture_embedding.parquet` | F34 | `micro_hardware_and_lines_culture.parquet` |
| `micro_pleural_culture_embedding.parquet` | F35 | `micro_pleural_culture.parquet` |
| `micro_peritoneal_culture_embedding.parquet` | F36 | `micro_peritoneal_culture.parquet` |
| `micro_joint_fluid_culture_embedding.parquet` | F37 | `micro_joint_fluid_culture.parquet` |
| `micro_fluid_culture_embedding.parquet` | F38 | `micro_fluid_culture.parquet` |
| `micro_bone_marrow_culture_embedding.parquet` | F39 | `micro_bone_marrow_culture.parquet` |
| `micro_csf_culture_embedding.parquet` | F40 | `micro_csf_culture.parquet` |
| `micro_fungal_tissue_wound_embedding.parquet` | F41 | `micro_fungal_tissue_wound.parquet` |
| `micro_fungal_respiratory_embedding.parquet` | F42 | `micro_fungal_respiratory.parquet` |
| `micro_fungal_fluid_embedding.parquet` | F43 | `micro_fungal_fluid.parquet` |
| `micro_mrsa_staph_screen_embedding.parquet` | F44 | `micro_mrsa_staph_screen.parquet` |
| `micro_resistance_screen_embedding.parquet` | F45 | `micro_resistance_screen.parquet` |
| `micro_cdiff_embedding.parquet` | F46 | `micro_cdiff.parquet` |
| `micro_stool_bacterial_embedding.parquet` | F47 | `micro_stool_bacterial.parquet` |
| `micro_stool_parasitology_embedding.parquet` | F48 | `micro_stool_parasitology.parquet` |
| `micro_herpesvirus_serology_embedding.parquet` | F49 | `micro_herpesvirus_serology.parquet` |
| `micro_hepatitis_hiv_embedding.parquet` | F50 | `micro_hepatitis_hiv.parquet` |
| `micro_syphilis_serology_embedding.parquet` | F51 | `micro_syphilis_serology.parquet` |
| `micro_misc_serology_embedding.parquet` | F52 | `micro_misc_serology.parquet` |
| `micro_herpesvirus_culture_antigen_embedding.parquet` | F53 | `micro_herpesvirus_culture_antigen.parquet` |
| `micro_gc_chlamydia_sti_embedding.parquet` | F54 | `micro_gc_chlamydia_sti.parquet` |
| `micro_vaginal_genital_flora_embedding.parquet` | F55 | `micro_vaginal_genital_flora.parquet` |
| `micro_throat_strep_embedding.parquet` | F56 | `micro_throat_strep.parquet` |

**Notes:** Embedding columns are discovered dynamically by `combine_dataset.py` from `EMBEDDINGS_DIR` — the file list above is the expected state, not a hardcoded dependency. Admissions with empty text input receive a zero vector (768 × 0.0 float32).

---

### 3.13 final_cdss_dataset.parquet

**Produced by:** `combine_dataset.py` (Step 11)  
**Row definition:** One row per hospital admission  

| Column | Type          | Nullable | Description |
|--------|---------------|----------|-------------|
| `subject_id` | int64         | No  | Patient identifier |
| `hadm_id` | int64         | No  | Hospital admission identifier |
| `split` | varchar       | No  | Split assignment: `train`, `dev`, or `test` |
| `y1_mortality` | int64         | No  | In-hospital mortality label (0 or 1) |
| `y2_readmission` | float64       | Yes | 30-day readmission label (0.0, 1.0, or NaN for deceased) |
| `demographic_vec` | float64[8]    | No  | F1: 8-dimensional demographics vector |
| `diag_history_embedding` | float32[768]  | No  | F2: diagnosis history embedding |
| `discharge_history_embedding` | float32[768]  | No  | F3: discharge summary history embedding |
| `triage_embedding` | float32[768]  | No  | F4: triage data embedding |
| `chief_complaint_embedding` | float32[768]  | No  | F5: chief complaint embedding |
| `lab_blood_gas_embedding` | float32[768]  | No  | F6: blood gas lab group embedding |
| `lab_blood_chemistry_embedding` | float32[768]  | No  | F7: blood chemistry lab group embedding |
| `lab_blood_hematology_embedding` | float32[768]  | No  | F8: blood hematology lab group embedding |
| `lab_urine_chemistry_embedding` | float32[768]  | No  | F9: urine chemistry lab group embedding |
| `lab_urine_hematology_embedding` | float32[768]  | No  | F10: urine hematology lab group embedding |
| `lab_other_body_fluid_chemistry_embedding` | float32[768]  | No  | F11: other body fluid chemistry embedding |
| `lab_other_body_fluid_hematology_embedding` | float32[768]  | No  | F12: other body fluid hematology embedding |
| `lab_ascites_embedding` | float32[768]  | No  | F13: ascites lab group embedding |
| `lab_pleural_embedding` | float32[768]  | No  | F14: pleural lab group embedding |
| `lab_csf_embedding` | float32[768]  | No  | F15: CSF lab group embedding |
| `lab_bone_marrow_embedding` | float32[768]  | No  | F16: bone marrow lab group embedding |
| `lab_joint_fluid_embedding` | float32[768]  | No  | F17: joint fluid lab group embedding |
| `lab_stool_embedding` | float32[768]  | No  | F18: stool lab group embedding |
| `radiology_embedding` | float32[768]  | No  | F19: radiology note embedding |
| `micro_blood_culture_routine_embedding` | float32[768]  | No  | F20: blood culture embedding |
| `micro_blood_bottle_gram_stain_embedding` | float32[768]  | No  | F21: blood bottle gram stain embedding |
| `micro_urine_culture_embedding` | float32[768]  | No  | F22: urine culture embedding |
| `micro_urine_viral_embedding` | float32[768]  | No  | F23: urine viral culture embedding |
| `micro_urinary_antigens_embedding` | float32[768]  | No  | F24: urinary antigens embedding |
| `micro_respiratory_non_invasive_embedding` | float32[768]  | No  | F25: respiratory non-invasive embedding |
| `micro_respiratory_invasive_embedding` | float32[768]  | No  | F26: respiratory invasive embedding |
| `micro_respiratory_afb_embedding` | float32[768]  | No  | F27: respiratory AFB/TB embedding |
| `micro_respiratory_viral_embedding` | float32[768]  | No  | F28: respiratory viral embedding |
| `micro_respiratory_pcp_legionella_embedding` | float32[768]  | No  | F29: PCP/Legionella embedding |
| `micro_gram_stain_respiratory_embedding` | float32[768]  | No  | F30: gram stain respiratory embedding |
| `micro_gram_stain_wound_tissue_embedding` | float32[768]  | No  | F31: gram stain wound/tissue embedding |
| `micro_gram_stain_csf_embedding` | float32[768]  | No  | F32: gram stain CSF embedding |
| `micro_wound_culture_embedding` | float32[768]  | No  | F33: wound culture embedding |
| `micro_hardware_and_lines_culture_embedding` | float32[768]  | No  | F34: hardware/lines culture embedding |
| `micro_pleural_culture_embedding` | float32[768]  | No  | F35: pleural culture embedding |
| `micro_peritoneal_culture_embedding` | float32[768]  | No  | F36: peritoneal culture embedding |
| `micro_joint_fluid_culture_embedding` | float32[768]  | No  | F37: joint fluid culture embedding |
| `micro_fluid_culture_embedding` | float32[768]  | No  | F38: fluid culture embedding |
| `micro_bone_marrow_culture_embedding` | float32[768]  | No  | F39: bone marrow culture embedding |
| `micro_csf_culture_embedding` | float32[768]  | No  | F40: CSF culture embedding |
| `micro_fungal_tissue_wound_embedding` | float32[768]  | No  | F41: fungal tissue/wound embedding |
| `micro_fungal_respiratory_embedding` | float32[768]  | No  | F42: fungal respiratory embedding |
| `micro_fungal_fluid_embedding` | float32[768]  | No  | F43: fungal fluid embedding |
| `micro_mrsa_staph_screen_embedding` | float32[768]  | No  | F44: MRSA/Staph screen embedding |
| `micro_resistance_screen_embedding` | float32[768]  | No  | F45: resistance screen embedding |
| `micro_cdiff_embedding` | float32[768]  | No  | F46: C. difficile embedding |
| `micro_stool_bacterial_embedding` | float32[768]  | No  | F47: stool bacterial culture embedding |
| `micro_stool_parasitology_embedding` | float32[768]  | No  | F48: stool parasitology embedding |
| `micro_herpesvirus_serology_embedding` | float32[768]  | No  | F49: herpesvirus serology embedding |
| `micro_hepatitis_hiv_embedding` | float32[768]  | No  | F50: hepatitis/HIV viral loads embedding |
| `micro_syphilis_serology_embedding` | float32[768]  | No  | F51: syphilis serology embedding |
| `micro_misc_serology_embedding` | float32[768]  | No  | F52: miscellaneous serology embedding |
| `micro_herpesvirus_culture_antigen_embedding` | float32[768]  | No  | F53: herpesvirus culture/antigen embedding |
| `micro_gc_chlamydia_sti_embedding` | float32[768]  | No  | F54: GC/Chlamydia STI embedding |
| `micro_vaginal_genital_flora_embedding` | float32[768]  | No  | F55: vaginal/genital flora embedding |
| `micro_throat_strep_embedding` | float32[768]  | No  | F56: throat strep embedding |

**Primary key:** `hadm_id`
**Total columns:** 61 (3 metadata + 2 labels + 1 structured + 55 embeddings)
**Total embedding dimensions per admission:** 55 × 768 = 42,240 float32 values
**Notes:** The Nullable column reflects the data contract, not the PyArrow schema declaration (PyArrow marks all columns `nullable=True` by default regardless of actual data content). All embedding columns and `demographic_vec` are non-nullable by contract — missing feature text produces a zero vector, never null. `y2_readmission` is the only nullable column (NaN for deceased patients). PyArrow stores `y1_mortality` as int64, `y2_readmission` and `demographic_vec` as float64 (double). All `*_embedding` columns are stored as `list<element: float>` (float32). Downstream consumers should accept these types and cast if needed rather than asserting specific bit-widths. Embedding columns are joined from `EMBEDDINGS_DIR` dynamically — the column list above reflects the expected full dataset. Column order is canonical and enforced by `combine_dataset.py` using `CANONICAL_COLUMNS` derived at runtime from `LAB_PANEL_CONFIG_PATH` and `MICRO_PANEL_CONFIG_PATH` — no column names are hardcoded in `combine_dataset.py`. If any expected column is missing at combine time, `combine_dataset.py` raises `ValueError` listing the missing columns.
