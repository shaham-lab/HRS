# CDSS-ML Preprocessing — Detailed Design

## Table of Contents

1. [Identifier Hierarchy](#1-identifier-hierarchy)
2. [Data Splits — Implementation](#2-data-splits--implementation)
3. [Module Implementation](#3-module-implementation)
   - [build_lab_panel_config.py](#buildlabpanelconfigpy)
   - [create_splits.py](#createsplitspy)
   - [extract_demographics.py](#extractdemographicspy)
   - [extract_diag_history.py](#extractdiaghistorypy)
   - [extract_discharge_history.py](#extractdischargehistorypy)
   - [extract_triage_and_complaint.py](#extracttriageandcomplaintpy)
   - [extract_labs.py](#extractlabspy)
   - [extract_radiology.py](#extractradiologypy)
   - [extract_y_data.py](#extractydatapy)
   - [embed_features.py](#embedfeaturespy)
   - [combine_dataset.py](#combinedatasetpy)
4. [Embedding Implementation Detail](#4-embedding-implementation-detail)
   - [Model](#model)
   - [Mean Pooling](#mean-pooling)
   - [Per-Feature Token Length Caps](#per-feature-token-length-caps)
   - [Admission-Slice Batching](#admission-slice-batching)
   - [Multi-GPU Execution Within a Slice](#multi-gpu-execution-within-a-slice)
   - [GPU Load Balancing](#gpu-load-balancing)
   - [Feature-Level Resume](#feature-level-resume)
   - [Record-Level Resume](#record-level-resume)
5. [Final Dataset Assembly](#5-final-dataset-assembly)
6. [Configuration Reference](#6-configuration-reference)
7. [Memory Requirements](#7-memory-requirements)

---

## 1. Identifier Hierarchy

MIMIC-IV uses three nested identifiers:

```
subject_id   (patient — persistent across all visits)
  └── hadm_id    (hospital admission — the unit of prediction)
        └── stay_id    (ICU stay within an admission)
```

**`hadm_id` is the primary join key** throughout the pipeline. All feature parquets carry `(subject_id, hadm_id)` as their key columns.

**`stay_id`** appears only indirectly in Phase 1 — via `chartevents` in the demographics fallback (joined on `hadm_id`). It becomes critical in the MDP phase for modelling clinical interventions.

### Missing `hadm_id`

Several tables contain records with null `hadm_id` (~10–20% of `labevents`, lower rates in `note` and `chartevents`). Handling is configurable via `HADM_LINKAGE_STRATEGY`:

| Strategy | Behaviour |
|----------|-----------|
| `"drop"` *(default)* | Exclude records with null `hadm_id`. Count and percentage logged per module. |
| `"link"` | Time-window linkage: match `charttime` against patient admission windows within `HADM_LINKAGE_TOLERANCE_HOURS`. Assign if exactly one match; assign closest if multiple; drop if none. All outcomes logged to `hadm_linkage_stats.json`. |

---

## 2. Data Splits — Implementation

`create_splits.py` performs a patient-level stratified 3-way split:

```
1. Load admissions → compute hospital_expire_flag per subject_id
2. Build patient table: subject_id, has_death (1 if any admission is Y1=1)
3. sklearn.model_selection.train_test_split with stratify=has_death, seed=42
4. Output: data_splits.parquet — one row per hadm_id with split column
```

All admissions of a patient go to the same split. Stratification on Y1 only — Y2 would create degenerate strata because deceased patients always have Y2=NaN.

**Output statistics:**

| Split | Admissions | Y1 positive |
|-------|------------|-------------|
| Train | 435,160 | 2.16% |
| Dev | 54,934 | 2.16% |
| Test | 55,934 | 2.16% |

---

## 3. Module Implementation

### `build_lab_panel_config.py`

Derives lab groups from `d_labitems` and writes `classifications/lab_panel_config.yaml`.

**Algorithm:**
1. Load `d_labitems` — retain `itemid`, `fluid`, `category`
2. Drop artefact rows where `fluid` in `["I", "Q", "fluid"]`
3. For each row, assign to a group using `_FLUID_CATEGORY_MAP` if the `(fluid, category)` pair has an explicit mapping
4. For fluids in `_SINGLE_GROUP_FLUIDS` (Ascites, Pleural, Cerebrospinal Fluid, Bone Marrow, Joint Fluid, Stool), all itemids regardless of category are merged under a single group key; `Cerebrospinal Fluid` is mapped to the key `csf` via `_SINGLE_GROUP_NAME_OVERRIDES`
5. Blood gas panels for non-blood fluids (`Urine`, `Other Body Fluid`, `Fluid`) are folded into their closest existing group rather than creating new groups
6. Write YAML: `{group_name: [itemid, ...], ...}`

**Output — 13 groups:**

| # | Group name | Fluid | Category | Items | Emb. dim |
|---|-----------|-------|----------|-------|---------|
| 1 | `blood_gas` | Blood | Blood Gas | 34 | 768 |
| 2 | `blood_chemistry` | Blood | Chemistry | 267 | 768 |
| 3 | `blood_hematology` | Blood | Hematology | 198 | 768 |
| 4 | `urine_chemistry` | Urine | Chemistry | 72 | 768 |
| 5 | `urine_hematology` | Urine | Hematology + Blood Gas | 65 | 768 |
| 6 | `other_body_fluid_chemistry` | Other Body Fluid | Chemistry | 65 | 768 |
| 7 | `other_body_fluid_hematology` | Other Body Fluid | Hematology + Blood Gas | 62 | 768 |
| 8 | `ascites` | Ascites | Chemistry + Hematology | 40 | 768 |
| 9 | `pleural` | Pleural | Chemistry + Hematology | 39 | 768 |
| 10 | `csf` | Cerebrospinal Fluid | Chemistry + Hematology | 34 | 768 |
| 11 | `bone_marrow` | Bone Marrow | Hematology | 44 | 768 |
| 12 | `joint_fluid` | Joint Fluid | Blood Gas + Chemistry + Hematology | 38 | 768 |
| 13 | `stool` | Stool | Chemistry + Hematology | 18 | 768 |

---

### `create_splits.py`

See [Section 2](#2-data-splits--implementation).

---

### `extract_demographics.py`

Produces `demographic_vec = [age, gender, height_cm, weight_kg, bmi, height_missing, weight_missing, bmi_missing]`.

**Age:**
```
age = anchor_age + (year(admittime) - anchor_year)
```
Corrects for MIMIC-IV's year-shift anonymisation.

**Gender:** `M = 1.0`, `F = 0.0`.

**Height / Weight source priority:**

| Priority | Source | Notes |
|----------|--------|-------|
| 1 | `omr.result_name` contains "Height" / "Weight" | `chartdate ≤ admittime`. Inches → cm (×2.54). Lbs → kg (×0.453592). |
| 2–5 | `chartevents` itemids: height 226707, 226730; weight 226512, 224639, 226531, 226846 | First value within admission window. Unit conversion per itemid. |

Plausibility filters: height 50–250 cm, weight 20–400 kg. Values outside range discarded before imputation.

After merge (pre-imputation): height missing 41.5%, weight missing 29.7%, BMI missing 43.2%.

**Imputation:**
- Missingness flags set **before** any imputation
- Method: sample from `N(mean, std)` per `(age_bin × gender)` stratum
- Age bins: `[18–29, 30–44, 45–64, 65–74, 75+]` — 11 strata (5 age × 2 gender, minus one absent stratum)
- Statistics computed on **train split only**, saved to `imputation_stats.json`
- Applied identically to dev/test; fallback to global stats if stratum absent
- Result: 226,873 height values imputed, 162,179 weight values imputed

**BMI:** Use OMR `bmi` value if present. Derive as `weight_kg / (height_cm / 100)²` if absent. **Never imputed independently.**

**Memory:** `chartevents` is streamed in 500k-row chunks — never loaded fully into memory.

---

### `extract_diag_history.py`

Builds a structured text block of prior-visit ICD diagnoses.

**Algorithm:**
1. Load `diagnoses_icd` + join `d_icd_diagnoses` for `long_title`
2. Load current admission `admittime` for each `hadm_id`
3. For each target admission, collect all prior admissions with `admittime < current admittime` for the same `subject_id`
4. Sort prior admissions chronologically; format as text block

**Text format:**
```
Past Diagnoses:

Visit (2018-03-12):
Chronic kidney disease, stage 3
Hypertension, unspecified

Visit (2019-07-24):
Acute kidney injury
```

First-time admissions (no prior visits) → empty string → zero vector at embedding time.

**Output:** 546,028 rows (322,399 with prior-visit content; 223,629 empty strings).

---

### `extract_discharge_history.py`

Concatenates prior-visit discharge notes with dated headers.

**Algorithm:**
1. Load `note/discharge`
2. For each target admission, collect prior discharge notes (`charttime < current admittime`)
3. Clean each note: strip everything before the first `"Allergies:"` marker (removes boilerplate header)
4. Concatenate chronologically with dated section headers

**Text format:**
```
Prior Discharge Summary (2018-03-12):
Allergies: Penicillin
[clinical note body...]

Prior Discharge Summary (2019-07-24):
Allergies: None known
[clinical note body...]
```

**Output:** 546,028 rows (259,218 with prior notes; 286,810 empty strings). Source: 331,793 discharge notes.

---

### `extract_triage_and_complaint.py`

Extracts two features from the ED visit: a structured triage text (F4) and raw chief complaint (F5).

**`hadm_id` resolution for ED visits:**
1. Primary: join `edstays` on `stay_id → hadm_id` (direct linkage)
2. Fallback: for `subject_id` with no direct link, find the inpatient admission where `admittime` is closest to and ≥ `intime`
3. Non-admitted ED visits (no matching `hadm_id`) are excluded

**F4 — Triage text template:**
```
Triage assessment: temperature {T}°C, heart rate {HR} bpm,
respiratory rate {RR} breaths/min, O2 saturation {O2}%,
blood pressure {SBP}/{DBP} mmHg, pain score {pain}/10, acuity level {acuity}.
```
Missing fields rendered as `"N/A"`.

**F5 — Chief complaint:**
- Primary source: `triage.chiefcomplaint`
- Fallback: `chartevents` itemid 223112
- No cleaning or templating — raw text only

---

### `extract_labs.py`

Produces a long-format parquet of all lab events within the admission window.

**Algorithm:**
1. Stream `labevents` in 500k-row chunks
2. For each chunk: join `admissions` on `hadm_id` to get `admittime`
3. Filter: `admittime ≤ charttime ≤ admittime + LAB_ADMISSION_WINDOW`
4. Format each retained event as a text line
5. Concatenate all chunks → write `labs_features.parquet`

**Text line format:**
```
[HH:MM] {label}: {value} {unit} (ref: lower-upper) [ABNORMAL]
```

- `[HH:MM]` = elapsed hours:minutes since `admittime` (not wall clock time)
- `valuenum` formatted to 2 decimal places when available; `value` text field otherwise
- `(ref: lower-upper)` omitted when either bound is null
- `[ABNORMAL]` appended when `flag == "abnormal"` **or** `valuenum` outside `[ref_range_lower, ref_range_upper]`

**Example (blood_chemistry group):**
```
[00:14] Glucose: 8.20 mmol/L [ABNORMAL]
[00:14] Sodium: 138.00 mEq/L
[00:14] Potassium: 6.10 mEq/L [ABNORMAL]
[08:32] Creatinine: 1.80 mg/dL [ABNORMAL]
```

**Output:** 16,838,777 rows for 400,754 admissions (from 79,440,362 raw events). 145,274 admissions have no events in the window → zero vectors at embedding time.

**Admission window:** `LAB_ADMISSION_WINDOW` — integer hours or `"full"`. Default: 24.

**Lab event counts per group (MIMIC-IV v3.1):**

| Group | Admissions with events | Notes |
|-------|----------------------|-------|
| `blood_hematology` | 375,727 | |
| `blood_chemistry` | 370,981 | |
| `blood_gas` | 78,524 | includes Fluid Blood Gas items (0 events) |
| `urine_hematology` | 73,887 | includes Urine Blood Gas items (0 events) |
| `urine_chemistry` | 47,818 | |
| `other_body_fluid_chemistry` | 4,200 | |
| `other_body_fluid_hematology` | 3,119 | includes Other Body Fluid Blood Gas items (1,279 events) |
| `ascites` | 2,630 | |
| `csf` | 4,016 | merged Chemistry (1,994) + Hematology (2,022) |
| `pleural` | 1,908 | |
| `joint_fluid` | 849 | |
| `bone_marrow` | 515 | |
| `stool` | 70 | |

---

### `extract_radiology.py`

Selects the most recent radiology note within the current admission window.

**Algorithm:**
1. Load `note/radiology` — 1,144,758 notes; drop null `hadm_id` (50.7% of raw notes)
2. Filter to notes within admission window (`admittime ≤ charttime ≤ dischtime`)
3. Per `hadm_id`, select the note with the latest `charttime`
4. Clean: strip everything before the first `"EXAMINATION:"` marker

**Output:** 546,028 rows (220,022 with a note; 326,006 empty strings).

---

### `extract_y_data.py`

Produces `y_labels.parquet` with Y1 and Y2 per admission.

**Y1:** Direct from `admissions.hospital_expire_flag`. Values: 0 (survived) or 1 (died in hospital).

**Y2:** For each admission where Y1 = 0, check whether any subsequent admission for the same `subject_id` has `admittime ≤ dischtime + 30 days`. Y2 = 1 if yes, 0 if no. **Y2 = NaN for all admissions where Y1 = 1** (deceased patients cannot be readmitted).

---

### `embed_features.py`

Embeds all 22 text features using Clinical_ModernBERT. Full implementation details in [Section 4](#4-embedding-implementation-detail).

**CLI entry point:**
```bash
python embed_features.py --config config/preprocessing.yaml \
                         --slice-index 0
```
Also callable as `run(config, slice_index)` from `run_pipeline.py`.

`--slice-index` specifies which admission slice this job processes (0-based). The total number of slices is derived at runtime: `n_slices = ceil(total_admissions / (BERT_SLICE_SIZE_PER_GPU × n_gpus))`. Omitting `--slice-index` defaults to `0`. To run the full dataset in a single job, set `BERT_SLICE_SIZE_PER_GPU` large enough to cover all admissions.

**Feature inputs:**

| Feature | Input parquet | Text column |
|---------|---------------|-------------|
| `diag_history` | `diag_history_features.parquet` | `diag_history_text` |
| `discharge_history` | `discharge_history_features.parquet` | `discharge_history_text` |
| `triage` | `triage_features.parquet` | `triage_text` |
| `chief_complaint` | `chief_complaint_features.parquet` | `chief_complaint_text` |
| `radiology` | `radiology_features.parquet` | `radiology_text` |
| `lab_{group}` ×13 | `labs_features.parquet` (filtered by group's itemid list) | `lab_text_line` (concatenated per hadm_id) |

**Labs processing:** `labs_df` (16.8M rows) is filtered to the current slice's `hadm_id` set, then scanned **once** before the 13-group embedding loop to build per-group text maps.

---

### `combine_dataset.py`

Builds `final_cdss_dataset.parquet` from `data_splits.parquet` as the admission universe.

**Algorithm:**
1. Start with `data_splits.parquet` (546,028 rows)
2. Left-join `y_labels.parquet` on `hadm_id`
3. Left-join `demographics_features.parquet` on `hadm_id`
4. Scan `EMBEDDINGS_DIR` for all `*.parquet` files dynamically
5. Left-join each embedding parquet on `hadm_id`
6. Write `final_cdss_dataset.parquet`

All joins are **left joins** — admissions missing a non-lab feature receive null for that column. Lab embedding columns are always a 768-float array (zero vector for admissions with no events — never null).

**Intentionally excluded:** `labs_features.parquet` (superseded by per-group embedding parquets), all raw text parquets (superseded by embedding parquets).

---

## 4. Embedding Implementation Detail

### Model

| Property | Value |
|----------|-------|
| Model identifier | `Simonlee711/Clinical_ModernBERT` |
| Pre-training corpus | PubMed abstracts, MIMIC-IV clinical notes, medical ontologies |
| Context window | 8,192 tokens (RoPE positional encoding + Flash Attention) |
| Hidden size | 768 |
| Load call | `BertModel.from_pretrained(model_name, add_pooling_layer=False)` |
| HF logging | `hf_logging.set_verbosity_error()` — suppresses non-error output |

The model is used as a **frozen feature extractor** — weights are never updated.

---

### Mean Pooling

```
token hidden states (final layer):
  t₁    t₂    t₃   ...   tₙ   [PAD] [PAD]
   │     │     │           │
   └─────┴─────┴─────...───┘
              mean
               │
               ▼
         768-d embedding vector
```

Mean pooling is chosen over `[CLS]` because Clinical_ModernBERT is not fine-tuned — the `[CLS]` token does not develop task-specific semantics in this regime. Mean pooling ensures every content token contributes equally, which is important for long texts (600-token discharge notes, 40-measurement lab timelines).

**Implementation:**
```python
attention_mask = batch["attention_mask"]           # (B, L)
hidden = model(**batch).last_hidden_state          # (B, L, 768)
mask = attention_mask.unsqueeze(-1).float()        # (B, L, 1)
embedding = (hidden * mask).sum(1) / mask.sum(1)  # (B, 768)
```

**Empty / missing features:** Zero vector of dimension 768 — never null in output parquet.

---

### Per-Feature Token Length Caps

Per-feature caps are applied as `min(BERT_MAX_LENGTH, cap)` at tokenisation time, using `padding="longest"` per batch (not global padding). This prevents short texts from being padded to the global 8,192 maximum.

| Feature | Cap (tokens) | Rationale |
|---------|-------------|-----------|
| `chief_complaint` | 64 | Single phrase, typically 5–15 words |
| `triage` | 256 | Structured template, bounded length |
| `diag_history` | 512 | ICD label list — verbose but bounded |
| `radiology` | 1,024 | Single report |
| All lab groups | 2,048 | Lab timeline — longer for complex admissions |
| `discharge_history` | 4,096 | Concatenated multi-visit notes |

Attention is O(n²) in sequence length. Padding chief complaint to 64 vs 8,192 tokens is ~16,000× less GPU compute per token.

**Dynamic batch size:** `_effective_batch_size(base_batch_size, max_length_cap)` scales the batch size inversely with `max_length_cap` to maintain a constant token budget per GPU forward pass.

---

### Admission-Slice Batching

#### Motivation

The full corpus of 546,028 admissions cannot be embedded in a single SLURM job — empirically, 136,507 admissions failed to complete within the 12-hour partition limit on 2 L4 GPUs. The safe per-GPU throughput is approximately **20,000 admissions per 12-hour window**.

The `BERT_SLICE_SIZE_PER_GPU` config key sets this limit. With 2 GPUs, each SLURM job covers `BERT_SLICE_SIZE_PER_GPU × n_gpus` admissions. The total number of slices is computed at runtime — no manual counting required. Reducing `BERT_SLICE_SIZE_PER_GPU` fits shorter partition windows (e.g. `L4-4h`); increasing it reduces the number of jobs needed on faster hardware.

#### Slice Layout (default: `BERT_SLICE_SIZE_PER_GPU = 20000`, 2 GPUs)

Each job covers 40,000 admissions (20,000 per GPU). 14 jobs cover the full 546,028.

| Slice | `hadm_id` rows | Per-GPU rows |
|-------|---------------|-------------|
| 0 | 0 – 39,999 | 20,000 |
| 1 | 40,000 – 79,999 | 20,000 |
| 2 | 80,000 – 119,999 | 20,000 |
| 3 | 120,000 – 159,999 | 20,000 |
| 4 | 160,000 – 199,999 | 20,000 |
| 5 | 200,000 – 239,999 | 20,000 |
| 6 | 240,000 – 279,999 | 20,000 |
| 7 | 280,000 – 319,999 | 20,000 |
| 8 | 320,000 – 359,999 | 20,000 |
| 9 | 360,000 – 399,999 | 20,000 |
| 10 | 400,000 – 439,999 | 20,000 |
| 11 | 440,000 – 479,999 | 20,000 |
| 12 | 480,000 – 519,999 | 20,000 |
| 13 | 520,000 – 546,027 | ~13,000 |

#### CLI Interface

```bash
python embed_features.py --config config/preprocessing.yaml \
                         --slice-index 5
```

- `BERT_SLICE_SIZE_PER_GPU` (config key): admissions per GPU per job. Default: `20000`. This is the single knob for tuning time-window fit.
- `--slice-index` (CLI arg, also `BERT_SLICE_INDEX` in config): 0-based index of the slice this job processes. `src/preprocessing/submit_all.sh` sets this per job automatically.
- `n_slices` is never passed manually — it is always computed as `ceil(total_admissions / (BERT_SLICE_SIZE_PER_GPU × n_gpus))`.

#### Slice Computation

```python
per_job = config["BERT_SLICE_SIZE_PER_GPU"] * n_gpus
n_slices = math.ceil(len(all_hadm_ids) / per_job)
slice_start = slice_index * per_job
slice_end   = min(slice_start + per_job, len(all_hadm_ids))
slice_hadm_ids = set(all_hadm_ids[slice_start:slice_end])
```

`all_hadm_ids` is the sorted list of all `hadm_id` values from `data_splits.parquet` — deterministic and reproducible across runs.

#### Output Appending

Each slice appends its rows to the same feature parquet using `fastparquet` append mode:

```python
import fastparquet as fp

if output_path.exists():
    fp.write(str(output_path), df_slice, compression="snappy", append=True)
else:
    fp.write(str(output_path), df_slice, compression="snappy")
```

This adds a new row group to the parquet file. The resulting file is a valid multi-row-group parquet readable by `pandas`/`pyarrow` without modification.

#### Slice-Level Completeness Check

`check_embed_status.py` determines whether a slice is complete by comparing the number of embedded rows in each output parquet against the expected cumulative row count after each slice. A slice is considered done if all 18 feature parquets contain rows for all `hadm_id` values in that slice's range.

Slices always run sequentially (chained via `--dependency=afterok`) — this ensures each slice appends cleanly to the previous slice's output without concurrent write conflicts.

---

### Multi-GPU Execution Within a Slice

Each embed SLURM job processes one slice using 2 GPUs. The slice's `hadm_ids` are split evenly between the two workers.

```
Main process (one slice: ~40k admissions)
├── Load all input parquets, filter to slice_hadm_ids
├── Split slice into GPU-0 half and GPU-1 half (~20k each)
├── Apply LPT load balancing across 18 features per GPU half (22 currently)
│
├── Worker 0 (cuda:0) — embeds its feature × hadm_id assignments
└── Worker 1 (cuda:1) — embeds its feature × hadm_id assignments
         │
         └── Both workers append results to shared output parquets
             (sequential per-feature writes, no concurrent file access)
```

`torch.multiprocessing.get_context("spawn")` is required for CUDA. Single-GPU path runs in-process.

---

### GPU Load Balancing

**Problem:** Naive round-robin assignment based on feature order can concentrate large features on the same GPU. For example, `diag_history` (546k texts × 512 tokens) and `blood_chemistry` (371k texts × 2,048 tokens) landing on GPU 0 causes GPU 0 to run for ~11 hours while GPU 1 finishes in ~2 hours.

**Solution — Longest Processing Time (LPT) scheduling:**

1. Estimate each feature's compute cost for this slice: `cost = len(slice_texts) × max_length_cap`
2. Sort all 18 features by cost **descending**
3. Assign features to GPUs using **greedy interleaving** — each feature goes to the GPU with the lowest accumulated cost so far

```
sorted features (by cost, high → low):
  discharge_history (40k × 4096)  → GPU 0  (cost: 164M)
  blood_hematology  (38k × 2048)  → GPU 1  (cost:  78M)
  blood_chemistry   (37k × 2048)  → GPU 0  (running: 240M) → GPU 1 lower → GPU 1
  diag_history      (40k × 512)   → GPU 0  (running: 184M) → GPU 0
  ...
```

**Implementation in `run()`:**
```python
tasks.sort(key=lambda t: len(t["texts"]) * t["max_length"], reverse=True)
gpu_loads = [0] * n_gpus
gpu_tasks = [[] for _ in range(n_gpus)]
for task in tasks:
    g = gpu_loads.index(min(gpu_loads))
    gpu_tasks[g].append(task)
    gpu_loads[g] += len(task["texts"]) * task["max_length"]
```

---

### Feature-Level Resume

Before embedding each feature within a slice, the code checks whether this slice's rows are already present in the output parquet. If the feature parquet exists and contains all expected `hadm_id` values for this slice, the feature is skipped entirely for this slice.

| Check | Catches |
|-------|---------|
| Output parquet exists | Slice never started for this feature |
| Slice's hadm_ids all present | Partial slice write |
| No null values for slice rows | Partial embedding within slice |

Features passing all checks are logged as `[SKIP slice=N feature=X]` and not re-embedded. Override with `BERT_FORCE_REEMBED: true` in config.

**Atomic slice writes:** within a slice, each feature's batch is written to a `.tmp` file then `os.replace()`-d onto the append target, preventing a partially written batch from being mistaken for a complete one.

---

### Record-Level Resume

Feature-level resume (above) handles the case where a complete slice was written for a feature. Record-level resume handles the case where a slice was interrupted mid-feature — some rows were appended, but not all.

**Design:**

```
For each feature in each slice:
  1. Compute slice_hadm_ids (the hadm_ids this slice/feature should produce)
  2. If output parquet exists: load its hadm_ids → build already_done set
  3. pending = slice_hadm_ids − already_done
  4. If pending is empty: SKIP (feature-level resume handles this)
  5. Embed pending texts in batches of BERT_CHECKPOINT_INTERVAL rows
  6. After each interval: append batch to output parquet via fastparquet
  7. On slice completion: log row count
```

**Append mechanics (fastparquet):**
```python
import fastparquet as fp

# First write for this feature (no prior rows at all)
fp.write(output_path, df_batch, compression="snappy")

# Append — adds a new row group to the existing file
fp.write(output_path, df_batch, compression="snappy", append=True)
```

`fastparquet` append mode adds row groups to the existing parquet file without reading the full file. The result is a valid multi-row-group parquet readable by `pandas`/`pyarrow`.

**Checkpoint interval:** `BERT_CHECKPOINT_INTERVAL` — rows between appends. Default: 10,000. Too small → high I/O overhead. Too large → large re-work on kill (at most `BERT_CHECKPOINT_INTERVAL` rows lost per interruption).

**Configuration keys:**

| Key | Default | Description |
|-----|---------|-------------|
| `BERT_SLICE_SIZE_PER_GPU` | `20000` | Admissions per GPU per SLURM job; determines total slice count |
| `BERT_SLICE_INDEX` | `0` | Slice index for this job (overridden by `--slice-index` CLI arg) |
| `BERT_CHECKPOINT_INTERVAL` | `10000` | Rows between within-feature incremental appends |
| `BERT_FORCE_REEMBED` | `false` | If true, ignores all slice/feature/record-level state |

**Resume hierarchy summary:**

| Level | Granularity | Trigger | Action |
|-------|-------------|---------|--------|
| Slice-level | Per SLURM job | `check_embed_status.py` at submit time | Job not submitted if slice complete |
| Feature-level | Per feature per slice | All slice `hadm_ids` present in output | Feature skipped for this slice |
| Record-level | Per checkpoint interval | Some (not all) slice `hadm_ids` present | Only missing rows embedded |

---

## 5. Final Dataset Assembly

### Join diagram

```
data_splits.parquet          (546,028 rows)
        │
        ├── LEFT JOIN y_labels.parquet              on hadm_id
        │       y1_mortality, y2_readmission
        │
        ├── LEFT JOIN demographics_features.parquet  on hadm_id
        │       demographic_vec  [8 floats]
        │
        ├── LEFT JOIN diag_history_embeddings.parquet
        ├── LEFT JOIN discharge_history_embeddings.parquet
        ├── LEFT JOIN triage_embeddings.parquet
        ├── LEFT JOIN chief_complaint_embeddings.parquet
        ├── LEFT JOIN radiology_embeddings.parquet
        │       {feature}_embedding  [768 floats each]
        │
        └── LEFT JOIN lab_{group}_embeddings.parquet  ×13
                lab_{group}_embedding  [768 floats, zero vector if no events]
```

### Output schema

| Column | Type | Source |
|--------|------|--------|
| `subject_id` | int64 | `data_splits.parquet` |
| `hadm_id` | int64 | `data_splits.parquet` |
| `split` | str | `data_splits.parquet` |
| `y1_mortality` | int8 | `y_labels.parquet` |
| `y2_readmission` | float32 (NaN for deceased) | `y_labels.parquet` |
| `demographic_vec` | float32[8] | `demographics_features.parquet` |
| `diag_history_embedding` | float32[768] | embeddings/ |
| `discharge_history_embedding` | float32[768] | embeddings/ |
| `triage_embedding` | float32[768] | embeddings/ |
| `chief_complaint_embedding` | float32[768] | embeddings/ |
| `radiology_embedding` | float32[768] | embeddings/ |
| `lab_blood_gas_embedding` | float32[768] | embeddings/ |
| `lab_blood_chemistry_embedding` | float32[768] | embeddings/ |
| `lab_blood_hematology_embedding` | float32[768] | embeddings/ |
| `lab_urine_chemistry_embedding` | float32[768] | embeddings/ |
| `lab_urine_hematology_embedding` | float32[768] | embeddings/ |
| `lab_other_body_fluid_chemistry_embedding` | float32[768] | embeddings/ |
| `lab_other_body_fluid_hematology_embedding` | float32[768] | embeddings/ |
| `lab_ascites_embedding` | float32[768] | embeddings/ |
| `lab_pleural_embedding` | float32[768] | embeddings/ |
| `lab_csf_embedding` | float32[768] | embeddings/ |
| `lab_bone_marrow_embedding` | float32[768] | embeddings/ |
| `lab_joint_fluid_embedding` | float32[768] | embeddings/ |
| `lab_stool_embedding` | float32[768] | embeddings/ |

**Total: 24 columns** (3 metadata + 2 labels + 1 structured vector + 18 embeddings)

---

## 6. Configuration Reference

All configuration in `config/preprocessing.yaml`. No module reads this file directly — `run_pipeline.py` loads it and passes the dict to each module's `run(config)`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `MIMIC_DATA_DIR` | str | — | Root of MIMIC-IV download (`hosp/`, `icu/` subdirs) |
| `MIMIC_NOTE_DIR` | str | `MIMIC_DATA_DIR` | Root of `mimic-iv-note` (`note/` subdir) |
| `MIMIC_ED_DIR` | str | `MIMIC_DATA_DIR` | Root of `mimic-iv-ed` (`ed/` subdir) |
| `SPLIT_TRAIN` | float | `0.80` | Patient fraction for train |
| `SPLIT_DEV` | float | `0.10` | Patient fraction for dev |
| `SPLIT_TEST` | float | `0.10` | Patient fraction for test |
| `BERT_MODEL_NAME` | str | `Simonlee711/Clinical_ModernBERT` | HuggingFace model identifier |
| `BERT_MAX_LENGTH` | int | `8192` | Global token limit; per-feature caps applied on top |
| `BERT_BATCH_SIZE` | int | `32` | Base batch size; auto-scaled per feature |
| `BERT_DEVICE` | str | `cuda` | Falls back to CPU if CUDA unavailable |
| `BERT_MAX_GPUS` | int\|null | `null` | Max GPUs to use; null = all available |
| `BERT_SLICE_SIZE_PER_GPU` | int | `20000` | Admissions per GPU per SLURM job; controls number of slices |
| `BERT_SLICE_INDEX` | int | `0` | Slice index for this job; overridden by `--slice-index` CLI arg |
| `BERT_FORCE_REEMBED` | bool | `false` | Bypass all slice/feature/record-level resume |
| `BERT_CHECKPOINT_INTERVAL` | int | `10000` | Rows between within-feature checkpoint appends |
| `LAB_ADMISSION_WINDOW` | int\|`"full"` | `24` | Hours of lab events from `admittime`; `"full"` = entire admission |
| `HADM_LINKAGE_STRATEGY` | str | `"drop"` | `"drop"` or `"link"` for null `hadm_id` records |
| `HADM_LINKAGE_TOLERANCE_HOURS` | int | `1` | Tolerance in hours for time-window linkage |
| `PREPROCESSING_DIR` | str | `data/preprocessing` | Root output directory |
| `FEATURES_DIR` | str | `data/preprocessing/features` | Raw feature parquets |
| `EMBEDDINGS_DIR` | str | `data/preprocessing/features/embeddings` | Embedding parquets |
| `CLASSIFICATIONS_DIR` | str | `data/preprocessing/classifications` | Labels, config artefacts |
| `HASH_REGISTRY_PATH` | str | `data/preprocessing/source_hashes.json` | MD5 registry for incremental runs |

---

## 7. Memory Requirements

| Process | Peak RAM | Bottleneck |
|---------|----------|------------|
| `pipeline_job` | ~45 GB | `extract_discharge_history` loading 331k notes |
| `embed_job` main process (per slice) | ~8 GB | Loading slice-filtered feature parquets + subset of 16.8M lab rows |
| `embed_job` per GPU worker | ~2.5 GB | Model weights (570 MB) + embedding accumulation + batch buffers |
| `embed_job` total (2 GPUs, one slice) | ~13 GB | — |
| `combine_job` | ~8 GB | Loading all 18 embedding parquets simultaneously |

All embed SLURM jobs allocated 64 GB for safe headroom. The main-process memory footprint is substantially lower per slice than for the full dataset, since feature parquets are filtered to `slice_hadm_ids` before spawning workers.

`labevents` (79M rows) and `chartevents` are never loaded fully — streamed in 500k-row chunks in the extract phase. The embed phase loads the already-filtered `labs_features.parquet` (16.8M rows total), further trimmed to the slice's admission subset.
