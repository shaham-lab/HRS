# MIMIC-IV Feature Set and Target Set — Reward Model Reference

> **Location:** `HRS/src/reward_model/mimic4_feature_set.md`
>
> This document is the authoritative reference for the feature slots and classification targets used by the Reward Model when trained on MIMIC-IV data. For the generic feature set contract see Section 4 of `reward_model_architecture.md`. For MIMIC-IV configuration defaults see Section 15 of `reward_model_architecture.md`.

---

## 1. Classification Targets (T=2)

| Index | Column | Definition | Population | NaN convention |
|-------|--------|------------|------------|----------------|
| 0 — Y1 | `y1_mortality` | In-hospital mortality (`admissions.hospital_expire_flag = 1`) | All admissions | Never NaN; ~8–10% positive rate |
| 1 — Y2 | `y2_readmission` | Unplanned readmission within 30 days of `dischtime` | Survivors only (`y1_mortality = 0`) | NaN when `y1_mortality = 1`; ~20% positive among survivors |

**NaN assignment rule:** `y2_readmission = NaN` must be set for all admissions where `y1_mortality = 1`. This is enforced by `extract_y_data.py` in `HRS/src/preprocessing` and validated by `Mimic4DataLoader._validate_labels()` at load time. Readmission is undefined for deceased patients — the reward model head for Y2 learns `P(readmitted | survived)`. The per-batch dynamic NaN mask in `RewardModelManager.compute_loss()` excludes deceased patients from the Y2 loss without removing rows from the dataset.

The following assertions must pass after NaN assignment:
```python
assert df.loc[df['y1_mortality'] == 1, 'y2_readmission'].isna().all()
assert df.loc[df['y1_mortality'] == 0, 'y2_readmission'].notna().all()
```

A surviving patient with a missing Y2 label indicates a data quality issue upstream — it must not be silently absorbed into the NaN masking.

---

## 2. Feature Summary

| Property | Value |
|----------|-------|
| Total feature slots (N) | 56 |
| Always-visible slots | 5 (F1–F5) |
| Maskable slots (M) | 51 (F6–F56) |
| Total input dimensionality (D) | 8 + (55 × 768) = **42,248** |
| Embedding model | Clinical_ModernBERT (`Simonlee711/Clinical_ModernBERT`) |
| Embedding dimensionality | 768 per slot (mean pooling over tokens) |
| Missing feature convention | Zero vector of the correct dimensionality |

---

## 3. Feature Table

| ID | Column name | Dim | RL visibility | Description |
|----|------------|-----|---------------|-------------|
| F1 | `demographic_vec` | 8 | Always visible | Age, gender, height, weight, BMI + 3 missingness flags |
| F2 | `diag_history_embedding` | 768 | Always visible | Diagnosis history from prior admissions |
| F3 | `discharge_history_embedding` | 768 | Always visible | Discharge summary history from prior admissions |
| F4 | `triage_embedding` | 768 | Always visible | Triage data from current visit |
| F5 | `chief_complaint_embedding` | 768 | Always visible | Chief complaint from current visit |
| F6 | `lab_blood_gas_embedding` | 768 | Maskable | Lab group: blood gas |
| F7 | `lab_blood_chemistry_embedding` | 768 | Maskable | Lab group: blood chemistry |
| F8 | `lab_blood_hematology_embedding` | 768 | Maskable | Lab group: blood hematology |
| F9 | `lab_urine_chemistry_embedding` | 768 | Maskable | Lab group: urine chemistry |
| F10 | `lab_urine_hematology_embedding` | 768 | Maskable | Lab group: urine hematology |
| F11 | `lab_other_body_fluid_chemistry_embedding` | 768 | Maskable | Lab group: other body fluid chemistry |
| F12 | `lab_other_body_fluid_hematology_embedding` | 768 | Maskable | Lab group: other body fluid hematology |
| F13 | `lab_ascites_embedding` | 768 | Maskable | Lab group: ascites |
| F14 | `lab_pleural_embedding` | 768 | Maskable | Lab group: pleural |
| F15 | `lab_csf_embedding` | 768 | Maskable | Lab group: cerebrospinal fluid |
| F16 | `lab_bone_marrow_embedding` | 768 | Maskable | Lab group: bone marrow |
| F17 | `lab_joint_fluid_embedding` | 768 | Maskable | Lab group: joint fluid |
| F18 | `lab_stool_embedding` | 768 | Maskable | Lab group: stool |
| F19 | `radiology_embedding` | 768 | Maskable | Radiology note from current visit |
| F20 | `micro_blood_culture_routine_embedding` | 768 | Maskable | Microbiology: blood culture routine |
| F21 | `micro_blood_bottle_gram_stain_embedding` | 768 | Maskable | Microbiology: blood bottle gram stain |
| F22 | `micro_urine_culture_embedding` | 768 | Maskable | Microbiology: urine culture |
| F23 | `micro_urine_viral_embedding` | 768 | Maskable | Microbiology: urine viral |
| F24 | `micro_urinary_antigens_embedding` | 768 | Maskable | Microbiology: urinary antigens |
| F25 | `micro_respiratory_non_invasive_embedding` | 768 | Maskable | Microbiology: respiratory non-invasive |
| F26 | `micro_respiratory_invasive_embedding` | 768 | Maskable | Microbiology: respiratory invasive |
| F27 | `micro_respiratory_afb_embedding` | 768 | Maskable | Microbiology: respiratory AFB |
| F28 | `micro_respiratory_viral_embedding` | 768 | Maskable | Microbiology: respiratory viral |
| F29 | `micro_respiratory_pcp_legionella_embedding` | 768 | Maskable | Microbiology: respiratory PCP/Legionella |
| F30 | `micro_gram_stain_respiratory_embedding` | 768 | Maskable | Microbiology: gram stain respiratory |
| F31 | `micro_gram_stain_wound_tissue_embedding` | 768 | Maskable | Microbiology: gram stain wound/tissue |
| F32 | `micro_gram_stain_csf_embedding` | 768 | Maskable | Microbiology: gram stain CSF |
| F33 | `micro_wound_culture_embedding` | 768 | Maskable | Microbiology: wound culture |
| F34 | `micro_hardware_and_lines_culture_embedding` | 768 | Maskable | Microbiology: hardware and lines culture |
| F35 | `micro_pleural_culture_embedding` | 768 | Maskable | Microbiology: pleural culture |
| F36 | `micro_peritoneal_culture_embedding` | 768 | Maskable | Microbiology: peritoneal culture |
| F37 | `micro_joint_fluid_culture_embedding` | 768 | Maskable | Microbiology: joint fluid culture |
| F38 | `micro_fluid_culture_embedding` | 768 | Maskable | Microbiology: fluid culture |
| F39 | `micro_bone_marrow_culture_embedding` | 768 | Maskable | Microbiology: bone marrow culture |
| F40 | `micro_csf_culture_embedding` | 768 | Maskable | Microbiology: CSF culture |
| F41 | `micro_fungal_tissue_wound_embedding` | 768 | Maskable | Microbiology: fungal tissue/wound |
| F42 | `micro_fungal_respiratory_embedding` | 768 | Maskable | Microbiology: fungal respiratory |
| F43 | `micro_fungal_fluid_embedding` | 768 | Maskable | Microbiology: fungal fluid |
| F44 | `micro_mrsa_staph_screen_embedding` | 768 | Maskable | Microbiology: MRSA/Staph screen |
| F45 | `micro_resistance_screen_embedding` | 768 | Maskable | Microbiology: resistance screen |
| F46 | `micro_cdiff_embedding` | 768 | Maskable | Microbiology: C. difficile |
| F47 | `micro_stool_bacterial_embedding` | 768 | Maskable | Microbiology: stool bacterial |
| F48 | `micro_stool_parasitology_embedding` | 768 | Maskable | Microbiology: stool parasitology |
| F49 | `micro_herpesvirus_serology_embedding` | 768 | Maskable | Microbiology: herpesvirus serology |
| F50 | `micro_hepatitis_hiv_embedding` | 768 | Maskable | Microbiology: hepatitis/HIV |
| F51 | `micro_syphilis_serology_embedding` | 768 | Maskable | Microbiology: syphilis serology |
| F52 | `micro_misc_serology_embedding` | 768 | Maskable | Microbiology: miscellaneous serology |
| F53 | `micro_herpesvirus_culture_antigen_embedding` | 768 | Maskable | Microbiology: herpesvirus culture/antigen |
| F54 | `micro_gc_chlamydia_sti_embedding` | 768 | Maskable | Microbiology: GC/Chlamydia/STI |
| F55 | `micro_vaginal_genital_flora_embedding` | 768 | Maskable | Microbiology: vaginal/genital flora |
| F56 | `micro_throat_strep_embedding` | 768 | Maskable | Microbiology: throat/Strep |

---

## Slot Group Summary

| Group | Slots | Count | Always visible | Dim each |
|-------|-------|-------|----------------|----------|
| Demographics | F1 | 1 | Yes | 8 |
| History and triage | F2–F5 | 4 | Yes | 768 |
| Lab groups | F6–F18 | 13 | No | 768 |
| Radiology | F19 | 1 | No | 768 |
| Microbiology panels | F20–F56 | 37 | No | 768 |
| **Total** | **F1–F56** | **56** | **5** | |

---

## Notes

- F1 (`demographic_vec`) is an 8-dimensional numeric vector, not a BERT embedding. Components: age, gender, height, weight, BMI, and 3 missingness flags for height/weight/BMI.
- All embedding slots (F2–F56) use mean pooling over Clinical_ModernBERT tokens. The same model and pooling strategy is applied to all text fields regardless of clinical domain.
- Missing features (no lab results for a group, no radiology note) are represented as a zero vector of the correct dimensionality — **never NaN**. NaN inside a feature vector propagates silently through the MLP and corrupts all downstream computations.
- The canonical column order defined here is enforced by `combine_dataset.py` in `HRS/src/preprocessing` and validated by `Mimic4DataLoader` at load time. Any upstream change to feature count or column order is detected at startup via the schema validation step.
- The feature index map (start/end indices for each slot within the 42,248-dim vector) is derived automatically at load time from this column order. See `PREPROCESSING_DATA_MODEL.md` Section 3.12 for the full schema specification.
