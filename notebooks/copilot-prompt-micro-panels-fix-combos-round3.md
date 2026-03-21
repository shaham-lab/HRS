# Fix Round 3 (Final): Add remaining missing combos to `PANELS_30`

## Context

After round 2, 4,042 rows remain unassigned. This is the final fix.
After applying these, ~2,400 rows will remain unassigned — all from combos
with fewer than 28 events each, representing the genuine long tail (0.14%
of total events). No further iteration needed and no new panels required.

Add each tuple to the `"combos"` list of the named panel. Do not change
anything else.

---

## `blood_culture_routine` — add 1 combo
```python
("ISOLATE FOR MIC", "BLOOD CULTURE"),  # 28
```

## `csf_culture` — add 1 combo
```python
("ANAEROBIC CULTURE", "CSF;SPINAL FLUID"),  # 36
```

## `fluid_culture` — add 2 combos
```python
("ANAEROBIC CULTURE", "DIALYSIS FLUID"),  # 45
("WOUND CULTURE",     "PERITONEAL FLUID"), # 32
```

## `fungal_fluid` — add 2 combos
```python
("ACID FAST CULTURE", "BILE"),  # 34
("ACID FAST SMEAR",   "BILE"),  # 34
```

## `fungal_respiratory` — add 2 combos
```python
("FUNGAL CULTURE",                  "BRONCHIAL BRUSH"),  # 60
("POTASSIUM HYDROXIDE PREPARATION", "BRONCHIAL BRUSH"),  # 35
```

## `fungal_tissue_wound` — add 5 combos
```python
("FUNGAL CULTURE",                  "BIOPSY"),    # 59
("FUNGAL CULTURE",                  "EAR"),       # 40
("POTASSIUM HYDROXIDE PREPARATION", "BIOPSY"),    # 35
("POTASSIUM HYDROXIDE PREPARATION", "ASPIRATE"),  # 35
("POTASSIUM HYDROXIDE PREPARATION", "FOREIGN BODY"), # 30
```

## `gram_stain_respiratory` — add 1 combo
```python
("GRAM STAIN", "BRONCHIAL BRUSH"),  # 54
```

## `gram_stain_wound_fluid` — add 4 combos
```python
("GRAM STAIN", "EAR"),   # 63
("GRAM STAIN", "FOOT CULTURE"),  # 38
("GRAM STAIN", "EYE"),   # 31
("GRAM STAIN", "URINE"), # 29
```

## `herpesvirus_culture_antigen` — add 6 combos
```python
("DIRECT ANTIGEN TEST FOR VARICELLA-ZOSTER VIRUS",        "Direct Antigen Test for Herpes Simplex Virus Types 1 & 2"),  # 56
("VIRAL CULTURE",                                          "THROAT FOR STREP"),  # 53
("VIRAL CULTURE: R/O CYTOMEGALOVIRUS",                     "BIOPSY"),  # 48
("CYTOMEGALOVIRUS EARLY ANTIGEN TEST (SHELL VIAL METHOD)", "VIRAL CULTURE: R/O CYTOMEGALOVIRUS"),  # 37
("VARICELLA-ZOSTER CULTURE",                               "TISSUE"),  # 29
("VIRAL CULTURE: R/O HERPES SIMPLEX VIRUS",                "Direct Antigen Test for Herpes Simplex Virus Types 1 & 2"),  # 28
```

## `mrsa_staph_screen` — add 1 combo
```python
("Staph aureus Screen", "SWAB"),  # 38
```

## `respiratory_pcp_legionella` — add 1 combo
```python
("LEGIONELLA CULTURE", "SWAB"),  # 31
```

## `respiratory_viral` — add 6 combos
```python
("CYTOMEGALOVIRUS EARLY ANTIGEN TEST (SHELL VIAL METHOD)", "BRONCHIAL WASHINGS"),                   # 56
("VIRAL CULTURE: R/O CYTOMEGALOVIRUS",                     "Mini-BAL"),                             # 45
("CYTOMEGALOVIRUS EARLY ANTIGEN TEST (SHELL VIAL METHOD)", "Rapid Respiratory Viral Screen & Culture"), # 42
("VIRAL CULTURE: R/O HERPES SIMPLEX VIRUS",                "BRONCHOALVEOLAR LAVAGE"),               # 40
("Respiratory Viral Antigen Screen",                       "ASPIRATE"),                             # 33
("VIRAL CULTURE",                                          "Influenza A/B by DFA"),                 # 30
```

## `stool_bacterial` — add 1 combo
```python
("CAMPYLOBACTER CULTURE", "STOOL (RECEIVED IN TRANSPORT SYSTEM)"),  # 41
```

## `urine_culture` — add 2 combos
```python
("VIRAL CULTURE: R/O CYTOMEGALOVIRUS", "URINE"),                # 37
("FLUID CULTURE",                      "URINE,SUPRAPUBIC ASPIRATE"), # 32
```

## `wound_culture` — add 5 combos
```python
("TISSUE",              "SWAB"),             # 61
("RESPIRATORY CULTURE", "FLUID,OTHER"),      # 60
("WOUND CULTURE",       "CORNEAL EYE SCRAPINGS"), # 50
("RESPIRATORY CULTURE", "THROAT FOR STREP"), # 41
("TISSUE CULTURE-TISSUE","TISSUE"),          # 37
```

---

## Expected result after applying

- Unassigned: ~2,400 rows (0.14% of total — genuine long tail, no further action needed)
- All 30 panels confirmed final
- No new panels required — all remaining combos map to existing panels
