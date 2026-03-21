# Fix Round 6 (Final): Last combos in `PANELS_30`

## Assessment

EAR and EYE specimens are left unassigned intentionally — they are not
sufficiently related to any existing panel to warrant inclusion.

After this fix, ~2,211 rows will remain unassigned (0.055% of 3.99M total —
genuine noise including EAR, EYE, and rare tail combos).
**The panel mapping is final after this round — no further iteration needed.**

---

## `throat_strep` — add 1 combo
```python
("R/O Beta Strep Group A", "SWAB"),  # 30
```

## `respiratory_viral` — add 9 combos
```python
("Respiratory Viral Antigen Screen", "SWAB"),                                # 30
("Respiratory Viral Culture",        "ASPIRATE"),                            # 30
("VIRAL CULTURE",                    "BRONCHIAL WASHINGS"),                  # 29
("Respiratory Viral Culture",        "SWAB"),                                # 27
("DIRECT INFLUENZA B ANTIGEN TEST",  "ASPIRATE"),                            # 25
("DIRECT INFLUENZA A ANTIGEN TEST",  "ASPIRATE"),                            # 25
("VIRAL CULTURE",                    "RAPID RESPIRATORY VIRAL ANTIGEN TEST"), # 25
("DIRECT INFLUENZA A ANTIGEN TEST",  "SWAB"),                                # 22
("DIRECT INFLUENZA B ANTIGEN TEST",  "SWAB"),                                # 22
```

## `respiratory_pcp_legionella` — add 1 combo
```python
("LEGIONELLA CULTURE", "PLEURAL FLUID"),  # 22
```

## `wound_culture` — add 4 combos
```python
("FLUID CULTURE",     "TISSUE"),       # 30
("ANAEROBIC CULTURE", "FLUID WOUND"),  # 26
("NOCARDIA CULTURE",  "SWAB"),         # 22
("TISSUE",            "ABSCESS"),      # 20
```

## `fluid_culture` — add 3 combos
```python
("ANAEROBIC CULTURE",       "FLUID RECEIVED IN BLOOD CULTURE BOTTLES"),  # 23
("Fluid Culture in Bottles","PERITONEAL FLUID"),                          # 23
("FLUID CULTURE",           "FLUID RECEIVED IN BLOOD CULTURE BOTTLES"),  # 21
```

## `csf_culture` — add 1 combo
```python
("VARICELLA-ZOSTER CULTURE", "CSF;SPINAL FLUID"),  # 22
```

## `fungal_tissue_wound` — add 3 combos
# Note: EYE and EAR specimens are intentionally excluded
```python
("ACID FAST CULTURE",                     "ASPIRATE"),  # 25
("MODIFIED ACID-FAST STAIN FOR NOCARDIA", "TISSUE"),   # 25
("ACID FAST SMEAR",                       "ASPIRATE"),  # 23
```

## `fungal_respiratory` — add 1 combo
```python
("FUNGAL CULTURE", "BRONCHIAL BRUSH - PROTECTED"),  # 23
```

## `fungal_fluid` — add 2 combos
```python
("ACID FAST CULTURE", "DIALYSIS FLUID"),  # 27
("ACID FAST SMEAR",   "DIALYSIS FLUID"),  # 27
```

## `herpesvirus_culture_antigen` — add 6 combos
```python
("Direct Antigen Test for Herpes Simplex Virus Types 1 & 2", "DIRECT ANTIGEN TEST FOR VARICELLA-ZOSTER VIRUS"),  # 28
("VARICELLA-ZOSTER CULTURE",               "Direct Antigen Test for Herpes Simplex Virus Types 1 & 2"),          # 27
("VIRAL CULTURE: R/O HERPES SIMPLEX VIRUS","STOOL"),                                                             # 26
("VIRAL CULTURE: R/O CYTOMEGALOVIRUS",     "SWAB"),                                                              # 24
("VARICELLA-ZOSTER CULTURE",               "VARICELLA-ZOSTER CULTURE"),                                          # 24
("VIRAL CULTURE",                          "STOOL (RECEIVED IN TRANSPORT SYSTEM)"),                              # 20
```

## `herpesvirus_serology` — add 3 combos
```python
("EPSTEIN-BARR VIRUS EBNA IgG AB", "SEROLOGY/BLOOD"),  # 28
("EPSTEIN-BARR VIRUS VCA-IgG AB",  "SEROLOGY/BLOOD"),  # 28
("EPSTEIN-BARR VIRUS VCA-IgM AB",  "SEROLOGY/BLOOD"),  # 28
```

## `misc_serology` — add 1 combo
```python
("TOXOPLASMA IgG ANTIBODY", "SEROLOGY/BLOOD"),  # 28
```

## `stool_parasitology` — add 1 combo
```python
("OVA + PARASITES", "BRONCHOALVEOLAR LAVAGE"),  # 28
```

## `EXCLUDED_TESTS` — add 1 test
```python
"POST-MORTEM ACID-FAST CULTURE",
```

## What NOT to add (intentionally left unassigned)
- `("FUNGAL CULTURE", "EYE")` — EYE specimens excluded
- `("POTASSIUM HYDROXIDE PREPARATION", "CORNEAL EYE SCRAPINGS")` — EYE excluded
- `("POTASSIUM HYDROXIDE PREPARATION", "EAR")` — EAR specimens excluded
- `("GRAM STAIN", "CORNEAL EYE SCRAPINGS")` — EYE excluded

---

## Final state after applying

- **Unassigned: ~2,211 rows** (0.055% of 3.99M — includes EAR/EYE by design)
- **30 panels confirmed final**
- The `PANELS_30` dict in the notebook is the canonical source of truth
  for the `extract_microbiology.py` extraction module
