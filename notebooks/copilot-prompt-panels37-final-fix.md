# Fix: Final Unassigned Combos in `PANELS_37` (Section 6)

## Context

After running the 37-panel notebook, 7,400 rows remain unassigned.
This prompt fixes all assignable combos. After applying, the residual
will be ~1,942 rows (417 intentional EAR/EYE + ~1,525 tail combos
with <14 events each — 0.049% of 3.99M total).

**The panel mapping is final after this fix.**

---

## `blood_bottle_gram_stain` — add 3 combos
```python
("BLOOD/FUNGAL CULTURE",         "FLUID RECEIVED IN BLOOD CULTURE BOTTLES"),  # 131
("BLOOD/AFB CULTURE",            "FLUID RECEIVED IN BLOOD CULTURE BOTTLES"),  # 110
("Anaerobic Bottle Gram Stain",  "Stem Cell - Blood Culture"),                #  15
```

## `urine_culture` — add 1 combo
```python
("FUNGAL CULTURE", "URINE"),  # 406
```

## `respiratory_invasive` — add 2 combos
```python
("FLUID CULTURE",       "BRONCHIAL WASHINGS"),   # 18
("ANAEROBIC CULTURE",   "BRONCHOALVEOLAR LAVAGE"), # 15
```

## `respiratory_afb` — add 1 combo
```python
("MTB Direct Amplification", "BRONCHIAL WASHINGS"),  # 18
```

## `respiratory_viral` — add 12 combos
```python
("VIRAL CULTURE: R/O CYTOMEGALOVIRUS",                     "BRONCHOALVEOLAR LAVAGE"),                   # 1289
("VIRAL CULTURE",                                          "BRONCHOALVEOLAR LAVAGE"),                   #  212
("Respiratory Viral Antigen Screen",                       "RAPID RESPIRATORY VIRAL ANTIGEN TEST"),     #   88
("Respiratory Viral Culture",                              "RAPID RESPIRATORY VIRAL ANTIGEN TEST"),     #   53
("CYTOMEGALOVIRUS EARLY ANTIGEN TEST (SHELL VIAL METHOD)", "Mini-BAL"),                                 #   26
("VARICELLA-ZOSTER CULTURE",                               "BRONCHOALVEOLAR LAVAGE"),                   #   17
("VIRAL CULTURE",                                          "SPUTUM"),                                   #   17
("Respiratory Virus Identification",                       "ASPIRATE"),                                 #   16
("DIRECT INFLUENZA A ANTIGEN TEST",                        "Influenza A/B by DFA - Bronch Lavage"),     #   15
("Respiratory Virus Identification",                       "RAPID RESPIRATORY VIRAL ANTIGEN TEST"),     #   15
("DIRECT INFLUENZA B ANTIGEN TEST",                        "Influenza A/B by DFA - Bronch Lavage"),     #   15
("DIRECT INFLUENZA B ANTIGEN TEST",                        "Rapid Respiratory Viral Screen & Culture"), #   14
```

## `respiratory_pcp_legionella` — add 1 combo
```python
("NOCARDIA CULTURE", "BRONCHIAL BRUSH"),  # 20
```

## `peritoneal_culture` — add 1 combo
```python
("WOUND CULTURE", "PERITONEAL FLUID"),  # 45
```

## `fungal_tissue_wound` — add 4 combos
```python
("ED Gram Stain for Yeast", "Swab"),       # 859  (lowercase 's' variant of SWAB)
("ACID FAST CULTURE",       "BIOPSY"),     # 162
("ACID FAST SMEAR",         "BIOPSY"),     # 162
("FUNGAL CULTURE",          "THROAT FOR STREP"),  # 19
```

## `fungal_fluid` — add 2 combos
```python
("ACID FAST SMEAR",    "BILE"),  # 46
("ACID FAST CULTURE",  "BILE"),  # 46
```

## `mrsa_staph_screen` — add 2 combos
```python
("MRSA SCREEN", "Staph aureus swab"),  # 17
("MRSA SCREEN", "STOOL"),             # 16
```

## `stool_parasitology` — add 2 combos
```python
("MICROSPORIDIA STAIN", "ASPIRATE"),   # 17
("OVA + PARASITES",     "TISSUE"),     # 14
```

## `herpesvirus_culture_antigen` — add 7 combos
```python
("VIRAL CULTURE: R/O HERPES SIMPLEX VIRUS",                "VIRAL CULTURE:R/O HERPES SIMPLEX VIRUS"),  # 875
("VIRAL CULTURE",                                          "TISSUE"),                                  # 221
("VIRAL CULTURE",                                          "SWAB"),                                    # 180
("VIRAL CULTURE",                                          "THROAT FOR STREP"),                        #  71
("VIRAL CULTURE",                                          "VIRAL CULTURE"),                           #  17
("Direct Antigen Test for Herpes Simplex Virus Types 1 & 2","VIRAL CULTURE:R/O HERPES SIMPLEX VIRUS"), #  15
("VIRAL CULTURE: R/O CYTOMEGALOVIRUS",                     "VIRAL CULTURE:R/O HERPES SIMPLEX VIRUS"),  #  15
```

## `misc_serology` — add 1 combo
```python
("TOXOPLASMA IgM ANTIBODY", "SEROLOGY/BLOOD"),  # 17
```

## `throat_strep` — add 1 combo
```python
("R/O Beta Strep Group A", "THROAT"),  # 14
```

## `wound_culture` — add 1 combo
```python
("TISSUE", "SWAB"),  # 120
```

---

## Intentionally NOT assigned (EAR / EYE — by design)
```
# 145  (FUNGAL CULTURE,                      EAR)
# 145  (FUNGAL CULTURE,                      EAR)  — duplicate row in output
#  64  (FUNGAL CULTURE,                      CORNEAL EYE SCRAPINGS)
#  55  (GRAM STAIN,                          EYE)
#  53  (GRAM STAIN,                          CORNEAL EYE SCRAPINGS)
#  22  (FUNGAL CULTURE,                      EYE)
#  22  (POTASSIUM HYDROXIDE PREPARATION,     CORNEAL EYE SCRAPINGS)
#  21  (POTASSIUM HYDROXIDE PREPARATION,     EAR)
#  20  (VIRAL CULTURE,                       EYE)
#  15  (WOUND CULTURE,                       EYE)
```
These remain unassigned intentionally. EAR and EYE specimens are not
clinically related to any existing panel.

---

## Expected result after applying

- **Unassigned: ~1,942 rows** (0.049% of 3.99M total)
  - 417 intentional EAR/EYE
  - ~1,525 tail combos with <14 events each
- **37 panels confirmed final**
- `PANELS_37` is the canonical source of truth for `extract_microbiology.py`
