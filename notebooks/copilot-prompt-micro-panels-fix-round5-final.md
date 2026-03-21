# Fix Round 5 (Final): Last remaining unassigned combos in `PANELS_30`

## Context

After round 4, 7,690 rows remain unassigned. This prompt fixes all combos
with ≥31 events. After applying, ~3,093 rows will remain unassigned —
all from combos with fewer than 31 events each (~0.08% of 3.99M total).
The panel mapping is then final — no further iteration needed.

---

## `blood_bottle_gram_stain` — add 4 combos
```python
("Aerobic Bottle Gram Stain",  "BLOOD CULTURE (POST-MORTEM)"),              # 231
("Anaerobic Bottle Gram Stain","BLOOD CULTURE (POST-MORTEM)"),              # 198
("ANAEROBIC BOTTLE",           "FLUID RECEIVED IN BLOOD CULTURE BOTTLES"),  # 135
("AEROBIC BOTTLE",             "FLUID RECEIVED IN BLOOD CULTURE BOTTLES"),  # 125
```

## `csf_culture` — add 2 combos
```python
("ACID FAST SMEAR",                          "CSF;SPINAL FLUID"),  # 105
("VIRAL CULTURE: R/O HERPES SIMPLEX VIRUS",  "CSF;SPINAL FLUID"),  #  73
```

## `fungal_fluid` — add 4 combos
```python
("FUNGAL CULTURE",                  "PROSTHETIC JOINT FLUID"),  # 45
("ACID FAST CULTURE",               "PROSTHETIC JOINT FLUID"),  # 34
("ACID FAST SMEAR",                 "PROSTHETIC JOINT FLUID"),  # 34
("POTASSIUM HYDROXIDE PREPARATION", "BILE"),                    # 31
```

## `fungal_tissue_wound` — add 3 combos
```python
("ED Gram Stain for Yeast",                         "Swab"),                   # 859
("POTASSIUM HYDROXIDE PREPARATION (HAIR/SKIN/NAILS)","NAIL SCRAPINGS"),         # 295
("FUNGAL CULTURE",                                   "CORNEAL EYE SCRAPINGS"),  #  64
```

## `gc_chlamydia_sti` — add 1 combo
```python
("R/O GC Only", "THROAT CULTURE"),  # 50
```

## `gram_stain_wound_fluid` — add 3 combos
```python
("GRAM STAIN", "CORNEAL EYE SCRAPINGS"),               # 53
("GRAM STAIN", "FLUID WOUND"),                         # 43
("GRAM STAIN", "FLUID RECEIVED IN BLOOD CULTURE BOTTLES"),  # 36
```

## `herpesvirus_culture_antigen` — add 8 combos
```python
("VARICELLA-ZOSTER CULTURE",                               "VIRAL CULTURE:R/O HERPES SIMPLEX VIRUS"),  # 68
("VIRAL CULTURE",                                          "BIOPSY"),                                   # 46
("Direct Antigen Test for Herpes Simplex Virus Types 1 & 2","SWAB"),                                   # 45
("CYTOMEGALOVIRUS EARLY ANTIGEN TEST (SHELL VIAL METHOD)", "BIOPSY"),                                   # 44
("VIRAL CULTURE: R/O HERPES SIMPLEX VIRUS",                "BIOPSY"),                                   # 38
("VIRAL CULTURE: R/O HERPES SIMPLEX VIRUS",                "THROAT FOR STREP"),                         # 31
("DIRECT ANTIGEN TEST FOR VARICELLA-ZOSTER VIRUS",         "SWAB"),                                     # 31
("VIRAL CULTURE: R/O HERPES SIMPLEX VIRUS",                "VIRAL CULTURE: R/O CYTOMEGALOVIRUS"),       # 31
```

## `herpesvirus_serology` — add 2 combos
```python
("CMV IgM ANTIBODY", "SEROLOGY/BLOOD"),  # 32
("CMV IgG ANTIBODY", "SEROLOGY/BLOOD"),  # 32
```

## `resistance_screen` — add 1 combo
```python
("Cipro Resistant Screen", "Cipro Resistant Screen"),  # 250
```

## `respiratory_viral` — add 2 combos
```python
("Respiratory Viral Antigen Screen", "RAPID RESPIRATORY VIRAL ANTIGEN TEST"),  # 88
("Respiratory Viral Culture",        "RAPID RESPIRATORY VIRAL ANTIGEN TEST"),  # 53
```

## `stool_parasitology` — add 6 combos
```python
("Cryptosporidium/Giardia (DFA)",    "STOOL (RECEIVED IN TRANSPORT SYSTEM)"),  # 768
("O&P MACROSCOPIC EXAM - ARTHROPOD", "ARTHROPOD"),                              # 105
("SCOTCH TAPE PREP/PADDLE",          "SCOTCH TAPE PREP/PADDLE"),                #  96
("OVA + PARASITES",                  "URINE"),                                  #  81
("O&P MACROSCOPIC EXAM - WORM",      "WORM"),                                   #  49
("OVA + PARASITES",                  "SPUTUM"),                                 #  31
```

## `vaginal_genital_flora` — add 1 combo
```python
("SMEAR FOR BACTERIAL VAGINOSIS", "Swab"),  # 151  (lowercase 's' variant)
```

## `wound_culture` — add 3 combos
```python
("ANAEROBIC CULTURE", "POSTMORTEM CULTURE"),  # 46
("ANAEROBIC CULTURE", "FOOT CULTURE"),        # 39
("NOCARDIA CULTURE",  "ABSCESS"),             # 31
```

---

## Expected result

- Unassigned: ~3,093 rows (0.08% of 3.99M total)
- All remaining unassigned are combos with <31 events — genuine statistical
  noise and data-entry quirks, not worth further iteration
- Panel mapping is **final**
