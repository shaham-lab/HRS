# Fix: Add missing (test_name, spec_type_desc) combos to `PANELS_30` in Section 6

## Context

After running Section 6 in `mimic4_microbiology_exploration.ipynb`, the coverage
summary showed 50,449 unassigned rows. All can be fixed by adding missing combos
to the existing `PANELS_30` dict. Do not restructure or rename any panels — only
add tuples to the `"combos"` lists of existing panels.

---

## Changes — add to existing panel combo lists

### `blood_bottle_gram_stain` — add 2 combos
```python
("Aerobic Bottle Gram Stain",    "FLUID RECEIVED IN BLOOD CULTURE BOTTLES"),
("Anaerobic Bottle Gram Stain",  "FLUID RECEIVED IN BLOOD CULTURE BOTTLES"),
```

### `urine_culture` — add 2 combos
```python
("FLUID CULTURE",       "URINE,KIDNEY"),
("ANAEROBIC CULTURE",   "URINE"),
```

### `respiratory_sputum_bal` — add 1 combo
```python
("RESPIRATORY CULTURE", "ASPIRATE"),
```

### `respiratory_afb` — add 4 combos
```python
("ACID FAST CULTURE", "BRONCHIAL BRUSH"),
("ACID FAST SMEAR",   "BRONCHIAL BRUSH"),
("ACID FAST CULTURE", "TRACHEAL ASPIRATE"),
("ACID FAST SMEAR",   "TRACHEAL ASPIRATE"),
```

### `respiratory_viral` — add 2 combos
```python
("VIRAL CULTURE",             "Rapid Respiratory Viral Screen & Culture"),
("Respiratory Viral Culture", "Influenza A/B by DFA"),
```

### `wound_culture` — add 8 combos
```python
("FLUID CULTURE",       "SWAB"),
("RESPIRATORY CULTURE", "SWAB"),
("RESPIRATORY CULTURE", "EAR"),
("RESPIRATORY CULTURE", "EYE"),
("RESPIRATORY CULTURE", "Staph aureus swab"),
("ANAEROBIC CULTURE",   "Foreign Body - Sonication Culture"),
("ANAEROBIC CULTURE",   "BONE MARROW"),
("ANAEROBIC CULTURE",   "FOREIGN BODY"),
("TISSUE",              "BIOPSY"),
```

### `fluid_culture` — add 5 combos  ← CRITICAL: largest single fix (9,004 + 3,153 events)
```python
("FLUID CULTURE",       "FLUID,OTHER"),
("FLUID CULTURE",       "PROSTHETIC JOINT FLUID"),
("ANAEROBIC CULTURE",   "FLUID,OTHER"),
("ANAEROBIC CULTURE",   "JOINT FLUID"),
("FLUID CULTURE",       "URINE,KIDNEY"),   # if not already added to urine_culture above
```
Note: `("FLUID CULTURE", "URINE,KIDNEY")` should go in `urine_culture` not here —
only add it to `fluid_culture` if you did not add it to `urine_culture`.

### `gram_stain_wound_fluid` — add 1 combo
```python
("GRAM STAIN", "ASPIRATE"),
```

### `fungal_tissue_wound` — add 10 combos
```python
("ACID FAST CULTURE", "TISSUE"),
("ACID FAST SMEAR",   "TISSUE"),
("ACID FAST CULTURE", "SWAB"),
("ACID FAST SMEAR",   "SWAB"),
("ACID FAST CULTURE", "ABSCESS"),
("ACID FAST SMEAR",   "ABSCESS"),
("ACID FAST CULTURE", "BONE MARROW"),
("ACID FAST SMEAR",   "BONE MARROW"),
("ACID FAST CULTURE", "FOREIGN BODY"),
("ACID FAST SMEAR",   "FOREIGN BODY"),
("ACID FAST CULTURE", "BIOPSY"),
("ACID FAST SMEAR",   "BIOPSY"),
("ACID FAST CULTURE", "FOOT CULTURE"),
("ACID FAST SMEAR",   "FOOT CULTURE"),
("ACID FAST CULTURE", "CORNEAL EYE SCRAPINGS"),
("ACID FAST SMEAR",   "CORNEAL EYE SCRAPINGS"),
```

### `fungal_fluid` — add 9 combos
```python
("ACID FAST CULTURE", "PLEURAL FLUID"),
("ACID FAST SMEAR",   "PLEURAL FLUID"),
("ACID FAST CULTURE", "PERITONEAL FLUID"),
("ACID FAST SMEAR",   "PERITONEAL FLUID"),
("ACID FAST CULTURE", "JOINT FLUID"),
("ACID FAST SMEAR",   "JOINT FLUID"),
("ACID FAST CULTURE", "FLUID,OTHER"),
("ACID FAST SMEAR",   "FLUID,OTHER"),
("ACID FAST CULTURE", "URINE"),
("POTASSIUM HYDROXIDE PREPARATION", "FLUID,OTHER"),
```

### `herpesvirus_culture_antigen` — add 7 combos
```python
("Direct Antigen Test for Herpes Simplex Virus Types 1 & 2",
 "Direct Antigen Test for Herpes Simplex Virus Types 1 & 2"),
("VIRAL CULTURE: R/O HERPES SIMPLEX VIRUS", "SWAB"),
("VIRAL CULTURE: R/O HERPES SIMPLEX VIRUS", "SKIN SCRAPINGS"),
("VARICELLA-ZOSTER CULTURE",                "SKIN SCRAPINGS"),
("VIRAL CULTURE",                           "THROAT CULTURE"),
("VIRAL CULTURE",                           "SWAB"),
("VIRAL CULTURE",                           "TISSUE"),
("VIRAL CULTURE",                           "SEROLOGY/BLOOD"),
```

### `stool_parasitology` — add 2 combos
```python
("VIRAL CULTURE",     "STOOL"),
("ACID FAST CULTURE", "STOOL"),
```

---

## After applying all fixes, re-run Section 6 cells 6.2 and 6.3

The unassigned count should drop from 50,449 to under 500
(only rare tail combos with <50 events each will remain unassigned).

## What NOT to change

- Do not rename, merge, or reorder any panels
- Do not change the `EXCLUDED_TESTS` set
- Do not modify any cells outside Section 6
- Do not change the panel descriptions
