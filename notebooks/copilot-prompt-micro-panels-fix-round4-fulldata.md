# Fix Round 4 (Full Dataset): Update `PANELS_30` for complete `microbiologyevents` table

## Context

The notebook now runs on the full dataset including outpatient rows (null
`hadm_id`). This reveals many high-volume combos that were invisible in the
admission-only analysis, particularly for STI panels, vaginal flora, throat
strep, and blood bottles. Add every tuple below to the named panel's `"combos"`
list. Also add the listed tests to `EXCLUDED_TESTS`.

---

## `blood_culture_routine` — add 5 combos
```python
("AEROBIC BOTTLE",      "BLOOD CULTURE"),
("ANAEROBIC BOTTLE",    "BLOOD CULTURE"),
("BLOOD/FUNGAL CULTURE","BLOOD CULTURE ( MYCO/F LYTIC BOTTLE)"),
("BLOOD/AFB CULTURE",   "BLOOD CULTURE ( MYCO/F LYTIC BOTTLE)"),
("ISOLATE FOR MIC",     "Isolate"),
```
Note: `("BLOOD/FUNGAL CULTURE", "BLOOD CULTURE")` and
`("BLOOD/AFB CULTURE", "BLOOD CULTURE")` — add these too if not already present.

## `urine_antigen_naat` — add 2 combos (large volume)
```python
("Chlamydia trachomatis, Nucleic Acid Probe, with Amplification",       "URINE"),  # 18,958
("NEISSERIA GONORRHOEAE (GC), NUCLEIC ACID PROBE, WITH AMPLIFICATION",  "URINE"),  # 16,800
```

## `urine_culture` — add 1 combo
```python
("ISOLATE FOR MIC", "URINE"),  # 21
```

## `respiratory_sputum_bal` — add 1 combo
```python
("ISOLATE FOR MIC", "SPUTUM"),  # 12
```

## `respiratory_pcp_legionella` — add 2 combos
```python
("IMMUNOFLUORESCENT TEST FOR PNEUMOCYSTIS CARINII", "SPUTUM"),               # 29
("IMMUNOFLUORESCENT TEST FOR PNEUMOCYSTIS CARINII", "BRONCHOALVEOLAR LAVAGE"), # 17
```

## `respiratory_viral` — add 1 combo
```python
("VIRAL CULTURE: R/O CYTOMEGALOVIRUS", "SPUTUM"),  # 1
```

## `wound_culture` — add 3 combos
```python
("RESPIRATORY CULTURE", "THROAT CULTURE"),       # 370
("RESPIRATORY CULTURE", "Staph aureus swab"),    # 292
("ISOLATE FOR MIC",     "TISSUE"),               # 10
```

## `fluid_culture` — add 1 combo
```python
("ISOLATE FOR MIC", "FLUID,OTHER"),  # 13
```

## `fungal_tissue_wound` — add 2 combos
```python
("ED Gram Stain for Yeast", "SWAB"),         # 252
("Malassezia furfur Culture", "SWAB"),       # 1
```

## `mrsa_staph_screen` — add 1 combo (large — was in admission data but missed)
```python
("Staph aureus Screen", "Staph aureus swab"),  # 30,354
```

## `resistance_screen` — add 2 combos
```python
("ED Gram Stain for Yeast", "Swab R/O Yeast Screen"),  # 859
("Cipro Resistant Screen",  "STOOL"),                  # 1
```

## `cdiff` — add 2 combos
```python
("CLOSTRIDIUM DIFFICILE TOXIN ASSAY",       "STOOL (RECEIVED IN TRANSPORT SYSTEM)"),  # 97
("C. difficile Toxin antigen assay",        "STOOL (RECEIVED IN TRANSPORT SYSTEM)"),  # small
```

## `stool_bacterial` — add 4 combos
```python
("FECAL CULTURE",                       "FECAL SWAB"),                              # 1,075
("CAMPYLOBACTER CULTURE",               "FECAL SWAB"),                              # 1,069
("CAMPYLOBACTER CULTURE",               "STOOL (RECEIVED IN TRANSPORT SYSTEM)"),    # 1,069
("FECAL CULTURE - R/O E.COLI 0157:H7",  "STOOL (RECEIVED IN TRANSPORT SYSTEM)"),   #   254
("FECAL CULTURE - R/O YERSINIA",        "STOOL (RECEIVED IN TRANSPORT SYSTEM)"),   #   272
("FECAL CULTURE - R/O VIBRIO",          "STOOL (RECEIVED IN TRANSPORT SYSTEM)"),   #   216
("STOOL SMEAR FOR POLYMORPHONUCLEAR LEUKOCYTES", "STOOL"),                          #     1
```

## `stool_parasitology` — add 3 combos
```python
("OVA + PARASITES",      "STOOL (RECEIVED IN TRANSPORT SYSTEM)"),  # 1,666
("CYCLOSPORA STAIN",     "STOOL (RECEIVED IN TRANSPORT SYSTEM)"),  #   365
("MICROSPORIDIA STAIN",  "STOOL (RECEIVED IN TRANSPORT SYSTEM)"),  #   326
```

## `herpesvirus_serology` — add 2 combos (large — outpatient volume)
```python
("VARICELLA-ZOSTER IgG SEROLOGY", "SEROLOGY/BLOOD"),  # 14,905
("MONOSPOT",                      "SEROLOGY/BLOOD"),  #  5,012
```

## `herpesvirus_culture_antigen` — add 2 combos
```python
("VARICELLA-ZOSTER CULTURE",            "SKIN SCRAPINGS"),  # 549
("VIRAL CULTURE: R/O CYTOMEGALOVIRUS",  "TISSUE"),          # 359
```

## `misc_serology` — add 4 combos (large — outpatient volume)
```python
("HELICOBACTER PYLORI ANTIBODY TEST", "SEROLOGY/BLOOD"),  # 11,957
("RUBELLA IgG SEROLOGY",              "SEROLOGY/BLOOD"),  # 10,938
("Lyme IgG",                          "Blood (LYME)"),    #  3,429
("Lyme IgM",                          "Blood (LYME)"),    #  3,429
```

## `gc_chlamydia_sti` — add 7 combos (largest new volume from outpatient)
```python
("Chlamydia trachomatis, Nucleic Acid Probe, with Amplification",       "SWAB"),            # 48,654
("NEISSERIA GONORRHOEAE (GC), NUCLEIC ACID PROBE, WITH AMPLIFICATION",  "SWAB"),            # 48,446
("GENITAL CULTURE",             "SWAB"),            #    277
("R/O GC Only",                 "THROAT FOR STREP"), #   146
("GENITAL CULTURE FOR TOXIC SHOCK", "SWAB"),         #   104
("R/O GC Only",                 "SWAB"),             #    95
("R/O GC Only",                 "THROAT"),           #    77
("R/O GC Only",                 "RECTAL - R/O GC"),  #    30
("R/O GC Only",                 "SWAB, R/O GC"),     #    23
```

## `vaginal_genital_flora` — add 6 combos (large new volume from outpatient)
```python
("SMEAR FOR BACTERIAL VAGINOSIS",           "SWAB"),               # 27,748
("YEAST VAGINITIS CULTURE",                 "SWAB"),               # 26,203
("R/O GROUP B BETA STREP",                  "ANORECTAL/VAGINAL"),   # 11,536
("R/O Group B Strep - Penicillin Allergy",  "ANORECTAL/VAGINAL"),   #  1,329
("TRICHOMONAS SALINE PREP",                 "SWAB"),               #     70
("R/O Group B Strep - Penicillin Allergy",  "SWAB"),               #     55
("TRICHOMONAS SALINE PREP",                 "URINE"),              #      9
```

## `throat_strep` — add 4 combos (large new volume from outpatient)
```python
("R/O Beta Strep Group A",   "THROAT FOR STREP"),   # 11,822
("R/O Beta Strep Group A",   "THROAT CULTURE"),     #    496
("GRAM STAIN- R/O THRUSH",   "THROAT FOR STREP"),   #    391
("GRAM STAIN- R/O THRUSH",   "SWAB"),               #     66
("GRAM STAIN- R/O THRUSH",   "THROAT CULTURE"),     #     50
```

---

## `EXCLUDED_TESTS` — add 7 new tests
```python
"FISH ANALYSIS, 10-30 CELLS",
"CHROMOSOME ANALYSIS - ADDITIONAL KARYOTYPE",
"TISSUE CULTURE-AMNIOTIC FLUID",
"TISSUE CULTURE-CVS",
"CVS NEEDLE ASPIRATION EVALUATION",
"Cryopreservation - Cells",
"Problem",
```

---

## Expected result after applying

Unassigned should drop to under 3,000 rows (0.1% of total ~3.2M rows).
The major gains are from the outpatient STI/vaginal/throat panels which
dominate the outpatient volume.
