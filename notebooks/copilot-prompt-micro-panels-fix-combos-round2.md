# Fix Round 2: Add remaining missing combos to `PANELS_30` in Section 6

## Context

After the first fix, 8,369 rows remain unassigned. This prompt covers all
combos with ≥60 events. Add each tuple to the `"combos"` list of the
named panel. Do not change anything else.

---

## `blood_culture_routine` — add 2 combos
```python
("BLOOD/FUNGAL CULTURE", "FLUID RECEIVED IN BLOOD CULTURE BOTTLES"),  # 110
("BLOOD/AFB CULTURE",    "FLUID RECEIVED IN BLOOD CULTURE BOTTLES"),  #  91
```

## `csf_culture` — add 1 combo
```python
("HIV-1 Viral Load/Ultrasensitive", "CSF;SPINAL FLUID"),  # 165
```

## `fluid_culture` — add 3 combos
```python
("WOUND CULTURE",   "BILE"),         # 170
("FLUID CULTURE",   "BONE MARROW"),  # 132
("FLUID CULTURE",   "ASPIRATE"),     #  83
```

## `fungal_fluid` — add 2 combos
```python
("FUNGAL CULTURE", "DIALYSIS FLUID"),  # 136
("FUNGAL CULTURE", "BILE"),            #  70
```

## `fungal_tissue_wound` — add 4 combos
```python
("FUNGAL CULTURE",   "ASPIRATE"),     # 157
("NOCARDIA CULTURE", "TISSUE"),       # 147
("FUNGAL CULTURE",   "BONE MARROW"),  #  77
("FUNGAL CULTURE",   "FOREIGN BODY"), #  65
```

## `gram_stain_wound_fluid` — add 2 combos
```python
("GRAM STAIN", "PROSTHETIC JOINT FLUID"),  # 119
("GRAM STAIN", "BIOPSY"),                  #  81
```

## `herpesvirus_culture_antigen` — add 7 combos
```python
("VIRAL CULTURE: R/O HERPES SIMPLEX VIRUS", "THROAT CULTURE"),                          # 140
("VIRAL CULTURE: R/O CYTOMEGALOVIRUS",       "VIRAL CULTURE: R/O CYTOMEGALOVIRUS"),      #  87
("VIRAL CULTURE",                            "FLUID,OTHER"),                             #  73
("VIRAL CULTURE",                            "ASPIRATE"),                                #  71
("VIRAL CULTURE: R/O HERPES SIMPLEX VIRUS",  "TISSUE"),                                  #  65
("VIRAL CULTURE",                            "PLEURAL FLUID"),                           #  64
("VIRAL CULTURE: R/O HERPES SIMPLEX VIRUS",  "VIRAL CULTURE:R/O HERPES SIMPLEX VIRUS"),  #  63
```

## `mrsa_staph_screen` — add 1 combo
```python
("MRSA SCREEN", "SWAB"),  # 89
```

## `resistance_screen` — add 2 combos
```python
("SWAB- R/O YEAST",      "Swab R/O Yeast Screen"),    # 101
("Swab - R/O Yeast - IC","Infection Control Yeast"),   #  74
```

## `respiratory_pcp_legionella` — add 4 combos
```python
("LEGIONELLA CULTURE",                                          "Mini-BAL"),   # 155
("NOCARDIA CULTURE",                                            "BRONCHIAL WASHINGS"),  # 129
("Immunofluorescent test for Pneumocystis jirovecii (carinii)", "TISSUE"),     #  94
("LEGIONELLA CULTURE",                                          "TISSUE"),     #  88
```

## `respiratory_viral` — add 3 combos
```python
("VIRAL CULTURE",                    "BRONCHOALVEOLAR LAVAGE"),                         # 156
("VIRAL CULTURE: R/O CYTOMEGALOVIRUS","Rapid Respiratory Viral Screen & Culture"),      #  99
("VIRAL CULTURE: R/O CYTOMEGALOVIRUS","BRONCHIAL WASHINGS"),                            #  68
```

## `urine_culture` — add 3 combos
```python
("VIRAL CULTURE",      "URINE"),        # 173
("FLUID CULTURE",      "URINE"),        # 172
("ANAEROBIC CULTURE",  "URINE,KIDNEY"), # 120
```

## `wound_culture` — add 6 combos
```python
("ANAEROBIC CULTURE",   "ASPIRATE"),        # 146
("WOUND CULTURE",       "FLUID,OTHER"),     # 122
("RESPIRATORY CULTURE", "THROAT CULTURE"),  # 105
("WOUND CULTURE",       "FLUID WOUND"),     #  97
("WOUND CULTURE",       "ASPIRATE"),        #  92
("ANAEROBIC CULTURE",   "BIOPSY"),          #  81
```

---

## After applying, re-run cells 6.2 and 6.3

Expected result: unassigned drops to under ~4,000 rows, all from
low-frequency tail combos with <60 events each — acceptable as
a catch-all residual.
