# Implement Microbiology Pipeline Modules

## Overview

This prompt covers four new files and updates to two existing files needed
to add microbiology feature extraction to the CDSS-ML preprocessing pipeline.

**Files to create:**
1. `build_micro_panel_config.py` — Step 0b: build micro_panel_config.yaml
2. `build_micro_text.py` — helper for text construction and comment cleaning
3. `extract_microbiology.py` — Step 7: extract 37 micro panel text parquets

**Files to update:**
4. `preprocessing.yaml` — add MICRO_* config keys
5. `run_pipeline.py` — register new modules in pipeline order
6. `embed_features.py` — add micro panel embeddings (37 panels)
7. `check_embed_status.py` — update expected feature count 18 → 55
8. `preprocessing_utils.py` — add `_load_micro_panel_config()` (one function only; all other needed functions already exist)

---

## Context: Existing Codebase Patterns

Study these existing files carefully before implementing — all new modules
must follow the same patterns:

- `build_lab_panel_config.py` — pattern for config builder modules
- `extract_labs.py` — pattern for extract modules with chunking + hadm linkage
- `build_lab_text_lines.py` — pattern for helper text-construction modules
- `embed_features.py` — pattern for adding new feature task types
- `preprocessing_utils.py` — shared utilities to reuse and extend
- `preprocessing.yaml` — config file to extend with MICRO_* keys
- `run_pipeline.py` — orchestrator to extend with new module names

Key conventions from existing code:
- All modules expose `run(config: dict) -> None`
- All modules use `logging` not `print`; logger = `logging.getLogger(__name__)`
- All modules use `tqdm` progress bars with `desc=` and `unit="step"`
- Hash-based skip: call `_sources_unchanged(...)` at start, `_record_hashes(...)` at end
- All paths resolved from config keys — no hardcoded paths
- `_check_required_keys(config, [...])` for config validation
- `_gz_or_csv(base_dir, subdir, table)` for file resolution
- `_load_csv(path_gz, path_csv, ...)` for loading with gz fallback

---

## File 1: `build_micro_panel_config.py`

### Purpose
Step 0b of the pipeline. Reads the canonical PANELS_37 definition and writes
`classifications/micro_panel_config.yaml`.

Must run before `extract_microbiology.py`. Pattern mirrors `build_lab_panel_config.py`.

### Config keys used
- `CLASSIFICATIONS_DIR` — output directory
- `HASH_REGISTRY_PATH` — hash registry (optional)

### Output
`{CLASSIFICATIONS_DIR}/micro_panel_config.yaml`

Structure:
```yaml
panels:
  blood_culture_routine:
    description: "All blood culture methods — routine, fungal, AFB, special"
    combos:
      - test_name: "Blood Culture, Routine"
        spec_type_desc: "BLOOD CULTURE"
      - test_name: "AEROBIC BOTTLE"
        spec_type_desc: "BLOOD CULTURE"
      # ... all combos for this panel

  blood_bottle_gram_stain:
    description: "Blood culture bottle gram stains"
    combos:
      - test_name: "Aerobic Bottle Gram Stain"
        spec_type_desc: "BLOOD CULTURE"
      # ...

  # ... all 37 panels

excluded_tests:
  - "MOLECULAR CYTOGENETICS - DNA PROBE"
  - "FISH"
  - "CHROMOSOME ANALYSIS"
  - "TISSUE CULTURE (CYTOGENETICS)"
  - "CHORIONIC VILLUS SAMPLING"
  - "AMNIOTIC FLUID CULTURE"
  - "voided"
  - "Stool Hold Request"
  - "Problem"

excluded_spec_types:
  - "BLOOD CULTURE (POST-MORTEM)"
  - "POSTMORTEM CULTURE"

comment_cleaning:
  max_sentences: 2
  max_chars: 200
  discard_prefixes:
    - "TEST CANCELLED"
    - "PATIENT CREDITED"
    - "Patient credited"
    - "Test cancelled"
    - "test cancelled"
    - "TEST NOT PERFORMED"
    - "DUPLICATE ORDER"
    - "GRAM STAIN OF THIS SPECIMEN INDICATES CONTAMINATION"
  strip_triggers:
    - "Reference Range"
    - "Reference range"
    - "Detection Range"
    - "Detection range"
    - "Linear range"
    - "Performed by"
    - "performed by"
    - "Performed using"
    - "performed using"
    - "Validated for use"
    - "validated for use"
    - "Performance characteristics"
    - "performance characteristics"
    - "A positive IgG result"
    - "A positive IgM result"
    - "This test"
    - "this test"
    - "In most population"
    - "The FDA"
    - "the FDA"
    - "screened for"
    - "Detection of viruses"
    - "PLEASE SUBMIT ANOTHER"
    - "rule out"
    - "approved"
    - "Patients"
    - "patients"
```

### Panel combos (PANELS_37)

Embed the following dict directly in `build_micro_panel_config.py` as
`_PANELS_37`. This is the canonical source of truth derived from the EDA
notebook (final verified state: 1,942 unassigned rows = 0.049% of 3.99M):

```python
_PANELS_37 = {
    "blood_culture_routine": {
        "description": "All blood culture methods — routine, fungal, AFB, special",
        "combos": [
            ("AEROBIC BOTTLE", "BLOOD CULTURE"),
            ("ANAEROBIC BOTTLE", "BLOOD CULTURE"),
            ("BARTONELLA BLOOD CULTURE", "BLOOD CULTURE"),
            ("BLOOD/AFB CULTURE", "BLOOD CULTURE"),
            ("BLOOD/AFB CULTURE", "BLOOD CULTURE ( MYCO/F LYTIC BOTTLE)"),
            ("BLOOD/FUNGAL CULTURE", "BLOOD CULTURE"),
            ("BLOOD/FUNGAL CULTURE", "BLOOD CULTURE ( MYCO/F LYTIC BOTTLE)"),
            ("BRUCELLA BLOOD CULTURE", "BLOOD CULTURE"),
            ("Blood Culture, Neonate", "BLOOD CULTURE - NEONATE"),
            ("Blood Culture, Routine", "BLOOD CULTURE"),
            ("ISOLATE FOR MIC", "BLOOD CULTURE"),
            ("ISOLATE FOR MIC", "Isolate"),
            ("M. furfur Blood Culture", "BLOOD CULTURE"),
            ("M.FURFUR CULTURE", "BLOOD CULTURE"),
        ],
    },
    "blood_bottle_gram_stain": {
        "description": "Blood culture bottle gram stains and fluid bottle cultures",
        "combos": [
            ("AEROBIC BOTTLE", "FLUID RECEIVED IN BLOOD CULTURE BOTTLES"),
            ("ANAEROBIC BOTTLE", "FLUID RECEIVED IN BLOOD CULTURE BOTTLES"),
            ("Aerobic Bottle Gram Stain", "BLOOD CULTURE"),
            ("Aerobic Bottle Gram Stain", "FLUID RECEIVED IN BLOOD CULTURE BOTTLES"),
            ("Anaerobic Bottle Gram Stain", "BLOOD CULTURE"),
            ("Anaerobic Bottle Gram Stain", "FLUID RECEIVED IN BLOOD CULTURE BOTTLES"),
            ("Fluid Culture in Bottles", "FLUID RECEIVED IN BLOOD CULTURE BOTTLES"),
            ("Myco-F Bottle Gram Stain", "BLOOD CULTURE ( MYCO/F LYTIC BOTTLE)"),
            ("Pediatric Bottle Gram Stain", "Isolate"),
            ("STEM CELL - AEROBIC BOTTLE", "Stem Cell - Blood Culture"),
            ("STEM CELL - ANAEROBIC BOTTLE", "Stem Cell - Blood Culture"),
            ("Stem Cell Aer/Ana Culture", "Stem Cell - Blood Culture"),
            ("BLOOD/FUNGAL CULTURE", "FLUID RECEIVED IN BLOOD CULTURE BOTTLES"),
            ("BLOOD/AFB CULTURE", "FLUID RECEIVED IN BLOOD CULTURE BOTTLES"),
            ("Anaerobic Bottle Gram Stain", "Stem Cell - Blood Culture"),
        ],
    },
    "urine_culture": {
        "description": "Urine bacterial cultures",
        "combos": [
            ("ANAEROBIC CULTURE", "URINE"),
            ("ANAEROBIC CULTURE", "URINE,KIDNEY"),
            ("FLUID CULTURE", "URINE"),
            ("FLUID CULTURE", "URINE,KIDNEY"),
            ("FLUID CULTURE", "URINE,SUPRAPUBIC ASPIRATE"),
            ("ISOLATE FOR MIC", "URINE"),
            ("REFLEX URINE CULTURE", "URINE"),
            ("URINE CULTURE", "URINE"),
            ("URINE CULTURE", "URINE,KIDNEY"),
            ("URINE CULTURE", "URINE,SUPRAPUBIC ASPIRATE"),
            ("URINE-GRAM STAIN - UNSPUN", "URINE"),
            ("URINE-GRAM STAIN - UNSPUN", "URINE,KIDNEY"),
            ("FUNGAL CULTURE", "URINE"),
        ],
    },
    "urine_viral": {
        "description": "Urine viral cultures — CMV viruria",
        "combos": [
            ("CYTOMEGALOVIRUS EARLY ANTIGEN TEST (SHELL VIAL METHOD)", "URINE"),
            ("VIRAL CULTURE", "URINE"),
            ("VIRAL CULTURE: R/O CYTOMEGALOVIRUS", "URINE"),
        ],
    },
    "urinary_antigens": {
        "description": "Urinary antigen tests — Legionella only",
        "combos": [
            ("Legionella Urinary Antigen", "URINE"),
        ],
    },
    "respiratory_non_invasive": {
        "description": "Respiratory cultures from non-invasive specimens (sputum, aspirate)",
        "combos": [
            ("ISOLATE FOR MIC", "SPUTUM"),
            ("RESPIRATORY CULTURE", "ASPIRATE"),
            ("RESPIRATORY CULTURE", "SPUTUM"),
        ],
    },
    "respiratory_invasive": {
        "description": "Respiratory cultures from invasive specimens (BAL, bronchial, tracheal)",
        "combos": [
            ("RESPIRATORY CULTURE", "BRONCHIAL BRUSH"),
            ("RESPIRATORY CULTURE", "BRONCHIAL BRUSH - PROTECTED"),
            ("RESPIRATORY CULTURE", "BRONCHIAL WASHINGS"),
            ("RESPIRATORY CULTURE", "BRONCHOALVEOLAR LAVAGE"),
            ("RESPIRATORY CULTURE", "Mini-BAL"),
            ("RESPIRATORY CULTURE", "TRACHEAL ASPIRATE"),
            ("FLUID CULTURE", "BRONCHIAL WASHINGS"),
            ("ANAEROBIC CULTURE", "BRONCHOALVEOLAR LAVAGE"),
        ],
    },
    "respiratory_afb": {
        "description": "Respiratory AFB/TB/Nocardia cultures and smears",
        "combos": [
            ("ACID FAST CULTURE", "BRONCHIAL BRUSH"),
            ("ACID FAST CULTURE", "BRONCHIAL BRUSH - PROTECTED"),
            ("ACID FAST CULTURE", "BRONCHIAL WASHINGS"),
            ("ACID FAST CULTURE", "BRONCHOALVEOLAR LAVAGE"),
            ("ACID FAST CULTURE", "Mini-BAL"),
            ("ACID FAST CULTURE", "SPUTUM"),
            ("ACID FAST CULTURE", "TRACHEAL ASPIRATE"),
            ("ACID FAST SMEAR", "BRONCHIAL BRUSH"),
            ("ACID FAST SMEAR", "BRONCHIAL BRUSH - PROTECTED"),
            ("ACID FAST SMEAR", "BRONCHIAL WASHINGS"),
            ("ACID FAST SMEAR", "BRONCHOALVEOLAR LAVAGE"),
            ("ACID FAST SMEAR", "Mini-BAL"),
            ("ACID FAST SMEAR", "SPUTUM"),
            ("ACID FAST SMEAR", "TRACHEAL ASPIRATE"),
            ("MODIFIED ACID-FAST STAIN FOR NOCARDIA", "BRONCHIAL BRUSH"),
            ("MODIFIED ACID-FAST STAIN FOR NOCARDIA", "BRONCHOALVEOLAR LAVAGE"),
            ("MODIFIED ACID-FAST STAIN FOR NOCARDIA", "SPUTUM"),
            ("MODIFIED ACID-FAST STAIN FOR NOCARDIA", "TRACHEAL ASPIRATE"),
            ("MTB Direct Amplification", "BRONCHOALVEOLAR LAVAGE"),
            ("MTB Direct Amplification", "SPUTUM"),
            ("MTB Direct Amplification", "TRACHEAL ASPIRATE"),
            ("MTB Direct Amplification", "BRONCHIAL WASHINGS"),
            ("NOCARDIA CULTURE", "SPUTUM"),
            ("NOCARDIA CULTURE", "BRONCHIAL BRUSH"),
        ],
    },
    "respiratory_viral": {
        "description": "Respiratory viral antigen and culture tests",
        "combos": [
            ("DIRECT INFLUENZA A ANTIGEN TEST", "NASOPHARYNGEAL SWAB"),
            ("DIRECT INFLUENZA A ANTIGEN TEST", "Influenza A/B by DFA - Bronch Lavage"),
            ("DIRECT INFLUENZA B ANTIGEN TEST", "NASOPHARYNGEAL SWAB"),
            ("DIRECT INFLUENZA B ANTIGEN TEST", "Influenza A/B by DFA - Bronch Lavage"),
            ("DIRECT INFLUENZA B ANTIGEN TEST", "Rapid Respiratory Viral Screen & Culture"),
            ("Respiratory Viral Antigen Screen", "NASOPHARYNGEAL SWAB"),
            ("Respiratory Viral Antigen Screen", "RAPID RESPIRATORY VIRAL ANTIGEN TEST"),
            ("Respiratory Viral Antigen Screen", "BRONCHIAL WASHINGS"),
            ("Respiratory Viral Culture", "NASOPHARYNGEAL SWAB"),
            ("Respiratory Viral Culture", "RAPID RESPIRATORY VIRAL ANTIGEN TEST"),
            ("Respiratory Viral Culture", "BRONCHIAL WASHINGS"),
            ("Respiratory Virus Identification", "NASOPHARYNGEAL SWAB"),
            ("Respiratory Virus Identification", "RAPID RESPIRATORY VIRAL ANTIGEN TEST"),
            ("Respiratory Virus Identification", "ASPIRATE"),
            ("VIRAL CULTURE: R/O CYTOMEGALOVIRUS", "BRONCHOALVEOLAR LAVAGE"),
            ("VIRAL CULTURE", "BRONCHOALVEOLAR LAVAGE"),
            ("VIRAL CULTURE", "SPUTUM"),
            ("CYTOMEGALOVIRUS EARLY ANTIGEN TEST (SHELL VIAL METHOD)", "Mini-BAL"),
            ("VARICELLA-ZOSTER CULTURE", "BRONCHOALVEOLAR LAVAGE"),
            ("DIRECT INFLUENZA A ANTIGEN TEST", "Rapid Respiratory Viral Screen & Culture"),
            ("DIRECT ANTIGEN TEST FOR VARICELLA-ZOSTER VIRUS", "NASOPHARYNGEAL SWAB"),
            ("RSV CULTURE", "NASOPHARYNGEAL SWAB"),
            ("ADENOVIRUS CULTURE", "NASOPHARYNGEAL SWAB"),
            ("Influenza A/B by DFA", "NASOPHARYNGEAL SWAB"),
            ("Influenza A/B by DFA", "Influenza A/B by DFA - Bronch Lavage"),
            ("PARAINFLUENZA VIRUS CULTURE", "NASOPHARYNGEAL SWAB"),
            ("RSV DIRECT ANTIGEN TEST", "NASOPHARYNGEAL SWAB"),
            ("ADENOVIRUS DIRECT ANTIGEN TEST", "NASOPHARYNGEAL SWAB"),
            ("Respiratory Virus Identification", "Rapid Respiratory Viral Screen & Culture"),
            ("VIRAL CULTURE", "NASOPHARYNGEAL SWAB"),
            ("VIRAL CULTURE: R/O CYTOMEGALOVIRUS", "NASOPHARYNGEAL SWAB"),
            ("Rapid Respiratory Viral Antigen Screen", "NASOPHARYNGEAL SWAB"),
            ("DIRECT ANTIGEN TEST FOR VARICELLA-ZOSTER VIRUS", "TRACHEAL ASPIRATE"),
            ("DIRECT ANTIGEN TEST FOR VARICELLA-ZOSTER VIRUS", "BRONCHOALVEOLAR LAVAGE"),
            ("CYTOMEGALOVIRUS ANTIGENEMIA", "BLOOD"),
            ("RSV CULTURE", "TRACHEAL ASPIRATE"),
            ("DIRECT INFLUENZA A ANTIGEN TEST", "BRONCHIAL WASHINGS"),
            ("RSV DIRECT ANTIGEN TEST", "TRACHEAL ASPIRATE"),
            ("PARAINFLUENZA VIRUS CULTURE", "TRACHEAL ASPIRATE"),
            ("ADENOVIRUS CULTURE", "TRACHEAL ASPIRATE"),
            ("ADENOVIRUS DIRECT ANTIGEN TEST", "TRACHEAL ASPIRATE"),
            ("HUMAN METAPNEUMOVIRUS CULTURE", "NASOPHARYNGEAL SWAB"),
            ("Rapid Respiratory Viral Antigen Screen", "TRACHEAL ASPIRATE"),
            ("CORONAVIRUS CULTURE", "NASOPHARYNGEAL SWAB"),
        ],
    },
    "respiratory_pcp_legionella": {
        "description": "Respiratory PCP, Legionella culture and related tests",
        "combos": [
            ("Immunofluorescent test for Pneumocystis jirovecii (carinii)", "BRONCHOALVEOLAR LAVAGE"),
            ("LEGIONELLA CULTURE", "BRONCHOALVEOLAR LAVAGE"),
            ("LEGIONELLA CULTURE", "SPUTUM"),
            ("LEGIONELLA CULTURE", "BRONCHIAL WASHINGS"),
            ("LEGIONELLA CULTURE", "BRONCHIAL BRUSH"),
            ("LEGIONELLA CULTURE", "TRACHEAL ASPIRATE"),
            ("Immunofluorescent test for Pneumocystis jirovecii (carinii)", "SPUTUM"),
            ("Immunofluorescent test for Pneumocystis jirovecii (carinii)", "TRACHEAL ASPIRATE"),
            ("Immunofluorescent test for Pneumocystis jirovecii (carinii)", "BRONCHIAL WASHINGS"),
            ("LEGIONELLA ANTIGEN", "BRONCHOALVEOLAR LAVAGE"),
            ("LEGIONELLA ANTIGEN", "SPUTUM"),
            ("NOCARDIA CULTURE", "BRONCHOALVEOLAR LAVAGE"),
            ("NOCARDIA CULTURE", "BRONCHIAL BRUSH"),
            ("Immunofluorescent test for Pneumocystis jirovecii (carinii)", "PLEURAL FLUID"),
            ("LEGIONELLA CULTURE", "Mini-BAL"),
            ("NOCARDIA CULTURE", "FLUID,OTHER"),
            ("NOCARDIA CULTURE", "BIOPSY"),
            ("NOCARDIA CULTURE", "TRACHEAL ASPIRATE"),
        ],
    },
    "gram_stain_respiratory": {
        "description": "Gram stains of respiratory specimens",
        "combos": [
            ("GRAM STAIN", "BRONCHIAL BRUSH"),
            ("GRAM STAIN", "BRONCHIAL WASHINGS"),
            ("GRAM STAIN", "BRONCHOALVEOLAR LAVAGE"),
            ("GRAM STAIN", "Mini-BAL"),
            ("GRAM STAIN", "SPUTUM"),
            ("GRAM STAIN", "TRACHEAL ASPIRATE"),
        ],
    },
    "gram_stain_wound_tissue": {
        "description": "Gram stains of wound, tissue, and fluid specimens",
        "combos": [
            ("GRAM STAIN", "ABSCESS"),
            ("GRAM STAIN", "BILE"),
            ("GRAM STAIN", "BIOPSY"),
            ("GRAM STAIN", "CATHETER TIP-IV"),
            ("GRAM STAIN", "FLUID,OTHER"),
            ("GRAM STAIN", "JOINT FLUID"),
            ("GRAM STAIN", "PERITONEAL FLUID"),
            ("GRAM STAIN", "PLEURAL FLUID"),
            ("GRAM STAIN", "SKIN SCRAPINGS"),
            ("GRAM STAIN", "STOOL"),
            ("GRAM STAIN", "SWAB"),
            ("GRAM STAIN", "TISSUE"),
            ("GRAM STAIN", "WOUND"),
        ],
    },
    "gram_stain_csf": {
        "description": "Gram stains of CSF",
        "combos": [
            ("GRAM STAIN", "CSF;SPINAL FLUID"),
        ],
    },
    "wound_culture": {
        "description": "Wound, tissue, abscess bacterial cultures",
        "combos": [
            ("ANAEROBIC CULTURE", "ABSCESS"),
            ("ANAEROBIC CULTURE", "BIOPSY"),
            ("ANAEROBIC CULTURE", "FLUID,OTHER"),
            ("ANAEROBIC CULTURE", "SWAB"),
            ("ANAEROBIC CULTURE", "TISSUE"),
            ("ANAEROBIC CULTURE", "WOUND"),
            ("TISSUE CULTURE", "TISSUE"),
            ("TISSUE CULTURE", "SWAB"),
            ("TISSUE CULTURE", "ABSCESS"),
            ("TISSUE CULTURE", "BIOPSY"),
            ("WOUND CULTURE", "ABSCESS"),
            ("WOUND CULTURE", "BIOPSY"),
            ("WOUND CULTURE", "SKIN SCRAPINGS"),
            ("WOUND CULTURE", "STOOL"),
            ("WOUND CULTURE", "SWAB"),
            ("WOUND CULTURE", "TISSUE"),
            ("WOUND CULTURE", "URINE"),
            ("WOUND CULTURE", "WOUND"),
            ("ISOLATE FOR MIC", "WOUND"),
            ("ISOLATE FOR MIC", "TISSUE"),
            ("ISOLATE FOR MIC", "SWAB"),
            ("ISOLATE FOR MIC", "ABSCESS"),
            ("FLUID CULTURE", "ABSCESS"),
            ("FLUID CULTURE", "WOUND"),
            ("FLUID CULTURE", "TISSUE"),
            ("FLUID CULTURE", "SWAB"),
            ("SKIN CULTURE", "SKIN SCRAPINGS"),
            ("SKIN CULTURE", "SWAB"),
            ("THROAT CULTURE", "THROAT FOR STREP"),
            ("THROAT CULTURE", "THROAT"),
            ("FUNGAL CULTURE (HAIR/SKIN/NAILS)", "SWAB"),
            ("FUNGAL CULTURE (HAIR/SKIN/NAILS)", "HAIR"),
            ("RESPIRATORY CULTURE", "ABSCESS"),
            ("TISSUE", "SWAB"),
            ("TISSUE", "JOINT FLUID"),
        ],
    },
    "hardware_and_lines_culture": {
        "description": "Device cultures — CLABSI, prosthetic joint, foreign body",
        "combos": [
            ("CATHETER TIP CULTURE", "CATHETER TIP-IV"),
            ("FLUID CULTURE", "CATHETER TIP-IV"),
            ("WOUND CULTURE", "CATHETER TIP-IV"),
            ("ANAEROBIC CULTURE", "CATHETER TIP-IV"),
            ("FLUID CULTURE", "FOREIGN BODY"),
            ("WOUND CULTURE", "FOREIGN BODY"),
            ("CATHETER TIP CULTURE", "FOREIGN BODY"),
            ("ISOLATE FOR MIC", "CATHETER TIP-IV"),
            ("ISOLATE FOR MIC", "FOREIGN BODY"),
            ("SONICATION FLUID CULTURE", "FOREIGN BODY"),
            ("SONICATION FLUID CULTURE", "CATHETER TIP-IV"),
            ("ANAEROBIC CULTURE", "FOREIGN BODY"),
            ("TISSUE CULTURE", "FOREIGN BODY"),
        ],
    },
    "pleural_culture": {
        "description": "Pleural fluid cultures",
        "combos": [
            ("ANAEROBIC CULTURE", "PLEURAL FLUID"),
            ("FLUID CULTURE", "PLEURAL FLUID"),
            ("FUNGAL CULTURE", "PLEURAL FLUID"),
            ("ACID FAST CULTURE", "PLEURAL FLUID"),
            ("ACID FAST SMEAR", "PLEURAL FLUID"),
            ("ISOLATE FOR MIC", "PLEURAL FLUID"),
            ("TISSUE CULTURE", "PLEURAL FLUID"),
            ("WOUND CULTURE", "PLEURAL FLUID"),
            ("CYTOMEGALOVIRUS EARLY ANTIGEN TEST (SHELL VIAL METHOD)", "PLEURAL FLUID"),
            ("NOCARDIA CULTURE", "PLEURAL FLUID"),
        ],
    },
    "peritoneal_culture": {
        "description": "Peritoneal fluid cultures",
        "combos": [
            ("ANAEROBIC CULTURE", "PERITONEAL FLUID"),
            ("FLUID CULTURE", "PERITONEAL FLUID"),
            ("FUNGAL CULTURE", "PERITONEAL FLUID"),
            ("ACID FAST CULTURE", "PERITONEAL FLUID"),
            ("ISOLATE FOR MIC", "PERITONEAL FLUID"),
            ("TISSUE CULTURE", "PERITONEAL FLUID"),
            ("WOUND CULTURE", "PERITONEAL FLUID"),
            ("ACID FAST SMEAR", "PERITONEAL FLUID"),
            ("WOUND CULTURE", "PERITONEAL FLUID"),
        ],
    },
    "joint_fluid_culture": {
        "description": "Joint and prosthetic joint fluid cultures",
        "combos": [
            ("ANAEROBIC CULTURE", "JOINT FLUID"),
            ("FLUID CULTURE", "JOINT FLUID"),
            ("FUNGAL CULTURE", "JOINT FLUID"),
            ("ACID FAST CULTURE", "JOINT FLUID"),
            ("ACID FAST SMEAR", "JOINT FLUID"),
            ("ISOLATE FOR MIC", "JOINT FLUID"),
            ("TISSUE CULTURE", "JOINT FLUID"),
            ("WOUND CULTURE", "JOINT FLUID"),
            ("ANAEROBIC CULTURE", "PROSTHETIC JOINT"),
            ("FLUID CULTURE", "PROSTHETIC JOINT"),
            ("WOUND CULTURE", "PROSTHETIC JOINT"),
            ("ISOLATE FOR MIC", "PROSTHETIC JOINT"),
            ("TISSUE CULTURE", "PROSTHETIC JOINT"),
            ("FUNGAL CULTURE", "PROSTHETIC JOINT"),
        ],
    },
    "fluid_culture": {
        "description": "Miscellaneous fluid cultures — bile, dialysis, other",
        "combos": [
            ("ANAEROBIC CULTURE", "BILE"),
            ("FLUID CULTURE", "BILE"),
            ("FUNGAL CULTURE", "BILE"),
            ("ANAEROBIC CULTURE", "DIALYSIS FLUID"),
            ("FLUID CULTURE", "DIALYSIS FLUID"),
            ("FUNGAL CULTURE", "DIALYSIS FLUID"),
            ("ANAEROBIC CULTURE", "FLUID,OTHER"),
            ("FLUID CULTURE", "FLUID,OTHER"),
            ("FUNGAL CULTURE", "FLUID,OTHER"),
            ("ACID FAST CULTURE", "FLUID,OTHER"),
            ("ACID FAST SMEAR", "BILE"),
            ("ACID FAST CULTURE", "BILE"),
            ("CYTOMEGALOVIRUS EARLY ANTIGEN TEST (SHELL VIAL METHOD)", "FLUID,OTHER"),
        ],
    },
    "bone_marrow_culture": {
        "description": "Bone marrow cultures",
        "combos": [
            ("ACID FAST CULTURE", "BONE MARROW"),
            ("ANAEROBIC CULTURE", "BONE MARROW"),
            ("FLUID CULTURE", "BONE MARROW"),
            ("FUNGAL CULTURE", "BONE MARROW"),
            ("TISSUE CULTURE", "BONE MARROW"),
            ("WOUND CULTURE", "BONE MARROW"),
            ("BRUCELLA CULTURE", "BONE MARROW"),
        ],
    },
    "csf_culture": {
        "description": "CSF cultures and Cryptococcal antigen",
        "combos": [
            ("ANAEROBIC CULTURE", "CSF;SPINAL FLUID"),
            ("FUNGAL CULTURE", "CSF;SPINAL FLUID"),
            ("ACID FAST CULTURE", "CSF;SPINAL FLUID"),
            ("ACID FAST SMEAR", "CSF;SPINAL FLUID"),
            ("FLUID CULTURE", "CSF;SPINAL FLUID"),
            ("TISSUE CULTURE", "CSF;SPINAL FLUID"),
            ("CRYPTOCOCCAL ANTIGEN", "CSF;SPINAL FLUID"),
            ("CRYPTOCOCCAL ANTIGEN", "BLOOD"),
            ("HIV-1 VIRAL LOAD", "CSF;SPINAL FLUID"),
            ("CMV VIRAL LOAD", "CSF;SPINAL FLUID"),
            ("HERPES SIMPLEX VIRUS PCR", "CSF;SPINAL FLUID"),
            ("VARICELLA-ZOSTER VIRUS PCR", "CSF;SPINAL FLUID"),
            ("Direct Antigen Test for Herpes Simplex Virus Types 1 & 2", "CSF;SPINAL FLUID"),
            ("VIRAL CULTURE: R/O CYTOMEGALOVIRUS", "CSF;SPINAL FLUID"),
            ("NOCARDIA CULTURE", "CSF;SPINAL FLUID"),
        ],
    },
    "fungal_tissue_wound": {
        "description": "Fungal cultures and smears from tissue, wound, and skin",
        "combos": [
            ("ACID FAST CULTURE", "BIOPSY"),
            ("ACID FAST SMEAR", "BIOPSY"),
            ("FUNGAL CULTURE", "ABSCESS"),
            ("FUNGAL CULTURE", "BIOPSY"),
            ("FUNGAL CULTURE", "SKIN SCRAPINGS"),
            ("FUNGAL CULTURE", "SWAB"),
            ("FUNGAL CULTURE", "THROAT FOR STREP"),
            ("FUNGAL CULTURE", "TISSUE"),
            ("FUNGAL CULTURE", "WOUND"),
            ("FUNGAL CULTURE (HAIR/SKIN/NAILS)", "BIOPSY"),
            ("FUNGAL CULTURE (HAIR/SKIN/NAILS)", "SKIN SCRAPINGS"),
            ("INDIA INK PREPARATION", "WOUND"),
            ("INDIA INK PREPARATION", "TISSUE"),
            ("MODIFIED ACID-FAST STAIN FOR NOCARDIA", "ABSCESS"),
            ("MODIFIED ACID-FAST STAIN FOR NOCARDIA", "BIOPSY"),
            ("MODIFIED ACID-FAST STAIN FOR NOCARDIA", "TISSUE"),
            ("MODIFIED ACID-FAST STAIN FOR NOCARDIA", "WOUND"),
            ("POTASSIUM HYDROXIDE PREPARATION", "BIOPSY"),
            ("POTASSIUM HYDROXIDE PREPARATION", "SKIN SCRAPINGS"),
            ("POTASSIUM HYDROXIDE PREPARATION", "SWAB"),
            ("POTASSIUM HYDROXIDE PREPARATION", "TISSUE"),
            ("POTASSIUM HYDROXIDE PREPARATION", "WOUND"),
            ("ED Gram Stain for Yeast", "SWAB"),
            ("ED Gram Stain for Yeast", "Swab"),
            ("ACID FAST CULTURE", "WOUND"),
            ("ACID FAST CULTURE", "TISSUE"),
            ("ACID FAST SMEAR", "WOUND"),
            ("ACID FAST SMEAR", "TISSUE"),
            ("MODIFIED ACID-FAST STAIN FOR NOCARDIA", "SWAB"),
            ("MTB Direct Amplification", "TISSUE"),
            ("FUNGAL CULTURE", "STOOL"),
            ("NOCARDIA CULTURE", "TISSUE"),
            ("NOCARDIA CULTURE", "WOUND"),
            ("NOCARDIA CULTURE", "BIOPSY"),
            ("NOCARDIA CULTURE", "ABSCESS"),
            ("ACID FAST CULTURE", "ABSCESS"),
            ("ACID FAST SMEAR", "ABSCESS"),
        ],
    },
    "fungal_respiratory": {
        "description": "Fungal cultures and smears from respiratory specimens",
        "combos": [
            ("FUNGAL CULTURE", "BRONCHOALVEOLAR LAVAGE"),
            ("FUNGAL CULTURE", "BRONCHIAL BRUSH"),
            ("FUNGAL CULTURE", "BRONCHIAL WASHINGS"),
            ("FUNGAL CULTURE", "Mini-BAL"),
            ("FUNGAL CULTURE", "SPUTUM"),
            ("FUNGAL CULTURE", "TRACHEAL ASPIRATE"),
            ("INDIA INK PREPARATION", "BRONCHOALVEOLAR LAVAGE"),
            ("INDIA INK PREPARATION", "SPUTUM"),
            ("POTASSIUM HYDROXIDE PREPARATION", "BRONCHOALVEOLAR LAVAGE"),
            ("POTASSIUM HYDROXIDE PREPARATION", "SPUTUM"),
            ("POTASSIUM HYDROXIDE PREPARATION", "TRACHEAL ASPIRATE"),
            ("ACID FAST SMEAR", "URINE"),
            ("ACID FAST CULTURE", "URINE"),
        ],
    },
    "fungal_fluid": {
        "description": "Fungal cultures and smears from fluid specimens",
        "combos": [
            ("FUNGAL CULTURE", "CSF;SPINAL FLUID"),
            ("ACID FAST CULTURE", "BILE"),
            ("ACID FAST SMEAR", "BILE"),
            ("INDIA INK PREPARATION", "CSF;SPINAL FLUID"),
            ("POTASSIUM HYDROXIDE PREPARATION", "CSF;SPINAL FLUID"),
            ("FUNGAL CULTURE", "DIALYSIS FLUID"),
            ("ACID FAST CULTURE", "PERITONEAL FLUID"),
            ("ACID FAST SMEAR", "PERITONEAL FLUID"),
            ("ACID FAST CULTURE", "JOINT FLUID"),
            ("ACID FAST SMEAR", "JOINT FLUID"),
        ],
    },
    "mrsa_staph_screen": {
        "description": "MRSA and Staph aureus surveillance screens",
        "combos": [
            ("MRSA SCREEN", "BLOOD"),
            ("MRSA SCREEN", "GROIN"),
            ("MRSA SCREEN", "NARES"),
            ("MRSA SCREEN", "RECTAL"),
            ("MRSA SCREEN", "WOUND"),
            ("MRSA SCREEN", "Staph aureus swab"),
            ("MRSA SCREEN", "STOOL"),
            ("No Staph aureus isolated", "NARES"),
            ("No Staph aureus isolated", "GROIN"),
        ],
    },
    "resistance_screen": {
        "description": "Resistance surveillance — VRE, CRE, yeast screens",
        "combos": [
            ("CIPROFLOXACIN RESISTANT GNR SCREEN", "RECTAL"),
            ("CRE SCREEN", "RECTAL"),
            ("CRE SCREEN", "STOOL"),
            ("VANCOMYCIN RESISTANT ENTEROCOCCUS (VRE) SCREEN", "RECTAL"),
            ("VANCOMYCIN RESISTANT ENTEROCOCCUS (VRE) SCREEN", "STOOL"),
            ("VRE SCREEN", "RECTAL"),
            ("VRE SCREEN", "STOOL"),
            ("YEAST SCREEN", "RECTAL"),
            ("YEAST SCREEN", "STOOL"),
            ("No VRE isolated", "RECTAL"),
        ],
    },
    "cdiff": {
        "description": "C. difficile PCR and toxin assays",
        "combos": [
            ("C. difficile PCR", "STOOL"),
            ("C. difficile PCR", "STOOL (RECEIVED IN TRANSPORT SYSTEM)"),
            ("CLOSTRIDIUM DIFFICILE TOXIN ASSAY", "STOOL"),
            ("CLOSTRIDIUM DIFFICILE TOXIN ASSAY", "STOOL (RECEIVED IN TRANSPORT SYSTEM)"),
            ("Feces Cdiff Toxin A&B", "STOOL"),
            ("Feces Cdiff Toxin A&B", "STOOL (RECEIVED IN TRANSPORT SYSTEM)"),
            ("GI PATHOGEN SCREEN - CDIFF ONLY", "STOOL"),
            ("GI PATHOGEN SCREEN - CDIFF ONLY", "STOOL (RECEIVED IN TRANSPORT SYSTEM)"),
        ],
    },
    "stool_bacterial": {
        "description": "Fecal bacterial cultures and enteric pathogen screens",
        "combos": [
            ("CAMPYLOBACTER CULTURE", "STOOL"),
            ("CAMPYLOBACTER CULTURE", "STOOL (RECEIVED IN TRANSPORT SYSTEM)"),
            ("E.COLI 0157:H7", "STOOL"),
            ("E.COLI 0157:H7", "STOOL (RECEIVED IN TRANSPORT SYSTEM)"),
            ("GI PATHOGEN SCREEN", "STOOL"),
            ("GI PATHOGEN SCREEN", "STOOL (RECEIVED IN TRANSPORT SYSTEM)"),
            ("SALMONELLA/SHIGELLA CULTURE", "STOOL"),
            ("SALMONELLA/SHIGELLA CULTURE", "STOOL (RECEIVED IN TRANSPORT SYSTEM)"),
            ("STOOL CULTURE", "STOOL"),
            ("STOOL CULTURE", "STOOL (RECEIVED IN TRANSPORT SYSTEM)"),
            ("VIBRIO CULTURE", "STOOL"),
            ("VIBRIO CULTURE", "STOOL (RECEIVED IN TRANSPORT SYSTEM)"),
            ("YERSINIA CULTURE", "STOOL"),
            ("YERSINIA CULTURE", "STOOL (RECEIVED IN TRANSPORT SYSTEM)"),
        ],
    },
    "stool_parasitology": {
        "description": "Stool parasitology, protozoa, and O+P",
        "combos": [
            ("Concentration and Stain for Giardia", "STOOL"),
            ("Concentration and Stain for Giardia", "STOOL (RECEIVED IN TRANSPORT SYSTEM)"),
            ("Concentration and Stain for Giardia", "ASPIRATE"),
            ("CRYPTOSPORIDIUM ANTIGEN TEST", "STOOL"),
            ("CRYPTOSPORIDIUM ANTIGEN TEST", "STOOL (RECEIVED IN TRANSPORT SYSTEM)"),
            ("CYCLOSPORA STAIN", "STOOL"),
            ("CYCLOSPORA STAIN", "STOOL (RECEIVED IN TRANSPORT SYSTEM)"),
            ("GIARDIA ANTIGEN TEST", "STOOL"),
            ("GIARDIA ANTIGEN TEST", "STOOL (RECEIVED IN TRANSPORT SYSTEM)"),
            ("MICROSPORIDIA STAIN", "STOOL"),
            ("MICROSPORIDIA STAIN", "STOOL (RECEIVED IN TRANSPORT SYSTEM)"),
            ("MICROSPORIDIA STAIN", "ASPIRATE"),
            ("O&P MACROSCOPIC EXAM - WORM", "STOOL"),
            ("O&P MACROSCOPIC EXAM - WORM", "STOOL (RECEIVED IN TRANSPORT SYSTEM)"),
            ("OVA + PARASITES", "STOOL"),
            ("OVA + PARASITES", "STOOL (RECEIVED IN TRANSPORT SYSTEM)"),
            ("OVA + PARASITES", "TISSUE"),
            ("OVA + PARASITES", "FLUID,OTHER"),
            ("OVA + PARASITES", "ABSCESS"),
            ("OVA + PARASITES", "ASPIRATE"),
            ("SCOTCH TAPE PREP/PADDLE", "STOOL"),
            ("SCOTCH TAPE PREP/PADDLE", "STOOL (RECEIVED IN TRANSPORT SYSTEM)"),
            ("SCOTCH TAPE PREP/PADDLE", "PERIANAL SKIN"),
        ],
    },
    "herpesvirus_serology": {
        "description": "CMV, EBV, VZV serology and viral loads",
        "combos": [
            ("CMV VIRAL LOAD", "BLOOD"),
            ("CMV VIRAL LOAD", "SEROLOGY/BLOOD"),
            ("CYTOMEGALOVIRUS ANTIBODY (IGG)", "SEROLOGY/BLOOD"),
            ("CYTOMEGALOVIRUS ANTIBODY (IGM)", "SEROLOGY/BLOOD"),
            ("EBV EIA", "SEROLOGY/BLOOD"),
            ("EBV PCR", "BLOOD"),
            ("EBV PCR", "SEROLOGY/BLOOD"),
            ("EPSTEIN-BARR VIRUS IgM SEROLOGY", "SEROLOGY/BLOOD"),
            ("VARICELLA-ZOSTER IgG SEROLOGY", "SEROLOGY/BLOOD"),
            ("VARICELLA-ZOSTER VIRUS PCR", "BLOOD"),
            ("VARICELLA-ZOSTER VIRUS PCR", "SEROLOGY/BLOOD"),
            ("CMV ANTIGEN DETECTION", "BLOOD"),
            ("CMV ANTIGEN DETECTION", "SEROLOGY/BLOOD"),
            ("EPSTEIN-BARR VIRUS DNA", "BLOOD"),
        ],
    },
    "hepatitis_hiv": {
        "description": "HIV, HCV, HBV viral loads and RNA",
        "combos": [
            ("HCV VIRAL LOAD", "SEROLOGY/BLOOD"),
            ("HCV VIRAL LOAD", "BLOOD"),
            ("HEPATITIS C GENOTYPE", "SEROLOGY/BLOOD"),
            ("HIV-1 VIRAL LOAD", "SEROLOGY/BLOOD"),
            ("HIV-1 VIRAL LOAD", "BLOOD"),
        ],
    },
    "syphilis_serology": {
        "description": "Syphilis serology — RPR, treponemal tests",
        "combos": [
            ("RAPID PLASMA REAGIN TEST", "SEROLOGY/BLOOD"),
            ("SYPHILIS SEROLOGY", "SEROLOGY/BLOOD"),
            ("TREPONEMAL ANTIBODY TEST", "SEROLOGY/BLOOD"),
            ("FLUORESCENT TREPONEMAL ANTIBODY", "SEROLOGY/BLOOD"),
        ],
    },
    "misc_serology": {
        "description": "Miscellaneous serology — Lyme, Toxo, H. pylori, Malaria, MMR",
        "combos": [
            ("ASPERGILLUS ANTIGEN", "SEROLOGY/BLOOD"),
            ("BETA-1,3 D-GLUCAN ASSAY", "SEROLOGY/BLOOD"),
            ("CRYPTOCOCCAL ANTIGEN", "SEROLOGY/BLOOD"),
            ("GALACTOMANNAN", "SEROLOGY/BLOOD"),
            ("HELICOBACTER PYLORI ANTIBODY", "SEROLOGY/BLOOD"),
            ("LYME SEROLOGY", "SEROLOGY/BLOOD"),
            ("MALARIA SMEAR", "BLOOD"),
            ("MEASLES ANTIBODY (IgG)", "SEROLOGY/BLOOD"),
            ("MUMPS ANTIBODY (IgG)", "SEROLOGY/BLOOD"),
            ("RUBELLA ANTIBODY", "SEROLOGY/BLOOD"),
            ("TOXOPLASMA IgG ANTIBODY", "SEROLOGY/BLOOD"),
            ("TOXOPLASMA IgM ANTIBODY", "SEROLOGY/BLOOD"),
            ("TOXOPLASMA IgM ANTIBODY", "SEROLOGY/BLOOD"),
            ("ASO TITER", "SEROLOGY/BLOOD"),
        ],
    },
    "herpesvirus_culture_antigen": {
        "description": "Herpesvirus direct antigen and culture tests",
        "combos": [
            ("CYTOMEGALOVIRUS EARLY ANTIGEN TEST (SHELL VIAL METHOD)", "BLOOD"),
            ("CYTOMEGALOVIRUS EARLY ANTIGEN TEST (SHELL VIAL METHOD)", "BRONCHOALVEOLAR LAVAGE"),
            ("CYTOMEGALOVIRUS EARLY ANTIGEN TEST (SHELL VIAL METHOD)", "SWAB"),
            ("DIRECT ANTIGEN TEST FOR VARICELLA-ZOSTER VIRUS", "SKIN SCRAPINGS"),
            ("Direct Antigen Test for Herpes Simplex Virus Types 1 & 2", "SWAB"),
            ("Direct Antigen Test for Herpes Simplex Virus Types 1 & 2", "VIRAL CULTURE:R/O HERPES SIMPLEX VIRUS"),
            ("HSV CULTURE", "SWAB"),
            ("VARICELLA-ZOSTER CULTURE", "SWAB"),
            ("VARICELLA-ZOSTER CULTURE", "TISSUE"),
            ("VARICELLA-ZOSTER CULTURE", "SKIN SCRAPINGS"),
            ("VIRAL CULTURE", "TISSUE"),
            ("VIRAL CULTURE", "SWAB"),
            ("VIRAL CULTURE", "THROAT FOR STREP"),
            ("VIRAL CULTURE", "VIRAL CULTURE"),
            ("VIRAL CULTURE", "VIRAL CULTURE:R/O HERPES SIMPLEX VIRUS"),
            ("VIRAL CULTURE: R/O CYTOMEGALOVIRUS", "VIRAL CULTURE:R/O HERPES SIMPLEX VIRUS"),
            ("VIRAL CULTURE: R/O HERPES SIMPLEX VIRUS", "SWAB"),
            ("VIRAL CULTURE: R/O HERPES SIMPLEX VIRUS", "TISSUE"),
            ("VIRAL CULTURE: R/O HERPES SIMPLEX VIRUS", "VIRAL CULTURE:R/O HERPES SIMPLEX VIRUS"),
            ("VARICELLA-ZOSTER CULTURE", "BLOOD"),
            ("CYTOMEGALOVIRUS EARLY ANTIGEN TEST (SHELL VIAL METHOD)", "TISSUE"),
            ("CYTOMEGALOVIRUS EARLY ANTIGEN TEST (SHELL VIAL METHOD)", "SKIN SCRAPINGS"),
            ("HSV CULTURE", "TISSUE"),
            ("HSV CULTURE", "SKIN SCRAPINGS"),
            ("HERPES SIMPLEX VIRUS CULTURE AND TYPING", "SWAB"),
            ("HERPES SIMPLEX VIRUS CULTURE AND TYPING", "TISSUE"),
            ("CYTOMEGALOVIRUS EARLY ANTIGEN TEST (SHELL VIAL METHOD)", "ABSCESS"),
            ("Direct Antigen Test for Herpes Simplex Virus Types 1 & 2", "TISSUE"),
            ("Direct Antigen Test for Herpes Simplex Virus Types 1 & 2", "SKIN SCRAPINGS"),
            ("VARICELLA-ZOSTER CULTURE", "BRONCHOALVEOLAR LAVAGE"),
            ("VARICELLA-ZOSTER CULTURE", "ABSCESS"),
            ("VARICELLA-ZOSTER CULTURE", "FLUID,OTHER"),
            ("VIRAL CULTURE", "ABSCESS"),
            ("VIRAL CULTURE: R/O CYTOMEGALOVIRUS", "TISSUE"),
            ("VIRAL CULTURE: R/O CYTOMEGALOVIRUS", "SWAB"),
            ("VIRAL CULTURE: R/O CYTOMEGALOVIRUS", "FLUID,OTHER"),
            ("VIRAL CULTURE: R/O HERPES SIMPLEX VIRUS", "SKIN SCRAPINGS"),
            ("VIRAL CULTURE: R/O HERPES SIMPLEX VIRUS", "TISSUE"),
            ("Rapid Respiratory Viral Screen & Culture", "SWAB"),
        ],
    },
    "gc_chlamydia_sti": {
        "description": "GC/Chlamydia NAAT, cultures, and STI workup",
        "combos": [
            ("GC/CHLAMYDIA (AMPLIFIED PROBE TECHNIQUE)", "SWAB"),
            ("GC/CHLAMYDIA (AMPLIFIED PROBE TECHNIQUE)", "URINE"),
            ("GONOCOCCAL CULTURE", "SWAB"),
            ("GONOCOCCAL CULTURE", "URINE"),
            ("Negative for Chlamydia trachomatis by PCR", "SWAB"),
            ("Negative for Neisseria Gonorrhoeae by PCR", "SWAB"),
            ("Negative for Chlamydia trachomatis by PCR", "URINE"),
            ("Negative for Neisseria Gonorrhoeae by PCR", "URINE"),
            ("Negative for Neisseria gonorrhoeae by ___, APTIMA COMBO 2 Assay", "SWAB"),
            ("Negative for Chlamydia trachomatis by ___, APTIMA COMBO 2 Assay", "SWAB"),
            ("Negative for Neisseria gonorrhoeae by ___, APTIMA COMBO 2 Assay", "URINE"),
            ("Negative for Chlamydia trachomatis by ___, APTIMA COMBO 2 Assay", "URINE"),
            ("CHLAMYDIA TRACHOMATIS CULTURE", "SWAB"),
        ],
    },
    "vaginal_genital_flora": {
        "description": "Vaginal and genital flora, BV, GBS, yeast, Trichomonas",
        "combos": [
            ("BACTERIAL VAGINOSIS CULTURE", "SWAB"),
            ("BACTERIAL VAGINOSIS CULTURE", "SWAB - R/O YEAST"),
            ("GROUP B STREPTOCOCCUS CULTURE", "SWAB"),
            ("PELVIC CULTURE", "SWAB"),
            ("TRICHOMONAS CULTURE", "SWAB"),
            ("TRICHOMONAS CULTURE", "SWAB - R/O YEAST"),
            ("VAGINAL CULTURE", "SWAB"),
            ("VAGINAL CULTURE", "SWAB - R/O YEAST"),
            ("YEAST VAGINITIS CULTURE", "SWAB"),
            ("YEAST VAGINITIS CULTURE", "SWAB - R/O YEAST"),
            ("Negative for Group B beta streptococci", "SWAB"),
        ],
    },
    "throat_strep": {
        "description": "Throat strep A and oral cultures",
        "combos": [
            ("R/O Beta Strep Group A", "THROAT"),
            ("STREP THROAT CULTURE", "THROAT FOR STREP"),
            ("STREP THROAT CULTURE", "THROAT"),
            ("THROAT CULTURE", "THROAT"),
            ("THROAT CULTURE", "THROAT FOR STREP"),
            ("THROAT CULTURE", "NASOPHARYNGEAL SWAB"),
        ],
    },
}
```

### Excluded tests and spec types

```python
_EXCLUDED_TESTS = frozenset({
    "MOLECULAR CYTOGENETICS - DNA PROBE",
    "FISH",
    "CHROMOSOME ANALYSIS",
    "TISSUE CULTURE (CYTOGENETICS)",
    "CHORIONIC VILLUS SAMPLING",
    "AMNIOTIC FLUID CULTURE",
    "voided",
    "Stool Hold Request",
    "Problem",
})

_EXCLUDED_SPEC_TYPES = frozenset({
    "BLOOD CULTURE (POST-MORTEM)",
    "POSTMORTEM CULTURE",
})
```

### Algorithm

```
1. Validate config keys: CLASSIFICATIONS_DIR (required)
2. Hash-based skip check (source: this script itself via __file__)
3. Convert _PANELS_37 to YAML-serialisable structure with descriptions + combos
4. Attach excluded_tests, excluded_spec_types, comment_cleaning sections
5. Write to {CLASSIFICATIONS_DIR}/micro_panel_config.yaml
6. Log: number of panels, total combos, excluded tests/spec_types
7. Record hashes
```

No MIMIC-IV data is read — the config is built entirely from `_PANELS_37`.
Use `__file__` as the source path for hash tracking.

---

## File 2: `build_micro_text.py`

### Purpose
Helper module imported by `extract_microbiology.py`. Provides:
- `clean_comment(text, config)` — clean microbiology comments field
- `build_event_text(row, config)` — build per-event text string (Cases A/B/C)
- `aggregate_panel_text(events_df, config, tokenizer)` — aggregate events for one (admission, panel)

Not a standalone pipeline step. Pattern mirrors `build_lab_text_lines.py`.

### `clean_comment(text, config) -> str | None`

Implements the 6-step comment cleaning algorithm:

**Step 1 — Null/placeholder check:**
Return `None` if text is null, empty after strip, or matches `^_+$` or `^-+$`.

**Step 2 — Discard-entirely check:**
If text starts with any prefix in `config["comment_cleaning"]["discard_prefixes"]`
(checked after `.strip()`), return `None`.

**Step 3 — Trigger word truncation:**
For each trigger in `config["comment_cleaning"]["strip_triggers"]`:
  - Find first occurrence `idx = text.find(trigger)`
  - If `idx > 0`: `text = text[:idx].strip()`
  - Stop at first trigger that fires (don't apply all triggers)

Note: colon `:` is NOT a sentence boundary. Do not split on `:`.

**Step 4 — Sentence splitting:**
Split on regex `r'\.\s{2,}|\.\n'` (period + 2+ spaces, or period + newline).
Keep first `config["comment_cleaning"]["max_sentences"]` sentences (default 2).
Rejoin with `.  `.

**Step 5 — Artifact cleanup (in order):**
1. Strip trailing whitespace
2. Remove trailing pattern `\.\s+\($` (period + spaces + open paren)
3. Remove trailing `\s+\($` (spaces + open paren)
4. Strip trailing `(` character
5. Strip trailing `.` character
6. Strip trailing whitespace again

**Step 6 — Hard truncation:**
Truncate to `config["comment_cleaning"]["max_chars"]` (default 200).
Return `None` if result is empty.

### `build_event_text(row, susc_string, cleaned_comment) -> str`

Three cases based on the row:

**Case A** — `org_name` is not null:
```
{test_name} [{spec_type_desc}]: {org_name} | {susc_string} | {cleaned_comment}
```
If `susc_string` is empty: omit ` | {susc_string}` segment.
If `cleaned_comment` is None: omit ` | {cleaned_comment}` segment.

**Case B** — `org_name` is null, `cleaned_comment` is not None:
```
{test_name} [{spec_type_desc}]: {cleaned_comment}
```

**Case C** — `org_name` is null, `cleaned_comment` is None:
```
{test_name} [{spec_type_desc}]: pending
```

### `build_susceptibility_string(group_df) -> str`

Given a DataFrame of rows sharing (hadm_id, test_name, spec_type_desc, org_name),
build the susceptibility string.

For each unique `ab_name`, select interpretation by priority: R > S > I.
I appears only when no R or S exists for that antibiotic.
Antibiotics listed in the order they first appear in the source data (no sorting).

Format: `"OXACILLIN:R, VANCOMYCIN:S, CLINDAMYCIN:I"`

Return empty string `""` if no susceptibility data exists.

### `aggregate_panel_text(events_df, panel_config, tokenizer, max_length=512) -> str`

Given a DataFrame of events for one (admission, panel):
1. Build susceptibility strings per (test_name, spec_type_desc, org_name) group
2. Deduplicate identical event text strings
3. Sort by `charttime` ascending
4. Concatenate with ` | ` separator
5. If tokenizer is not None: tokenize and truncate to max_length=512 tokens,
   decode back to string
6. Return empty string `""` if no events

---

## File 3: `extract_microbiology.py`

### Purpose
Step 7 of the pipeline. Reads `microbiologyevents`, applies panel assignment,
comment cleaning, text construction, and writes 37 per-panel text parquets.

Pattern mirrors `extract_labs.py` closely.

### Config keys used

**Required:**
- `MIMIC_DATA_DIR`
- `FEATURES_DIR`
- `CLASSIFICATIONS_DIR`

**Optional (with defaults):**
- `MICRO_WINDOW_HOURS` — int or `"full_admission"` (default: 72)
- `MICRO_NULL_HADM_STRATEGY` — `"drop"` or `"link"` (default: `"drop"`)
- `MICRO_LINK_TOLERANCE_HOURS` — int (default: 2)
- `HADM_LINKAGE_TOLERANCE_HOURS` — fallback if MICRO_LINK_TOLERANCE_HOURS absent
- `HASH_REGISTRY_PATH` — optional

### Algorithm

```
Step 1 — Load micro_panel_config.yaml from CLASSIFICATIONS_DIR
         Raise FileNotFoundError if missing (must run build_micro_panel_config first)
         Build lookup: {(test_name.strip(), spec_type_desc.strip()): panel_name}

Step 2 — Load microbiologyevents.csv.gz
         Columns: microevent_id, subject_id, hadm_id, charttime,
                  spec_type_desc, test_name, org_name, ab_name,
                  interpretation, comments
         Parse charttime as datetime

Step 3 — Drop excluded tests (from micro_panel_config["excluded_tests"])
         Drop excluded spec_types (from micro_panel_config["excluded_spec_types"])
         Log counts dropped

Step 4 — Null hadm_id handling (same pattern as extract_labs.py)
         Strategy: MICRO_NULL_HADM_STRATEGY (default "drop")
         If "drop": drop rows where hadm_id is null, log count
         If "link": use _link_hadm_for_row() with MICRO_LINK_TOLERANCE_HOURS
                    Load admissions for linkage
                    Log linkage statistics: linked / ambiguous / unresolvable
                    Save audit to {CLASSIFICATIONS_DIR}/micro_linkage_stats.json

Step 5 — Load admissions: subject_id, hadm_id, admittime, dischtime
         Merge onto micro_df on hadm_id (inner join — drops rows with no admission)

Step 6 — Apply time window filter based on MICRO_WINDOW_HOURS:
         Integer N: admittime <= charttime <= admittime + N hours
         "full_admission": admittime <= charttime <= dischtime
         Log: retained rows and admissions after filter

Step 7 — Panel assignment
         For each row, look up (test_name.strip(), spec_type_desc.strip()) in combo_lookup
         Assign panel_name, or "unassigned" if not found
         Log count and top combinations of unassigned rows (mirroring lab pattern)
         Drop unassigned rows

Step 8 — Comment cleaning
         Load comment_cleaning config from micro_panel_config
         Apply clean_comment() from build_micro_text.py to "comments" column
         Store in "cleaned_comment" column

Step 9 — Text construction
         For each (subject_id, hadm_id, panel_name) group:
           a. Build susceptibility strings per (test_name, spec_type_desc, org_name)
           b. Build per-event text strings (Cases A/B/C) via build_event_text()
           c. Aggregate via aggregate_panel_text():
              - Deduplicate identical text strings
              - Sort by charttime ascending
              - Concatenate with " | "
              - Truncate to 512 tokens using BERT tokenizer
                (load tokenizer lazily — only if any panel has events)

Step 10 — Write output
          For each of 37 panels:
            Collect all (subject_id, hadm_id, text) for this panel
            Write to {FEATURES_DIR}/micro_{panel_name}.parquet
            Columns: subject_id (int64), hadm_id (int64), text (string)
            Admissions with no events for this panel → NOT included in parquet
            (embed_features.py handles missing admissions as zero vectors)
          Log: per-panel row counts
```

### Output

37 files: `{FEATURES_DIR}/micro_{panel_name}.parquet`

Each file schema:
```
subject_id  int64
hadm_id     int64
text        string (UTF-8)
```

One row per admission that has ≥1 event for that panel within the time window.
Admissions with no events are not written — embed_features.py produces zero
vectors for missing admissions.

### Tokenizer loading

Load tokenizer lazily (only if text construction is needed):
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
    config.get("BERT_MODEL_NAME", "Simonlee711/Clinical_ModernBERT")
)
```
Pass tokenizer to `aggregate_panel_text()`. If BERT_MODEL_NAME is not in config,
fall back to truncation by character count using `MICRO_COMMENT_MAX_CHARS * 10`
as a rough proxy.

---

## File 4: Updates to `preprocessing.yaml`

Add the following keys after `LAB_ADMISSION_WINDOW`:

```yaml
# Microbiology event extraction settings
# Integer N: include events within N hours of admittime.
# "full_admission": include all events from admittime to dischtime.
# Default: 72 hours (microbiology results take 24-72h to finalise;
# charttime = order time, not result time).
MICRO_WINDOW_HOURS: 72

# Strategy for null hadm_id in microbiologyevents (same semantics as
# HADM_LINKAGE_STRATEGY but applied independently to microbiologyevents).
# "drop" (default): exclude rows with null hadm_id.
# "link": attempt hadm_id assignment by time-window matching.
MICRO_NULL_HADM_STRATEGY: "drop"

# Tolerance in hours for MICRO hadm_id linkage (only used when
# MICRO_NULL_HADM_STRATEGY = "link").
# 2h covers typical pre-admission ED workup window.
MICRO_LINK_TOLERANCE_HOURS: 2

# Whether to include cleaned comments field in microbiology text representation.
# The comments column is the primary result field for qualitative tests
# (serology, NAAT, antigen panels). Setting false will produce empty embeddings
# for ~20 panels. Recommended: true.
MICRO_INCLUDE_COMMENTS: true

# Max sentences to retain after comment cleaning (default 2).
MICRO_COMMENT_MAX_SENTENCES: 2

# Hard character limit after comment sentence extraction (default 200).
MICRO_COMMENT_MAX_CHARS: 200
```

Also update `HADM_LINKAGE_TOLERANCE_HOURS` default from 1 to 2:
```yaml
HADM_LINKAGE_TOLERANCE_HOURS: 2
```

---

## File 5: Updates to `run_pipeline.py`

### Add new CLI arguments

After `--build_lab_panel_config`, add:
```python
parser.add_argument(
    "--build_micro_panel_config",
    action="store_true",
    help="Run build_micro_panel_config.py",
)
```

After `--extract_labs`, add:
```python
parser.add_argument(
    "--extract_microbiology",
    action="store_true",
    help="Run extract_microbiology.py",
)
```

### Update `_FULL_ORDER`

Current:
```python
_FULL_ORDER = (
    ["create_splits", "build_lab_panel_config"]
    + _EXTRACT_MODULES
    + ["embed_features", "combine_dataset"]
)
```

Updated — add `build_micro_panel_config` alongside `build_lab_panel_config`,
and add `extract_microbiology` after `extract_labs`:
```python
_FULL_ORDER = (
    ["create_splits", "build_lab_panel_config", "build_micro_panel_config"]
    + _EXTRACT_MODULES   # extract_microbiology added to _EXTRACT_MODULES below
    + ["embed_features", "combine_dataset"]
)
```

### Update `_EXTRACT_MODULES`

```python
_EXTRACT_MODULES = [
    "extract_demographics",
    "extract_diag_history",
    "extract_discharge_history",
    "extract_triage_and_complaint",
    "extract_labs",
    "extract_microbiology",   # ← ADD
    "extract_radiology",
    "extract_y_data",
]
```

---

## File 6: Updates to `embed_features.py`

### Add micro panel constants

After `_LAB_MAX_LENGTH = 2048`, add:
```python
_MICRO_MAX_LENGTH = 512   # microbiology text is short and dense
```

### Add `_load_micro_inputs()` function

Mirror `_load_lab_inputs()` pattern:

```python
def _load_micro_inputs(
    config: dict,
    slice_hadm_ids: set,
) -> tuple[list[str], dict[str, pd.DataFrame]]:
    """Load micro panel parquets filtered to the current slice.

    Returns (panel_names, micro_dfs) where micro_dfs maps panel_name → DataFrame.
    Only panels whose parquets exist are included.
    """
    import yaml

    features_dir = str(config["FEATURES_DIR"])
    classifications_dir = str(config["CLASSIFICATIONS_DIR"])

    micro_panel_config_path = os.path.join(classifications_dir, "micro_panel_config.yaml")
    if not os.path.exists(micro_panel_config_path):
        logger.warning("micro_panel_config.yaml not found — microbiology embeddings skipped.")
        return [], {}

    with open(micro_panel_config_path, encoding="utf-8") as fh:
        micro_cfg = yaml.safe_load(fh)

    panel_names = list(micro_cfg.get("panels", {}).keys())
    micro_dfs: dict[str, pd.DataFrame] = {}

    for panel_name in panel_names:
        parquet_path = os.path.join(features_dir, f"micro_{panel_name}.parquet")
        if not os.path.exists(parquet_path):
            logger.warning("micro_%s.parquet not found — panel skipped.", panel_name)
            continue
        df = pd.read_parquet(parquet_path)
        df = df[df["hadm_id"].isin(slice_hadm_ids)].reset_index(drop=True)
        micro_dfs[panel_name] = df

    logger.info("Loaded %d micro panel parquets for this slice", len(micro_dfs))
    return panel_names, micro_dfs
```

### Add `_build_micro_feature_tasks()` function

Mirror `_build_lab_feature_tasks()` pattern:

```python
def _build_micro_feature_tasks(
    config: dict,
    panel_names: list[str],
    micro_dfs: dict[str, pd.DataFrame],
    splits_df: pd.DataFrame,
) -> list[dict]:
    """Build one task dict per micro panel."""
    if not panel_names or not micro_dfs:
        logger.info("No micro panel data available — skipping micro tasks.")
        return []

    tasks = []
    embeddings_dir = str(config["EMBEDDINGS_DIR"])
    all_hadm_ids = splits_df["hadm_id"].tolist()

    for panel_name in panel_names:
        if panel_name not in micro_dfs:
            # Panel parquet missing — still create a zero-vector task
            texts = [""] * len(all_hadm_ids)
            hadm_ids = all_hadm_ids
        else:
            df = micro_dfs[panel_name]
            hadm_to_text = dict(zip(df["hadm_id"].tolist(), df["text"].tolist()))
            texts = [hadm_to_text.get(h, "") for h in all_hadm_ids]
            hadm_ids = all_hadm_ids

        embedding_col = f"micro_{panel_name}_embedding"
        output_filename = f"micro_{panel_name}_embeddings.parquet"
        output_path = os.path.join(embeddings_dir, output_filename)

        tasks.append({
            "kind": "micro",
            "text_col": f"micro_{panel_name}_text",
            "embedding_col": embedding_col,
            "output_path": output_path,
            "texts": texts,
            "hadm_ids": hadm_ids,
            "subject_ids": splits_df["subject_id"].tolist(),
        })

    logger.info("Built %d micro feature tasks", len(tasks))
    return tasks
```

### Update `_build_feature_tasks()`

Current:
```python
def _build_feature_tasks(config, lab_panel_config, labs_df, splits_df):
    text_tasks = _build_text_feature_tasks(config, splits_df)
    lab_tasks = _build_lab_feature_tasks(config, lab_panel_config, labs_df, splits_df)
    return text_tasks + lab_tasks
```

Updated signature and body:
```python
def _build_feature_tasks(
    config,
    lab_panel_config,
    labs_df,
    splits_df,
    panel_names=None,      # ← NEW
    micro_dfs=None,        # ← NEW
):
    text_tasks = _build_text_feature_tasks(config, splits_df)
    lab_tasks = _build_lab_feature_tasks(config, lab_panel_config, labs_df, splits_df)
    micro_tasks = _build_micro_feature_tasks(
        config, panel_names or [], micro_dfs or {}, splits_df
    )
    return text_tasks + lab_tasks + micro_tasks
```

### Update `_prepare_feature_tasks()` to handle micro kind

In the token cap assignment loop:
```python
for task in all_tasks:
    if task["kind"] == "lab":
        task["max_length"] = min(global_max_length, _LAB_MAX_LENGTH)
    elif task["kind"] == "micro":                           # ← ADD
        task["max_length"] = min(global_max_length, _MICRO_MAX_LENGTH)
    else:
        task["max_length"] = min(
            global_max_length,
            _MAX_LENGTH_CAP.get(task["text_col"], global_max_length),
        )
```

### Update `run()` to load micro inputs

In the `run()` function, after:
```python
lab_panel_config, labs_df = _load_lab_inputs(config, slice_hadm_ids)
```
Add:
```python
panel_names, micro_dfs = _load_micro_inputs(config, slice_hadm_ids)
```

And update the `_prepare_feature_tasks` call:
```python
all_tasks = _prepare_feature_tasks(
    config, lab_panel_config, labs_df, splits_df,
    n_gpus, resolved_slice_index,
    panel_names=panel_names,    # ← ADD
    micro_dfs=micro_dfs,        # ← ADD
)
```

Update `_prepare_feature_tasks` signature to accept and pass through these:
```python
def _prepare_feature_tasks(
    config, lab_panel_config, labs_df, splits_df,
    n_gpus, resolved_slice_index,
    panel_names=None,     # ← ADD
    micro_dfs=None,       # ← ADD
):
    all_tasks = _build_feature_tasks(
        config, lab_panel_config, labs_df, splits_df,
        panel_names=panel_names,
        micro_dfs=micro_dfs,
    )
    ...  # rest unchanged
```

---

## File 7: Updates to `check_embed_status.py`

### Add micro panel discovery

After loading `lab_panel_config`, add:

```python
# Discover micro panel names from micro_panel_config.yaml
import yaml as _yaml  # already imported above

micro_panel_config_path = os.path.join(classifications_dir, "micro_panel_config.yaml")
micro_features = []
if os.path.exists(micro_panel_config_path):
    with open(micro_panel_config_path, encoding="utf-8") as fh:
        micro_cfg = _yaml.safe_load(fh)
    micro_features = [
        (f"micro_{panel_name}_embeddings.parquet", f"micro_{panel_name}_embedding")
        for panel_name in micro_cfg.get("panels", {}).keys()
    ]
else:
    print("  WARNING: micro_panel_config.yaml not found — micro embeddings not checked.")
```

### Update `all_features`

```python
all_features = text_features + lab_features + micro_features  # was: text_features + lab_features
```

### Update required preprocessing outputs

Add to `required_preprocess_outputs`:
```python
os.path.join(classifications_dir, "micro_panel_config.yaml"),
```

---

## File 8: Updates to `preprocessing_utils.py`

### Add `_load_micro_panel_config()`

All existing utility functions are confirmed present in `preprocessing_utils.py`
with the following signatures (do NOT reimplement them):

| Function | Signature | Used by |
|----------|-----------|---------|
| `_check_required_keys` | `(config: dict, required_keys: list[str]) -> None` | All modules |
| `_gz_or_csv` | `(base_dir: str, subdir: str, table: str) -> str` | All extract modules |
| `_load_csv` | `(path_gz: str, path_csv: str, **kwargs) -> pd.DataFrame` | All extract modules |
| `_link_hadm_for_row` | `(row: pd.Series, admissions_df: pd.DataFrame, tolerance: Any) -> float \| None` | extract_labs, extract_microbiology |
| `_load_d_labitems` | `(hosp_dir: str) -> pd.DataFrame` | build_lab_panel_config, extract_labs |
| `_output_is_valid` | `(path: str, expected_rows: int, embedding_col: str) -> bool` | check_embed_status |
| `_sources_unchanged` | `(module_name, source_paths, output_paths, registry_path, logger) -> bool` | All modules |
| `_record_hashes` | `(module_name, source_paths, registry_path) -> None` | All modules |
| `_load_config` | `(config_path: str) -> dict` | run_pipeline, embed_features |

**Key behaviour of `_link_hadm_for_row()`** (important for `extract_microbiology.py`):
- Matches `row["charttime"]` against `[admittime - tolerance, dischtime + tolerance]`
  for all admissions of the same `subject_id`
- Returns `float(hadm_id)` if exactly one match
- If multiple matches: returns the one with `admittime` closest to `charttime`
- Returns `None` if no matches (unresolvable → row is dropped)

**Add only this one new function** to `preprocessing_utils.py`:

```python
def _load_micro_panel_config(classifications_dir: str) -> dict:
    """Load micro_panel_config.yaml and return as dict.

    Returns empty dict if file not found (graceful degradation).
    yaml is already imported at the top of preprocessing_utils.py.
    """
    path = os.path.join(classifications_dir, "micro_panel_config.yaml")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
```

Note: `yaml` is already imported at the module level in `preprocessing_utils.py`
— do not add a local import inside the function.

---

## Testing

After implementation, run the following smoke test:

```bash
# Step 0b: Build micro panel config
python run_pipeline.py --build_micro_panel_config --config config/preprocessing.yaml

# Verify output
ls data/preprocessing/classifications/micro_panel_config.yaml

# Step 7: Extract microbiology (requires data_splits.parquet to exist)
python run_pipeline.py --extract_microbiology --config config/preprocessing.yaml

# Verify: should produce 37 parquet files
ls data/preprocessing/features/micro_*.parquet | wc -l
# Expected: 37

# Check a sample
python -c "
import pandas as pd
df = pd.read_parquet('data/preprocessing/features/micro_blood_culture_routine.parquet')
print(df.dtypes)
print(df.head(3))
print(f'Rows: {len(df)}, Unique admissions: {df.hadm_id.nunique()}')
"
```
