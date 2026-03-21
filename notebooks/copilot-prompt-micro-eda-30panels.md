# Add Section 6: 30-Panel (Test × Specimen) Groupings to `mimic4_microbiology_exploration.ipynb`

## Context

The notebook currently ends at **Section 5** (cell 10), which defines a 7-panel
mapping by `spec_type_desc` alone. The analysis has since evolved to a finer-grained
30-panel design based on `(test_name, spec_type_desc)` combinations.

Add a new **Section 6** immediately after the existing cell 10. Do not modify
any existing cells.

---

## What to add

Add two new cells after cell 10:

- **Cell 11** — markdown header and description
- **Cell 12** — code cell with the full 30-panel analysis

---

## Cell 11 — Markdown

```markdown
## Section 6: 30-Panel (Test × Specimen) Groupings

The 7-panel design grouped by specimen type alone conflates clinically distinct
events. For example, GRAM STAIN on CSF (meningitis workup) and GRAM STAIN on
sputum (pneumonia workup) belong in different feature slots. This section defines
30 fine-grained panels based on `(test_name, spec_type_desc)` combinations,
then computes coverage, positivity rate, and event counts for each panel.

Each panel becomes one 768-d BERT embedding feature in the CDSS model (F20–F49).
Cytogenetics, chromosome analysis, FISH, and post-mortem tests are excluded
(0% positivity, not relevant to mortality/readmission prediction).
```

---

## Cell 12 — Code

```python
# =============================================================================
# Section 6: 30-Panel (test_name × spec_type_desc) grouping
# =============================================================================

# ---------------------------------------------------------------------------
# 6.1 — Define the 30 panels as lists of (test_name, spec_type_desc) pairs
# ---------------------------------------------------------------------------
# Each panel is a dict with keys: 'desc' and 'combos'
# 'combos' is a list of (test_name, spec_type_desc) tuples.
# Rows matching ANY combo in the list are assigned to that panel.
# ---------------------------------------------------------------------------

PANELS_30 = {
    "blood_culture_routine": {
        "desc": "All blood culture methods — routine, fungal, AFB, Brucella, special",
        "combos": [
            ("Blood Culture, Routine",                          "BLOOD CULTURE"),
            ("BLOOD/FUNGAL CULTURE",                           "BLOOD CULTURE ( MYCO/F LYTIC BOTTLE)"),
            ("BLOOD/AFB CULTURE",                              "BLOOD CULTURE ( MYCO/F LYTIC BOTTLE)"),
            ("BRUCELLA BLOOD CULTURE",                         "BLOOD CULTURE"),
            ("M. furfur Blood Culture",                        "BLOOD CULTURE"),
            ("BARTONELLA BLOOD CULTURE",                       "BLOOD CULTURE"),
            ("Blood Culture, Neonate",                         "BLOOD CULTURE - NEONATE"),
            ("Blood Culture, Post Mortem",                     "BLOOD CULTURE (POST-MORTEM)"),
            ("AEROBIC BOTTLE",                                 "BLOOD CULTURE"),
            ("ANAEROBIC BOTTLE",                               "BLOOD CULTURE"),
            ("ISOLATE FOR MIC",                                "Isolate"),
            ("M.FURFUR CULTURE",                               "BLOOD CULTURE"),
        ],
    },
    "blood_bottle_gram_stain": {
        "desc": "Gram stains read from blood culture bottles",
        "combos": [
            ("Aerobic Bottle Gram Stain",   "BLOOD CULTURE"),
            ("Anaerobic Bottle Gram Stain", "BLOOD CULTURE"),
            ("Myco-F Bottle Gram Stain",    "BLOOD CULTURE ( MYCO/F LYTIC BOTTLE)"),
            ("Pediatric Bottle Gram Stain", "Isolate"),
            ("Fluid Culture in Bottles",    "FLUID RECEIVED IN BLOOD CULTURE BOTTLES"),
            ("Stem Cell Aer/Ana Culture",   "Stem Cell - Blood Culture"),
            ("STEM CELL - AEROBIC BOTTLE",  "Stem Cell - Blood Culture"),
            ("STEM CELL - ANAEROBIC BOTTLE","Stem Cell - Blood Culture"),
        ],
    },
    "urine_culture": {
        "desc": "Urine cultures + reflex + urine gram stain",
        "combos": [
            ("URINE CULTURE",              "URINE"),
            ("URINE CULTURE",              "URINE,KIDNEY"),
            ("URINE CULTURE",              "URINE,SUPRAPUBIC ASPIRATE"),
            ("REFLEX URINE CULTURE",       "URINE"),
            ("URINE-GRAM STAIN - UNSPUN",  "URINE"),
            ("URINE-GRAM STAIN - UNSPUN",  "URINE,KIDNEY"),
        ],
    },
    "urine_antigen_naat": {
        "desc": "Legionella urinary antigen + GC/Chlamydia NAAT from urine",
        "combos": [
            ("Legionella Urinary Antigen ",                                          "URINE"),
            ("Chlamydia trachomatis, Nucleic Acid Probe, with Amplification",        "URINE"),
            ("NEISSERIA GONORRHOEAE (GC), NUCLEIC ACID PROBE, WITH AMPLIFICATION",  "URINE"),
        ],
    },
    "respiratory_sputum_bal": {
        "desc": "Bacterial respiratory cultures — sputum, BAL, bronchial, tracheal",
        "combos": [
            ("RESPIRATORY CULTURE", "SPUTUM"),
            ("RESPIRATORY CULTURE", "BRONCHOALVEOLAR LAVAGE"),
            ("RESPIRATORY CULTURE", "BRONCHIAL WASHINGS"),
            ("RESPIRATORY CULTURE", "Mini-BAL"),
            ("RESPIRATORY CULTURE", "BRONCHIAL BRUSH - PROTECTED"),
            ("RESPIRATORY CULTURE", "BRONCHIAL BRUSH"),
            ("RESPIRATORY CULTURE", "TRACHEAL ASPIRATE"),
        ],
    },
    "respiratory_afb": {
        "desc": "AFB culture/smear + MTB PCR + Nocardia — TB/mycobacterial workup",
        "combos": [
            ("ACID FAST CULTURE",                                           "SPUTUM"),
            ("ACID FAST SMEAR",                                             "SPUTUM"),
            ("ACID FAST CULTURE",                                           "BRONCHOALVEOLAR LAVAGE"),
            ("ACID FAST SMEAR",                                             "BRONCHOALVEOLAR LAVAGE"),
            ("ACID FAST CULTURE",                                           "BRONCHIAL WASHINGS"),
            ("ACID FAST SMEAR",                                             "BRONCHIAL WASHINGS"),
            ("ACID FAST CULTURE",                                           "Mini-BAL"),
            ("ACID FAST SMEAR",                                             "Mini-BAL"),
            ("MTB Direct Amplification",                                    "SPUTUM"),
            ("MTB Direct Amplification",                                    "BRONCHOALVEOLAR LAVAGE"),
            ("GEN-PROBE AMPLIFIED M. TUBERCULOSIS DIRECT TEST (MTD)",       "SPUTUM"),
            ("GEN-PROBE AMPLIFIED M. TUBERCULOSIS DIRECT TEST (MTD)",       "BRONCHOALVEOLAR LAVAGE"),
            ("NOCARDIA CULTURE",                                            "BRONCHOALVEOLAR LAVAGE"),
            ("NOCARDIA CULTURE",                                            "SPUTUM"),
            ("NOCARDIA CULTURE",                                            "Mini-BAL"),
            ("MODIFIED ACID-FAST STAIN FOR NOCARDIA",                       "SPUTUM"),
            ("MODIFIED ACID-FAST STAIN FOR NOCARDIA",                       "BRONCHOALVEOLAR LAVAGE"),
        ],
    },
    "respiratory_viral": {
        "desc": "Influenza A/B, RSV, respiratory viral antigen screen and culture",
        "combos": [
            ("Respiratory Viral Antigen Screen",    "Rapid Respiratory Viral Screen & Culture"),
            ("Respiratory Viral Culture",           "Rapid Respiratory Viral Screen & Culture"),
            ("Respiratory Virus Identification",    "Rapid Respiratory Viral Screen & Culture"),
            ("DIRECT INFLUENZA A ANTIGEN TEST",     "Influenza A/B by DFA"),
            ("DIRECT INFLUENZA B ANTIGEN TEST",     "Influenza A/B by DFA"),
            ("DIRECT RSV ANTIGEN TEST",             "Rapid Respiratory Viral Screen & Culture"),
            ("Respiratory Viral Antigen Screen",    "BRONCHOALVEOLAR LAVAGE"),
            ("Respiratory Viral Culture",           "BRONCHOALVEOLAR LAVAGE"),
            ("Respiratory Viral Antigen Screen",    "Influenza A/B by DFA"),
            ("Respiratory Virus Identification",    "Influenza A/B by DFA"),
            ("DIRECT INFLUENZA A ANTIGEN TEST",     "RAPID RESPIRATORY VIRAL ANTIGEN TEST"),
            ("DIRECT INFLUENZA B ANTIGEN TEST",     "RAPID RESPIRATORY VIRAL ANTIGEN TEST"),
        ],
    },
    "respiratory_pcp_legionella": {
        "desc": "PCP immunofluorescence + Legionella culture",
        "combos": [
            ("Immunofluorescent test for Pneumocystis jirovecii (carinii)", "SPUTUM"),
            ("Immunofluorescent test for Pneumocystis jirovecii (carinii)", "BRONCHOALVEOLAR LAVAGE"),
            ("Immunofluorescent test for Pneumocystis jirovecii (carinii)", "BRONCHIAL WASHINGS"),
            ("Immunofluorescent test for Pneumocystis jirovecii (carinii)", "Mini-BAL"),
            ("LEGIONELLA CULTURE",                                          "SPUTUM"),
            ("LEGIONELLA CULTURE",                                          "BRONCHOALVEOLAR LAVAGE"),
            ("LEGIONELLA CULTURE",                                          "BRONCHIAL WASHINGS"),
        ],
    },
    "gram_stain_respiratory": {
        "desc": "Gram stain from sputum, BAL, bronchial — pneumonia workup",
        "combos": [
            ("GRAM STAIN", "SPUTUM"),
            ("GRAM STAIN", "BRONCHOALVEOLAR LAVAGE"),
            ("GRAM STAIN", "BRONCHIAL WASHINGS"),
            ("GRAM STAIN", "Mini-BAL"),
            ("GRAM STAIN", "TRACHEAL ASPIRATE"),
        ],
    },
    "gram_stain_wound_fluid": {
        "desc": "Gram stain from wound, tissue, abscess, peritoneal, pleural, joint",
        "combos": [
            ("GRAM STAIN", "SWAB"),
            ("GRAM STAIN", "TISSUE"),
            ("GRAM STAIN", "ABSCESS"),
            ("GRAM STAIN", "PERITONEAL FLUID"),
            ("GRAM STAIN", "PLEURAL FLUID"),
            ("GRAM STAIN", "FLUID,OTHER"),
            ("GRAM STAIN", "JOINT FLUID"),
            ("GRAM STAIN", "BILE"),
            ("GRAM STAIN", "CATHETER TIP-IV"),
            ("GRAM STAIN", "FOREIGN BODY"),
            ("GRAM STAIN", "DIALYSIS FLUID"),
            ("GRAM STAIN", "BONE MARROW"),
        ],
    },
    "gram_stain_csf": {
        "desc": "Gram stain from CSF — meningitis / encephalitis workup",
        "combos": [
            ("GRAM STAIN", "CSF;SPINAL FLUID"),
        ],
    },
    "wound_culture": {
        "desc": "Wound, abscess, catheter, tissue, anaerobic, sonication cultures",
        "combos": [
            ("WOUND CULTURE",                           "SWAB"),
            ("WOUND CULTURE",                           "ABSCESS"),
            ("WOUND CULTURE",                           "CATHETER TIP-IV"),
            ("WOUND CULTURE",                           "FOREIGN BODY"),
            ("WOUND CULTURE",                           "FOOT CULTURE"),
            ("WOUND CULTURE",                           "TISSUE"),
            ("ANAEROBIC CULTURE",                       "SWAB"),
            ("ANAEROBIC CULTURE",                       "ABSCESS"),
            ("ANAEROBIC CULTURE",                       "TISSUE"),
            ("TISSUE",                                  "TISSUE"),
            ("Sonication culture, prosthetic joint",    "Foreign Body - Sonication Culture"),
            ("Sonication culture, prosthetic joint",    "FOREIGN BODY"),
        ],
    },
    "fluid_culture": {
        "desc": "Cultures from peritoneal, pleural, biliary, joint, dialysis fluids",
        "combos": [
            ("FLUID CULTURE",                           "PERITONEAL FLUID"),
            ("FLUID CULTURE",                           "PLEURAL FLUID"),
            ("FLUID CULTURE",                           "BILE"),
            ("FLUID CULTURE",                           "JOINT FLUID"),
            ("FLUID CULTURE",                           "DIALYSIS FLUID"),
            ("FLUID CULTURE",                           "ABSCESS"),
            ("FLUID CULTURE",                           "FLUID,OTHER"),
            ("ANAEROBIC CULTURE",                       "PERITONEAL FLUID"),
            ("ANAEROBIC CULTURE",                       "PLEURAL FLUID"),
            ("ANAEROBIC CULTURE",                       "BILE"),
            ("ANAEROBIC CULTURE",                       "JOINT FLUID"),
            ("Anaerobic culture, Prosthetic Joint Fluid","PROSTHETIC JOINT FLUID"),
        ],
    },
    "csf_culture": {
        "desc": "Cultures + viral + Cryptococcal antigen from CSF and serum",
        "combos": [
            ("FLUID CULTURE",                       "CSF;SPINAL FLUID"),
            ("FUNGAL CULTURE",                      "CSF;SPINAL FLUID"),
            ("ACID FAST CULTURE",                   "CSF;SPINAL FLUID"),
            ("VIRAL CULTURE",                       "CSF;SPINAL FLUID"),
            ("Enterovirus Culture",                 "CSF;SPINAL FLUID"),
            ("CRYPTOCOCCAL ANTIGEN",                "CSF;SPINAL FLUID"),
            ("QUANTITATIVE CRYPTOCOCCAL ANTIGEN",   "CSF;SPINAL FLUID"),
            ("CRYPTOCOCCAL ANTIGEN",                "SEROLOGY/BLOOD"),
            ("QUANTITATIVE CRYPTOCOCCAL ANTIGEN",   "SEROLOGY/BLOOD"),
        ],
    },
    "fungal_tissue_wound": {
        "desc": "Fungal culture + KOH from tissue, wound, skin",
        "combos": [
            ("FUNGAL CULTURE",                                      "TISSUE"),
            ("FUNGAL CULTURE",                                      "SWAB"),
            ("FUNGAL CULTURE",                                      "ABSCESS"),
            ("FUNGAL CULTURE",                                      "JOINT FLUID"),
            ("FUNGAL CULTURE",                                      "SKIN SCRAPINGS"),
            ("POTASSIUM HYDROXIDE PREPARATION",                     "TISSUE"),
            ("POTASSIUM HYDROXIDE PREPARATION",                     "SWAB"),
            ("POTASSIUM HYDROXIDE PREPARATION",                     "ABSCESS"),
            ("POTASSIUM HYDROXIDE PREPARATION (HAIR/SKIN/NAILS)",   "SKIN SCRAPINGS"),
            ("FUNGAL CULTURE (HAIR/SKIN/NAILS)",                    "SKIN SCRAPINGS"),
            ("FUNGAL CULTURE (HAIR/SKIN/NAILS)",                    "NAIL SCRAPINGS"),
        ],
    },
    "fungal_respiratory": {
        "desc": "Fungal culture + KOH from respiratory specimens",
        "combos": [
            ("FUNGAL CULTURE",                  "SPUTUM"),
            ("FUNGAL CULTURE",                  "BRONCHOALVEOLAR LAVAGE"),
            ("FUNGAL CULTURE",                  "BRONCHIAL WASHINGS"),
            ("FUNGAL CULTURE",                  "Mini-BAL"),
            ("POTASSIUM HYDROXIDE PREPARATION", "BRONCHOALVEOLAR LAVAGE"),
            ("POTASSIUM HYDROXIDE PREPARATION", "BRONCHIAL WASHINGS"),
            ("POTASSIUM HYDROXIDE PREPARATION", "Mini-BAL"),
            ("POTASSIUM HYDROXIDE PREPARATION", "SPUTUM"),
        ],
    },
    "fungal_fluid": {
        "desc": "Fungal culture from peritoneal, pleural, joint, other fluids",
        "combos": [
            ("FUNGAL CULTURE",                  "PERITONEAL FLUID"),
            ("FUNGAL CULTURE",                  "PLEURAL FLUID"),
            ("FUNGAL CULTURE",                  "URINE"),
            ("FUNGAL CULTURE",                  "FLUID,OTHER"),
            ("POTASSIUM HYDROXIDE PREPARATION", "PERITONEAL FLUID"),
            ("POTASSIUM HYDROXIDE PREPARATION", "PLEURAL FLUID"),
            ("POTASSIUM HYDROXIDE PREPARATION", "CSF;SPINAL FLUID"),
            ("POTASSIUM HYDROXIDE PREPARATION", "JOINT FLUID"),
        ],
    },
    "mrsa_staph_screen": {
        "desc": "MRSA + Staph aureus screen + Preop PCR",
        "combos": [
            ("MRSA SCREEN",         "MRSA SCREEN"),
            ("Staph aureus Screen", "Staph aureus swab"),
            ("Staph aureus Screen", "MRSA SCREEN"),
            ("Staph aureus Preop PCR", "Staph aureus swab"),
            ("Staph aureus Preop PCR", "MRSA SCREEN"),
        ],
    },
    "resistance_screen": {
        "desc": "VRE + CRE surveillance screens",
        "combos": [
            ("R/O VANCOMYCIN RESISTANT ENTEROCOCCUS",                  "SWAB"),
            ("R/O VANCOMYCIN RESISTANT ENTEROCOCCUS",                  "STOOL"),
            ("Carbapenemase Resistant Enterobacteriaceae Screen",       "CRE Screen"),
            ("Carbapenemase Resistant Enterobacteriaceae Screen",       "SWAB"),
            ("CRE/ESBL/AMP-C Screening",                               "SWAB"),
        ],
    },
    "cdiff": {
        "desc": "All C. difficile tests — PCR + toxin A&B + toxin assay",
        "combos": [
            ("C. difficile PCR",                        "STOOL"),
            ("CLOSTRIDIUM DIFFICILE TOXIN A & B TEST",  "STOOL"),
            ("CLOSTRIDIUM DIFFICILE TOXIN ASSAY",       "STOOL"),
            ("C. difficile Toxin antigen assay",        "STOOL"),
        ],
    },
    "stool_bacterial": {
        "desc": "Fecal cultures + Campylobacter + E.coli O157 + Yersinia + Vibrio + Shiga",
        "combos": [
            ("FECAL CULTURE",                       "STOOL"),
            ("FECAL CULTURE",                       "STOOL (RECEIVED IN TRANSPORT SYSTEM)"),
            ("CAMPYLOBACTER CULTURE",               "STOOL"),
            ("FECAL CULTURE - R/O E.COLI 0157:H7", "STOOL"),
            ("FECAL CULTURE - R/O YERSINIA",        "STOOL"),
            ("FECAL CULTURE - R/O VIBRIO",          "STOOL"),
            ("SHIGA TOXIN (EHEC)",                  "STOOL"),
        ],
    },
    "stool_parasitology": {
        "desc": "OVA+parasites + Crypto/Giardia + Cyclospora + Microsporidia",
        "combos": [
            ("OVA + PARASITES",                     "STOOL"),
            ("OVA + PARASITES",                     "STOOL (RECEIVED IN TRANSPORT SYSTEM)"),
            ("Cryptosporidium/Giardia (DFA)",       "STOOL"),
            ("CYCLOSPORA STAIN",                    "STOOL"),
            ("MICROSPORIDIA STAIN",                 "STOOL"),
            ("O&P MACROSCOPIC EXAM - WORM",         "STOOL"),
            ("O&P MACROSCOPIC EXAM - ARTHROPOD",    "STOOL"),
            ("Concentration and Stain for Giardia", "STOOL"),
            ("Acid Fast Stain for Cryptosporidium", "STOOL"),
        ],
    },
    "herpesvirus_serology": {
        "desc": "CMV/EBV/VZV serology and viral loads from blood",
        "combos": [
            ("CMV Viral Load",                  "Immunology (CMV)"),
            ("CMV Viral Load",                  "IMMUNOLOGY"),
            ("CMV IgG ANTIBODY",                "Blood (CMV AB)"),
            ("CMV IgM ANTIBODY",                "Blood (CMV AB)"),
            ("EPSTEIN-BARR VIRUS VCA-IgG AB",   "Blood (EBV)"),
            ("EPSTEIN-BARR VIRUS VCA-IgM AB",   "Blood (EBV)"),
            ("EPSTEIN-BARR VIRUS EBNA IgG AB",  "Blood (EBV)"),
            ("VARICELLA-ZOSTER IgG SEROLOGY",   "SEROLOGY/BLOOD"),
            ("MONOSPOT",                        "SEROLOGY/BLOOD"),
        ],
    },
    "hepatitis_hiv": {
        "desc": "HIV/HCV/HBV viral loads + HCV genotyping",
        "combos": [
            ("HCV VIRAL LOAD",              "IMMUNOLOGY"),
            ("HIV-1 Viral Load/Ultrasensitive", "IMMUNOLOGY"),
            ("HBV Viral Load",              "IMMUNOLOGY"),
            ("HCV GENOTYPE",                "IMMUNOLOGY"),
            ("Reflex HCV Qual PCR",         "IMMUNOLOGY"),
        ],
    },
    "syphilis_serology": {
        "desc": "RPR + quantitative RPR + treponemal antibody from blood",
        "combos": [
            ("RAPID PLASMA REAGIN TEST",    "SEROLOGY/BLOOD"),
            ("RPR w/check for Prozone",     "SEROLOGY/BLOOD"),
            ("QUANTITATIVE RPR",            "SEROLOGY/BLOOD"),
            ("TREPONEMAL ANTIBODY TEST",    "SEROLOGY/BLOOD"),
        ],
    },
    "misc_serology": {
        "desc": "Lyme, Toxo, H.pylori, Malaria, Rubella, Mumps, ASO",
        "combos": [
            ("LYME SEROLOGY",                   "SEROLOGY/BLOOD"),
            ("Lyme IgM",                        "Blood (LYME)"),
            ("Lyme IgG",                        "Blood (LYME)"),
            ("TOXOPLASMA IgG ANTIBODY",         "Blood (Toxo)"),
            ("TOXOPLASMA IgM ANTIBODY",         "Blood (Toxo)"),
            ("HELICOBACTER PYLORI ANTIBODY TEST","SEROLOGY/BLOOD"),
            ("Malaria Antigen Test",            "Blood (Malaria)"),
            ("RUBELLA IgG SEROLOGY",            "SEROLOGY/BLOOD"),
            ("Rubella IgG/IgM Antibody",        "SEROLOGY/BLOOD"),
            ("RUBEOLA ANTIBODY, IgG",           "SEROLOGY/BLOOD"),
            ("MUMPS IgG ANTIBODY",              "SEROLOGY/BLOOD"),
            ("ASO Screen",                      "SEROLOGY/BLOOD"),
            ("ASO TITER",                       "SEROLOGY/BLOOD"),
        ],
    },
    "herpesvirus_culture_antigen": {
        "desc": "CMV/HSV/VZV direct antigen tests + cultures from skin/swab/resp",
        "combos": [
            ("CYTOMEGALOVIRUS EARLY ANTIGEN TEST (SHELL VIAL METHOD)", "BRONCHOALVEOLAR LAVAGE"),
            ("CYTOMEGALOVIRUS EARLY ANTIGEN TEST (SHELL VIAL METHOD)", "TISSUE"),
            ("VIRAL CULTURE: R/O CYTOMEGALOVIRUS",                     "BRONCHOALVEOLAR LAVAGE"),
            ("VIRAL CULTURE: R/O CYTOMEGALOVIRUS",                     "TISSUE"),
            ("VIRAL CULTURE:R/O HERPES SIMPLEX VIRUS",                 "SWAB"),
            ("VIRAL CULTURE:R/O HERPES SIMPLEX VIRUS",                 "SKIN SCRAPINGS"),
            ("Direct Antigen Test for Herpes Simplex Virus Types 1 & 2","SKIN SCRAPINGS"),
            ("DIRECT ANTIGEN TEST FOR VARICELLA-ZOSTER VIRUS",         "DIRECT ANTIGEN TEST FOR VARICELLA-ZOSTER VIRUS"),
            ("VARICELLA-ZOSTER CULTURE",                               "SWAB"),
            ("VARICELLA-ZOSTER CULTURE",                               "THROAT CULTURE"),
        ],
    },
    "gc_chlamydia_sti": {
        "desc": "GC/Chlamydia NAAT + genital cultures from swab",
        "combos": [
            ("Chlamydia trachomatis, Nucleic Acid Probe, with Amplification",       "SWAB"),
            ("NEISSERIA GONORRHOEAE (GC), NUCLEIC ACID PROBE, WITH AMPLIFICATION", "SWAB"),
            ("CHLAMYDIA CULTURE",           "SWAB"),
            ("GENITAL CULTURE",             "SWAB"),
            ("R/O GC Only",                 "SWAB"),
            ("GENITAL CULTURE FOR TOXIC SHOCK", "SWAB"),
        ],
    },
    "vaginal_genital_flora": {
        "desc": "BV + GBS + yeast vaginitis + Trichomonas",
        "combos": [
            ("YEAST VAGINITIS CULTURE",             "SWAB"),
            ("YEAST VAGINITIS CULTURE",             "ANORECTAL/VAGINAL"),
            ("SMEAR FOR BACTERIAL VAGINOSIS",       "SWAB"),
            ("SMEAR FOR BACTERIAL VAGINOSIS",       "ANORECTAL/VAGINAL"),
            ("R/O GROUP B BETA STREP",              "ANORECTAL/VAGINAL"),
            ("R/O GROUP B BETA STREP",              "SWAB"),
            ("R/O Group B Strep - Penicillin Allergy", "ANORECTAL/VAGINAL"),
            ("TRICHOMONAS SALINE PREP",             "SWAB"),
        ],
    },
    "throat_strep": {
        "desc": "Strep Group A + thrush gram stain from throat",
        "combos": [
            ("R/O Beta Strep Group A",      "THROAT FOR STREP"),
            ("R/O Beta Strep Group A",      "THROAT CULTURE"),
            ("GRAM STAIN- R/O THRUSH",      "THROAT FOR STREP"),
            ("GRAM STAIN- R/O THRUSH",      "SWAB"),
        ],
    },
}

# Cytogenetics / chromosomal tests to exclude entirely
EXCLUDED_TESTS = {
    "MOLECULAR CYTOGENETICS - DNA PROBE",
    "MOLECULAR CYTOGENETICS-DNA Probe",
    "INTERPHASE FISH ANALYSIS, 100-300 CELLS",
    "INTERPHASE FISH ANALYSIS, 25-99 CELLS",
    "FISH ANALYSIS, 3-5 CELLS",
    "CHROMOSOME ANALYSIS-BONE MARROW",
    "CHROMOSOME ANALYSIS-TISSUE",
    "CHROMOSOME ANALYSIS-NEOPLASTIC BLOOD",
    "CHROMOSOME ANALYSIS-AMNIOTIC FLUID",
    "CHROMOSOME ANALYSIS-BLOOD",
    "CHROMOSOME ANALYSIS-FLUID",
    "CHROMOSOME ANALYSIS-CVS",
    "Tissue Culture-Bone Marrow",
    "Tissue Culture - Neoplastic Blood",
    "TISSUE CULTURE-LYMPHOCYTE",
    "TISSUE CULTURE-FLUID",
    "Additional Cells and Karyotype",
    "ADDITIONAL CELLS COUNTED",
    "FOCUSED ANALYSIS",
    "Deparaffinization, Lysis of Cells",
    "Tissue culture for additional cells",
    "voided",
    "Stool Hold Request",
    "POST MORTEM MYCOLOGY CULTURE",
    "POST-MORTEM VIRAL CULTURE",
    "POSTMORTEM CULTURE",
    "Blood Culture, Post Mortem",
}

# ---------------------------------------------------------------------------
# 6.2 — Assign panel labels to micro_df rows
# ---------------------------------------------------------------------------

# Build lookup: (test_name, spec_type_desc) → panel_name
combo_to_panel = {}
for panel_name, panel_info in PANELS_30.items():
    for test, spec in panel_info["combos"]:
        combo_to_panel[(test, spec)] = panel_name

def assign_panel_30(row):
    if row['test_name'] in EXCLUDED_TESTS:
        return 'EXCLUDED'
    key = (row['test_name'], row['spec_type_desc'])
    return combo_to_panel.get(key, 'unassigned')

micro_df['panel_30'] = micro_df.apply(assign_panel_30, axis=1)

# ---------------------------------------------------------------------------
# 6.3 — Coverage summary table
# ---------------------------------------------------------------------------

print("=" * 105)
print("30-PANEL COVERAGE SUMMARY")
print("=" * 105)
print(f"\n{'#':<4} {'Panel':<35} {'Combos':>6} {'Events':>10} {'Admissions':>11} "
      f"{'Coverage%':>10} {'Pos%':>7}")
print("-" * 85)

summary_rows = []
for i, (panel_name, panel_info) in enumerate(PANELS_30.items(), 1):
    p_df = micro_df[micro_df['panel_30'] == panel_name]
    n_events    = len(p_df)
    n_adm       = p_df['hadm_id'].nunique()
    coverage    = n_adm / total_admissions * 100
    pos_rate    = p_df['org_name'].notna().mean() * 100 if n_events > 0 else 0
    n_combos    = len(panel_info["combos"])
    summary_rows.append({
        'rank': i, 'panel': panel_name, 'combos': n_combos,
        'events': n_events, 'admissions': n_adm,
        'coverage_pct': coverage, 'pos_rate': pos_rate,
        'desc': panel_info["desc"],
    })
    print(f"{i:<4} {panel_name:<35} {n_combos:>6} {n_events:>10,} {n_adm:>11,} "
          f"{coverage:>9.2f}% {pos_rate:>6.1f}%")

summary_df = pd.DataFrame(summary_rows)

# Unassigned and excluded
n_unassigned = (micro_df['panel_30'] == 'unassigned').sum()
n_excluded   = (micro_df['panel_30'] == 'EXCLUDED').sum()
total_assigned = summary_df['events'].sum()
print("-" * 85)
print(f"{'Unassigned':<40} {n_unassigned:>10,}")
print(f"{'Excluded (cytogenetics/admin)':<40} {n_excluded:>10,}")
print(f"{'TOTAL assigned':<40} {total_assigned:>10,}  "
      f"(of {len(micro_df):,} admission-linked rows)")
print()
print(f"Panel count      : {len(PANELS_30)}")
print(f"Total admissions : {total_admissions:,}")
print(f"Admissions with ≥1 panel assigned : "
      f"{micro_df[micro_df['panel_30'].isin(PANELS_30)]['hadm_id'].nunique():,}")

# ---------------------------------------------------------------------------
# 6.4 — Bar chart: events per panel, sorted descending
# ---------------------------------------------------------------------------

chart_df = summary_df.sort_values('events', ascending=True)
palette  = sns.color_palette('tab20', n_colors=len(chart_df))

fig, ax = plt.subplots(figsize=(14, 11))
bars = ax.barh(chart_df['panel'], chart_df['events'],
               color=palette, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, chart_df['events']):
    ax.text(bar.get_width() + chart_df['events'].max() * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f'{val:,}', va='center', fontsize=8)
ax.set_title('Total Events per Panel — 30-Panel (Test × Specimen) Design', fontsize=13)
ax.set_xlabel('Number of Events', fontsize=11)
ax.set_ylabel('Panel', fontsize=11)
ax.tick_params(axis='y', labelsize=9)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# 6.5 — Coverage vs positivity scatter plot
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(13, 9))
scatter = ax.scatter(
    summary_df['coverage_pct'],
    summary_df['pos_rate'],
    s=summary_df['events'] / summary_df['events'].max() * 1500 + 30,
    c=range(len(summary_df)),
    cmap='tab20',
    alpha=0.8,
    edgecolors='black',
    linewidth=0.5,
)

for _, row in summary_df.iterrows():
    ax.annotate(
        row['panel'],
        xy=(row['coverage_pct'], row['pos_rate']),
        xytext=(4, 2),
        textcoords='offset points',
        fontsize=7,
        alpha=0.85,
    )

ax.axvline(x=5,  color='gray', linestyle='--', linewidth=0.8, alpha=0.6, label='5% coverage')
ax.axhline(y=20, color='gray', linestyle=':',  linewidth=0.8, alpha=0.6, label='20% positivity')
ax.set_xlabel('Admission Coverage (% of all admissions with ≥1 event in panel)', fontsize=11)
ax.set_ylabel('Positivity Rate (% of events with organism identified)', fontsize=11)
ax.set_title('30-Panel Signal Map: Coverage vs Positivity\n(bubble size ∝ event count)', fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# 6.6 — Text summary for copy-paste
# ---------------------------------------------------------------------------

print("\n" + "=" * 105)
print("PANEL DESCRIPTIONS")
print("=" * 105)
for row in summary_rows:
    print(f"  {row['rank']:<3} {row['panel']:<35}  {row['desc']}")
```

---

## Requirements

- Add the two cells **after** the existing cell 10 (Section 5) — do not modify any existing cells
- `micro_df` and `total_admissions` are already loaded in cell 2 — do not reload them
- `PANELS_30` dict must be defined exactly as written — the combo tuples must match the exact `test_name` and `spec_type_desc` strings from `microbiologyevents.csv`
- The `panel_30` column is added to `micro_df` in-place so it is available for use in later cells
- All imports (`pandas`, `matplotlib`, `seaborn`, `numpy`) are already available from cell 2
