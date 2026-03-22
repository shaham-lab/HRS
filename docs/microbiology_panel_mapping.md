# Microbiology Panel Groupings — Complete Test × Specimen Mapping

**37 panels.** Applied corrections:

1. **Post-mortem → EXCLUDED** — prevents target leakage (Y1 = in-hospital mortality)
2. **Fluid panels split** — `pleural_culture`, `peritoneal_culture`, `joint_fluid_culture`, `fluid_culture` mirror F8/F9/F12 lab architecture
3. **`bone_marrow_culture`** — mirrors F11 lab slot
4. **`hardware_and_lines_culture`** — separates CLABSI/device workup from wound cultures
5. **`respiratory_sputum_bal` split** — `respiratory_invasive` (BAL/bronchial/tracheal) vs `respiratory_non_invasive` (sputum/aspirate)
6. **`urine_antigen_naat` split** — `urinary_antigens` (Legionella only); GC/Chlamydia NAAT on urine absorbed into `gc_chlamydia_sti`
7. **`urine_viral`** — CMV viruria isolated from routine UTI workup

**473 assigned combos across 37 panels · 38 excluded**

| # | Test Name | Specimen Type | Panel |
|---|-----------|---------------|-------|
| 1 | AEROBIC BOTTLE | BLOOD CULTURE | blood_culture_routine |
|  | ANAEROBIC BOTTLE | BLOOD CULTURE | blood_culture_routine |
|  | BARTONELLA BLOOD CULTURE | BLOOD CULTURE | blood_culture_routine |
|  | BLOOD/AFB CULTURE | BLOOD CULTURE | blood_culture_routine |
|  | BLOOD/AFB CULTURE | BLOOD CULTURE ( MYCO/F LYTIC BOTTLE) | blood_culture_routine |
|  | BLOOD/FUNGAL CULTURE | BLOOD CULTURE | blood_culture_routine |
|  | BLOOD/FUNGAL CULTURE | BLOOD CULTURE ( MYCO/F LYTIC BOTTLE) | blood_culture_routine |
|  | BRUCELLA BLOOD CULTURE | BLOOD CULTURE | blood_culture_routine |
|  | Blood Culture, Neonate | BLOOD CULTURE - NEONATE | blood_culture_routine |
|  | Blood Culture, Routine | BLOOD CULTURE | blood_culture_routine |
|  | ISOLATE FOR MIC | BLOOD CULTURE | blood_culture_routine |
|  | ISOLATE FOR MIC | Isolate | blood_culture_routine |
|  | M. furfur Blood Culture | BLOOD CULTURE | blood_culture_routine |
|  | M.FURFUR CULTURE | BLOOD CULTURE | blood_culture_routine |
| 2 | AEROBIC BOTTLE | FLUID RECEIVED IN BLOOD CULTURE BOTTLES | blood_bottle_gram_stain |
|  | ANAEROBIC BOTTLE | FLUID RECEIVED IN BLOOD CULTURE BOTTLES | blood_bottle_gram_stain |
|  | Aerobic Bottle Gram Stain | BLOOD CULTURE | blood_bottle_gram_stain |
|  | Aerobic Bottle Gram Stain | FLUID RECEIVED IN BLOOD CULTURE BOTTLES | blood_bottle_gram_stain |
|  | Anaerobic Bottle Gram Stain | BLOOD CULTURE | blood_bottle_gram_stain |
|  | Anaerobic Bottle Gram Stain | FLUID RECEIVED IN BLOOD CULTURE BOTTLES | blood_bottle_gram_stain |
|  | Fluid Culture in Bottles | FLUID RECEIVED IN BLOOD CULTURE BOTTLES | blood_bottle_gram_stain |
|  | Myco-F Bottle Gram Stain | BLOOD CULTURE ( MYCO/F LYTIC BOTTLE) | blood_bottle_gram_stain |
|  | Pediatric Bottle Gram Stain | Isolate | blood_bottle_gram_stain |
|  | STEM CELL - AEROBIC BOTTLE | Stem Cell - Blood Culture | blood_bottle_gram_stain |
|  | STEM CELL - ANAEROBIC BOTTLE | Stem Cell - Blood Culture | blood_bottle_gram_stain |
|  | Stem Cell Aer/Ana Culture | Stem Cell - Blood Culture | blood_bottle_gram_stain |
| 3 | ANAEROBIC CULTURE | URINE | urine_culture |
|  | ANAEROBIC CULTURE | URINE,KIDNEY | urine_culture |
|  | FLUID CULTURE | URINE | urine_culture |
|  | FLUID CULTURE | URINE,KIDNEY | urine_culture |
|  | FLUID CULTURE | URINE,SUPRAPUBIC ASPIRATE | urine_culture |
|  | ISOLATE FOR MIC | URINE | urine_culture |
|  | REFLEX URINE CULTURE | URINE | urine_culture |
|  | URINE CULTURE | URINE | urine_culture |
|  | URINE CULTURE | URINE,KIDNEY | urine_culture |
|  | URINE CULTURE | URINE,SUPRAPUBIC ASPIRATE | urine_culture |
|  | URINE-GRAM STAIN - UNSPUN | URINE | urine_culture |
|  | URINE-GRAM STAIN - UNSPUN | URINE,KIDNEY | urine_culture |
| 4 | CYTOMEGALOVIRUS EARLY ANTIGEN TEST (SHELL VIAL METHOD) | URINE | urine_viral |
|  | VIRAL CULTURE | URINE | urine_viral |
|  | VIRAL CULTURE: R/O CYTOMEGALOVIRUS | URINE | urine_viral |
| 5 | Legionella Urinary Antigen | URINE | urinary_antigens |
| 6 | ISOLATE FOR MIC | SPUTUM | respiratory_non_invasive |
|  | RESPIRATORY CULTURE | ASPIRATE | respiratory_non_invasive |
|  | RESPIRATORY CULTURE | SPUTUM | respiratory_non_invasive |
| 7 | RESPIRATORY CULTURE | BRONCHIAL BRUSH | respiratory_invasive |
|  | RESPIRATORY CULTURE | BRONCHIAL BRUSH - PROTECTED | respiratory_invasive |
|  | RESPIRATORY CULTURE | BRONCHIAL WASHINGS | respiratory_invasive |
|  | RESPIRATORY CULTURE | BRONCHOALVEOLAR LAVAGE | respiratory_invasive |
|  | RESPIRATORY CULTURE | Mini-BAL | respiratory_invasive |
|  | RESPIRATORY CULTURE | TRACHEAL ASPIRATE | respiratory_invasive |
| 8 | ACID FAST CULTURE | BRONCHIAL BRUSH | respiratory_afb |
|  | ACID FAST CULTURE | BRONCHIAL BRUSH - PROTECTED | respiratory_afb |
|  | ACID FAST CULTURE | BRONCHIAL WASHINGS | respiratory_afb |
|  | ACID FAST CULTURE | BRONCHOALVEOLAR LAVAGE | respiratory_afb |
|  | ACID FAST CULTURE | Mini-BAL | respiratory_afb |
|  | ACID FAST CULTURE | SPUTUM | respiratory_afb |
|  | ACID FAST CULTURE | TRACHEAL ASPIRATE | respiratory_afb |
|  | ACID FAST SMEAR | BRONCHIAL BRUSH | respiratory_afb |
|  | ACID FAST SMEAR | BRONCHIAL BRUSH - PROTECTED | respiratory_afb |
|  | ACID FAST SMEAR | BRONCHIAL WASHINGS | respiratory_afb |
|  | ACID FAST SMEAR | BRONCHOALVEOLAR LAVAGE | respiratory_afb |
|  | ACID FAST SMEAR | Mini-BAL | respiratory_afb |
|  | ACID FAST SMEAR | SPUTUM | respiratory_afb |
|  | ACID FAST SMEAR | TRACHEAL ASPIRATE | respiratory_afb |
|  | GEN-PROBE AMPLIFIED M. TUBERCULOSIS DIRECT TEST (MTD) | BRONCHOALVEOLAR LAVAGE | respiratory_afb |
|  | GEN-PROBE AMPLIFIED M. TUBERCULOSIS DIRECT TEST (MTD) | SPUTUM | respiratory_afb |
|  | MODIFIED ACID-FAST STAIN FOR NOCARDIA | BRONCHOALVEOLAR LAVAGE | respiratory_afb |
|  | MODIFIED ACID-FAST STAIN FOR NOCARDIA | SPUTUM | respiratory_afb |
|  | MTB Direct Amplification | BRONCHOALVEOLAR LAVAGE | respiratory_afb |
|  | MTB Direct Amplification | SPUTUM | respiratory_afb |
|  | NOCARDIA CULTURE | BRONCHOALVEOLAR LAVAGE | respiratory_afb |
|  | NOCARDIA CULTURE | Mini-BAL | respiratory_afb |
|  | NOCARDIA CULTURE | SPUTUM | respiratory_afb |
| 9 | CYTOMEGALOVIRUS EARLY ANTIGEN TEST (SHELL VIAL METHOD) | BRONCHIAL WASHINGS | respiratory_viral |
|  | CYTOMEGALOVIRUS EARLY ANTIGEN TEST (SHELL VIAL METHOD) | BRONCHOALVEOLAR LAVAGE | respiratory_viral |
|  | CYTOMEGALOVIRUS EARLY ANTIGEN TEST (SHELL VIAL METHOD) | Rapid Respiratory Viral Screen & Culture | respiratory_viral |
|  | DIRECT INFLUENZA A ANTIGEN TEST | ASPIRATE | respiratory_viral |
|  | DIRECT INFLUENZA A ANTIGEN TEST | Influenza A/B by DFA | respiratory_viral |
|  | DIRECT INFLUENZA A ANTIGEN TEST | RAPID RESPIRATORY VIRAL ANTIGEN TEST | respiratory_viral |
|  | DIRECT INFLUENZA A ANTIGEN TEST | SWAB | respiratory_viral |
|  | DIRECT INFLUENZA B ANTIGEN TEST | ASPIRATE | respiratory_viral |
|  | DIRECT INFLUENZA B ANTIGEN TEST | Influenza A/B by DFA | respiratory_viral |
|  | DIRECT INFLUENZA B ANTIGEN TEST | RAPID RESPIRATORY VIRAL ANTIGEN TEST | respiratory_viral |
|  | DIRECT INFLUENZA B ANTIGEN TEST | SWAB | respiratory_viral |
|  | DIRECT RSV ANTIGEN TEST | Rapid Respiratory Viral Screen & Culture | respiratory_viral |
|  | Respiratory Viral Antigen Screen | ASPIRATE | respiratory_viral |
|  | Respiratory Viral Antigen Screen | BRONCHOALVEOLAR LAVAGE | respiratory_viral |
|  | Respiratory Viral Antigen Screen | Influenza A/B by DFA | respiratory_viral |
|  | Respiratory Viral Antigen Screen | Rapid Respiratory Viral Screen & Culture | respiratory_viral |
|  | Respiratory Viral Antigen Screen | SWAB | respiratory_viral |
|  | Respiratory Viral Culture | ASPIRATE | respiratory_viral |
|  | Respiratory Viral Culture | BRONCHOALVEOLAR LAVAGE | respiratory_viral |
|  | Respiratory Viral Culture | Influenza A/B by DFA | respiratory_viral |
|  | Respiratory Viral Culture | Rapid Respiratory Viral Screen & Culture | respiratory_viral |
|  | Respiratory Viral Culture | SWAB | respiratory_viral |
|  | Respiratory Virus Identification | Influenza A/B by DFA | respiratory_viral |
|  | Respiratory Virus Identification | Rapid Respiratory Viral Screen & Culture | respiratory_viral |
|  | VIRAL CULTURE | BRONCHIAL WASHINGS | respiratory_viral |
|  | VIRAL CULTURE | Influenza A/B by DFA | respiratory_viral |
|  | VIRAL CULTURE | RAPID RESPIRATORY VIRAL ANTIGEN TEST | respiratory_viral |
|  | VIRAL CULTURE | Rapid Respiratory Viral Screen & Culture | respiratory_viral |
|  | VIRAL CULTURE: R/O CYTOMEGALOVIRUS | BRONCHIAL WASHINGS | respiratory_viral |
|  | VIRAL CULTURE: R/O CYTOMEGALOVIRUS | Mini-BAL | respiratory_viral |
|  | VIRAL CULTURE: R/O CYTOMEGALOVIRUS | Rapid Respiratory Viral Screen & Culture | respiratory_viral |
|  | VIRAL CULTURE: R/O CYTOMEGALOVIRUS | SPUTUM | respiratory_viral |
|  | VIRAL CULTURE: R/O HERPES SIMPLEX VIRUS | BRONCHOALVEOLAR LAVAGE | respiratory_viral |
| 10 | IMMUNOFLUORESCENT TEST FOR PNEUMOCYSTIS CARINII | BRONCHOALVEOLAR LAVAGE | respiratory_pcp_legionella |
|  | IMMUNOFLUORESCENT TEST FOR PNEUMOCYSTIS CARINII | SPUTUM | respiratory_pcp_legionella |
|  | Immunofluorescent test for Pneumocystis jirovecii (carinii) | BRONCHIAL WASHINGS | respiratory_pcp_legionella |
|  | Immunofluorescent test for Pneumocystis jirovecii (carinii) | BRONCHOALVEOLAR LAVAGE | respiratory_pcp_legionella |
|  | Immunofluorescent test for Pneumocystis jirovecii (carinii) | Mini-BAL | respiratory_pcp_legionella |
|  | Immunofluorescent test for Pneumocystis jirovecii (carinii) | SPUTUM | respiratory_pcp_legionella |
|  | Immunofluorescent test for Pneumocystis jirovecii (carinii) | TISSUE | respiratory_pcp_legionella |
|  | LEGIONELLA CULTURE | BRONCHIAL WASHINGS | respiratory_pcp_legionella |
|  | LEGIONELLA CULTURE | BRONCHOALVEOLAR LAVAGE | respiratory_pcp_legionella |
|  | LEGIONELLA CULTURE | Mini-BAL | respiratory_pcp_legionella |
|  | LEGIONELLA CULTURE | PLEURAL FLUID | respiratory_pcp_legionella |
|  | LEGIONELLA CULTURE | SPUTUM | respiratory_pcp_legionella |
|  | LEGIONELLA CULTURE | SWAB | respiratory_pcp_legionella |
|  | LEGIONELLA CULTURE | TISSUE | respiratory_pcp_legionella |
|  | NOCARDIA CULTURE | BRONCHIAL WASHINGS | respiratory_pcp_legionella |
|  | NOCARDIA CULTURE | TRACHEAL ASPIRATE | respiratory_pcp_legionella |
| 11 | GRAM STAIN | BRONCHIAL BRUSH | gram_stain_respiratory |
|  | GRAM STAIN | BRONCHIAL WASHINGS | gram_stain_respiratory |
|  | GRAM STAIN | BRONCHOALVEOLAR LAVAGE | gram_stain_respiratory |
|  | GRAM STAIN | Mini-BAL | gram_stain_respiratory |
|  | GRAM STAIN | SPUTUM | gram_stain_respiratory |
|  | GRAM STAIN | TRACHEAL ASPIRATE | gram_stain_respiratory |
| 12 | GRAM STAIN | ABSCESS | gram_stain_wound_tissue |
|  | GRAM STAIN | ASPIRATE | gram_stain_wound_tissue |
|  | GRAM STAIN | BILE | gram_stain_wound_tissue |
|  | GRAM STAIN | BIOPSY | gram_stain_wound_tissue |
|  | GRAM STAIN | DIALYSIS FLUID | gram_stain_wound_tissue |
|  | GRAM STAIN | EAR | gram_stain_wound_tissue |
|  | GRAM STAIN | FLUID WOUND | gram_stain_wound_tissue |
|  | GRAM STAIN | FLUID,OTHER | gram_stain_wound_tissue |
|  | GRAM STAIN | FOOT CULTURE | gram_stain_wound_tissue |
|  | GRAM STAIN | STOOL | gram_stain_wound_tissue |
|  | GRAM STAIN | SWAB | gram_stain_wound_tissue |
|  | GRAM STAIN | TISSUE | gram_stain_wound_tissue |
|  | GRAM STAIN | URINE | gram_stain_wound_tissue |
| 13 | GRAM STAIN | CSF;SPINAL FLUID | gram_stain_csf |
| 14 | ANAEROBIC CULTURE | ABSCESS | wound_culture |
|  | ANAEROBIC CULTURE | ASPIRATE | wound_culture |
|  | ANAEROBIC CULTURE | BIOPSY | wound_culture |
|  | ANAEROBIC CULTURE | FLUID WOUND | wound_culture |
|  | ANAEROBIC CULTURE | FLUID,OTHER | wound_culture |
|  | ANAEROBIC CULTURE | FOOT CULTURE | wound_culture |
|  | ANAEROBIC CULTURE | POSTMORTEM CULTURE | wound_culture |
|  | ANAEROBIC CULTURE | SWAB | wound_culture |
|  | ANAEROBIC CULTURE | TISSUE | wound_culture |
|  | FLUID CULTURE | SWAB | wound_culture |
|  | FLUID CULTURE | TISSUE | wound_culture |
|  | ISOLATE FOR MIC | TISSUE | wound_culture |
|  | MODIFIED ACID-FAST STAIN FOR NOCARDIA | TISSUE | wound_culture |
|  | NOCARDIA CULTURE | ABSCESS | wound_culture |
|  | NOCARDIA CULTURE | SWAB | wound_culture |
|  | NOCARDIA CULTURE | TISSUE | wound_culture |
|  | RESPIRATORY CULTURE | EAR | wound_culture |
|  | RESPIRATORY CULTURE | EYE | wound_culture |
|  | RESPIRATORY CULTURE | FLUID,OTHER | wound_culture |
|  | RESPIRATORY CULTURE | SWAB | wound_culture |
|  | RESPIRATORY CULTURE | Staph aureus swab | wound_culture |
|  | RESPIRATORY CULTURE | THROAT CULTURE | wound_culture |
|  | RESPIRATORY CULTURE | THROAT FOR STREP | wound_culture |
|  | TISSUE | ABSCESS | wound_culture |
|  | TISSUE | BIOPSY | wound_culture |
|  | TISSUE | TISSUE | wound_culture |
|  | TISSUE CULTURE-TISSUE | TISSUE | wound_culture |
|  | WOUND CULTURE | ABSCESS | wound_culture |
|  | WOUND CULTURE | ASPIRATE | wound_culture |
|  | WOUND CULTURE | CORNEAL EYE SCRAPINGS | wound_culture |
|  | WOUND CULTURE | FLUID WOUND | wound_culture |
|  | WOUND CULTURE | FLUID,OTHER | wound_culture |
|  | WOUND CULTURE | FOOT CULTURE | wound_culture |
|  | WOUND CULTURE | SWAB | wound_culture |
|  | WOUND CULTURE | TISSUE | wound_culture |
| 15 | ACID FAST CULTURE | FOREIGN BODY | hardware_and_lines_culture |
|  | ACID FAST SMEAR | FOREIGN BODY | hardware_and_lines_culture |
|  | ANAEROBIC CULTURE | FOREIGN BODY | hardware_and_lines_culture |
|  | ANAEROBIC CULTURE | Foreign Body - Sonication Culture | hardware_and_lines_culture |
|  | FUNGAL CULTURE | FOREIGN BODY | hardware_and_lines_culture |
|  | GRAM STAIN | CATHETER TIP-IV | hardware_and_lines_culture |
|  | GRAM STAIN | FLUID RECEIVED IN BLOOD CULTURE BOTTLES | hardware_and_lines_culture |
|  | GRAM STAIN | FOREIGN BODY | hardware_and_lines_culture |
|  | POTASSIUM HYDROXIDE PREPARATION | FOREIGN BODY | hardware_and_lines_culture |
|  | Sonication culture, prosthetic joint | FOREIGN BODY | hardware_and_lines_culture |
|  | Sonication culture, prosthetic joint | Foreign Body - Sonication Culture | hardware_and_lines_culture |
|  | WOUND CULTURE | CATHETER TIP-IV | hardware_and_lines_culture |
|  | WOUND CULTURE | FOREIGN BODY | hardware_and_lines_culture |
| 16 | ACID FAST CULTURE | PLEURAL FLUID | pleural_culture |
|  | ACID FAST SMEAR | PLEURAL FLUID | pleural_culture |
|  | ANAEROBIC CULTURE | PLEURAL FLUID | pleural_culture |
|  | FLUID CULTURE | PLEURAL FLUID | pleural_culture |
|  | FUNGAL CULTURE | PLEURAL FLUID | pleural_culture |
|  | GRAM STAIN | PLEURAL FLUID | pleural_culture |
|  | NOCARDIA CULTURE | PLEURAL FLUID | pleural_culture |
|  | POTASSIUM HYDROXIDE PREPARATION | PLEURAL FLUID | pleural_culture |
|  | Respiratory Viral Culture | PLEURAL FLUID | pleural_culture |
|  | VIRAL CULTURE | PLEURAL FLUID | pleural_culture |
| 17 | ACID FAST CULTURE | PERITONEAL FLUID | peritoneal_culture |
|  | ACID FAST SMEAR | PERITONEAL FLUID | peritoneal_culture |
|  | ANAEROBIC CULTURE | PERITONEAL FLUID | peritoneal_culture |
|  | FLUID CULTURE | PERITONEAL FLUID | peritoneal_culture |
|  | FUNGAL CULTURE | PERITONEAL FLUID | peritoneal_culture |
|  | Fluid Culture in Bottles | PERITONEAL FLUID | peritoneal_culture |
|  | GRAM STAIN | PERITONEAL FLUID | peritoneal_culture |
|  | POTASSIUM HYDROXIDE PREPARATION | PERITONEAL FLUID | peritoneal_culture |
| 18 | ACID FAST CULTURE | JOINT FLUID | joint_fluid_culture |
|  | ACID FAST CULTURE | PROSTHETIC JOINT FLUID | joint_fluid_culture |
|  | ACID FAST SMEAR | JOINT FLUID | joint_fluid_culture |
|  | ACID FAST SMEAR | PROSTHETIC JOINT FLUID | joint_fluid_culture |
|  | ANAEROBIC CULTURE | JOINT FLUID | joint_fluid_culture |
|  | Anaerobic culture, Prosthetic Joint Fluid | PROSTHETIC JOINT FLUID | joint_fluid_culture |
|  | FLUID CULTURE | JOINT FLUID | joint_fluid_culture |
|  | FLUID CULTURE | PROSTHETIC JOINT FLUID | joint_fluid_culture |
|  | FUNGAL CULTURE | JOINT FLUID | joint_fluid_culture |
|  | FUNGAL CULTURE | PROSTHETIC JOINT FLUID | joint_fluid_culture |
|  | GRAM STAIN | JOINT FLUID | joint_fluid_culture |
|  | GRAM STAIN | PROSTHETIC JOINT FLUID | joint_fluid_culture |
|  | POTASSIUM HYDROXIDE PREPARATION | JOINT FLUID | joint_fluid_culture |
|  | Sonication culture, prosthetic joint | PROSTHETIC JOINT FLUID | joint_fluid_culture |
| 19 | ANAEROBIC CULTURE | BILE | fluid_culture |
|  | ANAEROBIC CULTURE | DIALYSIS FLUID | fluid_culture |
|  | ANAEROBIC CULTURE | FLUID RECEIVED IN BLOOD CULTURE BOTTLES | fluid_culture |
|  | FLUID CULTURE | ABSCESS | fluid_culture |
|  | FLUID CULTURE | ASPIRATE | fluid_culture |
|  | FLUID CULTURE | BILE | fluid_culture |
|  | FLUID CULTURE | DIALYSIS FLUID | fluid_culture |
|  | FLUID CULTURE | FLUID RECEIVED IN BLOOD CULTURE BOTTLES | fluid_culture |
|  | FLUID CULTURE | FLUID WOUND | fluid_culture |
|  | FLUID CULTURE | FLUID,OTHER | fluid_culture |
|  | ISOLATE FOR MIC | FLUID,OTHER | fluid_culture |
|  | WOUND CULTURE | BILE | fluid_culture |
| 20 | ACID FAST CULTURE | BONE MARROW | bone_marrow_culture |
|  | ACID FAST SMEAR | BONE MARROW | bone_marrow_culture |
|  | ANAEROBIC CULTURE | BONE MARROW | bone_marrow_culture |
|  | FLUID CULTURE | BONE MARROW | bone_marrow_culture |
|  | FUNGAL CULTURE | BONE MARROW | bone_marrow_culture |
|  | GRAM STAIN | BONE MARROW | bone_marrow_culture |
|  | POTASSIUM HYDROXIDE PREPARATION | BONE MARROW | bone_marrow_culture |
| 21 | ACID FAST CULTURE | CSF;SPINAL FLUID | csf_culture |
|  | ACID FAST SMEAR | CSF;SPINAL FLUID | csf_culture |
|  | ANAEROBIC CULTURE | CSF;SPINAL FLUID | csf_culture |
|  | CRYPTOCOCCAL ANTIGEN | CSF;SPINAL FLUID | csf_culture |
|  | CRYPTOCOCCAL ANTIGEN | SEROLOGY/BLOOD | csf_culture |
|  | Enterovirus Culture | CSF;SPINAL FLUID | csf_culture |
|  | FLUID CULTURE | CSF;SPINAL FLUID | csf_culture |
|  | FUNGAL CULTURE | CSF;SPINAL FLUID | csf_culture |
|  | HIV-1 Viral Load/Ultrasensitive | CSF;SPINAL FLUID | csf_culture |
|  | POTASSIUM HYDROXIDE PREPARATION | CSF;SPINAL FLUID | csf_culture |
|  | QUANTITATIVE CRYPTOCOCCAL ANTIGEN | CSF;SPINAL FLUID | csf_culture |
|  | QUANTITATIVE CRYPTOCOCCAL ANTIGEN | SEROLOGY/BLOOD | csf_culture |
|  | VARICELLA-ZOSTER CULTURE | CSF;SPINAL FLUID | csf_culture |
|  | VIRAL CULTURE | CSF;SPINAL FLUID | csf_culture |
|  | VIRAL CULTURE: R/O HERPES SIMPLEX VIRUS | CSF;SPINAL FLUID | csf_culture |
| 22 | ACID FAST CULTURE | ABSCESS | fungal_tissue_wound |
|  | ACID FAST CULTURE | ASPIRATE | fungal_tissue_wound |
|  | ACID FAST CULTURE | CORNEAL EYE SCRAPINGS | fungal_tissue_wound |
|  | ACID FAST CULTURE | FOOT CULTURE | fungal_tissue_wound |
|  | ACID FAST CULTURE | SWAB | fungal_tissue_wound |
|  | ACID FAST CULTURE | TISSUE | fungal_tissue_wound |
|  | ACID FAST SMEAR | ABSCESS | fungal_tissue_wound |
|  | ACID FAST SMEAR | ASPIRATE | fungal_tissue_wound |
|  | ACID FAST SMEAR | CORNEAL EYE SCRAPINGS | fungal_tissue_wound |
|  | ACID FAST SMEAR | FOOT CULTURE | fungal_tissue_wound |
|  | ACID FAST SMEAR | SWAB | fungal_tissue_wound |
|  | ACID FAST SMEAR | TISSUE | fungal_tissue_wound |
|  | ED Gram Stain for Yeast | SWAB | fungal_tissue_wound |
|  | FUNGAL CULTURE | ABSCESS | fungal_tissue_wound |
|  | FUNGAL CULTURE | ASPIRATE | fungal_tissue_wound |
|  | FUNGAL CULTURE | BIOPSY | fungal_tissue_wound |
|  | FUNGAL CULTURE | SKIN SCRAPINGS | fungal_tissue_wound |
|  | FUNGAL CULTURE | SWAB | fungal_tissue_wound |
|  | FUNGAL CULTURE | TISSUE | fungal_tissue_wound |
|  | FUNGAL CULTURE (HAIR/SKIN/NAILS) | NAIL SCRAPINGS | fungal_tissue_wound |
|  | FUNGAL CULTURE (HAIR/SKIN/NAILS) | SKIN SCRAPINGS | fungal_tissue_wound |
|  | Malassezia furfur Culture | SWAB | fungal_tissue_wound |
|  | POTASSIUM HYDROXIDE PREPARATION | ABSCESS | fungal_tissue_wound |
|  | POTASSIUM HYDROXIDE PREPARATION | ASPIRATE | fungal_tissue_wound |
|  | POTASSIUM HYDROXIDE PREPARATION | BIOPSY | fungal_tissue_wound |
|  | POTASSIUM HYDROXIDE PREPARATION | SWAB | fungal_tissue_wound |
|  | POTASSIUM HYDROXIDE PREPARATION | TISSUE | fungal_tissue_wound |
|  | POTASSIUM HYDROXIDE PREPARATION (HAIR/SKIN/NAILS) | NAIL SCRAPINGS | fungal_tissue_wound |
|  | POTASSIUM HYDROXIDE PREPARATION (HAIR/SKIN/NAILS) | SKIN SCRAPINGS | fungal_tissue_wound |
|  | POTASSIUM HYDROXIDE PREPARATION (HAIR/SKIN/NAILS) | SWAB | fungal_tissue_wound |
|  | POTASSIUM HYDROXIDE PREPARATION (HAIR/SKIN/NAILS) | TISSUE | fungal_tissue_wound |
| 23 | FUNGAL CULTURE | BRONCHIAL BRUSH | fungal_respiratory |
|  | FUNGAL CULTURE | BRONCHIAL BRUSH - PROTECTED | fungal_respiratory |
|  | FUNGAL CULTURE | BRONCHIAL WASHINGS | fungal_respiratory |
|  | FUNGAL CULTURE | BRONCHOALVEOLAR LAVAGE | fungal_respiratory |
|  | FUNGAL CULTURE | Mini-BAL | fungal_respiratory |
|  | FUNGAL CULTURE | SPUTUM | fungal_respiratory |
|  | FUNGAL CULTURE | TRACHEAL ASPIRATE | fungal_respiratory |
|  | POTASSIUM HYDROXIDE PREPARATION | BRONCHIAL BRUSH | fungal_respiratory |
|  | POTASSIUM HYDROXIDE PREPARATION | BRONCHIAL WASHINGS | fungal_respiratory |
|  | POTASSIUM HYDROXIDE PREPARATION | BRONCHOALVEOLAR LAVAGE | fungal_respiratory |
|  | POTASSIUM HYDROXIDE PREPARATION | Mini-BAL | fungal_respiratory |
|  | POTASSIUM HYDROXIDE PREPARATION | SPUTUM | fungal_respiratory |
|  | POTASSIUM HYDROXIDE PREPARATION | TRACHEAL ASPIRATE | fungal_respiratory |
| 24 | ACID FAST CULTURE | DIALYSIS FLUID | fungal_fluid |
|  | ACID FAST CULTURE | FLUID,OTHER | fungal_fluid |
|  | ACID FAST CULTURE | URINE | fungal_fluid |
|  | ACID FAST SMEAR | DIALYSIS FLUID | fungal_fluid |
|  | ACID FAST SMEAR | FLUID,OTHER | fungal_fluid |
|  | FUNGAL CULTURE | BILE | fungal_fluid |
|  | FUNGAL CULTURE | DIALYSIS FLUID | fungal_fluid |
|  | FUNGAL CULTURE | FLUID,OTHER | fungal_fluid |
|  | POTASSIUM HYDROXIDE PREPARATION | BILE | fungal_fluid |
|  | POTASSIUM HYDROXIDE PREPARATION | FLUID,OTHER | fungal_fluid |
| 25 | MRSA SCREEN | MRSA SCREEN | mrsa_staph_screen |
|  | MRSA SCREEN | SWAB | mrsa_staph_screen |
|  | Staph aureus Preop PCR | MRSA SCREEN | mrsa_staph_screen |
|  | Staph aureus Preop PCR | Staph aureus swab | mrsa_staph_screen |
|  | Staph aureus Screen | MRSA SCREEN | mrsa_staph_screen |
|  | Staph aureus Screen | SWAB | mrsa_staph_screen |
|  | Staph aureus Screen | Staph aureus swab | mrsa_staph_screen |
| 26 | CRE/ESBL/AMP-C Screening | SWAB | resistance_screen |
|  | Carbapenemase Resistant Enterobacteriaceae Screen | CRE Screen | resistance_screen |
|  | Carbapenemase Resistant Enterobacteriaceae Screen | SWAB | resistance_screen |
|  | Cipro Resistant Screen | Cipro Resistant Screen | resistance_screen |
|  | Cipro Resistant Screen | STOOL | resistance_screen |
|  | ED Gram Stain for Yeast | Swab R/O Yeast Screen | resistance_screen |
|  | R/O VANCOMYCIN RESISTANT ENTEROCOCCUS | STOOL | resistance_screen |
|  | R/O VANCOMYCIN RESISTANT ENTEROCOCCUS | SWAB | resistance_screen |
|  | SWAB- R/O YEAST | Swab R/O Yeast Screen | resistance_screen |
|  | Swab - R/O Yeast - IC | Infection Control Yeast | resistance_screen |
| 27 | C. difficile PCR | STOOL | cdiff |
|  | C. difficile PCR | STOOL (RECEIVED IN TRANSPORT SYSTEM) | cdiff |
|  | C. difficile Toxin antigen assay | STOOL | cdiff |
|  | C. difficile Toxin antigen assay | STOOL (RECEIVED IN TRANSPORT SYSTEM) | cdiff |
|  | CLOSTRIDIUM DIFFICILE TOXIN A & B TEST | STOOL | cdiff |
|  | CLOSTRIDIUM DIFFICILE TOXIN A & B TEST | STOOL (RECEIVED IN TRANSPORT SYSTEM) | cdiff |
|  | CLOSTRIDIUM DIFFICILE TOXIN ASSAY | STOOL | cdiff |
|  | CLOSTRIDIUM DIFFICILE TOXIN ASSAY | STOOL (RECEIVED IN TRANSPORT SYSTEM) | cdiff |
| 28 | CAMPYLOBACTER CULTURE | FECAL SWAB | stool_bacterial |
|  | CAMPYLOBACTER CULTURE | STOOL | stool_bacterial |
|  | CAMPYLOBACTER CULTURE | STOOL (RECEIVED IN TRANSPORT SYSTEM) | stool_bacterial |
|  | FECAL CULTURE | FECAL SWAB | stool_bacterial |
|  | FECAL CULTURE | STOOL | stool_bacterial |
|  | FECAL CULTURE | STOOL (RECEIVED IN TRANSPORT SYSTEM) | stool_bacterial |
|  | FECAL CULTURE - R/O E.COLI 0157:H7 | STOOL | stool_bacterial |
|  | FECAL CULTURE - R/O E.COLI 0157:H7 | STOOL (RECEIVED IN TRANSPORT SYSTEM) | stool_bacterial |
|  | FECAL CULTURE - R/O VIBRIO | STOOL | stool_bacterial |
|  | FECAL CULTURE - R/O VIBRIO | STOOL (RECEIVED IN TRANSPORT SYSTEM) | stool_bacterial |
|  | FECAL CULTURE - R/O YERSINIA | STOOL | stool_bacterial |
|  | FECAL CULTURE - R/O YERSINIA | STOOL (RECEIVED IN TRANSPORT SYSTEM) | stool_bacterial |
|  | SHIGA TOXIN (EHEC) | STOOL | stool_bacterial |
|  | STOOL SMEAR FOR POLYMORPHONUCLEAR LEUKOCYTES | STOOL | stool_bacterial |
| 29 | ACID FAST CULTURE | STOOL | stool_parasitology |
|  | Acid Fast Stain for Cryptosporidium | ASPIRATE | stool_parasitology |
|  | Acid Fast Stain for Cryptosporidium | STOOL | stool_parasitology |
|  | CYCLOSPORA STAIN | STOOL | stool_parasitology |
|  | CYCLOSPORA STAIN | STOOL (RECEIVED IN TRANSPORT SYSTEM) | stool_parasitology |
|  | Concentration and Stain for Giardia | STOOL | stool_parasitology |
|  | Cryptosporidium/Giardia (DFA) | STOOL | stool_parasitology |
|  | Cryptosporidium/Giardia (DFA) | STOOL (RECEIVED IN TRANSPORT SYSTEM) | stool_parasitology |
|  | MICROSPORIDIA STAIN | STOOL | stool_parasitology |
|  | MICROSPORIDIA STAIN | STOOL (RECEIVED IN TRANSPORT SYSTEM) | stool_parasitology |
|  | O&P MACROSCOPIC EXAM - ARTHROPOD | ARTHROPOD | stool_parasitology |
|  | O&P MACROSCOPIC EXAM - ARTHROPOD | STOOL | stool_parasitology |
|  | O&P MACROSCOPIC EXAM - WORM | STOOL | stool_parasitology |
|  | O&P MACROSCOPIC EXAM - WORM | WORM | stool_parasitology |
|  | OVA + PARASITES | BRONCHOALVEOLAR LAVAGE | stool_parasitology |
|  | OVA + PARASITES | SPUTUM | stool_parasitology |
|  | OVA + PARASITES | STOOL | stool_parasitology |
|  | OVA + PARASITES | STOOL (RECEIVED IN TRANSPORT SYSTEM) | stool_parasitology |
|  | OVA + PARASITES | URINE | stool_parasitology |
|  | SCOTCH TAPE PREP/PADDLE | SCOTCH TAPE PREP/PADDLE | stool_parasitology |
|  | VIRAL CULTURE | STOOL | stool_parasitology |
| 30 | CMV IgG ANTIBODY | Blood (CMV AB) | herpesvirus_serology |
|  | CMV IgG ANTIBODY | SEROLOGY/BLOOD | herpesvirus_serology |
|  | CMV IgM ANTIBODY | Blood (CMV AB) | herpesvirus_serology |
|  | CMV IgM ANTIBODY | SEROLOGY/BLOOD | herpesvirus_serology |
|  | CMV Viral Load | IMMUNOLOGY | herpesvirus_serology |
|  | CMV Viral Load | Immunology (CMV) | herpesvirus_serology |
|  | EPSTEIN-BARR VIRUS EBNA IgG AB | Blood (EBV) | herpesvirus_serology |
|  | EPSTEIN-BARR VIRUS EBNA IgG AB | SEROLOGY/BLOOD | herpesvirus_serology |
|  | EPSTEIN-BARR VIRUS VCA-IgG AB | Blood (EBV) | herpesvirus_serology |
|  | EPSTEIN-BARR VIRUS VCA-IgG AB | SEROLOGY/BLOOD | herpesvirus_serology |
|  | EPSTEIN-BARR VIRUS VCA-IgM AB | Blood (EBV) | herpesvirus_serology |
|  | EPSTEIN-BARR VIRUS VCA-IgM AB | SEROLOGY/BLOOD | herpesvirus_serology |
|  | MONOSPOT | SEROLOGY/BLOOD | herpesvirus_serology |
|  | VARICELLA-ZOSTER IgG SEROLOGY | SEROLOGY/BLOOD | herpesvirus_serology |
| 31 | HBV Viral Load | IMMUNOLOGY | hepatitis_hiv |
|  | HCV GENOTYPE | IMMUNOLOGY | hepatitis_hiv |
|  | HCV VIRAL LOAD | IMMUNOLOGY | hepatitis_hiv |
|  | HIV-1 Viral Load/Ultrasensitive | IMMUNOLOGY | hepatitis_hiv |
|  | Reflex HCV Qual PCR | IMMUNOLOGY | hepatitis_hiv |
| 32 | QUANTITATIVE RPR | SEROLOGY/BLOOD | syphilis_serology |
|  | RAPID PLASMA REAGIN TEST | SEROLOGY/BLOOD | syphilis_serology |
|  | RPR w/check for Prozone | SEROLOGY/BLOOD | syphilis_serology |
|  | TREPONEMAL ANTIBODY TEST | SEROLOGY/BLOOD | syphilis_serology |
| 33 | ASO Screen | SEROLOGY/BLOOD | misc_serology |
|  | ASO TITER | SEROLOGY/BLOOD | misc_serology |
|  | HELICOBACTER PYLORI ANTIBODY TEST | SEROLOGY/BLOOD | misc_serology |
|  | LYME SEROLOGY | SEROLOGY/BLOOD | misc_serology |
|  | Lyme IgG | Blood (LYME) | misc_serology |
|  | Lyme IgM | Blood (LYME) | misc_serology |
|  | MUMPS IgG ANTIBODY | SEROLOGY/BLOOD | misc_serology |
|  | Malaria Antigen Test | Blood (Malaria) | misc_serology |
|  | RUBELLA IgG SEROLOGY | SEROLOGY/BLOOD | misc_serology |
|  | RUBEOLA ANTIBODY, IgG | SEROLOGY/BLOOD | misc_serology |
|  | Rubella IgG/IgM Antibody | SEROLOGY/BLOOD | misc_serology |
|  | TOXOPLASMA IgG ANTIBODY | Blood (Toxo) | misc_serology |
|  | TOXOPLASMA IgG ANTIBODY | SEROLOGY/BLOOD | misc_serology |
|  | TOXOPLASMA IgM ANTIBODY | Blood (Toxo) | misc_serology |
| 34 | CYTOMEGALOVIRUS EARLY ANTIGEN TEST (SHELL VIAL METHOD) | BIOPSY | herpesvirus_culture_antigen |
|  | CYTOMEGALOVIRUS EARLY ANTIGEN TEST (SHELL VIAL METHOD) | TISSUE | herpesvirus_culture_antigen |
|  | CYTOMEGALOVIRUS EARLY ANTIGEN TEST (SHELL VIAL METHOD) | VIRAL CULTURE: R/O CYTOMEGALOVIRUS | herpesvirus_culture_antigen |
|  | DIRECT ANTIGEN TEST FOR VARICELLA-ZOSTER VIRUS | DIRECT ANTIGEN TEST FOR VARICELLA-ZOSTER VIRUS | herpesvirus_culture_antigen |
|  | DIRECT ANTIGEN TEST FOR VARICELLA-ZOSTER VIRUS | Direct Antigen Test for Herpes Simplex Virus Types 1 & 2 | herpesvirus_culture_antigen |
|  | DIRECT ANTIGEN TEST FOR VARICELLA-ZOSTER VIRUS | SWAB | herpesvirus_culture_antigen |
|  | Direct Antigen Test for Herpes Simplex Virus Types 1 & 2 | DIRECT ANTIGEN TEST FOR VARICELLA-ZOSTER VIRUS | herpesvirus_culture_antigen |
|  | Direct Antigen Test for Herpes Simplex Virus Types 1 & 2 | Direct Antigen Test for Herpes Simplex Virus Types 1 & 2 | herpesvirus_culture_antigen |
|  | Direct Antigen Test for Herpes Simplex Virus Types 1 & 2 | SKIN SCRAPINGS | herpesvirus_culture_antigen |
|  | Direct Antigen Test for Herpes Simplex Virus Types 1 & 2 | SWAB | herpesvirus_culture_antigen |
|  | VARICELLA-ZOSTER CULTURE | Direct Antigen Test for Herpes Simplex Virus Types 1 & 2 | herpesvirus_culture_antigen |
|  | VARICELLA-ZOSTER CULTURE | SKIN SCRAPINGS | herpesvirus_culture_antigen |
|  | VARICELLA-ZOSTER CULTURE | SWAB | herpesvirus_culture_antigen |
|  | VARICELLA-ZOSTER CULTURE | THROAT CULTURE | herpesvirus_culture_antigen |
|  | VARICELLA-ZOSTER CULTURE | TISSUE | herpesvirus_culture_antigen |
|  | VARICELLA-ZOSTER CULTURE | VARICELLA-ZOSTER CULTURE | herpesvirus_culture_antigen |
|  | VARICELLA-ZOSTER CULTURE | VIRAL CULTURE:R/O HERPES SIMPLEX VIRUS | herpesvirus_culture_antigen |
|  | VIRAL CULTURE | ASPIRATE | herpesvirus_culture_antigen |
|  | VIRAL CULTURE | BIOPSY | herpesvirus_culture_antigen |
|  | VIRAL CULTURE | FLUID,OTHER | herpesvirus_culture_antigen |
|  | VIRAL CULTURE | STOOL (RECEIVED IN TRANSPORT SYSTEM) | herpesvirus_culture_antigen |
|  | VIRAL CULTURE | THROAT CULTURE | herpesvirus_culture_antigen |
|  | VIRAL CULTURE: R/O CYTOMEGALOVIRUS | BIOPSY | herpesvirus_culture_antigen |
|  | VIRAL CULTURE: R/O CYTOMEGALOVIRUS | SWAB | herpesvirus_culture_antigen |
|  | VIRAL CULTURE: R/O CYTOMEGALOVIRUS | TISSUE | herpesvirus_culture_antigen |
|  | VIRAL CULTURE: R/O CYTOMEGALOVIRUS | VIRAL CULTURE: R/O CYTOMEGALOVIRUS | herpesvirus_culture_antigen |
|  | VIRAL CULTURE: R/O HERPES SIMPLEX VIRUS | BIOPSY | herpesvirus_culture_antigen |
|  | VIRAL CULTURE: R/O HERPES SIMPLEX VIRUS | Direct Antigen Test for Herpes Simplex Virus Types 1 & 2 | herpesvirus_culture_antigen |
|  | VIRAL CULTURE: R/O HERPES SIMPLEX VIRUS | SKIN SCRAPINGS | herpesvirus_culture_antigen |
|  | VIRAL CULTURE: R/O HERPES SIMPLEX VIRUS | STOOL | herpesvirus_culture_antigen |
|  | VIRAL CULTURE: R/O HERPES SIMPLEX VIRUS | SWAB | herpesvirus_culture_antigen |
|  | VIRAL CULTURE: R/O HERPES SIMPLEX VIRUS | THROAT CULTURE | herpesvirus_culture_antigen |
|  | VIRAL CULTURE: R/O HERPES SIMPLEX VIRUS | THROAT FOR STREP | herpesvirus_culture_antigen |
|  | VIRAL CULTURE: R/O HERPES SIMPLEX VIRUS | TISSUE | herpesvirus_culture_antigen |
|  | VIRAL CULTURE: R/O HERPES SIMPLEX VIRUS | VIRAL CULTURE: R/O CYTOMEGALOVIRUS | herpesvirus_culture_antigen |
| 35 | CHLAMYDIA CULTURE | SWAB | gc_chlamydia_sti |
|  | Chlamydia trachomatis, Nucleic Acid Probe, with Amplification | SWAB | gc_chlamydia_sti |
|  | Chlamydia trachomatis, Nucleic Acid Probe, with Amplification | URINE | gc_chlamydia_sti |
|  | GENITAL CULTURE | SWAB | gc_chlamydia_sti |
|  | GENITAL CULTURE FOR TOXIC SHOCK | SWAB | gc_chlamydia_sti |
|  | NEISSERIA GONORRHOEAE (GC), NUCLEIC ACID PROBE, WITH AMPLIFICATION | SWAB | gc_chlamydia_sti |
|  | NEISSERIA GONORRHOEAE (GC), NUCLEIC ACID PROBE, WITH AMPLIFICATION | URINE | gc_chlamydia_sti |
|  | R/O GC Only | RECTAL - R/O GC | gc_chlamydia_sti |
|  | R/O GC Only | SWAB | gc_chlamydia_sti |
|  | R/O GC Only | SWAB, R/O GC | gc_chlamydia_sti |
|  | R/O GC Only | THROAT | gc_chlamydia_sti |
|  | R/O GC Only | THROAT CULTURE | gc_chlamydia_sti |
|  | R/O GC Only | THROAT FOR STREP | gc_chlamydia_sti |
| 36 | R/O GROUP B BETA STREP | ANORECTAL/VAGINAL | vaginal_genital_flora |
|  | R/O GROUP B BETA STREP | SWAB | vaginal_genital_flora |
|  | R/O Group B Strep - Penicillin Allergy | ANORECTAL/VAGINAL | vaginal_genital_flora |
|  | R/O Group B Strep - Penicillin Allergy | SWAB | vaginal_genital_flora |
|  | SMEAR FOR BACTERIAL VAGINOSIS | ANORECTAL/VAGINAL | vaginal_genital_flora |
|  | SMEAR FOR BACTERIAL VAGINOSIS | SWAB | vaginal_genital_flora |
|  | SMEAR FOR BACTERIAL VAGINOSIS | Swab | vaginal_genital_flora |
|  | TRICHOMONAS SALINE PREP | SWAB | vaginal_genital_flora |
|  | TRICHOMONAS SALINE PREP | URINE | vaginal_genital_flora |
|  | YEAST VAGINITIS CULTURE | ANORECTAL/VAGINAL | vaginal_genital_flora |
|  | YEAST VAGINITIS CULTURE | SWAB | vaginal_genital_flora |
| 37 | GRAM STAIN- R/O THRUSH | SWAB | throat_strep |
|  | GRAM STAIN- R/O THRUSH | THROAT CULTURE | throat_strep |
|  | GRAM STAIN- R/O THRUSH | THROAT FOR STREP | throat_strep |
|  | R/O Beta Strep Group A | SWAB | throat_strep |
|  | R/O Beta Strep Group A | THROAT CULTURE | throat_strep |
|  | R/O Beta Strep Group A | THROAT FOR STREP | throat_strep |
| — | ADDITIONAL CELLS COUNTED | BONE MARROW - CYTOGENETICS | EXCLUDED |
|  | Additional Cells and Karyotype | BONE MARROW - CYTOGENETICS | EXCLUDED |
|  | Aerobic Bottle Gram Stain | BLOOD CULTURE (POST-MORTEM) | EXCLUDED |
|  | Anaerobic Bottle Gram Stain | BLOOD CULTURE (POST-MORTEM) | EXCLUDED |
|  | Blood Culture, Post Mortem | BLOOD CULTURE (POST-MORTEM) | EXCLUDED |
|  | CHROMOSOME ANALYSIS - ADDITIONAL KARYOTYPE | BONE MARROW - CYTOGENETICS | EXCLUDED |
|  | CHROMOSOME ANALYSIS-AMNIOTIC FLUID | AMNIOTIC FLUID | EXCLUDED |
|  | CHROMOSOME ANALYSIS-BLOOD | NEOPLASTIC BLOOD | EXCLUDED |
|  | CHROMOSOME ANALYSIS-BONE MARROW | BONE MARROW - CYTOGENETICS | EXCLUDED |
|  | CHROMOSOME ANALYSIS-CVS | CHORIONIC VILLUS SAMPLE | EXCLUDED |
|  | CHROMOSOME ANALYSIS-FLUID | FLUID,OTHER | EXCLUDED |
|  | CHROMOSOME ANALYSIS-NEOPLASTIC BLOOD | NEOPLASTIC BLOOD | EXCLUDED |
|  | CHROMOSOME ANALYSIS-TISSUE | TISSUE | EXCLUDED |
|  | CVS NEEDLE ASPIRATION EVALUATION | CHORIONIC VILLUS SAMPLE | EXCLUDED |
|  | Cryopreservation - Cells | CHORIONIC VILLUS SAMPLE | EXCLUDED |
|  | Deparaffinization, Lysis of Cells | Touch Prep/Sections | EXCLUDED |
|  | FISH ANALYSIS, 10-30 CELLS | BONE MARROW - CYTOGENETICS | EXCLUDED |
|  | FISH ANALYSIS, 3-5 CELLS | BONE MARROW - CYTOGENETICS | EXCLUDED |
|  | FOCUSED ANALYSIS | BONE MARROW - CYTOGENETICS | EXCLUDED |
|  | INTERPHASE FISH ANALYSIS, 100-300 CELLS | BONE MARROW - CYTOGENETICS | EXCLUDED |
|  | INTERPHASE FISH ANALYSIS, 25-99 CELLS | BONE MARROW - CYTOGENETICS | EXCLUDED |
|  | MOLECULAR CYTOGENETICS - DNA PROBE | BONE MARROW - CYTOGENETICS | EXCLUDED |
|  | MOLECULAR CYTOGENETICS-DNA Probe | BONE MARROW - CYTOGENETICS | EXCLUDED |
|  | POST MORTEM MYCOLOGY CULTURE | POSTMORTEM CULTURE | EXCLUDED |
|  | POST-MORTEM ACID-FAST CULTURE | POSTMORTEM CULTURE | EXCLUDED |
|  | POST-MORTEM DIRECT ACID-FAST STAIN | POSTMORTEM CULTURE | EXCLUDED |
|  | POST-MORTEM VIRAL CULTURE | POSTMORTEM CULTURE | EXCLUDED |
|  | POSTMORTEM CULTURE | POSTMORTEM CULTURE | EXCLUDED |
|  | Problem | MICRO PROBLEM PATIENT | EXCLUDED |
|  | Stool Hold Request | STOOL | EXCLUDED |
|  | TISSUE CULTURE-AMNIOTIC FLUID | AMNIOTIC FLUID | EXCLUDED |
|  | TISSUE CULTURE-CVS | CHORIONIC VILLUS SAMPLE | EXCLUDED |
|  | TISSUE CULTURE-FLUID | FLUID,OTHER | EXCLUDED |
|  | TISSUE CULTURE-LYMPHOCYTE | NEOPLASTIC BLOOD | EXCLUDED |
|  | Tissue Culture - Neoplastic Blood | NEOPLASTIC BLOOD | EXCLUDED |
|  | Tissue Culture-Bone Marrow | BONE MARROW - CYTOGENETICS | EXCLUDED |
|  | Tissue culture for additional cells | TISSUE | EXCLUDED |
|  | voided | various | EXCLUDED |