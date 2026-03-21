# Fix `mimic4_microbiology_exploration.ipynb` — Three issues in Section 6

## Issue 1 — Add remaining assignable combos to `PANELS_30`

Add these tuples to the `"combos"` lists of the named panels inside the
`PANELS_30` dict. Do not change anything else in the dict.

### `respiratory_afb`
```python
("ACID FAST CULTURE", "BRONCHIAL BRUSH"),
("ACID FAST SMEAR",   "BRONCHIAL BRUSH"),
("ACID FAST CULTURE", "BRONCHIAL BRUSH - PROTECTED"),
("ACID FAST SMEAR",   "BRONCHIAL BRUSH - PROTECTED"),
("ACID FAST CULTURE", "TRACHEAL ASPIRATE"),
("ACID FAST SMEAR",   "TRACHEAL ASPIRATE"),
```

### `respiratory_pcp_legionella`
```python
("NOCARDIA CULTURE", "Mini-BAL"),
("NOCARDIA CULTURE", "TRACHEAL ASPIRATE"),
```

### `fungal_respiratory`
```python
("FUNGAL CULTURE",                  "TRACHEAL ASPIRATE"),
("POTASSIUM HYDROXIDE PREPARATION", "TRACHEAL ASPIRATE"),
```

### `respiratory_viral`
```python
("VIRAL CULTURE: R/O CYTOMEGALOVIRUS",                     "SPUTUM"),
("CYTOMEGALOVIRUS EARLY ANTIGEN TEST (SHELL VIAL METHOD)", "SPUTUM"),
("CYTOMEGALOVIRUS EARLY ANTIGEN TEST (SHELL VIAL METHOD)", "Mini-BAL"),
```

### `wound_culture`
```python
("RESPIRATORY CULTURE", "THROAT FOR STREP"),
```

### `cdiff`
```python
("CLOSTRIDIUM DIFFICILE TOXIN A & B TEST", "STOOL (RECEIVED IN TRANSPORT SYSTEM)"),
("C. difficile PCR",                       "STOOL (RECEIVED IN TRANSPORT SYSTEM)"),
```

### `stool_bacterial`
```python
("FECAL CULTURE",       "FECAL SWAB"),
("CAMPYLOBACTER CULTURE","FECAL SWAB"),
```

### `stool_parasitology`
```python
("OVA + PARASITES", "ASPIRATE"),
("OVA + PARASITES", "FLUID,OTHER"),
```

---

## Issue 2 — Run panel assignment on ALL rows, not just hadm_id-linked rows

### Current behaviour (wrong)

In Section 1 (cell 2), the dataframe is filtered immediately after loading:
```python
micro_df = micro_df.dropna(subset=['hadm_id'])
```

This means Section 6 assigns panels only to the ~1.76M admission-linked rows,
missing the ~1.4M outpatient rows that have no `hadm_id`.

### Fix

**In cell 2 (Section 1):** Do NOT drop rows with null `hadm_id` from
`micro_df`. Instead, keep the full dataframe for panel grouping analysis,
and create a separate filtered dataframe for admission-level analyses.

Replace:
```python
# Filter out rows where hadm_id is null (events not linked to an admission)
micro_df = micro_df.dropna(subset=['hadm_id'])
print(f"After dropping null hadm_id: {micro_df.shape[0]:,} rows remain")
print("\nColumn overview:")
print(micro_df.dtypes)
```

With:
```python
# Keep full dataset for panel grouping — includes outpatient rows (null hadm_id)
print(f"Full dataset (including outpatient rows): {micro_df.shape[0]:,} rows")
print("\nColumn overview:")
print(micro_df.dtypes)
print()

# Separate admission-linked subset for admission-level analyses (sections 4, 7, 9, 10, 11)
micro_adm_df = micro_df.dropna(subset=['hadm_id']).copy()
print(f"Admission-linked rows (hadm_id not null): {micro_adm_df.shape[0]:,}")
print(f"Outpatient rows (hadm_id null)           : {micro_df['hadm_id'].isna().sum():,}")
```

**In Section 6 cell (the assign_panel_30 step):** Change the dataframe used
for panel assignment from `micro_df` (which was already filtered) to the full
dataset. Since we now keep the full dataset in `micro_df`, the existing
`micro_df.apply(assign_panel_30, axis=1)` call is already correct — no change
needed here.

**In all other sections (4, 7, 9, 10, 11):** Replace every reference to
`micro_df` with `micro_adm_df` EXCEPT in Section 6 (which should use the full
`micro_df`). Specifically:

- Section 4 (cell 8): replace `micro_df` → `micro_adm_df` throughout
- Section 5 (cell 10): replace `micro_df` → `micro_adm_df` throughout
- Section 7 (cells for 7a–7d): replace `micro_df` → `micro_adm_df` throughout
- Section 9 (cells for 9a–9c): replace `micro_df` → `micro_adm_df` throughout
- Section 10 (cells for 10a–10c): replace `micro_df` → `micro_adm_df` throughout
- Section 11 (cell): replace `micro_df` → `micro_adm_df` throughout

Section 2 (heatmap) and Section 3 (long tail) use `micro_df` — keep them on
the full dataset since they analyse test/specimen structure, not admissions.

**Update the coverage summary in Section 6 (cell 6.3):** The admission coverage
percentage should still be computed against `total_admissions` (from `adm_df`)
for clinical interpretability. The panel event counts however should now reflect
ALL rows including outpatient. Update the summary print:

```python
# In the summary loop, change:
p_df = micro_df[micro_df['panel_30'] == panel_name]
# (already correct since micro_df is now the full dataset)

# Update the footer to show both total rows and admission-linked rows
print(f"Total rows analysed (incl. outpatient) : {len(micro_df):,}")
print(f"Admission-linked rows                  : {len(micro_adm_df):,}")
print(f"Admissions with ≥1 panel assigned      : "
      f"{micro_adm_df[micro_adm_df['panel_30'].isin(PANELS_30)]['hadm_id'].nunique():,}")
```

Note: `panel_30` must be assigned on the full `micro_df` first, then
`micro_adm_df` inherits the column automatically since it was created with
`.copy()` before the assignment. To ensure this works, move the
`micro_adm_df = micro_df.dropna(subset=['hadm_id']).copy()` line in cell 2
to AFTER the `micro_df['panel_30'] = micro_df.apply(assign_panel_30, axis=1)`
line in Section 6. Or alternatively, re-apply the panel assignment to
`micro_adm_df` after assigning to `micro_df`:
```python
micro_adm_df['panel_30'] = micro_adm_df['test_name'].map(
    lambda t: micro_df.loc[micro_df['test_name'] == t, 'panel_30'].iloc[0]
    if len(micro_df.loc[micro_df['test_name'] == t]) > 0 else 'unassigned'
)
```
The simplest approach: just run `assign_panel_30` on both dataframes
after defining it:
```python
micro_df['panel_30']     = micro_df.apply(assign_panel_30, axis=1)
micro_adm_df['panel_30'] = micro_adm_df.apply(assign_panel_30, axis=1)
```

---

## Issue 3 — Remove stale sections 6a, 6b, 6c

Sections 6a, 6b, and 6c analyse the old 7-panel `other` bucket. This is no
longer relevant now that we have the 30-panel design. Remove them entirely.

Find and delete the following cells from the notebook (they appear between
the Section 6 markdown header and the `PANELS_30` definition):

1. The markdown cell starting with:
   ```
   ## Section 6: `other` Panel Deep-Dive
   ```
   Replace its content with:
   ```
   ## Section 6: 30-Panel (Test × Specimen) Groupings
   ```
   (keep the description paragraph already written for the 30-panel section)

2. The code cell containing `subcategory_map` and `assign_subcat` — delete it

3. The code cell containing `6b: Horizontal bar chart` for sub-categories — delete it

4. The code cell containing `6c: Decision table` — delete it

The notebook should jump directly from the Section 6 markdown header into
the `PANELS_30` definition cell.

---

## Summary of changes

| Change | Where | What |
|---|---|---|
| Add ~20 missing combos | Section 6, `PANELS_30` dict | Respiratory and stool combos |
| Keep full dataset in `micro_df` | Cell 2 (Section 1) | Remove `dropna` on `micro_df` |
| Create `micro_adm_df` | Cell 2 (Section 1) | Filtered copy for admission analyses |
| Swap `micro_df` → `micro_adm_df` | Sections 4, 5, 7, 9, 10, 11 | Use admission-filtered df |
| Keep `micro_df` | Sections 2, 3, 6 | Full dataset for structure analysis |
| Apply `assign_panel_30` to both dfs | Section 6 | Both get `panel_30` column |
| Delete 6a/6b/6c cells | Section 6 | Remove stale 7-panel other analysis |
