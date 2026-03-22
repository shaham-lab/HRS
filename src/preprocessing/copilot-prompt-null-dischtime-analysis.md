# Add Null `dischtime` Analysis to Section 2 of `mimic4_data_exploration.ipynb`

## Context

Section 2 currently loads the admissions table and computes basic patient/admission
statistics. We need to add a cell that analyses null `dischtime` specifically for
surviving patients (Y1=0), to determine whether hard exclusion or NaN assignment is
the appropriate pipeline design choice for Y2 label computation.

---

## What to add

Add one new markdown cell and one new code cell at the **end of Section 2**,
after the existing "ADMISSIONS PER PATIENT" output cell and before Section 3.

---

## New markdown cell

```markdown
### Section 2.1: Null `dischtime` Analysis

Surviving patients with a null `dischtime` cannot have a valid 30-day readmission
label (Y2) computed. This cell quantifies how many such records exist and how they
are distributed, to inform the pipeline design decision on whether to exclude these
admissions or assign Y2=NaN.
```

---

## New code cell

```python
# =============================================================================
# Section 2.1: Null dischtime Analysis
# =============================================================================
# admissions_df is already loaded in Section 2

admissions_df['dischtime'] = pd.to_datetime(admissions_df['dischtime'], errors='coerce')

total_adm          = len(admissions_df)
deceased           = admissions_df['hospital_expire_flag'] == 1
survivors          = admissions_df['hospital_expire_flag'] == 0
null_disch         = admissions_df['dischtime'].isna()

n_deceased         = deceased.sum()
n_survivors        = survivors.sum()

# Survivors with null dischtime
surv_null          = (survivors & null_disch)
n_surv_null        = surv_null.sum()

# Deceased with null dischtime (informational)
dec_null           = (deceased & null_disch)
n_dec_null         = dec_null.sum()

print("=" * 80)
print("NULL dischtime ANALYSIS")
print("=" * 80)
print(f"  Total admissions                          : {total_adm:>10,}")
print(f"  Deceased admissions (Y1=1)                : {n_deceased:>10,}  ({n_deceased/total_adm*100:.2f}%)")
print(f"  Surviving admissions (Y1=0)               : {n_survivors:>10,}  ({n_survivors/total_adm*100:.2f}%)")
print()
print(f"  Survivors with null dischtime             : {n_surv_null:>10,}  ({n_surv_null/n_survivors*100:.3f}% of survivors)")
print(f"  Survivors with null dischtime             : {n_surv_null:>10,}  ({n_surv_null/total_adm*100:.3f}% of all admissions)")
print()
print(f"  Deceased with null dischtime (info only)  : {n_dec_null:>10,}  ({n_dec_null/n_deceased*100:.2f}% of deceased)")
print("=" * 80)

# Breakdown by admission type for survivors with null dischtime
if n_surv_null > 0:
    print("\nSurvivors with null dischtime — breakdown by admission_type:")
    print(admissions_df[surv_null]['admission_type'].value_counts().to_string())

    print("\nSurvivors with null dischtime — breakdown by discharge_location:")
    print(admissions_df[surv_null]['discharge_location'].value_counts().head(15).to_string())

    # Time range of admittimes for these records
    adm_times = pd.to_datetime(admissions_df[surv_null]['admittime'], errors='coerce')
    print(f"\nAdmittime range for affected records:")
    print(f"  Earliest: {adm_times.min()}")
    print(f"  Latest  : {adm_times.max()}")
else:
    print("\nNo surviving admissions with null dischtime found.")
```

---

## Requirements

- `admissions_df` is already defined earlier in Section 2 — do not reload it
- Do not modify any existing cells
- Add after the last existing cell in Section 2, before the Section 3 markdown cell
