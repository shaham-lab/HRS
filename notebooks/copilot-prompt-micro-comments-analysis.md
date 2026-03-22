# Add Comments Field Analysis Cell to `mimic4_microbiology_exploration.ipynb`

## Context

The `microbiologyevents` table has a `comments` column that may contain
the actual qualitative result for tests that have no organism growth and
no numerical value (e.g. serology, antigen, molecular tests). Before
deciding on a cleaning strategy for feature extraction, we need to
understand the actual content distribution of this field.

---

## What to add

Add one new markdown cell and one new code cell at the **end** of the
notebook, after the last existing cell.

---

## New markdown cell

```markdown
## Section 13: Comments Field Analysis

The `comments` column in `microbiologyevents` may serve as the primary
result field for qualitative tests (serology, antigen, molecular) where
there is no organism to grow and no numerical value. This section
analyses the actual content distribution to inform the text
representation cleaning strategy for `extract_microbiology.py`.
```

---

## New code cell

```python
# =============================================================================
# Section 13: Comments Field Analysis
# =============================================================================
# Use full dataset (micro_df) — comments exist for both inpatient and outpatient

comments_raw = micro_df['comments']

# --- 13.1: Overall presence ---
n_total      = len(comments_raw)
n_null       = comments_raw.isna().sum()
n_empty      = (comments_raw.fillna('').str.strip() == '').sum()
n_placeholder = comments_raw.fillna('').str.match(r'^_+$|^-+$').sum()
n_meaningful = n_total - n_null - (n_empty - n_null) - n_placeholder

print("=== 13.1: Comments Field Presence ===\n")
print(f"  Total rows                    : {n_total:>10,}")
print(f"  Null (NaN)                    : {n_null:>10,}  ({n_null/n_total*100:.1f}%)")
print(f"  Empty or whitespace-only      : {n_empty - n_null:>10,}  ({(n_empty-n_null)/n_total*100:.1f}%)")
print(f"  Placeholder only (___ or ---) : {n_placeholder:>10,}  ({n_placeholder/n_total*100:.1f}%)")
print(f"  Potentially meaningful        : {n_meaningful:>10,}  ({n_meaningful/n_total*100:.1f}%)")

# --- 13.2: Top 80 most frequent non-empty comment values ---
comments_clean = (
    comments_raw
    .dropna()
    .pipe(lambda s: s[s.str.strip() != ''])
    .pipe(lambda s: s[~s.str.match(r'^_+$|^-+$')])
    .str.strip()
)

top80 = comments_clean.value_counts().head(80)

print(f"\n=== 13.2: Top 80 Most Frequent Comment Values ===\n")
print(f"{'Rank':<5} {'Count':>8}  {'Comment'}")
print("-" * 80)
for rank, (val, cnt) in enumerate(top80.items(), 1):
    # Truncate long values for display
    display_val = val[:100].replace('\n', ' ') if len(val) > 100 else val.replace('\n', ' ')
    print(f"{rank:<5} {cnt:>8,}  {display_val}")

# --- 13.3: Comment length distribution ---
comment_lengths = comments_clean.str.len()
print(f"\n=== 13.3: Comment Length Distribution (non-empty) ===\n")
print(f"  Count  : {len(comment_lengths):,}")
print(f"  Min    : {comment_lengths.min()}")
print(f"  Median : {comment_lengths.median():.0f}")
print(f"  Mean   : {comment_lengths.mean():.0f}")
print(f"  p75    : {comment_lengths.quantile(0.75):.0f}")
print(f"  p90    : {comment_lengths.quantile(0.90):.0f}")
print(f"  p95    : {comment_lengths.quantile(0.95):.0f}")
print(f"  Max    : {comment_lengths.max()}")

# --- 13.4: Comments by panel ---
print(f"\n=== 13.4: Comment Presence by Panel ===\n")
print(f"{'Panel':<35} {'Total':>8} {'Has Comment':>12} {'Comment%':>9}")
print("-" * 70)

micro_df['_has_comment'] = (
    comments_raw.notna() &
    (comments_raw.str.strip() != '') &
    (~comments_raw.str.match(r'^_+$|^-+$').fillna(False))
)

for panel_name in PANELS_37.keys():
    p_df = micro_df[micro_df['panel_37'] == panel_name]
    n_p  = len(p_df)
    n_c  = p_df['_has_comment'].sum()
    pct  = n_c / n_p * 100 if n_p > 0 else 0
    print(f"  {panel_name:<33} {n_p:>8,} {n_c:>12,} {pct:>8.1f}%")

micro_df.drop(columns=['_has_comment'], inplace=True)

# --- 13.5: 30 random samples of meaningful comments ---
print(f"\n=== 13.5: 30 Random Samples of Non-Empty Comments ===\n")
sample_comments = comments_clean.sample(min(30, len(comments_clean)), random_state=42)
for i, (idx, val) in enumerate(sample_comments.items(), 1):
    display_val = val[:200].replace('\n', ' ') if len(val) > 200 else val.replace('\n', ' ')
    print(f"  {i:>2}. {display_val}")

# --- 13.6: Comments for panels with high comment rates (top 10) ---
print(f"\n=== 13.6: Sample Comments from Top Comment-Rate Panels ===\n")

panel_comment_rates = {}
for panel_name in PANELS_37.keys():
    p_df = micro_df[micro_df['panel_37'] == panel_name]
    n_p  = len(p_df)
    if n_p == 0:
        continue
    has_comment = (
        p_df['comments'].notna() &
        (p_df['comments'].str.strip() != '') &
        (~p_df['comments'].str.match(r'^_+$|^-+$').fillna(False))
    )
    panel_comment_rates[panel_name] = has_comment.mean()

top_comment_panels = sorted(
    panel_comment_rates.items(), key=lambda x: -x[1]
)[:10]

for panel_name, rate in top_comment_panels:
    p_df = micro_df[micro_df['panel_37'] == panel_name]
    has_comment = (
        p_df['comments'].notna() &
        (p_df['comments'].str.strip() != '') &
        (~p_df['comments'].str.match(r'^_+$|^-+$').fillna(False))
    )
    samples = (
        p_df[has_comment]['comments']
        .str.strip()
        .value_counts()
        .head(10)
    )
    print(f"  {panel_name} (comment rate: {rate*100:.1f}%)")
    for val, cnt in samples.items():
        display_val = val[:120].replace('\n', ' ')
        print(f"    {cnt:>6,}x  {display_val}")
    print()
```

---

## Requirements

- This cell must be added **after** the last existing cell in the notebook
- `micro_df` and `PANELS_37` are already defined in earlier cells — do not redefine them
- The `panel_37` column must already be assigned on `micro_df` (done in Section 6)
- Do not modify any existing cells
