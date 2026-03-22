# Add Section 14: Deep Comments Analysis to `mimic4_microbiology_exploration.ipynb`

## Context

Section 13 revealed that comments are the primary result field for 20+ panels
and have a consistent structure: meaningful result in the first 1-2 sentences
followed by boilerplate. This section goes deeper to identify all boilerplate
trigger patterns, validate the truncation strategy, and characterize what
remains after cleaning — informing the final comment cleaning rules for
`extract_microbiology.py`.

---

## What to add

Add one markdown cell and one code cell at the end of the notebook.

---

## Markdown cell

```markdown
## Section 14: Deep Comments Analysis

Deeper analysis of the `comments` field to finalize the cleaning strategy
for `extract_microbiology.py`. Goals:
1. Identify all sentence-level boilerplate patterns that should be stripped
2. Validate that first-1-2-sentence truncation captures the meaningful result
3. Find all discard-entirely patterns (cancelled, duplicate, not performed)
4. Characterize the cleaned comment vocabulary per panel
5. Identify any trigger words missed in the initial review
```

---

## Code cell

```python
# =============================================================================
# Section 14: Deep Comments Analysis
# =============================================================================

import re
from collections import Counter

# Work on non-empty comments from full dataset
comments_series = (
    micro_df['comments']
    .dropna()
    .pipe(lambda s: s[s.str.strip() != ''])
    .pipe(lambda s: s[~s.str.match(r'^_+$|^-+$')])
    .str.strip()
)
print(f"Working set: {len(comments_series):,} non-empty comments\n")

# ── 14.1: Identify all unique sentence openers across the corpus ──────────────
# Split each comment on sentence boundaries (.  or .\n but NOT on single space
# after period and NOT on colon). Collect all sentences beyond position 0.
# These are candidates for boilerplate patterns.

print("=" * 80)
print("14.1: Most Common Sentence[1] Openers (boilerplate candidates)")
print("=" * 80)
print("(Sentence[1] = second sentence after splitting on '.  ' or '.\\n')\n")

def split_sentences(text):
    """Split on period+2spaces or period+newline. Not on colon."""
    parts = re.split(r'\.\s{2,}|\.\n', text)
    return [p.strip() for p in parts if p.strip()]

sentence1_openers = Counter()
sentence2_openers = Counter()

for text in comments_series:
    sents = split_sentences(text)
    if len(sents) > 1:
        # Take first 6 words of sentence[1] as the opener
        opener = ' '.join(sents[1].split()[:6])
        sentence1_openers[opener] += 1
    if len(sents) > 2:
        opener2 = ' '.join(sents[2].split()[:6])
        sentence2_openers[opener2] += 1

print("Top 60 sentence[1] openers:")
print(f"{'Count':>8}  Opener")
print("-" * 70)
for opener, cnt in sentence1_openers.most_common(60):
    print(f"{cnt:>8,}  {opener}")

print(f"\nTop 30 sentence[2] openers:")
print(f"{'Count':>8}  Opener")
print("-" * 70)
for opener, cnt in sentence2_openers.most_common(30):
    print(f"{cnt:>8,}  {opener}")

# ── 14.2: Validate truncation at known trigger words ─────────────────────────
print("\n" + "=" * 80)
print("14.2: Trigger Word Validation")
print("=" * 80)
print("For each candidate trigger word, show: how many comments it appears in,")
print("at what sentence position (0=first, 1=second, etc.), and 3 example")
print("before/after snippets to confirm the truncation is correct.\n")

candidate_triggers = [
    "Reference Range",
    "Reference range",
    "Detection Range",
    "Linear range",
    "Performed by",
    "performed by",
    "Validated for use",
    "validated for use",
    "rule out",
    "indicates",
    "This test",
    "this test",
    "Clinical significance",
    "clinical significance",
    "If testing",
    "Please resubmit",
    "contact the",
    "Contact the",
    "A negative result",
    "A positive result",
    "In most population",
    "The FDA",
    "performance characteristics",
    "Performance characteristics",
    "minimum 14 day",
    "Minimum 14 day",
    "screened for",
    "Detection of viruses",
    "sent to",
    "Sent to",
    "per 1000X FIELD",
]

for trigger in candidate_triggers:
    mask = comments_series.str.contains(re.escape(trigger), case=True, na=False)
    n_affected = mask.sum()
    if n_affected == 0:
        continue

    # Find sentence position where trigger first appears
    position_counts = Counter()
    for text in comments_series[mask].head(500):
        sents = split_sentences(text)
        for i, sent in enumerate(sents):
            if trigger.lower() in sent.lower():
                position_counts[i] += 1
                break

    most_common_pos = position_counts.most_common(1)[0][0] if position_counts else '?'

    print(f"  '{trigger}'")
    print(f"    Appears in: {n_affected:,} comments  |  Most common sentence position: {most_common_pos}")

    # Show 3 examples: the full comment truncated at trigger
    examples = comments_series[mask].head(3)
    for ex in examples:
        idx = ex.lower().find(trigger.lower())
        before = ex[:idx].strip()[:120]
        after  = ex[idx:idx+80].strip()
        print(f"    BEFORE: {before}")
        print(f"    TRIGGER→: {after[:60]}...")
        print()

# ── 14.3: Discard-entirely pattern analysis ───────────────────────────────────
print("=" * 80)
print("14.3: Discard-Entirely Pattern Analysis")
print("=" * 80)
print("Comments that should be discarded entirely (no clinical value)\n")

discard_candidates = [
    "TEST CANCELLED",
    "PATIENT CREDITED",
    "Patient credited",
    "TEST NOT PERFORMED",
    "DUPLICATE ORDER",
    "Duplicate order",
    "cancel",
    "Cancel",
    "not accepted",
    "NOT ACCEPTED",
    "REJECTED",
    "rejected",
    "unable to process",
    "Unable to process",
    "wrong tube",
    "Wrong tube",
    "quantity not sufficient",
    "Quantity not sufficient",
    "QNS",
    "specimen not received",
    "Specimen not received",
    "improper",
    "Improper",
]

print(f"{'Pattern':<45} {'Count':>8}  Example")
print("-" * 100)
for pattern in discard_candidates:
    mask = comments_series.str.contains(re.escape(pattern), case=True, na=False)
    n = mask.sum()
    if n > 0:
        example = comments_series[mask].iloc[0][:80].replace('\n', ' ')
        print(f"  {pattern:<43} {n:>8,}  {example}")

# ── 14.4: After cleaning — what remains? ─────────────────────────────────────
print("\n" + "=" * 80)
print("14.4: Cleaned Comment Vocabulary — Top 100 Values After Truncation")
print("=" * 80)
print("Apply proposed cleaning: truncate at trigger words, keep first 2 sentences,")
print("discard cancellations. Show top 100 unique values that remain.\n")

STRIP_TRIGGERS = [
    "Reference Range", "Reference range", "Detection Range", "Linear range",
    "Performed by", "performed by", "Validated for use", "validated for use",
    "rule out", "indicates", "This test", "this test",
    "Clinical significance", "A negative result", "A positive result",
    "In most population", "The FDA", "performance characteristics",
    "minimum 14 day", "Minimum 14 day", "screened for",
    "Detection of viruses", "per 1000X FIELD",
]

DISCARD_PREFIXES = (
    'TEST CANCELLED', 'TEST NOT PERFORMED', 'DUPLICATE ORDER',
    'Patient credited', 'PATIENT CREDITED',
)

def clean_comment(text):
    if not text or str(text).strip() in ('', '___', '---'):
        return None
    text = str(text).strip()
    # Discard entirely
    if any(text.startswith(p) for p in DISCARD_PREFIXES):
        return None
    # Truncate at trigger words
    for trigger in STRIP_TRIGGERS:
        idx = text.find(trigger)
        if idx > 0:
            text = text[:idx].strip().rstrip('.')
    # Split into sentences (not on colon)
    sents = split_sentences(text)
    # Keep first 2 sentences
    result = '.  '.join(sents[:2])
    # Hard truncate
    result = result[:200].strip()
    return result if result else None

cleaned = comments_series.apply(clean_comment).dropna()
cleaned = cleaned[cleaned.str.strip() != '']

print(f"Comments before cleaning : {len(comments_series):,}")
print(f"Comments after cleaning  : {len(cleaned):,}  ({len(cleaned)/len(comments_series)*100:.1f}% retained)")
print(f"Discarded entirely       : {len(comments_series) - len(cleaned):,}")
print()

top100_cleaned = cleaned.value_counts().head(100)
print(f"{'Rank':<5} {'Count':>8}  Cleaned value")
print("-" * 90)
for rank, (val, cnt) in enumerate(top100_cleaned.items(), 1):
    display = val[:100].replace('\n', ' ')
    print(f"{rank:<5} {cnt:>8,}  {display}")

# ── 14.5: Length distribution after cleaning ─────────────────────────────────
print("\n" + "=" * 80)
print("14.5: Cleaned Comment Length Distribution")
print("=" * 80)

cleaned_lengths = cleaned.str.len()
print(f"  Count  : {len(cleaned_lengths):,}")
print(f"  Min    : {cleaned_lengths.min()}")
print(f"  Median : {cleaned_lengths.median():.0f}")
print(f"  Mean   : {cleaned_lengths.mean():.1f}")
print(f"  p75    : {cleaned_lengths.quantile(0.75):.0f}")
print(f"  p90    : {cleaned_lengths.quantile(0.90):.0f}")
print(f"  p95    : {cleaned_lengths.quantile(0.95):.0f}")
print(f"  Max    : {cleaned_lengths.max()}")

# ── 14.6: Before vs after comparison — 20 examples ───────────────────────────
print("\n" + "=" * 80)
print("14.6: Before vs After Cleaning — 20 Examples")
print("=" * 80)
print("Showing cases where cleaning changed the comment.\n")

changed_mask = comments_series.apply(clean_comment) != comments_series
changed = comments_series[changed_mask].head(20)

for i, (idx, original) in enumerate(changed.items(), 1):
    cleaned_val = clean_comment(original)
    print(f"  {i:>2}. BEFORE: {original[:150].replace(chr(10), ' ')}")
    print(f"      AFTER : {cleaned_val if cleaned_val else '[DISCARDED]'}")
    print()
```

---

## Requirements

- Add after the last existing cell (Section 13)
- `micro_df` and `PANELS_37` are already defined — do not redefine
- `panel_37` column must already be assigned on `micro_df` (Section 6)
- Do not modify any existing cells
- All imports (`re`, `Counter`) should be added at the top of the code cell
  if not already available in the notebook scope
