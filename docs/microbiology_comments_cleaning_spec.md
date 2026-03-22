# Microbiology Comments Field Cleaning Specification

## Background

The `comments` column in `microbiologyevents` serves as the **primary result
field** for qualitative tests — serology, antigen, molecular, gram stain, and
parasitology panels — where there is no organism growth and no numerical value.
For these panels the comment IS the test result (e.g. "NONREACTIVE",
"Negative for Chlamydia trachomatis by PCR", "NO MYCOBACTERIA ISOLATED").

For culture panels the comment complements `org_name` by adding CFU/mL counts,
flora descriptions, PMN cell counts, and contamination flags.

Comment presence by panel type:
- gc_chlamydia_sti, gram_stain_csf, urinary_antigens : 98% have comments
- syphilis_serology, stool_bacterial, fungal panels  : 94–96%
- misc_serology, respiratory panels                  : 88–94%
- urine_culture, wound_culture                       : 33–38%

**Conclusion: comments must be included in the text representation.**
Excluding them would mean embedding near-empty strings for 20+ panels.

---

## Comment Structure

Each comment has the following layered structure:

```
[RESULT SENTENCE(S)]  [BOILERPLATE SENTENCES...]
```

The clinically meaningful content is always in the **first 1–2 sentences**.
All subsequent sentences are one of:
- Reference range information
- Methodology/instrument notes
- Regulatory/approval disclaimers
- Interpretation paragraphs
- Specimen quality advice
- Detection range specifications

Examples:
```
"NONREACTIVE.  Reference Range: Non-Reactive."
 ↑ result      ↑ boilerplate — strip from here

"Negative for Chlamydia trachomatis by PCR."
 ↑ entire comment is result — keep as-is

"NO POLYMORPHONUCLEAR LEUKOCYTES SEEN.  NO MICROORGANISMS SEEN.  
 This is a concentrated smear made by cytospin method..."
 ↑ sentence 1 = result    ↑ sentence 2 = result    ↑ sentence 3 = boilerplate

"HIV-1 RNA is not detected.  Performed using the ___ HIV-1 Test v2.0.
 Detection Range: 20-10,000,000 copies/mL."
 ↑ sentence 1 = result    ↑ sentence 2 = boilerplate — trigger on "Performed using"
```

---

## Cleaning Algorithm

Apply the following steps **in order**:

### Step 1 — Null and placeholder check
If the comment is null, empty, whitespace-only, or matches `^_+$` or `^-+$`,
return `None` (treat as absent).

### Step 2 — Discard-entirely check
If the comment **starts with** any discard prefix (case-sensitive, checked
after stripping leading whitespace), discard the entire comment and return `None`.

Discard prefixes:
```
TEST CANCELLED
PATIENT CREDITED
Patient credited
Test cancelled
test cancelled
TEST NOT PERFORMED
DUPLICATE ORDER
GRAM STAIN OF THIS SPECIMEN INDICATES CONTAMINATION
```

### Step 3 — Trigger word truncation
Find the **first occurrence** of any trigger word/phrase in the comment.
If found at character position `idx > 0`, truncate: `comment = comment[:idx]`.

**Important:** The trigger must appear at `idx > 0` — if it appears at position 0
the entire comment would be discarded, which is undesirable. In that case skip
the trigger and proceed to the next one.

**Sentence boundary note:** Colon `:` is NOT a sentence boundary and must not
be used as a split point. Triggers fire on the substring match, not on sentence
structure.

Trigger words/phrases (apply in listed order — stop at first match):
```
Reference Range
Reference range
Detection Range
Detection range
Linear range
Performed by
performed by
Performed using
performed using
Validated for use
validated for use
Performance characteristics
performance characteristics
A positive IgG result
A positive IgM result
This test
this test
In most population
The FDA
the FDA
screened for
Detection of viruses
PLEASE SUBMIT ANOTHER
rule out
approved
Patients
patients
```

### Step 4 — Sentence splitting
Split the remaining text into sentences using the pattern:
```
\.  +   (period followed by 2 or more spaces)
\.\n    (period followed by newline)
```

**Do NOT split on:**
- Single space after period (e.g. "10,000-100,000 CFU/mL. Commensal Flora")
- Colon `:` 
- Semicolon `;`

Keep the **first 2 sentences** only. Discard any further sentences.

### Step 5 — Post-strip artifact cleanup
After truncation and splitting, trailing artifacts may remain from the trigger
removal (e.g. an opening parenthesis left behind from `"(Reference Range-Negative)"`).

Apply the following cleanup in order:
1. Strip trailing whitespace
2. Remove trailing pattern `\.  +\($` (period + spaces + open paren)
3. Remove trailing pattern `\s+\($` (spaces + open paren)
4. Strip trailing `(` character
5. Strip trailing `.` character (optional period)
6. Strip trailing whitespace again

### Step 6 — Length check
Hard-truncate to `MICRO_COMMENT_MAX_CHARS` characters (default: 200).
If the result is empty or whitespace-only after truncation, return `None`.

---

## Important: Comments That Must Pass Through Unchanged

Certain comment strings are definitive clinical results that must **never** be
discarded or truncated, even though they describe negative or contaminated
outcomes. The most important example is:

```
MIXED BACTERIAL FLORA ( >= 3 COLONY TYPES), CONSISTENT WITH SKIN AND/OR GENITAL CONTAMINATION.
MIXED BACTERIAL FLORA ( >= 3 COLONY TYPES), CONSISTENT WITH FECAL CONTAMINATION.
```

These comments (164,652 and 13,647 occurrences respectively) indicate a poorly
collected specimen yielding no targetable pathogen. This is clinically meaningful
signal — the RL agent must see it to learn that ordering routine urine or wound
cultures on certain patients yields contaminated, uninterpretable results.

None of the trigger words appear in these strings, so they will pass through the
cleaning algorithm correctly as-is. This note exists to prevent any future
implementer from accidentally adding `"CONTAMINATION"` or `"MIXED"` as discard
patterns.

---

## Configuration Keys

```yaml
MICRO_INCLUDE_COMMENTS: true
MICRO_COMMENT_MAX_SENTENCES: 2          # keep first N sentences after splitting
MICRO_COMMENT_MAX_CHARS: 200            # hard truncation after sentence extraction
```

The trigger list and discard prefix list are defined in `micro_panel_config.yaml`
so they can be extended without code changes:

```yaml
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

---

## Expected Results After Cleaning

| Before | After |
|--------|-------|
| `NONREACTIVE.  Reference Range: Non-Reactive.` | `NONREACTIVE` |
| `POSITIVE BY EIA.  A positive IgG result generally indicates past exposure and/or immunity.` | `POSITIVE BY EIA` |
| `Negative for Chlamydia trachomatis by PCR.` | `Negative for Chlamydia trachomatis by PCR` |
| `NO POLYMORPHONUCLEAR LEUKOCYTES SEEN.  NO MICROORGANISMS SEEN.  This is a concentrated smear...` | `NO POLYMORPHONUCLEAR LEUKOCYTES SEEN.  NO MICROORGANISMS SEEN` |
| `HIV-1 RNA is not detected.  Performed using the ___ HIV-1 Test v2.0.  Detection Range: ...` | `HIV-1 RNA is not detected` |
| `CMV DNA not detected.  Performed by PCR.  Detection Range: 600 - 100,000 copies/ml.` | `CMV DNA not detected` |
| `NEGATIVE BY EIA.            (Reference Range-Negative).` | `NEGATIVE BY EIA` |
| `NEGATIVE FOR LEGIONELLA SEROGROUP 1 ANTIGEN.            (Reference Range-Negative).  Performed by...` | `NEGATIVE FOR LEGIONELLA SEROGROUP 1 ANTIGEN` |
| `10,000-100,000 CFU/mL Commensal Respiratory Flora.` | `10,000-100,000 CFU/mL Commensal Respiratory Flora` |
| `2+   (1-5 per 1000X FIELD):   POLYMORPHONUCLEAR LEUKOCYTES.  NO MICROORGANISMS SEEN.  This is a concentrated smear...` | `2+   (1-5 per 1000X FIELD):   POLYMORPHONUCLEAR LEUKOCYTES.  NO MICROORGANISMS SEEN` |
| `NEGATIVE <1:10 BY IFA.  INTERPRETATION: RESULTS INDICATIVE OF PAST EBV INFECTION.  In most populations...` | `NEGATIVE <1:10 BY IFA.  INTERPRETATION: RESULTS INDICATIVE OF PAST EBV INFECTION` |
| `Culture workup discontinued. Further incubation showed contamination with mixed skin/genital flora. Clinical significance of isolate(s) uncertain.` | `Culture workup discontinued. Further incubation showed contamination with mixed skin/genital flora` |
| `TEST CANCELLED, PATIENT CREDITED.` | *(discarded)* |
| `Test cancelled by laboratory.  PATIENT CREDITED.` | *(discarded)* |
| `GRAM STAIN OF THIS SPECIMEN INDICATES CONTAMINATION WITH OROPHARYNGEAL SECRETIONS...` | *(discarded)* |

---

## Coverage Statistics (from EDA on full MIMIC-IV dataset)

| Metric | Value |
|--------|-------|
| Total rows | 3,988,224 |
| Null / empty / placeholder | 1,546,251 (38.8%) |
| Potentially meaningful | 2,441,973 (61.2%) |
| After cleaning (retained) | 2,415,215 (98.9% of meaningful) |
| Discarded entirely | 26,758 (1.1% of meaningful) |
| Median cleaned length | 20 chars |
| Mean cleaned length | 35.8 chars |
| p95 cleaned length | 95 chars |
| Max cleaned length | 200 chars (hard limit) |

---

## Integration with Text Representation

The cleaned comment is appended to the event text string as the last field,
separated by ` | `:

```
{test_name} [{spec_type_desc}]: {org_name or 'no growth'} | {susc_string} | {cleaned_comment}
```

If `cleaned_comment` is `None` after cleaning, the ` | {cleaned_comment}` part
is omitted entirely — no trailing separator.

If `org_name` is null AND `cleaned_comment` is the primary result
(serology/molecular/antigen panels), the format simplifies to:

```
{test_name} [{spec_type_desc}]: {cleaned_comment}
```

The extraction module detects this case by checking whether `org_name` is null
and `cleaned_comment` is not null.
