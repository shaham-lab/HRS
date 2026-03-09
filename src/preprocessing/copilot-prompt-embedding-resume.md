# GitHub Copilot Prompt — Resume interrupted embedding runs in embed_features.py

## Context

`embed_features.py` embeds 18 features (5 text + 13 lab groups). Each feature takes
minutes to hours. If the process is interrupted mid-way — power loss, OOM kill, Ctrl+C —
the entire module reruns from scratch on the next pipeline run because
`preprocessing_utils._sources_unchanged()` only skips at the module level: all 18 output
parquets must exist for the module to be skipped.

The fix is per-feature resumption: before embedding each feature, check whether its output
parquet already exists and is valid. If so, skip it and move on to the next. This requires
no changes to `preprocessing_utils.py` — it is implemented entirely inside `embed_features.py`.

---

## Change 1 — Add a `_output_is_valid` helper

Add this function near the top of `embed_features.py`, after the `_TEXT_FEATURES` constant:

```python
def _output_is_valid(path: str, expected_rows: int, embedding_col: str) -> bool:
    """
    Return True if a completed embedding parquet exists at `path` and is usable.

    Checks:
    - File exists
    - Can be read as a parquet
    - Has the expected number of rows (matches the input feature file)
    - Contains the expected embedding column
    - No null values in the embedding column (a partial write would leave nulls)
    """
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_parquet(path)
    except Exception:
        return False
    if len(df) != expected_rows:
        return False
    if embedding_col not in df.columns:
        return False
    if df[embedding_col].isnull().any():
        return False
    return True
```

---

## Change 2 — Skip already-completed text features

In `run()`, inside the text feature loop, add a skip check immediately after loading the
input DataFrame and before calling `_embed_texts`:

```python
for (input_filename, text_col, output_filename, embedding_col) in tqdm(...):
    input_path = os.path.join(features_dir, input_filename)
    if not os.path.exists(input_path):
        logger.warning("Feature file not found, skipping: %s", input_path)
        pbar.update(1)
        continue

    df = pd.read_parquet(input_path)
    output_path = os.path.join(embeddings_dir, output_filename)

    # ── Resume check ──────────────────────────────────────────────────────
    if _output_is_valid(output_path, expected_rows=len(df), embedding_col=embedding_col):
        logger.info(
            "  [SKIP] %s already complete (%d rows) — resuming from next feature.",
            output_filename, len(df),
        )
        pbar.update(1)
        continue
    # ──────────────────────────────────────────────────────────────────────

    logger.info(
        "  Embedding '%s': %d texts, effective max_length=%d",
        text_col, len(df), effective_max_length,
    )
    # ... rest of embed + save logic unchanged ...
```

---

## Change 3 — Skip already-completed lab group features

In `run()`, inside the lab group loop, add the same skip check after computing
`group_text` (after the left-join to `splits_df`) and before calling `_embed_texts`:

```python
for group_name, itemids in tqdm(lab_panel_config.items(), ...):
    output_filename = f"lab_{group_name}_embeddings.parquet"
    output_path = os.path.join(embeddings_dir, output_filename)
    embedding_col = f"lab_{group_name}_embedding"

    # ... group_df filter and group_text left-join logic (unchanged) ...

    # ── Resume check ──────────────────────────────────────────────────────
    if _output_is_valid(output_path, expected_rows=len(group_text), embedding_col=embedding_col):
        logger.info(
            "  [SKIP] %s already complete (%d rows) — resuming from next group.",
            output_filename, len(group_text),
        )
        pbar.update(1)
        continue
    # ──────────────────────────────────────────────────────────────────────

    logger.info(
        "  Embedding lab group '%s': %d admissions (%d with events), effective max_length=%d",
        group_name, len(group_text),
        int((group_text["text"] != "").sum()),
        LAB_MAX_LENGTH,
    )
    # ... rest of embed + save logic unchanged ...
```

---

## Change 4 — Write output immediately after each feature completes

This is already the case in the current code (each feature is saved before moving to the
next), so no change is needed here. The pattern is correct — the parquet is written
atomically by pandas before the loop advances.

To make the write slightly safer against partial writes (e.g. a kill signal mid-write),
use a write-to-temp-then-rename pattern:

```python
# Before
out_df.to_parquet(output_path, index=False)

# After — atomic write via temp file
tmp_path = output_path + ".tmp"
out_df.to_parquet(tmp_path, index=False)
os.replace(tmp_path, output_path)   # atomic on POSIX; overwrites atomically
```

Apply this pattern to both the text feature save and the lab group save in `run()`.
`_output_is_valid` will never see a `.tmp` file (it checks the final path), so partial
writes from a previous interrupted run will simply be absent and the feature will be
re-embedded cleanly.

---

## Change 5 — Log resume status at the start of the module

At the top of `run()`, before the model is loaded, scan all 18 expected outputs and log
how many are already complete. This tells the user immediately whether a full run or a
resume is about to happen — and avoids loading the BERT model at all if everything is
already done:

```python
# Build the full list of expected output paths
expected_outputs: list[tuple[str, str]] = []  # (output_path, embedding_col)

for (_, _, output_filename, embedding_col) in _TEXT_FEATURES:
    expected_outputs.append(
        (os.path.join(embeddings_dir, output_filename), embedding_col)
    )
# Lab group outputs cannot be checked here without loading lab_panel_config.yaml,
# so only the 5 text features are pre-checked; lab groups are checked inline in the loop.

# Count how many text feature outputs already exist (row count not yet known, so
# just check existence as a fast pre-screen)
n_text_done = sum(1 for (p, _) in expected_outputs if os.path.exists(p))
n_text_total = len(_TEXT_FEATURES)

if n_text_done > 0:
    logger.info(
        "Resume mode: %d / %d text feature embeddings already present — "
        "will verify and skip completed ones.",
        n_text_done, n_text_total,
    )
else:
    logger.info("Fresh run: no existing embedding outputs found.")
```

---

## Behaviour summary after this change

| Scenario | Behaviour |
|---|---|
| Fresh run | All 18 features embedded normally |
| Interrupted after feature 3 | Features 1–3 skipped (valid parquets exist); features 4–18 embedded |
| Interrupted mid-feature | That feature's `.tmp` file is absent; feature re-embedded from scratch |
| Source parquet changed | `preprocessing_utils._sources_unchanged()` invalidates the whole module; all 18 re-embedded |
| `--force-module embed_features` | All 18 re-embedded (FORCE_RERUN bypasses `_sources_unchanged` but NOT the per-feature skip) |

### Note on `--force-module` and per-feature skipping

The per-feature skip in `_output_is_valid` is independent of `FORCE_RERUN`. If you want
to force re-embedding of all features, delete the embeddings directory or individual
parquet files. Optionally, expose a `BERT_FORCE_REEMBED: false` config key that disables
the per-feature skip when set to `true`:

```python
force_reembed = bool(config.get("BERT_FORCE_REEMBED", False))

# In the skip check:
if not force_reembed and _output_is_valid(...):
    logger.info("  [SKIP] ...")
    continue
```

Add to `preprocessing.yaml`:

```yaml
# Set to true to force re-embedding of all features even if output parquets exist.
# Normally false — embed_features.py resumes from the last completed feature.
BERT_FORCE_REEMBED: false
```
