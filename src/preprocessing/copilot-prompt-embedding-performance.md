# GitHub Copilot Prompt — Improve embedding performance in embed_features.py

## Context

`embed_features.py` embeds ~500,000 admissions across 18 feature types using
`Simonlee711/Clinical_ModernBERT`. On GPU it is running far slower than expected.
A review of the code identified the following bottlenecks:

---

## Bottleneck 1 — Max token length is 8192 for ALL features (most critical)

`BERT_MAX_LENGTH: 8192` is applied uniformly to every feature. This is catastrophically
slow for short features:
- `chief_complaint_text` — typically 3–15 tokens
- `triage_text` — typically 30–60 tokens
- `diag_history_text` — typically 50–200 tokens

With `max_length=8192`, every batch is padded to 8192 tokens regardless of actual content.
The attention mechanism is O(n²) in sequence length — padding to 8192 instead of 64 is
**16,000× more compute** per token for short texts.

### Fix — per-feature max_length cap

Add a `_MAX_LENGTH_CAP` dict that overrides `BERT_MAX_LENGTH` per feature with a realistic
ceiling. Long clinical notes (discharge history, radiology) legitimately need a high cap;
short features do not.

```python
# Add this constant near the top of the file, after _TEXT_FEATURES:
_MAX_LENGTH_CAP: dict[str, int] = {
    "diag_history_text":       512,   # ICD label list — short
    "chief_complaint_text":    64,    # single phrase
    "triage_text":             256,   # structured vitals template
    "discharge_history_text":  4096,  # full clinical note — keep long
    "radiology_text":          1024,  # radiology report — medium
    # lab group texts: set dynamically below (see Bottleneck 3)
}
```

Update `_embed_texts` to accept `max_length` as before (no signature change needed — the
caller will now pass the per-feature cap instead of the global config value).

Update the call sites in `run()`:

```python
# Non-lab text features — pass per-feature cap
for (input_filename, text_col, output_filename, embedding_col) in ...:
    effective_max_length = min(
        max_length,  # global BERT_MAX_LENGTH from config
        _MAX_LENGTH_CAP.get(text_col, max_length),
    )
    embeddings = _embed_texts(
        texts, tokenizer, model, device, effective_max_length, batch_size
    )

# Lab group features — cap at 2048 (lab text lines are long but not note-length)
LAB_MAX_LENGTH = min(max_length, 2048)
embeddings_group = _embed_texts(
    texts_group, tokenizer, model, device, LAB_MAX_LENGTH, batch_size
)
```

Also log the effective max length used for each feature so it is visible in the run log:

```python
logger.info(
    "  Embedding '%s': %d texts, effective max_length=%d",
    text_col, len(texts), effective_max_length,
)
```

---

## Bottleneck 2 — Static padding to max_length; no dynamic padding per batch

The tokenizer call uses `max_length=max_length` with `padding=True`, which pads every
sequence in the batch to `max_length` tokens — even if the longest sequence in that batch
is only 40 tokens. This wastes compute on every batch of short texts.

### Fix — use `longest` padding strategy

Change the tokenizer call in `_embed_texts` to pad only to the longest sequence in each
batch:

```python
# Before
encoded = tokenizer(
    batch_texts_safe,
    padding=True,
    truncation=True,
    max_length=max_length,
    return_tensors="pt",
)

# After
encoded = tokenizer(
    batch_texts_safe,
    padding="longest",       # pad to longest sequence in THIS batch only
    truncation=True,
    max_length=max_length,   # still enforce the hard ceiling
    return_tensors="pt",
)
```

This is safe because `truncation=True` + `max_length` still enforces the hard ceiling, but
within that ceiling each batch is padded only as much as necessary.

---

## Bottleneck 3 — Batch size is not tuned for GPU memory

The default `BERT_BATCH_SIZE: 32` was chosen conservatively. With `max_length=8192` this is
appropriate (32 × 8192 tokens would OOM most GPUs). But with the per-feature caps from
Bottleneck 1, the effective sequence length drops dramatically — meaning batch size can be
increased significantly for short features, keeping the GPU saturated.

### Fix — dynamic batch size scaling

Add a helper that scales batch size inversely with effective sequence length, keeping a
constant token budget:

```python
def _effective_batch_size(base_batch_size: int, effective_max_length: int,
                           reference_length: int = 512) -> int:
    """
    Scale batch size to maintain a roughly constant token budget per GPU step.
    base_batch_size is calibrated for reference_length tokens.
    """
    scale = reference_length / max(effective_max_length, 1)
    scaled = int(base_batch_size * scale)
    return max(1, min(scaled, base_batch_size * 8))  # cap at 8× base
```

Call it at each feature:

```python
effective_batch_size = _effective_batch_size(batch_size, effective_max_length)
logger.info(
    "  effective_batch_size=%d (base=%d, max_length=%d)",
    effective_batch_size, batch_size, effective_max_length,
)
embeddings = _embed_texts(
    texts, tokenizer, model, device, effective_max_length, effective_batch_size
)
```

Add `BERT_BATCH_SIZE` guidance to `preprocessing.yaml` as a comment:

```yaml
# Batch size calibrated for max_length=512 on GPU with ≥8GB VRAM.
# embed_features.py scales this automatically per feature based on sequence length.
BERT_BATCH_SIZE: 32
```

---

## Bottleneck 4 — CPU-to-GPU transfer happens inside the batch loop; no prefetching

The line `encoded = {k: v.to(device) for k, v in encoded.items()}` transfers each batch
from CPU RAM to GPU VRAM synchronously inside the forward-pass loop. The GPU sits idle
while waiting for the transfer.

### Fix — use `non_blocking=True` for async transfers

```python
# Before
encoded = {k: v.to(device) for k, v in encoded.items()}

# After
encoded = {k: v.to(device, non_blocking=True) for k, v in encoded.items()}
```

`non_blocking=True` allows the CPU to continue preparing the next batch while the GPU
processes the current one (requires `pin_memory=True` on the source tensor, which PyTorch
handles automatically for CPU tensors transferred to CUDA).

Also move `model.eval()` out of `_embed_texts` to the call site in `run()` so it is called
once after model load, not once per feature:

```python
# In run(), after model.to(device):
model.eval()

# Remove model.eval() from _embed_texts
```

---

## Bottleneck 5 — Mean pooling runs on GPU but result is moved to CPU inside the batch loop

The line `(sum_embeddings / sum_mask).cpu().numpy()` does `.cpu()` inside the batch loop,
which synchronises GPU and CPU on every batch. Accumulate on GPU and transfer once at the
end.

### Fix — accumulate embeddings as GPU tensors, transfer once

```python
def _embed_texts(
    texts: list[str],
    tokenizer,
    model,
    device,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    """Return a (N, hidden_size) float32 array using mean pooling."""
    import torch  # type: ignore

    all_embeddings: list[torch.Tensor] = []   # keep on GPU until the end
    empty_indices: list[int] = []
    global_idx = 0

    n_batches = (len(texts) + batch_size - 1) // batch_size
    for start in tqdm(range(0, len(texts), batch_size), total=n_batches,
                      desc="Embedding batches", unit="batch", leave=False):
        batch_texts = texts[start: start + batch_size]
        batch_texts_safe = [t if isinstance(t, str) and t.strip() else " "
                            for t in batch_texts]
        empty_flags = [not (isinstance(t, str) and t.strip()) for t in batch_texts]

        for i, flag in enumerate(empty_flags):
            if flag:
                empty_indices.append(global_idx + i)
        global_idx += len(batch_texts)

        encoded = tokenizer(
            batch_texts_safe,
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device, non_blocking=True) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)

        attention_mask = encoded["attention_mask"]
        token_embeddings = outputs.last_hidden_state          # (B, L, H)
        mask_expanded = (
            attention_mask.unsqueeze(-1)
            .expand(token_embeddings.size())
            .to(token_embeddings.dtype)
        )
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)  # (B, H)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)           # (B, H)
        mean_embeddings = sum_embeddings / sum_mask                           # (B, H)

        all_embeddings.append(mean_embeddings)   # stays on GPU

    # Single GPU → CPU transfer for the entire feature
    result = torch.cat(all_embeddings, dim=0).cpu().numpy()   # (N, H)

    # Zero-out originally empty texts
    if empty_indices:
        result[empty_indices] = 0.0

    return result
```

Key changes:
- `all_embeddings` accumulates GPU tensors instead of numpy arrays
- `.cpu().numpy()` called once after `torch.cat` instead of once per batch
- Empty text zeroing uses vectorised index assignment instead of a per-item loop
- `leave=False` on the inner tqdm so finished batch bars don't stack up in the terminal

---

## Bottleneck 6 — Lab group filtering re-scans the full labs DataFrame 13 times

The line `group_df = labs_df[labs_df["itemid"].isin(itemids)]` runs 13 times, each time
scanning all rows of `labs_df`. For a large labs parquet this is expensive.

### Fix — pre-group labs_df once before the loop

```python
# Before the lab group loop, pre-compute a dict of per-group DataFrames:
logger.info("Pre-grouping labs by panel…")
labs_df["_group"] = None
itemid_to_group = {
    itemid: group_name
    for group_name, itemids in lab_panel_config.items()
    for itemid in itemids
}
labs_df["_group"] = labs_df["itemid"].map(itemid_to_group)
labs_by_group = {
    group_name: grp.drop(columns=["_group"])
    for group_name, grp in labs_df.groupby("_group")
    if group_name in lab_panel_config
}
logger.info("  Pre-grouped %d lab rows into %d groups", len(labs_df), len(labs_by_group))

# Inside the loop, replace:
#   group_df = labs_df[labs_df["itemid"].isin(itemids)]
# with:
group_df = labs_by_group.get(group_name, pd.DataFrame())
```

---

## Summary of config changes to `preprocessing.yaml`

Add these two new optional keys with their defaults. `embed_features.py` should read them
with `config.get()` fallbacks so they are backwards-compatible:

```yaml
# Per-feature max token length is capped automatically inside embed_features.py.
# Override the global ceiling only if you have a specific reason.
BERT_MAX_LENGTH: 8192          # global hard ceiling (per-feature caps applied automatically)

# Batch size for BERT inference. embed_features.py scales this per feature automatically.
# Increase if your GPU has more than 11GB VRAM.
BERT_BATCH_SIZE: 32
```

No new config keys are required — all optimisations are applied automatically inside
`embed_features.py` based on the existing config values.

---

## Expected speedup

| Bottleneck fixed | Estimated speedup |
|---|---|
| Per-feature max_length caps (chief complaint: 8192→64) | 10–100× for short features |
| Dynamic padding per batch (`padding="longest"`) | 2–5× for mixed-length batches |
| Dynamic batch size scaling | 2–4× GPU utilisation improvement |
| Non-blocking GPU transfer | 10–20% latency reduction |
| Single GPU→CPU transfer per feature | 5–15% reduction |
| Pre-grouping labs DataFrame | 5–10× for the lab group loop |

The dominant gain by far is Bottleneck 1. Chief complaint and triage texts padded to 64
and 256 tokens respectively instead of 8192 will be roughly **50–100× faster** for those
two features alone.
