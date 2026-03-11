# Fix: fastparquet fails to serialize embedding column in `embed_features.py`

## Problem

In `_worker()`, after calling `_embed_texts()`, the embedding matrix is assigned
to the DataFrame column using `list(embeddings)`. This converts the `(N, 768)`
numpy array into a Python list of numpy row-arrays, which fastparquet cannot
type-infer, causing the error:

```
Can't infer object conversion type: 0  [0.0, 0.0, 0.0, ...]
```

fastparquet requires each cell in an object-dtype embedding column to be an
explicit `np.ndarray` with a consistent dtype — not a generic Python list.

## Location

File: `src/preprocessing/embed_features.py`
Function: `_worker()`
The checkpoint loop inside the `for start in range(0, n_total, checkpoint_interval):` block.

**Find this line (only one occurrence):**
```python
batch_sh[embedding_col] = list(embeddings)
```

## Fix

Replace that single line with:

```python
batch_sh[embedding_col] = [
    np.asarray(embeddings[i], dtype=np.float32)
    for i in range(len(embeddings))
]
```

This ensures every cell is a `np.float32` array of shape `(768,)`, which
fastparquet can serialize correctly as an object-dtype column of fixed-length
arrays.

## What NOT to change

- Do not change `_embed_texts()` — it correctly returns a `(N, 768)` float32
  numpy array and should stay as-is.
- Do not change any other DataFrame assignments elsewhere in the file.
- Do not change the fastparquet write calls (`fp.write`).
- This is a one-line fix. No other logic changes are needed.

## Verification

After the fix, confirm the column dtype before the `fp.write` call by adding
a single assert (can be removed after testing):

```python
assert all(isinstance(v, np.ndarray) for v in batch_sh[embedding_col]), \
    f"Expected np.ndarray per cell, got {type(batch_sh[embedding_col].iloc[0])}"
```
