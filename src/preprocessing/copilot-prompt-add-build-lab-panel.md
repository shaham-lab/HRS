# GitHub Copilot Prompt — Add build_lab_panel_config to run_pipeline.py

## Problem

`build_lab_panel_config` is missing from `run_pipeline.py`. It is a required
pipeline step that must run before `extract_labs` — it reads `d_labitems` and
writes `classifications/lab_panel_config.yaml`, which `extract_labs` and
`embed_features` both depend on. Because it is absent from `_FULL_ORDER`,
`lab_panel_config.yaml` is never created when running `--all`, causing all 13
lab group embeddings to be silently skipped.

## Three changes required in `src/preprocessing/run_pipeline.py`

### Change 1 — Add `--build_lab_panel_config` argparse argument

In the `main()` function, after the `--create_splits` argument block (around
line 117), add this new argument block:

```python
    parser.add_argument(
        "--build_lab_panel_config",
        action="store_true",
        help="Run build_lab_panel_config.py",
    )
```

### Change 2 — Add `build_lab_panel_config` to `_FULL_ORDER`

Find this block (around line 227):

```python
    # Full pipeline order
    _FULL_ORDER = (
        ["create_splits"]
        + _EXTRACT_MODULES
        + ["embed_features", "combine_dataset"]
    )
```

Replace it with:

```python
    # Full pipeline order
    _FULL_ORDER = (
        ["create_splits", "build_lab_panel_config"]
        + _EXTRACT_MODULES
        + ["embed_features", "combine_dataset"]
    )
```

`build_lab_panel_config` must come after `create_splits` and before
`_EXTRACT_MODULES` because `extract_labs` reads `lab_panel_config.yaml`.

### Change 3 — Update the pipeline plan log comment (optional but helpful)

The `_FULL_ORDER` list now has 11 steps instead of 10. No other changes needed —
the dynamic module runner `_run_module` and the skip/force logic all work
automatically for any module name in `_FULL_ORDER`.

## Verification

After the change, confirm:

```bash
python src/preprocessing/run_pipeline.py --help | grep build_lab_panel_config
# Should print:  --build_lab_panel_config  Run build_lab_panel_config.py

python src/preprocessing/run_pipeline.py \
    --config config/preprocessing.yaml \
    --modules build_lab_panel_config
# Should run and create data/preprocessing/classifications/lab_panel_config.yaml

ls data/preprocessing/classifications/lab_panel_config.yaml
# Should exist
```

And the full pipeline plan when running `--all` should now show 11 steps with
`build_lab_panel_config` as step 2:

```
Pipeline plan — 11 module(s) to run:
  1. create_splits
  2. build_lab_panel_config
  3. extract_demographics
  4. extract_diag_history
  ...
```

## No other changes

Do not modify any other function, argument, or logic in `run_pipeline.py`.
The only three changes are: add the argparse argument, update `_FULL_ORDER`,
and nothing else.
