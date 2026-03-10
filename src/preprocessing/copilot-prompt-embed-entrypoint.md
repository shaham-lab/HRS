# GitHub Copilot Prompt — Add CLI entry point to embed_features.py

## Problem

`embed_features.py` currently only has a `run(config: dict)` function called by
`run_pipeline.py`. When `embed_job.sh` calls it directly as:

```bash
python ./src/preprocessing/embed_features.py --config config/preprocessing.yaml
```

...nothing happens and the script exits immediately because there is no `main()` and
no `if __name__ == "__main__"` block.

## Fix

Add a `main()` function and `if __name__ == "__main__": main()` block at the bottom
of `embed_features.py`, after the closing of `run()`.

The `main()` function must:
1. Parse `--config` CLI argument
2. Load and expand the YAML config (reusing the same logic as `run_pipeline.py`)
3. Set up logging in the same format as `run_pipeline.py`
4. Call `run(config)`

### Exact code to append at the end of `embed_features.py`, after line 311:

```python
def main() -> None:
    """
    CLI entry point for running embed_features standalone.

    Usage:
        python src/preprocessing/embed_features.py --config config/preprocessing.yaml

    This allows embed_features to be submitted as a dedicated SLURM job
    (embed_job.sh) independently of run_pipeline.py.
    """
    import argparse
    import yaml  # type: ignore

    parser = argparse.ArgumentParser(
        description="Embed CDSS text features using BERT.",
    )
    parser.add_argument(
        "--config",
        default="config/preprocessing.yaml",
        help="Path to preprocessing.yaml (default: config/preprocessing.yaml)",
    )
    args = parser.parse_args()

    # Set up logging in the same format as run_pipeline.py
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load config and expand ~ in path values
    if not os.path.exists(args.config):
        raise FileNotFoundError(
            f"Configuration file not found: {args.config}"
        )
    with open(args.config, "r", encoding="utf-8") as fh:
        config: dict = yaml.safe_load(fh)

    _PATH_KEYS = {
        "MIMIC_DATA_DIR", "MIMIC_NOTE_DIR", "MIMIC_ED_DIR",
        "PREPROCESSING_DIR", "FEATURES_DIR", "EMBEDDINGS_DIR",
        "CLASSIFICATIONS_DIR", "HASH_REGISTRY_PATH",
    }
    for key in _PATH_KEYS:
        if key in config and isinstance(config[key], str):
            config[key] = os.path.expanduser(config[key])

    logger.info("Loaded configuration from %s", args.config)
    run(config)


if __name__ == "__main__":
    main()
```

## No other changes

Do not modify `run()`, `_embed_texts()`, `_get_device()`, or any other existing
function. Only append `main()` and `if __name__ == "__main__": main()` at the end
of the file.

## Verification

After the change, this should work from the repo root:

```bash
python src/preprocessing/embed_features.py --config config/preprocessing.yaml
```

And `embed_job.sh` which contains:

```bash
python ./src/preprocessing/embed_features.py \
    --config config/preprocessing.yaml
```

will now correctly run the full embedding pipeline.
