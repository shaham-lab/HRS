# Configuration System Documentation

## Overview

The P-CAFE project uses a hierarchical configuration system that allows flexible parameter management across different deployment environments and use cases.

## Configuration Hierarchy

The configuration system follows a three-tier hierarchy where each level can override the previous one:

1. **Base Configuration** (`base_config.json`) - Default values for all parameters
2. **User Configuration** (`user_config.json`) - User-specific overrides (optional)
3. **Command-Line Arguments** - Runtime overrides via CLI flags

### Configuration Priority

```
Command-Line Args > user_config.json > base_config.json
```

## File Structure

### base_config.json

Contains default configuration for all modules. This file should be committed to version control and represents the baseline configuration.

Example structure:
```json
{
  "user_specific_project_path": "${USER_SPECIFIC_VALUE}",
  "code_rel_path": "Integration\\",
  "input_rel_path": "input\\",
  "output_rel_path": "output\\",
  "embedder_gueser": {
    "num_epochs": 1000,
    "hidden_dim1": 64,
    "hidden-dim2": 32,
    "lr": 1e-4,
    "weight_decay": 0.001,
    "val_trials_wo_im": 10,
    "fraction_mask": 0.1,
    "run_validation": 10,
    "batch_size": 128,
    "text_embed_dim": 768,
    "reduced_dim": 20
  }
}
```

### user_config.json

Optional file for user-specific configuration overrides. This file can be added to `.gitignore` to keep personal settings private.

Example:
```json
{
  "user_specific_project_path": "/home/myuser/projects/P-CAFE",
  "embedder_gueser": {
    "num_epochs": 5000,
    "batch_size": 256
  }
}
```

In this example:
- `user_specific_project_path` is overridden
- `num_epochs` and `batch_size` are overridden
- All other values from `base_config.json` remain unchanged

## Usage

### Using Default Configuration

Simply run the script without any arguments:

```bash
python src/embedder_guesser.py
```

This will use values from `base_config.json`, overridden by `user_config.json` if it exists.

### Overriding with Command-Line Arguments

You can override any configuration parameter via command-line arguments:

```bash
python src/embedder_guesser.py --num_epochs 10000 --batch_size 64 --lr 0.0001
```

### Complete Example

Given these configurations:

**base_config.json:**
```json
{
  "embedder_gueser": {
    "num_epochs": 1000,
    "batch_size": 128,
    "lr": 1e-4
  }
}
```

**user_config.json:**
```json
{
  "embedder_gueser": {
    "num_epochs": 5000
  }
}
```

**Command:**
```bash
python src/embedder_guesser.py --batch_size 256
```

**Effective Configuration:**
- `num_epochs`: 5000 (from user_config.json)
- `batch_size`: 256 (from command-line)
- `lr`: 1e-4 (from base_config.json)

## Available Parameters for embedder_guesser

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--directory` | str | Project path | Directory for saved models |
| `--hidden-dim1` | int | 64 | First hidden layer dimension |
| `--hidden-dim2` | int | 32 | Second hidden layer dimension |
| `--lr` | float | 1e-4 | Learning rate |
| `--weight_decay` | float | 0.001 | L2 weight penalty |
| `--num_epochs` | int | 100000 | Number of training epochs |
| `--val_trials_wo_im` | int | 500 | Validation trials without improvement before stopping |
| `--fraction_mask` | float | 0 | Fraction of features to mask during training |
| `--run_validation` | int | 100 | Run validation after this many epochs |
| `--batch_size` | int | 128 | Training batch size |
| `--text_embed_dim` | int | 768 | Text embedding dimension |
| `--reduced_dim` | int | 20 | Reduced dimension for text embedding |
| `--save_dir` | str | 'guesser_eICU' | Directory to save models |
| `--data` | str | 'pcafe_utils.load_time_Series()' | Dataset loader function |

## Programmatic Usage

You can also use the configuration system in your own scripts:

```python
from load_config import load_hierarchical_config

# Load with defaults
config = load_hierarchical_config()

# Load with custom paths
config = load_hierarchical_config(
    base_config_path="my_base_config.json",
    user_config_path="my_user_config.json"
)

# Load with CLI overrides
cli_overrides = {
    "embedder_gueser": {
        "num_epochs": 50
    }
}
config = load_hierarchical_config(cli_args=cli_overrides)
```

## Best Practices

1. **Don't modify base_config.json for personal use** - Use `user_config.json` instead
2. **Keep user_config.json private** - Add it to `.gitignore` if it contains sensitive paths
3. **Use command-line args for experiments** - Quick tests and parameter sweeps
4. **Document changes** - If you modify `base_config.json`, update this documentation

## Migration from Old System

If you were using the old configuration system with hardcoded paths:

**Old:**
```python
with open(r'C:\Users\kashann\...\user_config_naama.json', 'r') as f:
    config = json.load(f)
```

**New:**
```python
from load_config import load_hierarchical_config
config = load_hierarchical_config()
```

Create a `user_config.json` in the `src/` directory with your personal settings, and the system will automatically pick it up.
