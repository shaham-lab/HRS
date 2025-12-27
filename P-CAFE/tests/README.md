# P-CAFE Unit Tests

This directory contains unit tests for the P-CAFE package.

## Running Tests

To run all tests:

```bash
cd P-CAFE
python -m pytest tests/
```

Or using unittest:

```bash
cd P-CAFE
python -m unittest discover tests/
```

To run specific test files:

```bash
python -m unittest tests/test_pcafe_utils.py
python -m unittest tests/test_load_config.py
```

## Test Coverage

### test_pcafe_utils.py
Tests for utility functions in `src/pcafe_utils.py`:
- `add_noise`: Adding Gaussian noise to features
- `balance_class`: Balancing classes with noise
- `balance_class_no_noise`: Balancing classes without noise
- `balance_class_multi`: Balancing multiple classes
- `map_multiple_features`: Feature index mapping
- `map_multiple_features_for_logistic_mimic`: MIMIC-III specific feature mapping

### test_load_config.py
Tests for configuration loading in `src/load_config.py`:
- `deep_update`: Recursive dictionary merging
- `load_hierarchical_config`: Hierarchical configuration loading
- `load_config`: Legacy configuration loading

## Test Structure

Each test file follows the standard unittest framework:
- Test classes inherit from `unittest.TestCase`
- Test methods start with `test_`
- setUp/tearDown methods for test fixtures
- Descriptive test names and docstrings

## Adding New Tests

When adding new functionality to P-CAFE:
1. Create corresponding test methods in the appropriate test file
2. Follow the existing naming conventions
3. Include docstrings describing what is being tested
4. Test edge cases and error conditions
5. Run tests to verify they pass

## Dependencies

Tests require:
- Python 3.14.2+
- numpy
- pandas
- unittest (standard library)
- pytest (optional, for advanced features)
