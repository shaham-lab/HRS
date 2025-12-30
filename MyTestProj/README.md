# PyHealthTest - MIMIC-IV Mortality Prediction Benchmark

This folder contains a benchmark implementation for mortality prediction using MIMIC-IV data with the PyHealth library.

## Overview

The benchmark demonstrates the complete workflow for healthcare predictive modeling using PyHealth:

1. **Data Loading**: Load MIMIC-IV data from specific tables
2. **Task Definition**: Define the mortality prediction task
3. **Data Splitting**: Split data by patients to avoid data leakage
4. **Model Training**: Train and evaluate a prediction model

## Files

- `mimic4_mortality_benchmark.py`: Main script implementing the benchmark workflow
- `demo_workflow.py`: Demo script showing the expected workflow without requiring actual data
- `__init__.py`: Package initialization file
- `README.md`: This documentation file

## Requirements

The benchmark requires the following dependencies (already added to `environment.yml`):
- pyhealth
- torch
- pandas
- numpy
- scikit-learn

## Data Setup

This benchmark requires MIMIC-IV data to run. You need to:

1. Obtain access to MIMIC-IV dataset from PhysioNet: https://physionet.org/content/mimiciv/
2. Download the MIMIC-IV data
3. Extract the CSV files to a directory
4. Update the `mimic4_root` variable in `mimic4_mortality_benchmark.py` to point to your data directory

The script expects the following tables:
- `diagnoses_icd`: ICD diagnosis codes
- `procedures_icd`: ICD procedure codes  
- `prescriptions`: Medication prescriptions

## Usage

### Quick Demo (No Data Required)

To see the expected workflow and structure without actual MIMIC-IV data:

```bash
cd PyHealthTest
python demo_workflow.py
```

This will display the complete workflow with code snippets and expected outputs.

### Running the Full Benchmark

Once you have set up the MIMIC-IV data:

```bash
cd PyHealthTest
python mimic4_mortality_benchmark.py
```

## Workflow Details

### 1. Load MIMIC-IV Data

```python
mimic4_dataset = MIMIC4Dataset(
    root=mimic4_root,
    tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
    code_mapping=None,
    dev=False,
)
```

The `MIMIC4Dataset` class automatically:
- Loads the specified tables from CSV files
- Processes and structures the data for machine learning
- Creates patient-visit relationships

### 2. Define Mortality Prediction Task

```python
mimic4_sample_dataset = mimic4_dataset.set_task(mortality_prediction_mimic4_fn)
```

The `mortality_prediction_mimic4_fn` function:
- Creates binary labels for mortality prediction
- Prepares features from diagnoses, procedures, and prescriptions
- Structures the data for supervised learning

### 3. Split Data by Patients

```python
train_dataset, val_dataset, test_dataset = split_by_patient(
    mimic4_sample_dataset,
    [0.8, 0.1, 0.1]  # 80% train, 10% validation, 10% test
)
```

Splitting by patient ensures:
- All visits from the same patient stay in the same split
- Prevents data leakage between train/validation/test sets
- Provides more realistic evaluation

### 4. Run Prediction Task

```python
model = Transformer(
    dataset=mimic4_sample_dataset,
    feature_keys=["conditions", "procedures", "drugs"],
    label_key="label",
    mode="binary",
)

trainer = Trainer(model=model)
trainer.train(train_dataloader=train_dataset, val_dataloader=val_dataset, epochs=10)
results = trainer.evaluate(test_dataset)
```

The prediction task:
- Initializes a Transformer model (can be replaced with other PyHealth models)
- Trains the model on the training set
- Validates on the validation set
- Evaluates final performance on the test set

## Model Options

PyHealth supports various models that can be used instead of Transformer:

- **RNN-based**: RNN, LSTM, GRU
- **Attention-based**: Transformer, RETAIN
- **CNN-based**: CNN
- **Tree-based**: AdaCare
- **Graph-based**: SafeDrug, GAMENet

Simply replace the `Transformer` class with any of these models.

## Evaluation Metrics

The benchmark reports standard classification metrics:
- Accuracy
- Precision
- Recall
- F1-score
- AUROC (Area Under ROC Curve)
- AUPRC (Area Under Precision-Recall Curve)

## Notes

- The script includes error handling for missing data paths
- GPU acceleration is automatically used if available
- The `dev=False` parameter loads the full dataset; set to `True` for testing with a smaller subset
- Training time depends on dataset size and available hardware

## References

- PyHealth Documentation: https://pyhealth.readthedocs.io/
- MIMIC-IV Documentation: https://mimic.mit.edu/docs/iv/
- PyHealth Paper: https://arxiv.org/abs/2101.04209

## License

This benchmark implementation follows the same license as the parent HRS project.
