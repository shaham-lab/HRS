"""
MIMIC-IV Mortality Prediction Benchmark using PyHealth

This script demonstrates how to:
1. Load MIMIC-IV data from tables: diagnoses_icd, procedures_icd, prescriptions
2. Define the mortality_prediction_mimic4_fn task
3. Split the data by patients
4. Run a prediction task

Note: This script requires MIMIC-IV data to be available in the specified root directory.
"""

from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks import mortality_prediction_mimic4_fn
from pyhealth.datasets import split_by_patient
from pyhealth.models import Transformer
from pyhealth.trainer import Trainer
import torch


def main():
    """
    Main function to run the MIMIC-IV mortality prediction benchmark.
    """
    
    # Step 1: Load MIMIC-IV dataset
    # The dataset will automatically load the required tables
    print("Step 1: Loading MIMIC-IV dataset...")
    print("Tables to load: diagnoses_icd, procedures_icd, prescriptions")
    
    # TODO: Update the root path to point to your MIMIC-IV data directory
    # The MIMIC-IV data should be in CSV format with the standard MIMIC-IV structure
    mimic4_root = "/path/to/mimic4/data"  # Update this path
    
    try:
        # Initialize MIMIC4Dataset
        # This will load the specified tables from the MIMIC-IV database
        mimic4_dataset = MIMIC4Dataset(
            root=mimic4_root,
            tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
            code_mapping=None,  # Use default code mapping
            dev=False,  # Set to True for development/testing with smaller dataset
        )
        print(f"Dataset loaded successfully with {len(mimic4_dataset.patients)} patients")
        
        # Step 2: Define the mortality prediction task
        print("\nStep 2: Defining mortality prediction task...")
        
        # Apply the mortality prediction task function to the dataset
        # This function will create the labels and prepare the data for mortality prediction
        mimic4_sample_dataset = mimic4_dataset.set_task(mortality_prediction_mimic4_fn)
        print(f"Task defined successfully with {len(mimic4_sample_dataset)} samples")
        
        # Step 3: Split the data by patients
        print("\nStep 3: Splitting data by patients...")
        
        # Split the dataset into train, validation, and test sets
        # Splitting by patient ensures that all visits from the same patient
        # are in the same split (avoiding data leakage)
        train_dataset, val_dataset, test_dataset = split_by_patient(
            mimic4_sample_dataset,
            [0.8, 0.1, 0.1]  # 80% train, 10% validation, 10% test
        )
        
        print(f"Train set: {len(train_dataset)} samples")
        print(f"Validation set: {len(val_dataset)} samples")
        print(f"Test set: {len(test_dataset)} samples")
        
        # Step 4: Run prediction task
        print("\nStep 4: Setting up and running prediction task...")
        
        # Initialize a model (using Transformer as an example)
        # You can replace this with other PyHealth models:
        # - RNN, LSTM, GRU
        # - RETAIN
        # - CNN
        # - etc.
        model = Transformer(
            dataset=mimic4_sample_dataset,
            feature_keys=["conditions", "procedures", "drugs"],
            label_key="label",
            mode="binary",  # Binary classification for mortality prediction
        )
        
        print("Model initialized successfully")
        
        # Determine device (GPU if available, otherwise CPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            device=device,
        )
        
        print(f"Trainer initialized (device: {device})")
        
        # Train the model
        print("\nTraining model...")
        trainer.train(
            train_dataloader=train_dataset,
            val_dataloader=val_dataset,
            epochs=10,
            monitor="pr_auc_weighted",
        )
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        results = trainer.evaluate(test_dataset)
        
        print("\n" + "="*50)
        print("RESULTS")
        print("="*50)
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
        print("="*50)
        
    except FileNotFoundError as e:
        print(f"\nError: MIMIC-IV data not found at {mimic4_root}")
        print("Please update the 'mimic4_root' variable to point to your MIMIC-IV data directory")
        print(f"Error details: {e}")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("="*70)
    print("MIMIC-IV Mortality Prediction Benchmark")
    print("="*70)
    print("\nThis script demonstrates the PyHealth workflow for:")
    print("1. Loading MIMIC-IV data (diagnoses_icd, procedures_icd, prescriptions)")
    print("2. Defining the mortality prediction task")
    print("3. Splitting data by patients")
    print("4. Running the prediction task")
    print("\n" + "="*70 + "\n")
    
    main()
