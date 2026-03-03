"""
Unit tests for the CDSS preprocessing pipeline modules.

These tests validate core logic without requiring MIMIC-IV data by using
synthetic DataFrames.
"""

import json
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

# Add the preprocessing directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'preprocessing'))


class TestCreateSplits(unittest.TestCase):
    """Tests for create_splits.py logic."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.mimic_dir = os.path.join(self.tmp, "mimic")
        self.hosp_dir = os.path.join(self.mimic_dir, "hosp")
        os.makedirs(self.hosp_dir)
        self.classifications_dir = os.path.join(self.tmp, "classifications")

        # Create a small synthetic admissions table
        admissions = pd.DataFrame({
            "subject_id": list(range(1, 21)),
            "hadm_id": list(range(100, 120)),
            "hospital_expire_flag": [1, 0, 0, 1, 0] * 4,
        })
        admissions.to_csv(
            os.path.join(self.hosp_dir, "admissions.csv"), index=False
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp)

    def test_splits_created(self):
        """Splits parquet is created with correct columns and values."""
        import create_splits
        config = {
            "MIMIC_DATA_DIR": self.mimic_dir,
            "SPLIT_TRAIN": 0.7,
            "SPLIT_DEV": 0.15,
            "SPLIT_TEST": 0.15,
            "CLASSIFICATIONS_DIR": self.classifications_dir,
        }
        create_splits.run(config)
        out_path = os.path.join(self.classifications_dir, "data_splits.parquet")
        self.assertTrue(os.path.exists(out_path))
        df = pd.read_parquet(out_path)
        self.assertIn("subject_id", df.columns)
        self.assertIn("hadm_id", df.columns)
        self.assertIn("split", df.columns)
        self.assertTrue(set(df["split"].unique()).issubset({"train", "dev", "test"}))

    def test_splits_cover_all_admissions(self):
        """Every admission is assigned a split."""
        import create_splits
        config = {
            "MIMIC_DATA_DIR": self.mimic_dir,
            "SPLIT_TRAIN": 0.7,
            "SPLIT_DEV": 0.15,
            "SPLIT_TEST": 0.15,
            "CLASSIFICATIONS_DIR": self.classifications_dir,
        }
        create_splits.run(config)
        df = pd.read_parquet(
            os.path.join(self.classifications_dir, "data_splits.parquet")
        )
        self.assertEqual(len(df), 20)
        self.assertFalse(df["split"].isna().any())

    def test_invalid_split_ratios_raise(self):
        """Split ratios that don't sum to 1 raise ValueError."""
        import create_splits
        config = {
            "MIMIC_DATA_DIR": self.mimic_dir,
            "SPLIT_TRAIN": 0.5,
            "SPLIT_DEV": 0.2,
            "SPLIT_TEST": 0.2,  # sums to 0.9
            "CLASSIFICATIONS_DIR": self.classifications_dir,
        }
        with self.assertRaises(ValueError):
            create_splits.run(config)

    def test_missing_config_key_raises(self):
        """Missing config key raises KeyError."""
        import create_splits
        with self.assertRaises(KeyError):
            create_splits.run({"MIMIC_DATA_DIR": self.mimic_dir})


class TestExtractDiagHistory(unittest.TestCase):
    """Tests for extract_diag_history.py logic."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.mimic_dir = os.path.join(self.tmp, "mimic")
        self.hosp_dir = os.path.join(self.mimic_dir, "hosp")
        os.makedirs(self.hosp_dir)
        self.features_dir = os.path.join(self.tmp, "features")

        # Admissions: patient 1 has two visits; patient 2 has one
        admissions = pd.DataFrame({
            "subject_id": [1, 1, 2],
            "hadm_id": [10, 20, 30],
            "admittime": pd.to_datetime(
                ["2020-01-01", "2020-06-01", "2020-03-01"]
            ),
        })
        admissions.to_csv(os.path.join(self.hosp_dir, "admissions.csv"), index=False)

        # Diagnoses
        diagnoses = pd.DataFrame({
            "subject_id": [1, 1, 2],
            "hadm_id": [10, 20, 30],
            "icd_code": ["A01", "B02", "C03"],
            "icd_version": [9, 9, 10],
        })
        diagnoses.to_csv(
            os.path.join(self.hosp_dir, "diagnoses_icd.csv"), index=False
        )

        d_icd = pd.DataFrame({
            "icd_code": ["A01", "B02", "C03"],
            "icd_version": [9, 9, 10],
            "long_title": ["Typhoid fever", "Bronchitis", "COVID-19"],
        })
        d_icd.to_csv(
            os.path.join(self.hosp_dir, "d_icd_diagnoses.csv"), index=False
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp)

    def test_prior_diagnoses_extracted(self):
        """Second visit for patient 1 has prior diagnosis from first visit."""
        import extract_diag_history
        config = {
            "MIMIC_DATA_DIR": self.mimic_dir,
            "FEATURES_DIR": self.features_dir,
        }
        extract_diag_history.run(config)
        out_path = os.path.join(self.features_dir, "diag_history_features.parquet")
        self.assertTrue(os.path.exists(out_path))
        df = pd.read_parquet(out_path)

        # hadm_id=20 is the second visit for patient 1 – should have "Typhoid fever"
        row_20 = df[df["hadm_id"] == 20].iloc[0]
        self.assertIn("Typhoid fever", row_20["diag_history_text"])

        # hadm_id=10 is the first visit for patient 1 – no prior history
        row_10 = df[df["hadm_id"] == 10].iloc[0]
        self.assertEqual(row_10["diag_history_text"], "")

        # Patient 2 only has one visit – no prior history
        row_30 = df[df["hadm_id"] == 30].iloc[0]
        self.assertEqual(row_30["diag_history_text"], "")


class TestExtractYData(unittest.TestCase):
    """Tests for extract_y_data.py logic."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.mimic_dir = os.path.join(self.tmp, "mimic")
        self.hosp_dir = os.path.join(self.mimic_dir, "hosp")
        os.makedirs(self.hosp_dir)
        self.classifications_dir = os.path.join(self.tmp, "classifications")

        # Two patients: patient 1 dies in first visit; patient 2 has a
        # readmission within 30 days; patient 3 has no readmission
        admissions = pd.DataFrame({
            "subject_id": [1, 2, 2, 3],
            "hadm_id": [10, 20, 21, 30],
            "admittime": pd.to_datetime([
                "2020-01-01",
                "2020-01-01",
                "2020-01-20",   # within 30 days of hadm_id=20 discharge
                "2020-01-01",
            ]),
            "dischtime": pd.to_datetime([
                "2020-01-10",
                "2020-01-10",
                "2020-01-25",
                "2020-01-05",
            ]),
            "hospital_expire_flag": [1, 0, 0, 0],
        })
        admissions.to_csv(os.path.join(self.hosp_dir, "admissions.csv"), index=False)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp)

    def test_y1_mortality(self):
        """Y1 should match hospital_expire_flag."""
        import extract_y_data
        config = {
            "MIMIC_DATA_DIR": self.mimic_dir,
            "CLASSIFICATIONS_DIR": self.classifications_dir,
        }
        extract_y_data.run(config)
        df = pd.read_parquet(
            os.path.join(self.classifications_dir, "y_labels.parquet")
        )
        self.assertEqual(df.loc[df["hadm_id"] == 10, "y1_mortality"].iloc[0], 1)
        self.assertEqual(df.loc[df["hadm_id"] == 20, "y1_mortality"].iloc[0], 0)

    def test_y2_readmission(self):
        """Patient 2 (hadm_id=20) has a readmission within 30 days."""
        import extract_y_data
        config = {
            "MIMIC_DATA_DIR": self.mimic_dir,
            "CLASSIFICATIONS_DIR": self.classifications_dir,
        }
        extract_y_data.run(config)
        df = pd.read_parquet(
            os.path.join(self.classifications_dir, "y_labels.parquet")
        )
        self.assertEqual(df.loc[df["hadm_id"] == 20, "y2_readmission"].iloc[0], 1.0)
        self.assertEqual(df.loc[df["hadm_id"] == 30, "y2_readmission"].iloc[0], 0.0)

    def test_y2_nan_for_deaths(self):
        """Y2 should be NaN for admissions where the patient died."""
        import extract_y_data
        config = {
            "MIMIC_DATA_DIR": self.mimic_dir,
            "CLASSIFICATIONS_DIR": self.classifications_dir,
        }
        extract_y_data.run(config)
        df = pd.read_parquet(
            os.path.join(self.classifications_dir, "y_labels.parquet")
        )
        self.assertTrue(pd.isna(df.loc[df["hadm_id"] == 10, "y2_readmission"].iloc[0]))


class TestExtractRadiology(unittest.TestCase):
    """Tests for extract_radiology.py text cleaning and extraction."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.mimic_dir = os.path.join(self.tmp, "mimic")
        self.hosp_dir = os.path.join(self.mimic_dir, "hosp")
        self.note_dir = os.path.join(self.mimic_dir, "note")
        os.makedirs(self.hosp_dir)
        os.makedirs(self.note_dir)
        self.features_dir = os.path.join(self.tmp, "features")

        admissions = pd.DataFrame({
            "subject_id": [1, 2],
            "hadm_id": [10, 20],
            "admittime": pd.to_datetime(["2020-01-01", "2020-02-01"]),
            "dischtime": pd.to_datetime(["2020-01-10", "2020-02-10"]),
        })
        admissions.to_csv(os.path.join(self.hosp_dir, "admissions.csv"), index=False)

        notes = pd.DataFrame({
            "subject_id": [1, 2],
            "hadm_id": [10, 20],
            "charttime": pd.to_datetime(["2020-01-05", "2020-02-05"]),
            "text": [
                "HEADER INFO\nEXAMINATION: Chest X-Ray\nFindings: Normal",
                "No marker here",
            ],
        })
        notes.to_csv(os.path.join(self.note_dir, "radiology.csv"), index=False)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp)

    def test_examination_marker_removed(self):
        """Text before EXAMINATION: should be stripped."""
        import extract_radiology
        config = {
            "MIMIC_DATA_DIR": self.mimic_dir,
            "FEATURES_DIR": self.features_dir,
        }
        extract_radiology.run(config)
        df = pd.read_parquet(
            os.path.join(self.features_dir, "radiology_features.parquet")
        )
        row = df[df["hadm_id"] == 10].iloc[0]
        self.assertTrue(row["radiology_text"].startswith("EXAMINATION:"))
        self.assertNotIn("HEADER INFO", row["radiology_text"])

    def test_no_marker_returns_full_text(self):
        """Text without EXAMINATION: should be returned unchanged."""
        import extract_radiology
        config = {
            "MIMIC_DATA_DIR": self.mimic_dir,
            "FEATURES_DIR": self.features_dir,
        }
        extract_radiology.run(config)
        df = pd.read_parquet(
            os.path.join(self.features_dir, "radiology_features.parquet")
        )
        row = df[df["hadm_id"] == 20].iloc[0]
        self.assertEqual(row["radiology_text"], "No marker here")


class TestExtractDischargeHistory(unittest.TestCase):
    """Tests for extract_discharge_history.py text cleaning."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.mimic_dir = os.path.join(self.tmp, "mimic")
        self.hosp_dir = os.path.join(self.mimic_dir, "hosp")
        self.note_dir = os.path.join(self.mimic_dir, "note")
        os.makedirs(self.hosp_dir)
        os.makedirs(self.note_dir)
        self.features_dir = os.path.join(self.tmp, "features")

        admissions = pd.DataFrame({
            "subject_id": [1, 1],
            "hadm_id": [10, 20],
            "admittime": pd.to_datetime(["2020-01-01", "2020-06-01"]),
        })
        admissions.to_csv(os.path.join(self.hosp_dir, "admissions.csv"), index=False)

        notes = pd.DataFrame({
            "subject_id": [1],
            "hadm_id": [10],
            "charttime": pd.to_datetime(["2020-01-09"]),
            "text": ["Some text before\nAllergies: None\nRest of note"],
        })
        notes.to_csv(os.path.join(self.note_dir, "discharge.csv"), index=False)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp)

    def test_allergies_marker_removal(self):
        """Text before 'Allergies:' is stripped from prior notes."""
        import extract_discharge_history
        config = {
            "MIMIC_DATA_DIR": self.mimic_dir,
            "FEATURES_DIR": self.features_dir,
        }
        extract_discharge_history.run(config)
        df = pd.read_parquet(
            os.path.join(self.features_dir, "discharge_history_features.parquet")
        )
        # hadm_id=20 is the second visit; prior note from hadm_id=10 should be present
        row = df[df["hadm_id"] == 20].iloc[0]
        self.assertIn("Allergies:", row["discharge_history_text"])
        self.assertNotIn("Some text before", row["discharge_history_text"])

    def test_first_visit_empty(self):
        """First visit has no prior discharge history."""
        import extract_discharge_history
        config = {
            "MIMIC_DATA_DIR": self.mimic_dir,
            "FEATURES_DIR": self.features_dir,
        }
        extract_discharge_history.run(config)
        df = pd.read_parquet(
            os.path.join(self.features_dir, "discharge_history_features.parquet")
        )
        row = df[df["hadm_id"] == 10].iloc[0]
        self.assertEqual(row["discharge_history_text"], "")


class TestCombineDataset(unittest.TestCase):
    """Tests for combine_dataset.py."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.features_dir = os.path.join(self.tmp, "features")
        self.embeddings_dir = os.path.join(self.tmp, "embeddings")
        self.classifications_dir = os.path.join(self.tmp, "classifications")
        os.makedirs(self.features_dir)
        os.makedirs(self.embeddings_dir)
        os.makedirs(self.classifications_dir)

        splits = pd.DataFrame({
            "subject_id": [1, 2],
            "hadm_id": [10, 20],
            "split": ["train", "dev"],
        })
        splits.to_parquet(
            os.path.join(self.classifications_dir, "data_splits.parquet"), index=False
        )

        labels = pd.DataFrame({
            "subject_id": [1, 2],
            "hadm_id": [10, 20],
            "y1_mortality": [0, 1],
            "y2_readmission": [1.0, float("nan")],
        })
        labels.to_parquet(
            os.path.join(self.classifications_dir, "y_labels.parquet"), index=False
        )

        demographics = pd.DataFrame({
            "subject_id": [1, 2],
            "hadm_id": [10, 20],
            "demographic_vec": [[1.0] * 8, [2.0] * 8],
        })
        demographics.to_parquet(
            os.path.join(self.features_dir, "demographics_features.parquet"),
            index=False,
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp)

    def test_final_dataset_created(self):
        """Final dataset parquet is created."""
        import combine_dataset
        config = {
            "FEATURES_DIR": self.features_dir,
            "EMBEDDINGS_DIR": self.embeddings_dir,
            "CLASSIFICATIONS_DIR": self.classifications_dir,
        }
        combine_dataset.run(config)
        out_path = os.path.join(self.tmp, "input", "final_cdss_dataset.parquet")
        # The output path depends on parent of classifications_dir
        actual_out = os.path.join(
            os.path.dirname(self.classifications_dir), "final_cdss_dataset.parquet"
        )
        self.assertTrue(os.path.exists(actual_out))

    def test_split_column_present(self):
        """Final dataset must contain a 'split' column."""
        import combine_dataset
        config = {
            "FEATURES_DIR": self.features_dir,
            "EMBEDDINGS_DIR": self.embeddings_dir,
            "CLASSIFICATIONS_DIR": self.classifications_dir,
        }
        combine_dataset.run(config)
        out = pd.read_parquet(
            os.path.join(
                os.path.dirname(self.classifications_dir), "final_cdss_dataset.parquet"
            )
        )
        self.assertIn("split", out.columns)
        self.assertIn("y1_mortality", out.columns)
        self.assertEqual(len(out), 2)


class TestConfigLoading(unittest.TestCase):
    """Tests for run_pipeline.py config loading."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp)

    def test_valid_config_loads(self):
        """Valid YAML config is loaded correctly."""
        import yaml
        cfg_path = os.path.join(self.tmp, "test.yaml")
        cfg = {
            "MIMIC_DATA_DIR": "/data/mimic",
            "SPLIT_TRAIN": 0.7,
            "SPLIT_DEV": 0.15,
            "SPLIT_TEST": 0.15,
            "BERT_MODEL_NAME": "bert-base-uncased",
            "BERT_MAX_LENGTH": 512,
            "BERT_BATCH_SIZE": 32,
            "BERT_DEVICE": "cpu",
            "FEATURES_DIR": "input/features",
            "EMBEDDINGS_DIR": "input/embeddings",
            "CLASSIFICATIONS_DIR": "input/classifications",
        }
        with open(cfg_path, "w") as fh:
            yaml.dump(cfg, fh)

        import run_pipeline
        loaded = run_pipeline._load_config(cfg_path)
        self.assertEqual(loaded["MIMIC_DATA_DIR"], "/data/mimic")
        self.assertAlmostEqual(loaded["SPLIT_TRAIN"], 0.7)

    def test_missing_config_raises(self):
        """Missing config file raises FileNotFoundError."""
        import run_pipeline
        with self.assertRaises(FileNotFoundError):
            run_pipeline._load_config(os.path.join(self.tmp, "nonexistent.yaml"))


if __name__ == "__main__":
    unittest.main()
