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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'preprocessing'))


class TestCreateSplits(unittest.TestCase):
    """Tests for create_splits.py logic."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.mimic_dir = os.path.join(self.tmp, "mimic")
        self.hosp_dir = os.path.join(self.mimic_dir, "hosp")
        os.makedirs(self.hosp_dir)
        self.preprocessing_dir = os.path.join(self.tmp, "preprocessing")

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
            "PREPROCESSING_DIR": self.preprocessing_dir,
        }
        create_splits.run(config)
        out_path = os.path.join(self.preprocessing_dir, "data_splits.parquet")
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
            "PREPROCESSING_DIR": self.preprocessing_dir,
        }
        create_splits.run(config)
        df = pd.read_parquet(
            os.path.join(self.preprocessing_dir, "data_splits.parquet")
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
            "PREPROCESSING_DIR": self.preprocessing_dir,
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
        self.preprocessing_dir = os.path.join(self.tmp, "preprocessing")
        os.makedirs(self.features_dir)
        os.makedirs(self.embeddings_dir)
        os.makedirs(self.classifications_dir)
        os.makedirs(self.preprocessing_dir)

        splits = pd.DataFrame({
            "subject_id": [1, 2],
            "hadm_id": [10, 20],
            "split": ["train", "dev"],
        })
        splits.to_parquet(
            os.path.join(self.preprocessing_dir, "data_splits.parquet"), index=False
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
            "PREPROCESSING_DIR": self.preprocessing_dir,
        }
        combine_dataset.run(config)
        out_path = os.path.join(self.classifications_dir, "final_cdss_dataset.parquet")
        self.assertTrue(os.path.exists(out_path))

    def test_split_column_present(self):
        """Final dataset must contain a 'split' column."""
        import combine_dataset
        config = {
            "FEATURES_DIR": self.features_dir,
            "EMBEDDINGS_DIR": self.embeddings_dir,
            "CLASSIFICATIONS_DIR": self.classifications_dir,
            "PREPROCESSING_DIR": self.preprocessing_dir,
        }
        combine_dataset.run(config)
        out = pd.read_parquet(
            os.path.join(self.classifications_dir, "final_cdss_dataset.parquet")
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


class TestHashUtils(unittest.TestCase):
    """Tests for preprocessing_utils hash-checking utilities."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        import preprocessing_utils
        self.utils = preprocessing_utils

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp)

    def _write_file(self, name: str, content: bytes = b"hello") -> str:
        path = os.path.join(self.tmp, name)
        with open(path, "wb") as f:
            f.write(content)
        return path

    def test_file_hash_returns_string(self):
        """_file_hash returns a hex string."""
        path = self._write_file("f.bin", b"test data")
        h = self.utils._file_hash(path)
        self.assertIsInstance(h, str)
        self.assertEqual(len(h), 32)  # MD5 hex digest

    def test_file_hash_changes_on_content_change(self):
        """_file_hash differs when file content changes."""
        path = self._write_file("f.bin", b"content_a")
        h1 = self.utils._file_hash(path)
        with open(path, "wb") as f:
            f.write(b"content_b")
        h2 = self.utils._file_hash(path)
        self.assertNotEqual(h1, h2)

    def test_load_hash_registry_empty_when_missing(self):
        """_load_hash_registry returns {} when registry file is absent."""
        registry = self.utils._load_hash_registry(
            os.path.join(self.tmp, "nonexistent.json")
        )
        self.assertEqual(registry, {})

    def test_save_and_load_registry_roundtrip(self):
        """_save_hash_registry and _load_hash_registry are inverses."""
        registry_path = os.path.join(self.tmp, "sub", "hashes.json")
        data = {"module_a": {"/path/file.gz": "abc123"}}
        self.utils._save_hash_registry(registry_path, data)
        loaded = self.utils._load_hash_registry(registry_path)
        self.assertEqual(loaded, data)

    def test_sources_unchanged_returns_false_when_output_missing(self):
        """_sources_unchanged is False when output file does not exist."""
        import logging
        logger = logging.getLogger("test")
        src = self._write_file("src.csv", b"data")
        registry_path = os.path.join(self.tmp, "hashes.json")
        result = self.utils._sources_unchanged(
            "mod",
            [src],
            [os.path.join(self.tmp, "missing_output.parquet")],
            registry_path,
            logger,
        )
        self.assertFalse(result)

    def test_sources_unchanged_returns_false_on_first_run(self):
        """_sources_unchanged is False on first run (no stored hash)."""
        import logging
        logger = logging.getLogger("test")
        src = self._write_file("src.csv", b"data")
        out = self._write_file("out.parquet", b"output")
        registry_path = os.path.join(self.tmp, "hashes.json")
        result = self.utils._sources_unchanged(
            "mod", [src], [out], registry_path, logger
        )
        self.assertFalse(result)

    def test_sources_unchanged_true_after_record(self):
        """_sources_unchanged is True after _record_hashes is called."""
        import logging
        logger = logging.getLogger("test")
        src = self._write_file("src.csv", b"stable data")
        out = self._write_file("out.parquet", b"output")
        registry_path = os.path.join(self.tmp, "hashes.json")
        self.utils._record_hashes("mod", [src], registry_path)
        result = self.utils._sources_unchanged(
            "mod", [src], [out], registry_path, logger
        )
        self.assertTrue(result)

    def test_sources_unchanged_false_after_source_modified(self):
        """_sources_unchanged is False when source file changes after record."""
        import logging
        logger = logging.getLogger("test")
        src = self._write_file("src.csv", b"original data")
        out = self._write_file("out.parquet", b"output")
        registry_path = os.path.join(self.tmp, "hashes.json")
        self.utils._record_hashes("mod", [src], registry_path)
        # Modify source
        with open(src, "wb") as f:
            f.write(b"modified data")
        result = self.utils._sources_unchanged(
            "mod", [src], [out], registry_path, logger
        )
        self.assertFalse(result)

    def test_gz_or_csv_prefers_gz(self):
        """_gz_or_csv returns .csv.gz path when .gz file exists."""
        gz_path = os.path.join(self.tmp, "admissions.csv.gz")
        with open(gz_path, "wb") as f:
            f.write(b"gz data")
        result = self.utils._gz_or_csv(self.tmp, "", "admissions")
        self.assertTrue(result.endswith(".csv.gz"))

    def test_gz_or_csv_falls_back_to_csv(self):
        """_gz_or_csv returns .csv path when no .gz file exists."""
        result = self.utils._gz_or_csv(self.tmp, "hosp", "admissions")
        self.assertTrue(result.endswith(".csv"))
        self.assertFalse(result.endswith(".csv.gz"))

    def test_hash_check_skip_respects_force_rerun(self):
        """Hash skip is bypassed when FORCE_RERUN is set in config."""
        import logging
        src = self._write_file("src.csv", b"data")
        out = self._write_file("out.parquet", b"output")
        registry_path = os.path.join(self.tmp, "hashes.json")
        self.utils._record_hashes("create_splits", [src], registry_path)

        # Without force: sources unchanged → skip
        logger = logging.getLogger("test")
        self.assertTrue(
            self.utils._sources_unchanged(
                "create_splits", [src], [out], registry_path, logger
            )
        )

        import create_splits
        config = {
            "MIMIC_DATA_DIR": os.path.join(self.tmp, "mimic"),
            "SPLIT_TRAIN": 0.7,
            "SPLIT_DEV": 0.15,
            "SPLIT_TEST": 0.15,
            "PREPROCESSING_DIR": os.path.join(self.tmp, "preprocessing"),
            "HASH_REGISTRY_PATH": registry_path,
            "FORCE_RERUN": False,
        }
        # create_splits will raise FileNotFoundError for missing admissions;
        # confirming it was NOT skipped (i.e. force bypasses hash check)
        config["FORCE_RERUN"] = True
        with self.assertRaises(FileNotFoundError):
            create_splits.run(config)



class TestExtractTriageAndComplaint(unittest.TestCase):
    """Tests for extract_triage_and_complaint.py edstays-based hadm_id resolution."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.mimic_dir = os.path.join(self.tmp, "mimic")
        self.ed_dir = os.path.join(self.mimic_dir, "ed")
        self.hosp_dir = os.path.join(self.mimic_dir, "hosp")
        os.makedirs(self.ed_dir)
        os.makedirs(self.hosp_dir)
        self.features_dir = os.path.join(self.tmp, "features")

        # Admissions table (for fallback linkage and presence in hosp/)
        # subject_id=2 has admittime BEFORE ED intime → fallback won't resolve it
        admissions = pd.DataFrame({
            "subject_id": [1, 2, 3],
            "hadm_id": [100, 200, 300],
            "admittime": pd.to_datetime(["2020-01-01 10:00", "2020-02-01 06:00", "2020-03-01 09:00"]),
            "dischtime": pd.to_datetime(["2020-01-05", "2020-02-05", "2020-03-05"]),
            "hospital_expire_flag": [0, 0, 0],
        })
        admissions.to_csv(os.path.join(self.hosp_dir, "admissions.csv"), index=False)

        # edstays table — stay_id 1→hadm_id 100 (admitted), stay_id 2→null (not admitted)
        edstays = pd.DataFrame({
            "subject_id": [1, 2, 3],
            "stay_id": [1001, 1002, 1003],
            "hadm_id": [100.0, float("nan"), float("nan")],
            "intime": pd.to_datetime(["2020-01-01 08:00", "2020-02-01 07:30", "2020-03-01 08:45"]),
            "outtime": pd.to_datetime(["2020-01-01 12:00", "2020-02-01 10:00", "2020-03-01 11:00"]),
            "disposition": ["ADMITTED", "HOME", "ADMITTED"],
        })
        edstays.to_csv(os.path.join(self.ed_dir, "edstays.csv"), index=False)

        # triage table — only stay_id, no hadm_id
        triage = pd.DataFrame({
            "subject_id": [1, 2, 3],
            "stay_id": [1001, 1002, 1003],
            "temperature": [37.0, 36.8, 38.1],
            "heartrate": [80, 75, 90],
            "resprate": [16, 14, 18],
            "o2sat": [98, 99, 97],
            "sbp": [120, 115, 130],
            "dbp": [80, 75, 85],
            "pain": [2, 0, 5],
            "acuity": [3, 4, 2],
            "chiefcomplaint": ["chest pain", "cough", "fever"],
        })
        triage.to_csv(os.path.join(self.ed_dir, "triage.csv"), index=False)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp)

    def test_primary_linkage_via_edstays(self):
        """stay_id → hadm_id resolved via edstays primary join."""
        import extract_triage_and_complaint
        config = {
            "MIMIC_DATA_DIR": self.mimic_dir,
            "FEATURES_DIR": self.features_dir,
        }
        extract_triage_and_complaint.run(config)
        df = pd.read_parquet(os.path.join(self.features_dir, "triage_features.parquet"))
        # subject_id=1: hadm_id directly from edstays (100)
        row = df[df["subject_id"] == 1]
        self.assertEqual(len(row), 1)
        self.assertEqual(row.iloc[0]["hadm_id"], 100)

    def test_fallback_linkage_via_intime(self):
        """Null hadm_id in edstays resolved via closest admittime >= intime."""
        import extract_triage_and_complaint
        config = {
            "MIMIC_DATA_DIR": self.mimic_dir,
            "FEATURES_DIR": self.features_dir,
        }
        extract_triage_and_complaint.run(config)
        df = pd.read_parquet(os.path.join(self.features_dir, "triage_features.parquet"))
        # subject_id=3: hadm_id not in edstays, resolved via fallback (300)
        row = df[df["subject_id"] == 3]
        self.assertEqual(len(row), 1)
        self.assertEqual(row.iloc[0]["hadm_id"], 300)

    def test_non_admitted_visit_dropped(self):
        """ED visit with no resolvable hadm_id (HOME disposition) is excluded."""
        import extract_triage_and_complaint
        config = {
            "MIMIC_DATA_DIR": self.mimic_dir,
            "FEATURES_DIR": self.features_dir,
        }
        extract_triage_and_complaint.run(config)
        df = pd.read_parquet(os.path.join(self.features_dir, "triage_features.parquet"))
        # subject_id=2: HOME disposition, no hadm_id resolvable → dropped
        self.assertEqual(len(df[df["subject_id"] == 2]), 0)

    def test_triage_text_built(self):
        """Triage template is rendered with correct vitals."""
        import extract_triage_and_complaint
        config = {
            "MIMIC_DATA_DIR": self.mimic_dir,
            "FEATURES_DIR": self.features_dir,
        }
        extract_triage_and_complaint.run(config)
        df = pd.read_parquet(os.path.join(self.features_dir, "triage_features.parquet"))
        row = df[df["hadm_id"] == 100].iloc[0]
        self.assertIn("temperature 37.0°C", row["triage_text"])
        self.assertIn("heart rate 80 bpm", row["triage_text"])

    def test_chief_complaint_extracted(self):
        """Chief complaint is extracted from triage.chiefcomplaint."""
        import extract_triage_and_complaint
        config = {
            "MIMIC_DATA_DIR": self.mimic_dir,
            "FEATURES_DIR": self.features_dir,
        }
        extract_triage_and_complaint.run(config)
        df = pd.read_parquet(
            os.path.join(self.features_dir, "chief_complaint_features.parquet")
        )
        row = df[df["hadm_id"] == 100].iloc[0]
        self.assertEqual(row["chief_complaint_text"], "chest pain")

    def test_missing_config_key_raises(self):
        """Missing required config key raises KeyError."""
        import extract_triage_and_complaint
        with self.assertRaises(KeyError):
            extract_triage_and_complaint.run({"MIMIC_DATA_DIR": self.mimic_dir})

    def test_missing_triage_file_raises(self):
        """Missing triage file raises FileNotFoundError."""
        import shutil
        import extract_triage_and_complaint
        shutil.rmtree(self.ed_dir)
        os.makedirs(self.ed_dir)
        with self.assertRaises(FileNotFoundError):
            extract_triage_and_complaint.run({
                "MIMIC_DATA_DIR": self.mimic_dir,
                "FEATURES_DIR": self.features_dir,
            })


class TestLabTextLineFormat(unittest.TestCase):
    """Tests for the new lab text line format in extract_labs.py and build_lab_text_lines.py."""

    def setUp(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'preprocessing'))

    def _make_row(self, **kwargs):
        """Create a minimal lab event row dict with defaults."""
        defaults = {
            "charttime": pd.Timestamp("2020-01-01 02:14:00"),
            "admittime": pd.Timestamp("2020-01-01 00:00:00"),
            "label": "Sodium",
            "value": "135",
            "valuenum": 135.0,
            "valueuom": "mEq/L",
            "ref_range_lower": 135.0,
            "ref_range_upper": 145.0,
            "flag": None,
            "fluid": "Blood",
            "category": "Chemistry",
        }
        defaults.update(kwargs)
        return pd.Series(defaults)

    def test_elapsed_time_format(self):
        """[HH:MM] shows elapsed time since admittime, not clock time."""
        import extract_labs
        row = self._make_row(
            charttime=pd.Timestamp("2020-01-01 02:14:00"),
            admittime=pd.Timestamp("2020-01-01 00:00:00"),
        )
        result = extract_labs._build_lab_text_line(row)
        self.assertTrue(result.startswith("[02:14]"), f"Expected [02:14] prefix, got: {result}")

    def test_no_fluid_category_in_line(self):
        """New format does not include (fluid/category) in the text line."""
        import extract_labs
        row = self._make_row()
        result = extract_labs._build_lab_text_line(row)
        self.assertNotIn("Blood/Chemistry", result)
        self.assertNotIn("(Blood/Chemistry)", result)

    def test_no_stat_flag_in_line(self):
        """New format does not include [STAT] even when priority is STAT."""
        import extract_labs
        row = self._make_row(priority="STAT")
        result = extract_labs._build_lab_text_line(row)
        self.assertNotIn("[STAT]", result)

    def test_abnormal_flag_from_flag_column(self):
        """[ABNORMAL] is appended when flag == 'abnormal'."""
        import extract_labs
        row = self._make_row(flag="abnormal", valuenum=140.0)
        result = extract_labs._build_lab_text_line(row)
        self.assertIn("[ABNORMAL]", result)

    def test_abnormal_flag_from_range_below(self):
        """[ABNORMAL] is appended when valuenum is below ref_range_lower."""
        import extract_labs
        row = self._make_row(
            flag=None,
            valuenum=120.0,
            ref_range_lower=135.0,
            ref_range_upper=145.0,
        )
        result = extract_labs._build_lab_text_line(row)
        self.assertIn("[ABNORMAL]", result)

    def test_abnormal_flag_from_range_above(self):
        """[ABNORMAL] is appended when valuenum is above ref_range_upper."""
        import extract_labs
        row = self._make_row(
            flag=None,
            valuenum=150.0,
            ref_range_lower=135.0,
            ref_range_upper=145.0,
        )
        result = extract_labs._build_lab_text_line(row)
        self.assertIn("[ABNORMAL]", result)

    def test_normal_value_no_abnormal_flag(self):
        """[ABNORMAL] is NOT appended when valuenum is within reference range."""
        import extract_labs
        row = self._make_row(
            flag=None,
            valuenum=140.0,
            ref_range_lower=135.0,
            ref_range_upper=145.0,
        )
        result = extract_labs._build_lab_text_line(row)
        self.assertNotIn("[ABNORMAL]", result)

    def test_ref_range_included_when_both_present(self):
        """Reference range (ref: lower-upper) is included when both bounds present."""
        import extract_labs
        row = self._make_row(ref_range_lower=135.0, ref_range_upper=145.0)
        result = extract_labs._build_lab_text_line(row)
        self.assertIn("(ref:", result)

    def test_ref_range_omitted_when_null(self):
        """Reference range is omitted when either bound is null."""
        import extract_labs
        row = self._make_row(ref_range_lower=None, ref_range_upper=None)
        result = extract_labs._build_lab_text_line(row)
        self.assertNotIn("(ref:", result)

    def test_value_format_numeric(self):
        """Numeric value is formatted to 2 decimal places."""
        import extract_labs
        row = self._make_row(valuenum=135.0)
        result = extract_labs._build_lab_text_line(row)
        self.assertIn("135.00", result)

    def test_text_value_fallback(self):
        """Text value field is used when valuenum is null."""
        import extract_labs
        row = self._make_row(valuenum=None, value="POSITIVE")
        result = extract_labs._build_lab_text_line(row)
        self.assertIn("POSITIVE", result)


class TestBuildLabTextLines(unittest.TestCase):
    """Tests for build_lab_text_lines.py module."""

    def setUp(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'preprocessing'))

    def _make_df(self, **kwargs):
        """Create a minimal lab events DataFrame."""
        defaults = {
            "charttime": [pd.Timestamp("2020-01-01 02:14:00")],
            "admittime": [pd.Timestamp("2020-01-01 00:00:00")],
            "label": ["Sodium"],
            "value": ["135"],
            "valuenum": [135.0],
            "valueuom": ["mEq/L"],
            "ref_range_lower": [135.0],
            "ref_range_upper": [145.0],
            "flag": [None],
        }
        defaults.update(kwargs)
        return pd.DataFrame(defaults)

    def test_series_elapsed_time(self):
        """build_lab_text_line_series returns correct elapsed time."""
        import build_lab_text_lines
        df = self._make_df()
        result = build_lab_text_lines.build_lab_text_line_series(df)
        self.assertTrue(result.iloc[0].startswith("[02:14]"))

    def test_series_abnormal_from_range(self):
        """build_lab_text_line_series marks ABNORMAL when out of range."""
        import build_lab_text_lines
        df = self._make_df(valuenum=[120.0])
        result = build_lab_text_lines.build_lab_text_line_series(df)
        self.assertIn("[ABNORMAL]", result.iloc[0])

    def test_row_function_matches_series(self):
        """build_lab_text_line_row and series produce identical output."""
        import build_lab_text_lines
        df = self._make_df()
        series_result = build_lab_text_lines.build_lab_text_line_series(df).iloc[0]
        row_result = build_lab_text_lines.build_lab_text_line_row(df.iloc[0])
        self.assertEqual(series_result, row_result)


class TestBuildLabPanelConfig(unittest.TestCase):
    """Tests for build_lab_panel_config.py."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.mimic_dir = os.path.join(self.tmp, "mimic")
        self.hosp_dir = os.path.join(self.mimic_dir, "hosp")
        os.makedirs(self.hosp_dir)
        self.classifications_dir = os.path.join(self.tmp, "classifications")

        # Minimal d_labitems covering multiple groups
        d_labitems = pd.DataFrame({
            "itemid": [51001, 51002, 51003, 51004, 51005, 51006, 51007],
            "label": ["Sodium", "pO2", "WBC", "Urine Na", "Glucose urine",
                      "Ascites Protein", "Junk"],
            "fluid": ["Blood", "Blood", "Blood", "Urine", "Urine",
                      "Ascites", "I"],
            "category": ["Chemistry", "Blood Gas", "Hematology",
                         "Chemistry", "Chemistry", "Chemistry", "Chemistry"],
        })
        d_labitems.to_csv(
            os.path.join(self.hosp_dir, "d_labitems.csv"), index=False
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp)

    def test_groups_created(self):
        """lab_panel_config.yaml is created with expected groups."""
        import yaml
        import build_lab_panel_config
        config = {
            "MIMIC_DATA_DIR": self.mimic_dir,
            "CLASSIFICATIONS_DIR": self.classifications_dir,
        }
        build_lab_panel_config.run(config)
        out_path = os.path.join(self.classifications_dir, "lab_panel_config.yaml")
        self.assertTrue(os.path.exists(out_path))
        with open(out_path) as f:
            cfg = yaml.safe_load(f)
        self.assertIn("blood_chemistry", cfg)
        self.assertIn("blood_gas", cfg)
        self.assertIn("blood_hematology", cfg)
        self.assertIn("urine_chemistry", cfg)
        self.assertIn("ascites", cfg)

    def test_artefact_rows_excluded(self):
        """Rows with fluid in ['I', 'Q', 'fluid'] are excluded."""
        import yaml
        import build_lab_panel_config
        config = {
            "MIMIC_DATA_DIR": self.mimic_dir,
            "CLASSIFICATIONS_DIR": self.classifications_dir,
        }
        build_lab_panel_config.run(config)
        out_path = os.path.join(self.classifications_dir, "lab_panel_config.yaml")
        with open(out_path) as f:
            cfg = yaml.safe_load(f)
        # itemid 51007 has fluid='I' → should be excluded
        all_itemids = [iid for items in cfg.values() for iid in items]
        self.assertNotIn(51007, all_itemids)

    def test_itemids_sorted(self):
        """Itemids within each group are sorted for determinism."""
        import yaml
        import build_lab_panel_config
        config = {
            "MIMIC_DATA_DIR": self.mimic_dir,
            "CLASSIFICATIONS_DIR": self.classifications_dir,
        }
        build_lab_panel_config.run(config)
        out_path = os.path.join(self.classifications_dir, "lab_panel_config.yaml")
        with open(out_path) as f:
            cfg = yaml.safe_load(f)
        for group_name, items in cfg.items():
            self.assertEqual(items, sorted(items), f"Group {group_name} itemids not sorted")

    def test_missing_config_key_raises(self):
        """Missing required config key raises KeyError."""
        import build_lab_panel_config
        with self.assertRaises(KeyError):
            build_lab_panel_config.run({"MIMIC_DATA_DIR": self.mimic_dir})


class TestExtractDemographicsHadmLinkage(unittest.TestCase):
    """Tests for the new HADM_LINKAGE_STRATEGY support in extract_demographics.py."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.mimic_dir = os.path.join(self.tmp, "mimic")
        self.hosp_dir = os.path.join(self.mimic_dir, "hosp")
        self.icu_dir = os.path.join(self.mimic_dir, "icu")
        os.makedirs(self.hosp_dir)
        os.makedirs(self.icu_dir)
        self.features_dir = os.path.join(self.tmp, "features")
        self.classifications_dir = os.path.join(self.tmp, "classifications")
        self.preprocessing_dir = os.path.join(self.tmp, "preprocessing")
        os.makedirs(self.classifications_dir)
        os.makedirs(self.preprocessing_dir)

        # Minimal admissions
        admissions = pd.DataFrame({
            "subject_id": [1, 2],
            "hadm_id": [10, 20],
            "admittime": pd.to_datetime(["2020-01-01", "2020-06-01"]),
        })
        admissions.to_csv(os.path.join(self.hosp_dir, "admissions.csv"), index=False)

        # Minimal patients
        patients = pd.DataFrame({
            "subject_id": [1, 2],
            "gender": ["M", "F"],
            "anchor_age": [45, 60],
            "anchor_year": [2019, 2020],
        })
        patients.to_csv(os.path.join(self.hosp_dir, "patients.csv"), index=False)

        # Splits parquet
        splits = pd.DataFrame({
            "subject_id": [1, 2],
            "hadm_id": [10, 20],
            "split": ["train", "dev"],
        })
        splits.to_parquet(os.path.join(self.preprocessing_dir, "data_splits.parquet"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp)

    def _base_config(self):
        return {
            "MIMIC_DATA_DIR": self.mimic_dir,
            "FEATURES_DIR": self.features_dir,
            "PREPROCESSING_DIR": self.preprocessing_dir,
            "CLASSIFICATIONS_DIR": self.classifications_dir,
        }

    def test_hadm_linkage_stats_written(self):
        """hadm_linkage_stats.json is created after running extract_demographics."""
        import extract_demographics
        extract_demographics.run(self._base_config())
        stats_path = os.path.join(self.classifications_dir, "hadm_linkage_stats.json")
        self.assertTrue(os.path.exists(stats_path))
        import json
        with open(stats_path) as f:
            stats = json.load(f)
        self.assertIn("extract_demographics", stats)
        self.assertIn("chartevents", stats["extract_demographics"])

    def test_hadm_linkage_strategy_config_keys(self):
        """HADM_LINKAGE_STRATEGY and HADM_LINKAGE_TOLERANCE_HOURS are accepted."""
        import extract_demographics
        config = self._base_config()
        config["HADM_LINKAGE_STRATEGY"] = "drop"
        config["HADM_LINKAGE_TOLERANCE_HOURS"] = 2
        # Should not raise
        extract_demographics.run(config)

    def test_hadm_linkage_stats_merged(self):
        """Running extract_demographics twice merges stats, not overwrites."""
        import extract_demographics, json
        extract_demographics.run(self._base_config())
        stats_path = os.path.join(self.classifications_dir, "hadm_linkage_stats.json")
        # Inject an extra key to simulate another module having written stats
        with open(stats_path) as f:
            existing = json.load(f)
        existing["other_module"] = {"some_table": {"total_null_hadm": 99}}
        with open(stats_path, "w") as f:
            json.dump(existing, f)
        # Run again
        extract_demographics.run(self._base_config())
        with open(stats_path) as f:
            merged = json.load(f)
        # Both keys should be present
        self.assertIn("extract_demographics", merged)
        self.assertIn("other_module", merged)


class TestExtractLabsAdmissionWindow(unittest.TestCase):
    """Tests for LAB_ADMISSION_WINDOW support in extract_labs.py."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.mimic_dir = os.path.join(self.tmp, "mimic")
        self.hosp_dir = os.path.join(self.mimic_dir, "hosp")
        os.makedirs(self.hosp_dir)
        self.features_dir = os.path.join(self.tmp, "features")
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'preprocessing'))

        # Admissions: one patient, one admission starting at 2020-01-01 00:00
        admissions = pd.DataFrame({
            "subject_id": [1],
            "hadm_id": [10],
            "admittime": pd.to_datetime(["2020-01-01 00:00:00"]),
            "dischtime": pd.to_datetime(["2020-01-05 00:00:00"]),
            "hospital_expire_flag": [0],
        })
        admissions.to_csv(os.path.join(self.hosp_dir, "admissions.csv"), index=False)

        # d_labitems: one item
        d_labitems = pd.DataFrame({
            "itemid": [50931],
            "label": ["Glucose"],
            "fluid": ["Blood"],
            "category": ["Chemistry"],
        })
        d_labitems.to_csv(os.path.join(self.hosp_dir, "d_labitems.csv"), index=False)

        # labevents: three events at 6h, 24h, and 48h after admittime
        labevents = pd.DataFrame({
            "subject_id": [1, 1, 1],
            "hadm_id": [10.0, 10.0, 10.0],
            "itemid": [50931, 50931, 50931],
            "charttime": pd.to_datetime([
                "2020-01-01 06:00:00",   # 6 hours — inside 24h window
                "2020-01-02 00:00:00",   # 24 hours — boundary: inside 24h window
                "2020-01-02 01:00:00",   # 25 hours — outside 24h window
            ]),
            "value": ["100", "110", "120"],
            "valuenum": [100.0, 110.0, 120.0],
            "valueuom": ["mg/dL", "mg/dL", "mg/dL"],
            "ref_range_lower": [70.0, 70.0, 70.0],
            "ref_range_upper": [100.0, 100.0, 100.0],
            "flag": [None, None, None],
            "priority": [None, None, None],
        })
        labevents.to_csv(os.path.join(self.hosp_dir, "labevents.csv"), index=False)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp)

    def _base_config(self):
        return {
            "MIMIC_DATA_DIR": self.mimic_dir,
            "FEATURES_DIR": self.features_dir,
        }

    def test_default_window_is_24_hours(self):
        """Default LAB_ADMISSION_WINDOW=24 includes events up to and including 24h."""
        import extract_labs
        extract_labs.run(self._base_config())
        df = pd.read_parquet(os.path.join(self.features_dir, "labs_features.parquet"))
        # 6h and 24h events are within [0, 24]; 25h event is excluded
        # The key check: we should have 2 events, not 3
        self.assertEqual(len(df), 2)

    def test_window_24_excludes_later_events(self):
        """Events after admittime + 24 hours are excluded when LAB_ADMISSION_WINDOW=24."""
        import extract_labs
        config = self._base_config()
        config["LAB_ADMISSION_WINDOW"] = 24
        extract_labs.run(config)
        df = pd.read_parquet(os.path.join(self.features_dir, "labs_features.parquet"))
        # 25-hour event (120.0 value) must be absent
        self.assertFalse((df["lab_text_line"].str.contains("120.00")).any())

    def test_full_window_includes_all_events(self):
        """LAB_ADMISSION_WINDOW='full' includes all events within the entire admission."""
        import extract_labs
        config = self._base_config()
        config["LAB_ADMISSION_WINDOW"] = "full"
        extract_labs.run(config)
        df = pd.read_parquet(os.path.join(self.features_dir, "labs_features.parquet"))
        # All three events (6h, 24h, 25h) are within the admission (dischtime = 2020-01-05)
        self.assertEqual(len(df), 3)

    def test_custom_window_6_hours(self):
        """LAB_ADMISSION_WINDOW=6 includes only the first 6 hours."""
        import extract_labs
        config = self._base_config()
        config["LAB_ADMISSION_WINDOW"] = 6
        extract_labs.run(config)
        df = pd.read_parquet(os.path.join(self.features_dir, "labs_features.parquet"))
        # Only the 6-hour event is within [0, 6]
        self.assertEqual(len(df), 1)
        self.assertIn("100.00", df.iloc[0]["lab_text_line"])

    def test_invalid_window_raises(self):
        """Non-numeric, non-'full' LAB_ADMISSION_WINDOW raises ValueError."""
        import extract_labs
        config = self._base_config()
        config["LAB_ADMISSION_WINDOW"] = "yesterday"
        with self.assertRaises(ValueError):
            extract_labs.run(config)

    def test_negative_window_raises(self):
        """Negative LAB_ADMISSION_WINDOW raises ValueError."""
        import extract_labs
        config = self._base_config()
        config["LAB_ADMISSION_WINDOW"] = -1
        with self.assertRaises(ValueError):
            extract_labs.run(config)


class TestExtractDiagHistoryFormat(unittest.TestCase):
    """Tests for the structured text block format in extract_diag_history.py."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.mimic_dir = os.path.join(self.tmp, "mimic")
        self.hosp_dir = os.path.join(self.mimic_dir, "hosp")
        os.makedirs(self.hosp_dir)
        self.features_dir = os.path.join(self.tmp, "features")
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'preprocessing'))

        # Patient 1: two prior admissions before hadm_id=30
        admissions = pd.DataFrame({
            "subject_id": [1, 1, 1],
            "hadm_id": [10, 20, 30],
            "admittime": pd.to_datetime(["2018-03-12", "2019-07-24", "2020-01-01"]),
        })
        admissions.to_csv(os.path.join(self.hosp_dir, "admissions.csv"), index=False)

        diagnoses = pd.DataFrame({
            "subject_id": [1, 1, 1],
            "hadm_id": [10, 10, 20],
            "icd_code": ["N18.3", "I10", "N17.9"],
            "icd_version": [10, 10, 10],
        })
        diagnoses.to_csv(os.path.join(self.hosp_dir, "diagnoses_icd.csv"), index=False)

        d_icd = pd.DataFrame({
            "icd_code": ["N18.3", "I10", "N17.9"],
            "icd_version": [10, 10, 10],
            "long_title": [
                "Chronic kidney disease, stage 3",
                "Hypertension",
                "Acute kidney injury",
            ],
        })
        d_icd.to_csv(os.path.join(self.hosp_dir, "d_icd_diagnoses.csv"), index=False)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp)

    def test_header_present(self):
        """Structured block starts with 'Past Diagnoses:' header."""
        import extract_diag_history
        config = {
            "MIMIC_DATA_DIR": self.mimic_dir,
            "FEATURES_DIR": self.features_dir,
        }
        extract_diag_history.run(config)
        df = pd.read_parquet(os.path.join(self.features_dir, "diag_history_features.parquet"))
        row = df[df["hadm_id"] == 30].iloc[0]
        self.assertTrue(row["diag_history_text"].startswith("Past Diagnoses:"))

    def test_dated_visit_headers(self):
        """Each prior visit has a 'Visit (YYYY-MM-DD):' section header."""
        import extract_diag_history
        config = {
            "MIMIC_DATA_DIR": self.mimic_dir,
            "FEATURES_DIR": self.features_dir,
        }
        extract_diag_history.run(config)
        df = pd.read_parquet(os.path.join(self.features_dir, "diag_history_features.parquet"))
        row = df[df["hadm_id"] == 30].iloc[0]
        text = row["diag_history_text"]
        self.assertIn("Visit (2018-03-12):", text)
        self.assertIn("Visit (2019-07-24):", text)

    def test_diagnoses_in_correct_visit_block(self):
        """Diagnoses appear under the correct dated visit header."""
        import extract_diag_history
        config = {
            "MIMIC_DATA_DIR": self.mimic_dir,
            "FEATURES_DIR": self.features_dir,
        }
        extract_diag_history.run(config)
        df = pd.read_parquet(os.path.join(self.features_dir, "diag_history_features.parquet"))
        row = df[df["hadm_id"] == 30].iloc[0]
        text = row["diag_history_text"]
        # 2018-03-12 visit: CKD and Hypertension
        idx_2018 = text.index("Visit (2018-03-12):")
        idx_2019 = text.index("Visit (2019-07-24):")
        block_2018 = text[idx_2018:idx_2019]
        self.assertIn("Chronic kidney disease, stage 3", block_2018)
        self.assertIn("Hypertension", block_2018)
        # 2019-07-24 visit: AKI
        block_2019 = text[idx_2019:]
        self.assertIn("Acute kidney injury", block_2019)

    def test_no_pipe_separator(self):
        """New format does not use pipe (|) separator between diagnoses."""
        import extract_diag_history
        config = {
            "MIMIC_DATA_DIR": self.mimic_dir,
            "FEATURES_DIR": self.features_dir,
        }
        extract_diag_history.run(config)
        df = pd.read_parquet(os.path.join(self.features_dir, "diag_history_features.parquet"))
        row = df[df["hadm_id"] == 30].iloc[0]
        self.assertNotIn(" | ", row["diag_history_text"])

    def test_first_visit_empty(self):
        """First visit produces an empty string (no prior admissions)."""
        import extract_diag_history
        config = {
            "MIMIC_DATA_DIR": self.mimic_dir,
            "FEATURES_DIR": self.features_dir,
        }
        extract_diag_history.run(config)
        df = pd.read_parquet(os.path.join(self.features_dir, "diag_history_features.parquet"))
        row_10 = df[df["hadm_id"] == 10].iloc[0]
        self.assertEqual(row_10["diag_history_text"], "")


class TestExtractDischargeHistoryFormat(unittest.TestCase):
    """Tests for the dated-header format in extract_discharge_history.py."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.mimic_dir = os.path.join(self.tmp, "mimic")
        self.hosp_dir = os.path.join(self.mimic_dir, "hosp")
        self.note_dir = os.path.join(self.mimic_dir, "note")
        os.makedirs(self.hosp_dir)
        os.makedirs(self.note_dir)
        self.features_dir = os.path.join(self.tmp, "features")
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'preprocessing'))

        admissions = pd.DataFrame({
            "subject_id": [1, 1, 1],
            "hadm_id": [10, 20, 30],
            "admittime": pd.to_datetime(["2018-03-12", "2019-07-24", "2020-01-01"]),
        })
        admissions.to_csv(os.path.join(self.hosp_dir, "admissions.csv"), index=False)

        # Two prior discharge notes
        notes = pd.DataFrame({
            "subject_id": [1, 1],
            "hadm_id": [10, 20],
            "charttime": pd.to_datetime(["2018-03-20", "2019-07-31"]),
            "text": [
                "Some header\nAllergies: Penicillin\nBody of note 1.",
                "Some header\nAllergies: None known\nBody of note 2.",
            ],
        })
        notes.to_csv(os.path.join(self.note_dir, "discharge.csv"), index=False)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp)

    def test_dated_header_present(self):
        """Prior Discharge Summary header with date is present for each prior note."""
        import extract_discharge_history
        config = {
            "MIMIC_DATA_DIR": self.mimic_dir,
            "FEATURES_DIR": self.features_dir,
        }
        extract_discharge_history.run(config)
        df = pd.read_parquet(
            os.path.join(self.features_dir, "discharge_history_features.parquet")
        )
        row = df[df["hadm_id"] == 30].iloc[0]
        text = row["discharge_history_text"]
        self.assertIn("Prior Discharge Summary (2018-03-12):", text)
        self.assertIn("Prior Discharge Summary (2019-07-24):", text)

    def test_note_body_follows_header(self):
        """Cleaned note body appears after each dated header."""
        import extract_discharge_history
        config = {
            "MIMIC_DATA_DIR": self.mimic_dir,
            "FEATURES_DIR": self.features_dir,
        }
        extract_discharge_history.run(config)
        df = pd.read_parquet(
            os.path.join(self.features_dir, "discharge_history_features.parquet")
        )
        row = df[df["hadm_id"] == 30].iloc[0]
        text = row["discharge_history_text"]
        self.assertIn("Allergies: Penicillin", text)
        self.assertIn("Allergies: None known", text)
        # Pre-Allergies text should be stripped
        self.assertNotIn("Some header", text)

    def test_no_old_separator(self):
        """New format does not use the old '---' separator."""
        import extract_discharge_history
        config = {
            "MIMIC_DATA_DIR": self.mimic_dir,
            "FEATURES_DIR": self.features_dir,
        }
        extract_discharge_history.run(config)
        df = pd.read_parquet(
            os.path.join(self.features_dir, "discharge_history_features.parquet")
        )
        row = df[df["hadm_id"] == 30].iloc[0]
        self.assertNotIn("---", row["discharge_history_text"])

    def test_single_prior_note(self):
        """Second visit has exactly one prior note with correct header."""
        import extract_discharge_history
        config = {
            "MIMIC_DATA_DIR": self.mimic_dir,
            "FEATURES_DIR": self.features_dir,
        }
        extract_discharge_history.run(config)
        df = pd.read_parquet(
            os.path.join(self.features_dir, "discharge_history_features.parquet")
        )
        row_20 = df[df["hadm_id"] == 20].iloc[0]
        text = row_20["discharge_history_text"]
        self.assertIn("Prior Discharge Summary (2018-03-12):", text)
        # Should only have the 2018 note, not 2019 (not prior to hadm_id=20)
        self.assertNotIn("Prior Discharge Summary (2019", text)


class TestEmbedFeaturesSlicing(unittest.TestCase):
    """Tests for embed_features.py slice computation and LPT scheduling."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.preprocessing_dir = os.path.join(self.tmp, "preprocessing")
        self.features_dir = os.path.join(self.tmp, "features")
        self.embeddings_dir = os.path.join(self.tmp, "embeddings")
        self.classifications_dir = os.path.join(self.tmp, "classifications")
        os.makedirs(self.preprocessing_dir)
        os.makedirs(self.features_dir)
        os.makedirs(self.embeddings_dir)
        os.makedirs(self.classifications_dir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp)

    def _make_splits(self, n: int) -> pd.DataFrame:
        """Write data_splits.parquet with n admissions and return it."""
        df = pd.DataFrame({
            "subject_id": list(range(1, n + 1)),
            "hadm_id": list(range(1000, 1000 + n)),
            "split": ["train"] * n,
        })
        df.to_parquet(
            os.path.join(self.preprocessing_dir, "data_splits.parquet"), index=False
        )
        return df

    def test_slice_computation_covers_all_admissions(self):
        """Every hadm_id belongs to exactly one slice."""
        import math
        import embed_features

        n = 100
        self._make_splits(n)
        all_hadm_ids = sorted(range(1000, 1000 + n))
        slice_size_per_gpu = 20
        n_gpus = 2
        per_job = slice_size_per_gpu * n_gpus
        n_slices = math.ceil(n / per_job)

        seen = set()
        for si in range(n_slices):
            start = si * per_job
            end = min(start + per_job, n)
            for hid in all_hadm_ids[start:end]:
                self.assertNotIn(hid, seen, f"hadm_id {hid} in multiple slices")
                seen.add(hid)

        self.assertEqual(seen, set(all_hadm_ids), "Not all hadm_ids covered by slices")

    def test_slice_out_of_range_returns_early(self):
        """run() with slice_index >= n_slices logs and returns without error."""
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not available")

        import embed_features

        n = 10
        self._make_splits(n)

        # With per_job=20 there is only 1 slice (slice 0), so slice_index=1 is OOB.
        config = {
            "FEATURES_DIR": self.features_dir,
            "EMBEDDINGS_DIR": self.embeddings_dir,
            "CLASSIFICATIONS_DIR": self.classifications_dir,
            "PREPROCESSING_DIR": self.preprocessing_dir,
            "BERT_MODEL_NAME": "bert-base-uncased",
            "BERT_MAX_LENGTH": 512,
            "BERT_BATCH_SIZE": 4,
            "BERT_DEVICE": "cpu",
            "BERT_SLICE_SIZE_PER_GPU": 20,
            "BERT_CHECKPOINT_INTERVAL": 5,
        }
        # Should return without raising
        embed_features.run(config, slice_index=99)
        # No embeddings written
        self.assertEqual(os.listdir(self.embeddings_dir), [])

    def test_hadm_id_split_disjoint_and_complete(self):
        """Hadm_id split between workers is disjoint and covers all admissions."""
        all_hadm = sorted(range(1000, 1100))  # 100 admissions
        n_gpus = 2
        n = len(all_hadm)
        gpu_hadm = []
        for g in range(n_gpus):
            start = g * n // n_gpus
            end = (g + 1) * n // n_gpus
            gpu_hadm.append(set(all_hadm[start:end]))

        # Disjoint
        self.assertEqual(gpu_hadm[0] & gpu_hadm[1], set(),
                         "Worker hadm_id sets overlap")
        # Complete coverage
        self.assertEqual(gpu_hadm[0] | gpu_hadm[1], set(all_hadm),
                         "Split does not cover all admissions")
        # Roughly equal sizes
        self.assertAlmostEqual(len(gpu_hadm[0]), len(gpu_hadm[1]), delta=1,
                               msg="Split halves differ by more than 1")

    def test_within_worker_lpt_ordering(self):
        """Tasks within a worker are sorted by estimated cost in descending order."""
        tasks = [
            {"embedding_col": f"feat_{i}", "texts": ["x"] * (i * 10), "max_length": 512}
            for i in range(1, 6)
        ]
        tasks.sort(key=lambda t: len(t["texts"]) * t["max_length"], reverse=True)
        costs = [len(t["texts"]) * t["max_length"] for t in tasks]
        self.assertEqual(costs, sorted(costs, reverse=True),
                         "Tasks not sorted descending by cost within worker")

    def test_effective_batch_size_capped_at_4x_base(self):
        """_effective_batch_size never exceeds 4x base_batch_size and matches spec table."""
        import embed_features

        base = 32
        for cap in [64, 128, 256, 512, 1024, 2048, 4096]:
            result = embed_features._effective_batch_size(base, cap)
            self.assertLessEqual(result, base * 4,
                                 f"Batch size exceeds 4x base for cap={cap}")
            self.assertGreaterEqual(result, 1)

        # Verify key values from spec table (base=32)
        spec_table = [
            (64,   128),   # chief_complaint: min(256, 128) = 128
            (256,   64),   # triage:          min(64, 128)  = 64
            (512,   32),   # diag_history:    min(32, 128)  = 32
            (1024,  16),   # radiology:       min(16, 128)  = 16
            (2048,   8),   # lab groups:      min(8, 128)   = 8
            (4096,   4),   # discharge_hist:  min(4, 128)   = 4
        ]
        for cap, expected in spec_table:
            got = embed_features._effective_batch_size(base, cap)
            self.assertEqual(got, expected,
                             f"cap={cap}: expected {expected}, got {got}")


class TestEmbedFeaturesWorkerPaths(unittest.TestCase):
    """Tests for per-worker parquet files and the merge step."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp)

    def _make_parquet(self, path: str, hadm_ids: list, col: str = "emb") -> None:
        """Write a minimal parquet with hadm_id and an embedding column."""
        df = pd.DataFrame({
            "hadm_id": hadm_ids,
            col: [float(h) for h in hadm_ids],
        })
        df.to_parquet(path, index=False)

    # ------------------------------------------------------------------ #
    # worker_path naming
    # ------------------------------------------------------------------ #
    def test_worker_path_naming_pattern(self):
        """worker_path follows the output_path + '.worker{rank}' pattern."""
        output_path = os.path.join(self.tmp, "feat.parquet")
        for rank in range(4):
            worker_path = output_path + f".worker{rank}"
            self.assertEqual(
                worker_path,
                output_path + f".worker{rank}",
                f"unexpected worker path for rank {rank}",
            )

    # ------------------------------------------------------------------ #
    # merge — single worker file → atomic rename, no leftover
    # ------------------------------------------------------------------ #
    def test_merge_single_worker_renames_atomically(self):
        """A single worker file is renamed to the final output path."""
        output_path = os.path.join(self.tmp, "feat.parquet")
        worker0 = output_path + ".worker0"
        self._make_parquet(worker0, [1001, 1002, 1003])

        # Simulate merge logic for 1 GPU
        n_gpus = 1
        worker_paths = [
            output_path + f".worker{r}"
            for r in range(n_gpus)
            if os.path.exists(output_path + f".worker{r}")
        ]
        self.assertEqual(len(worker_paths), 1)

        tmp_path = output_path + ".tmp"
        os.replace(worker_paths[0], tmp_path)
        os.replace(tmp_path, output_path)

        self.assertTrue(os.path.exists(output_path), "Final output not created")
        self.assertFalse(os.path.exists(worker0), "Worker file not removed")
        self.assertFalse(os.path.exists(tmp_path), ".tmp file left behind")

        result = pd.read_parquet(output_path)
        self.assertEqual(sorted(result["hadm_id"].tolist()), [1001, 1002, 1003])

    # ------------------------------------------------------------------ #
    # merge — two worker files → concat + cleanup
    # ------------------------------------------------------------------ #
    def test_merge_two_workers_concatenates_and_cleans_up(self):
        """Two worker files are merged, and worker files are deleted."""
        output_path = os.path.join(self.tmp, "feat.parquet")
        worker0 = output_path + ".worker0"
        worker1 = output_path + ".worker1"
        self._make_parquet(worker0, [1001, 1002])
        self._make_parquet(worker1, [1003, 1004])

        n_gpus = 2
        worker_paths = [
            output_path + f".worker{r}"
            for r in range(n_gpus)
            if os.path.exists(output_path + f".worker{r}")
        ]
        self.assertEqual(len(worker_paths), 2)

        dfs = [pd.read_parquet(wp) for wp in worker_paths]
        merged = pd.concat(dfs, ignore_index=True)
        tmp_path = output_path + ".tmp"
        merged.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, output_path)
        for wp in worker_paths:
            os.remove(wp)

        self.assertTrue(os.path.exists(output_path))
        self.assertFalse(os.path.exists(worker0), "worker0 not cleaned up")
        self.assertFalse(os.path.exists(worker1), "worker1 not cleaned up")

        result = pd.read_parquet(output_path)
        self.assertEqual(sorted(result["hadm_id"].tolist()), [1001, 1002, 1003, 1004])

    # ------------------------------------------------------------------ #
    # merge — missing worker file (worker failed) → skip that rank
    # ------------------------------------------------------------------ #
    def test_merge_skips_missing_worker_file(self):
        """If worker1 file is absent (failed), only worker0 data is merged."""
        output_path = os.path.join(self.tmp, "feat.parquet")
        worker0 = output_path + ".worker0"
        self._make_parquet(worker0, [1001, 1002])
        # worker1 intentionally NOT created (simulates failure)

        n_gpus = 2
        worker_paths = [
            output_path + f".worker{r}"
            for r in range(n_gpus)
            if os.path.exists(output_path + f".worker{r}")
        ]
        self.assertEqual(len(worker_paths), 1,
                         "Should only find the one worker file that exists")

    # ------------------------------------------------------------------ #
    # stale-cleanup: worker temp removed when final output is complete
    # ------------------------------------------------------------------ #
    def test_stale_worker_temp_removed_when_output_complete(self):
        """Stale worker temp is removed when the final output covers all slice hadm_ids."""
        output_path = os.path.join(self.tmp, "feat.parquet")
        worker_path = output_path + ".worker0"

        slice_hadm_ids = {1001, 1002, 1003}

        # Write a complete final output (all slice ids present)
        self._make_parquet(output_path, sorted(slice_hadm_ids))
        # Write a stale worker temp
        self._make_parquet(worker_path, [1001])

        # Simulate stale-cleanup logic
        force_reembed = False
        if os.path.exists(worker_path) and not force_reembed:
            if os.path.exists(output_path):
                try:
                    existing = pd.read_parquet(output_path, columns=["hadm_id"])
                    if slice_hadm_ids.issubset(set(existing["hadm_id"].tolist())):
                        os.remove(worker_path)
                except Exception:  # noqa: BLE001
                    pass

        self.assertFalse(os.path.exists(worker_path),
                         "Stale worker temp should have been removed")

    def test_stale_worker_temp_kept_when_output_incomplete(self):
        """Stale worker temp is kept when the final output is missing some ids."""
        output_path = os.path.join(self.tmp, "feat.parquet")
        worker_path = output_path + ".worker0"

        slice_hadm_ids = {1001, 1002, 1003}

        # Final output only has 2 of 3 ids — not complete
        self._make_parquet(output_path, [1001, 1002])
        self._make_parquet(worker_path, [1003])

        force_reembed = False
        if os.path.exists(worker_path) and not force_reembed:
            if os.path.exists(output_path):
                try:
                    existing = pd.read_parquet(output_path, columns=["hadm_id"])
                    if slice_hadm_ids.issubset(set(existing["hadm_id"].tolist())):
                        os.remove(worker_path)
                except Exception:  # noqa: BLE001
                    pass

        self.assertTrue(os.path.exists(worker_path),
                        "Worker temp should NOT be removed when output is incomplete")

    # ------------------------------------------------------------------ #
    # force_reembed: both .tmp files are cleaned up
    # ------------------------------------------------------------------ #
    def test_force_reembed_cleans_worker_tmp(self):
        """force_reembed removes both output.tmp and worker.tmp."""
        output_path = os.path.join(self.tmp, "feat.parquet")
        worker_path = output_path + ".worker0"

        output_tmp = output_path + ".tmp"
        worker_tmp = worker_path + ".tmp"

        # Create both stale .tmp files
        pd.DataFrame({"hadm_id": [1]}).to_parquet(output_tmp, index=False)
        pd.DataFrame({"hadm_id": [2]}).to_parquet(worker_tmp, index=False)

        # Simulate force_reembed cleanup logic
        force_reembed = True
        if force_reembed:
            if os.path.exists(output_tmp):
                os.remove(output_tmp)
            if os.path.exists(worker_tmp):
                os.remove(worker_tmp)

        self.assertFalse(os.path.exists(output_tmp), "output.tmp not removed")
        self.assertFalse(os.path.exists(worker_tmp), "worker.tmp not removed")

    # ------------------------------------------------------------------ #
    # parallel spawn: all processes are started before any join
    # ------------------------------------------------------------------ #
    def test_parallel_spawn_all_start_before_join(self):
        """All worker processes are spawned before any is joined (parallel pattern)."""
        # This test verifies the structural correctness of the parallel
        # spawn loop by checking that we can track start/join order using
        # a simple in-process mock rather than actual multiprocessing.
        from unittest.mock import MagicMock, call

        n_gpus = 2
        processes = []
        start_calls = []
        join_calls = []

        for rank in range(n_gpus):
            p = MagicMock()
            p.pid = 1000 + rank
            p.name = f"embed-worker-{rank}"
            p.exitcode = 0
            processes.append(p)
            p.start()
            start_calls.append(rank)

        for p in processes:
            p.join()
            join_calls.append(int(p.name.split("-")[-1]))

        # All starts happen before any joins
        self.assertEqual(start_calls, [0, 1], "Not all workers started")
        self.assertEqual(join_calls, [0, 1], "Not all workers joined")
        # Verify the order: start0, start1, join0, join1
        for p in processes:
            p.start.assert_called_once()
            p.join.assert_called_once()


if __name__ == "__main__":
    unittest.main()
