"""MIMIC-IV specific dataset loader and helpers."""

import logging
from typing import Dict, List, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.reward_model.data_loader import DataLoader
from src.reward_model.schema_error import SchemaError

logger = logging.getLogger(__name__)


_EXPECTED_COLUMNS: List[str] = [
    "subject_id",
    "hadm_id",
    "split",
    "y1_mortality",
    "y2_readmission",
    "demographic_vec",
    "diag_history_embedding",
    "discharge_history_embedding",
    "triage_embedding",
    "chief_complaint_embedding",
    "lab_blood_gas_embedding",
    "lab_blood_chemistry_embedding",
    "lab_blood_hematology_embedding",
    "lab_urine_chemistry_embedding",
    "lab_urine_hematology_embedding",
    "lab_other_body_fluid_chemistry_embedding",
    "lab_other_body_fluid_hematology_embedding",
    "lab_ascites_embedding",
    "lab_pleural_embedding",
    "lab_csf_embedding",
    "lab_bone_marrow_embedding",
    "lab_joint_fluid_embedding",
    "lab_stool_embedding",
    "radiology_embedding",
    "micro_blood_culture_routine_embedding",
    "micro_blood_bottle_gram_stain_embedding",
    "micro_urine_culture_embedding",
    "micro_urine_viral_embedding",
    "micro_urinary_antigens_embedding",
    "micro_respiratory_non_invasive_embedding",
    "micro_respiratory_invasive_embedding",
    "micro_respiratory_afb_embedding",
    "micro_respiratory_viral_embedding",
    "micro_respiratory_pcp_legionella_embedding",
    "micro_gram_stain_respiratory_embedding",
    "micro_gram_stain_wound_tissue_embedding",
    "micro_gram_stain_csf_embedding",
    "micro_wound_culture_embedding",
    "micro_hardware_and_lines_culture_embedding",
    "micro_pleural_culture_embedding",
    "micro_peritoneal_culture_embedding",
    "micro_joint_fluid_culture_embedding",
    "micro_fluid_culture_embedding",
    "micro_bone_marrow_culture_embedding",
    "micro_csf_culture_embedding",
    "micro_fungal_tissue_wound_embedding",
    "micro_fungal_respiratory_embedding",
    "micro_fungal_fluid_embedding",
    "micro_mrsa_staph_screen_embedding",
    "micro_resistance_screen_embedding",
    "micro_cdiff_embedding",
    "micro_stool_bacterial_embedding",
    "micro_stool_parasitology_embedding",
    "micro_herpesvirus_serology_embedding",
    "micro_hepatitis_hiv_embedding",
    "micro_syphilis_serology_embedding",
    "micro_misc_serology_embedding",
    "micro_herpesvirus_culture_antigen_embedding",
    "micro_gc_chlamydia_sti_embedding",
    "micro_vaginal_genital_flora_embedding",
    "micro_throat_strep_embedding",
]

_HISTORY_TRIAGE_EMBEDDINGS = {
    "diag_history_embedding",
    "discharge_history_embedding",
    "triage_embedding",
    "chief_complaint_embedding",
}

_LAB_EMBEDDINGS = {
    "lab_blood_gas_embedding",
    "lab_blood_chemistry_embedding",
    "lab_blood_hematology_embedding",
    "lab_urine_chemistry_embedding",
    "lab_urine_hematology_embedding",
    "lab_other_body_fluid_chemistry_embedding",
    "lab_other_body_fluid_hematology_embedding",
    "lab_ascites_embedding",
    "lab_pleural_embedding",
    "lab_csf_embedding",
    "lab_bone_marrow_embedding",
    "lab_joint_fluid_embedding",
    "lab_stool_embedding",
}

_RADIOLOGY_EMBEDDINGS = {"radiology_embedding"}

_MICRO_EMBEDDINGS = {
    "micro_blood_culture_routine_embedding",
    "micro_blood_bottle_gram_stain_embedding",
    "micro_urine_culture_embedding",
    "micro_urine_viral_embedding",
    "micro_urinary_antigens_embedding",
    "micro_respiratory_non_invasive_embedding",
    "micro_respiratory_invasive_embedding",
    "micro_respiratory_afb_embedding",
    "micro_respiratory_viral_embedding",
    "micro_respiratory_pcp_legionella_embedding",
    "micro_gram_stain_respiratory_embedding",
    "micro_gram_stain_wound_tissue_embedding",
    "micro_gram_stain_csf_embedding",
    "micro_wound_culture_embedding",
    "micro_hardware_and_lines_culture_embedding",
    "micro_pleural_culture_embedding",
    "micro_peritoneal_culture_embedding",
    "micro_joint_fluid_culture_embedding",
    "micro_fluid_culture_embedding",
    "micro_bone_marrow_culture_embedding",
    "micro_csf_culture_embedding",
    "micro_fungal_tissue_wound_embedding",
    "micro_fungal_respiratory_embedding",
    "micro_fungal_fluid_embedding",
    "micro_mrsa_staph_screen_embedding",
    "micro_resistance_screen_embedding",
    "micro_cdiff_embedding",
    "micro_stool_bacterial_embedding",
    "micro_stool_parasitology_embedding",
    "micro_herpesvirus_serology_embedding",
    "micro_hepatitis_hiv_embedding",
    "micro_syphilis_serology_embedding",
    "micro_misc_serology_embedding",
    "micro_herpesvirus_culture_antigen_embedding",
    "micro_gc_chlamydia_sti_embedding",
    "micro_vaginal_genital_flora_embedding",
    "micro_throat_strep_embedding",
}


class Mimic4DataLoader(DataLoader):
    """MIMIC-IV–specific dataset loader."""

    def _validate_schema(self, parquet_file: pq.ParquetFile) -> None:
        schema = parquet_file.schema_arrow
        expected_columns = self._get_expected_columns()
        self._validate_column_order(schema, expected_columns)
        self._validate_label_columns(schema)
        self._validate_demographic_vec(schema)
        self._validate_embedding_columns(schema)
        self._validate_null_counts(parquet_file, ["y1_mortality"])

    def _read_label_table(self, parquet_file: pq.ParquetFile) -> pa.Table:
        return parquet_file.read(columns=["y1_mortality", "y2_readmission"])

    def _validate_labels(self, label_table: pa.Table) -> None:
        df = label_table.to_pandas()
        mortality = df["y1_mortality"].astype(float)
        readmit = df["y2_readmission"].astype(float)
        deceased_mask = mortality == 1.0
        survivor_mask = mortality == 0.0

        if readmit[deceased_mask].notna().any():
            raise SchemaError("y2_readmission must be NaN for deceased rows; produced by extract_y_data.py")
        if survivor_mask.any() and readmit[survivor_mask].isna().any():
            raise SchemaError("y2_readmission must be non-null for survivors; produced by extract_y_data.py")

    def _build_feature_index_map(self, parquet_file: pq.ParquetFile) -> Dict[str, Tuple[int, int]]:
        columns = parquet_file.schema_arrow.names
        feature_columns: List[str] = []
        for name in columns:
            if name in {"subject_id", "hadm_id", "split", "y1_mortality", "y2_readmission"}:
                continue
            feature_columns.append(name)

        if "demographic_vec" not in feature_columns:
            raise SchemaError(
                "demographic_vec column missing from dataset; expected per PREPROCESSING_DATA_MODEL.md Section 3.12"
            )

        offset = 0
        index_map: Dict[str, Tuple[int, int]] = {}
        embedding_columns: List[str] = []

        for col in feature_columns:
            if col == "demographic_vec":
                width = 8
            elif col.endswith("_embedding"):
                width = 768
                embedding_columns.append(col)
            else:
                raise SchemaError(
                    f"Unexpected feature column '{col}' encountered while building index map; "
                    "expected only demographic_vec and *_embedding columns per PREPROCESSING_DATA_MODEL.md Section 3.12"
                )
            index_map[col] = (offset, offset + width)
            offset += width

        missing_expected = set(_EXPECTED_COLUMNS) - set(columns)
        if missing_expected:
            raise SchemaError(
                f"Missing expected columns {sorted(missing_expected)} per PREPROCESSING_DATA_MODEL.md Section 3.12"
            )

        history_count = len(set(embedding_columns) & _HISTORY_TRIAGE_EMBEDDINGS)
        lab_count = len(set(embedding_columns) & _LAB_EMBEDDINGS)
        radiology_count = len(set(embedding_columns) & _RADIOLOGY_EMBEDDINGS)
        micro_count = len(set(embedding_columns) & _MICRO_EMBEDDINGS)

        if len(embedding_columns) != 55 or (
            history_count != 4 or lab_count != 13 or radiology_count != 1 or micro_count != 37
        ):
            raise SchemaError(
                "Embedding column breakdown mismatch — expected 55 embedding columns with counts "
                "(history/triage=4, lab=13, radiology=1, microbiology=37) as defined in "
                "PREPROCESSING_DATA_MODEL.md Section 3.12"
            )

        return index_map

    def _compute_pos_weights(self, label_table: pa.Table, train_rows: List[int]) -> Tuple[float, float]:
        if self._config.POS_WEIGHT_Y1 is not None and self._config.POS_WEIGHT_Y2 is not None:
            return float(self._config.POS_WEIGHT_Y1), float(self._config.POS_WEIGHT_Y2)

        labels_df = label_table.to_pandas()
        train_y_df = labels_df.iloc[train_rows].reset_index(drop=True)
        y1 = train_y_df["y1_mortality"].astype(float)
        pos_y1 = float((y1 == 1).sum())
        neg_y1 = float((y1 == 0).sum())
        if pos_y1 == 0 or neg_y1 == 0:
            raise SchemaError("y1_mortality must contain both positive and negative examples")
        pos_weight_y1 = neg_y1 / pos_y1

        survivors = train_y_df[train_y_df["y1_mortality"] == 0]
        y2 = survivors["y2_readmission"].astype(float)
        pos_y2 = float((y2 == 1).sum())
        neg_y2 = float((y2 == 0).sum())
        if pos_y2 == 0 or neg_y2 == 0:
            raise SchemaError("y2_readmission must contain both positive and negative examples for survivors")
        pos_weight_y2 = neg_y2 / pos_y2

        return float(pos_weight_y1), float(pos_weight_y2)

    @staticmethod
    def _get_expected_columns() -> List[str]:
        """Return the ordered list of expected dataset columns per PREPROCESSING_DATA_MODEL.md Section 3.12."""
        return list(_EXPECTED_COLUMNS)

    @staticmethod
    def _validate_column_order(schema: pa.Schema, expected_columns: List[str]) -> None:
        names = schema.names
        if names != expected_columns:
            raise SchemaError(
                "Column order mismatch; expected canonical order from PREPROCESSING_DATA_MODEL.md Section 3.12. "
                "Re-run preprocessing to regenerate final_cdss_dataset.parquet"
            )

    @staticmethod
    def _validate_label_columns(schema: pa.Schema) -> None:
        for name in ("y1_mortality", "y2_readmission"):
            field = schema.field(name)
            if not pa.types.is_float32(field.type):
                raise SchemaError(f"{name} dtype mismatch; expected float32 produced by extract_y_data.py")
            if field.type != pa.float32():
                raise SchemaError(f"{name} type mismatch; expected float32 produced by extract_y_data.py")

    @staticmethod
    def _validate_embedding_columns(schema: pa.Schema) -> None:
        embedding_columns = [name for name in schema.names if name.endswith("_embedding")]
        for name in embedding_columns:
            field = schema.field(name)
            if not (pa.types.is_fixed_size_list(field.type) and pa.types.is_float32(field.type.value_type)):
                raise SchemaError(f"{name} type mismatch; expected float32[768] produced by combine_dataset.py")
            if field.type.list_size != 768:
                raise SchemaError(f"{name} length mismatch; expected fixed_size_list[768] produced by combine_dataset.py")

    @staticmethod
    def _validate_demographic_vec(schema: pa.Schema) -> None:
        field = schema.field("demographic_vec")
        if not (pa.types.is_fixed_size_list(field.type) and pa.types.is_float32(field.type.value_type)):
            raise SchemaError("demographic_vec type mismatch; expected float32[8] produced by combine_dataset.py")
        if field.type.list_size != 8:
            raise SchemaError("demographic_vec length mismatch; expected fixed_size_list[8] produced by combine_dataset.py")

    @staticmethod
    def _validate_null_counts(parquet_file: pq.ParquetFile, columns: List[str]) -> None:
        for col in columns:
            nulls = 0
            idx = parquet_file.schema_arrow.get_field_index(col)
            for rg in range(parquet_file.metadata.num_row_groups):
                stats = parquet_file.metadata.row_group(rg).column(idx).statistics
                if stats is None or not stats.has_null_count:
                    raise SchemaError(
                        f"Missing null-count statistics for column {col}; "
                        "cannot validate null counts — re-run preprocessing "
                        "to regenerate statistics"
                    )
                nulls += stats.null_count
            if nulls != 0:
                producer = "combine_dataset.py" if col.endswith("_embedding") else "extract_y_data.py"
                raise SchemaError(f"Null values found in {col} produced by {producer}")
