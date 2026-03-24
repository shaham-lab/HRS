import logging
import math
import os
from collections import OrderedDict, namedtuple
from typing import Dict, Iterable, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


logger = logging.getLogger(__name__)


class SchemaError(ValueError):
    """Raised when the upstream preprocessing schema contract is violated."""


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


class RewardModelConfig(BaseModel):
    INPUT_DIM: int
    LAYER_WIDTHS: List[int]
    DROPOUT_RATE: float
    ACTIVATION: str

    MAX_EPOCHS: int
    BATCH_SIZE_PER_GPU: int
    NUM_GPUS: int = 2
    LEARNING_RATE: float
    WEIGHT_DECAY: float
    ADAM_BETA1: float
    ADAM_BETA2: float
    LR_WARMUP_EPOCHS: int
    LR_MIN: float
    EARLY_STOPPING_PATIENCE: int
    CHECKPOINT_KEEP_N: int

    LOSS_WEIGHT_Y1: float
    LOSS_WEIGHT_Y2: float
    POS_WEIGHT_Y1: Optional[float] = Field(default=None)
    POS_WEIGHT_Y2: Optional[float] = Field(default=None)

    MASKING_START_RATIOS: Dict[str, float]
    MASKING_END_RATIOS: Dict[str, float]
    MASKING_TRANSITION_MIDPOINT_EPOCH: int
    MASKING_TRANSITION_SHAPE: str
    MASKING_K: int

    DATASET_PATH: str
    DATASET_ROW_GROUP_CACHE_SIZE: int = 2

    CHECKPOINT_DIR: str
    METRICS_PATH: str
    CALIBRATION_PARAMS_PATH: str
    EXPORT_PATH: str

    @field_validator("ACTIVATION")
    @classmethod
    def _validate_activation(cls, value: str) -> str:
        if value not in {"relu", "leaky_relu"}:
            raise ValueError("ACTIVATION must be one of {'relu', 'leaky_relu'}")
        return value

    @field_validator("MASKING_START_RATIOS", "MASKING_END_RATIOS")
    @classmethod
    def _validate_ratios(cls, value: Dict[str, float]) -> Dict[str, float]:
        expected_keys = {"random", "adversarial", "none"}
        if set(value.keys()) != expected_keys:
            raise ValueError(
                f"Masking ratios must define exactly {expected_keys}, received {set(value.keys())}"
            )
        return value

    @model_validator(mode="after")
    def _expand_paths(self) -> "RewardModelConfig":
        for field_name, field_value in self.__dict__.items():
            if field_value is None:
                continue
            if field_name.endswith(("_PATH", "_DIR", "_FILE")):
                expanded = os.path.abspath(os.path.expanduser(str(field_value)))
                object.__setattr__(self, field_name, expanded)
        start_sum = sum(self.MASKING_START_RATIOS.values())
        end_sum = sum(self.MASKING_END_RATIOS.values())
        if not math.isclose(start_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"MASKING_START_RATIOS must sum to 1.0, found {start_sum}")
        if not math.isclose(end_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"MASKING_END_RATIOS must sum to 1.0, found {end_sum}")
        return self

    @field_validator("MASKING_TRANSITION_SHAPE")
    @classmethod
    def _validate_transition_shape(cls, value: str) -> str:
        if value != "sigmoid":
            raise ValueError("MASKING_TRANSITION_SHAPE must be 'sigmoid'")
        return value

    class Config:
        extra = "forbid"


def load_and_validate_config(path: str) -> RewardModelConfig:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    try:
        return RewardModelConfig(**data)
    except ValidationError as exc:
        raise SchemaError(f"Invalid reward model configuration: {exc}") from exc


def build_feature_index_map(columns: Iterable[str]) -> Dict[str, Tuple[int, int]]:
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


def compute_pos_weights(df_train) -> Tuple[float, float]:
    y1 = df_train["y1_mortality"].astype(float)
    pos_y1 = float((y1 == 1).sum())
    neg_y1 = float((y1 == 0).sum())
    if pos_y1 == 0 or neg_y1 == 0:
        raise SchemaError("y1_mortality must contain both positive and negative examples")
    pos_weight_y1 = neg_y1 / pos_y1

    survivors = df_train[df_train["y1_mortality"] == 0]
    y2 = survivors["y2_readmission"].astype(float)
    pos_y2 = float((y2 == 1).sum())
    neg_y2 = float((y2 == 0).sum())
    if pos_y2 == 0 or neg_y2 == 0:
        raise SchemaError("y2_readmission must contain both positive and negative examples for survivors")
    pos_weight_y2 = neg_y2 / pos_y2

    return float(pos_weight_y1), float(pos_weight_y2)


def sigmoid_crossover(
    epoch: int,
    total_epochs: int,
    start_ratios: Dict[str, float],
    end_ratios: Dict[str, float],
    midpoint: float,
) -> Tuple[float, float, float]:
    clamped_epoch = max(0, min(epoch, total_epochs))
    scale = total_epochs if total_epochs > 0 else 1.0
    progress = 1.0 / (1.0 + math.exp(-(clamped_epoch - midpoint) / scale))
    probs = []
    for key in ("random", "adversarial", "none"):
        start = start_ratios[key]
        end = end_ratios[key]
        probs.append(start + (end - start) * progress)
    return tuple(probs)  # type: ignore[return-value]


def get_device(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def unwrap_ddp(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def broadcast_tensor(tensor: torch.Tensor, src_rank: int) -> torch.Tensor:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast(tensor, src=src_rank)
    return tensor


class ParquetDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        parquet_file: pq.ParquetFile,
        row_indices: List[int],
        feature_index_map: Dict[str, Tuple[int, int]],
        cache_size: int,
    ) -> None:
        self._parquet_file = parquet_file
        self._row_indices = list(row_indices)
        self._feature_index_map = feature_index_map
        self._cache_size = max(1, cache_size)
        self._cache: OrderedDict[int, pa.Table] = OrderedDict()
        self._columns_needed = list(feature_index_map.keys()) + ["y1_mortality", "y2_readmission"]

        metadata = parquet_file.metadata
        self._row_group_boundaries: List[Tuple[int, int]] = []
        start = 0
        for i in range(metadata.num_row_groups):
            num_rows = metadata.row_group(i).num_rows
            self._row_group_boundaries.append((start, start + num_rows))
            start += num_rows

    def __len__(self) -> int:
        return len(self._row_indices)

    def __getitem__(self, idx: int):
        row_idx = self._row_indices[idx]
        rg_index, rg_start = self._locate_row_group(row_idx)
        table = self._get_row_group(rg_index)
        offset = row_idx - rg_start
        row = table.slice(offset, 1)

        features = []
        for col in self._feature_index_map.keys():
            value = row[col].to_pylist()[0]
            features.append(torch.tensor(value, dtype=torch.float32))
        X = torch.cat(features, dim=0)

        y1_value = row["y1_mortality"].to_pylist()[0]
        y1_tensor = torch.tensor(y1_value, dtype=torch.int8)

        y2_value = row["y2_readmission"].to_pylist()[0]
        y2_value = float("nan") if y2_value is None else y2_value
        y2_tensor = torch.tensor(y2_value, dtype=torch.float32)

        return X, y1_tensor, y2_tensor

    def _locate_row_group(self, row_idx: int) -> Tuple[int, int]:
        for i, (start, end) in enumerate(self._row_group_boundaries):
            if start <= row_idx < end:
                return i, start
        raise IndexError(f"Row index {row_idx} out of bounds for dataset")

    def _get_row_group(self, rg_index: int) -> pa.Table:
        if rg_index in self._cache:
            table = self._cache.pop(rg_index)
            self._cache[rg_index] = table
            return table

        table = self._parquet_file.read_row_group(rg_index, columns=self._columns_needed)
        self._cache[rg_index] = table
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return table


DatasetBundle = namedtuple(
    "DatasetBundle",
    ["train_dataset", "dev_dataset", "test_dataset", "feature_index_map", "pos_weight_y1", "pos_weight_y2"],
)


def get_expected_columns() -> List[str]:
    return list(_EXPECTED_COLUMNS)
