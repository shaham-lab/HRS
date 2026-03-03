"""Shared utilities for the preprocessing pipeline."""

import os
from typing import cast

import pandas as pd


def _load_csv(path_gz: str, path_csv: str, **kwargs) -> pd.DataFrame:
    if os.path.exists(path_gz):
        return cast(pd.DataFrame, pd.read_csv(path_gz, **kwargs))
    if os.path.exists(path_csv):
        return cast(pd.DataFrame, pd.read_csv(path_csv, **kwargs))
    raise FileNotFoundError(
        f"Neither {path_gz} nor {path_csv} found."
    )
