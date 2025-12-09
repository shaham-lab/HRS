import abc
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any


# --- 1. The Contract (Interface) ---
class DataSource(abc.ABC):
    """
    Abstract Base Class.
    Now initialized with a configuration dictionary, not just a path.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abc.abstractmethod
    def get_name(self) -> str:
        pass

    @abc.abstractmethod
    def fetch_data(self) -> pd.DataFrame:
        pass


# --- 2. The Engine (Processor) ---
class DataProcessor:
    def __init__(self, config: Dict[str, Any]):
        # Read output directory from config, default to ./output if missing
        self.output_dir = Path(config.get('pipeline', {}).get('output_dir', './output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, source: DataSource):
        print(f"--- Starting Pipeline for Source: {source.get_name()} ---")

        # 1. Fetch Data
        df = source.fetch_data()

        # 2. Validation
        required_cols = {'subject_id', 'itemid', 'reward'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Missing columns. Found {df.columns}, expected {required_cols}")

        # 3. Create Mappings (Action -> Index)
        unique_actions = df['itemid'].unique()
        action_map = {int(k): i for i, k in enumerate(unique_actions)}
        df['action_index'] = df['itemid'].map(action_map)

        # 4. Save Data (Parquet)
        data_path = self.output_dir / f"{source.get_name()}_processed.parquet"
        df.to_parquet(data_path, index=False)

        # 5. Save Metadata
        metadata = {
            "source": source.get_name(),
            "config_used": source.config,  # Audit trail: save config used to generate this data
            "num_actions": len(unique_actions),
            "state_features": [c for c in df.columns if c not in ['subject_id', 'itemid', 'action_index', 'reward']],
            "action_map": action_map
        }

        meta_path = self.output_dir / f"{source.get_name()}_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        print(f"✅ Data saved to: {data_path}")