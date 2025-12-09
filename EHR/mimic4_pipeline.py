import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Any

# Import generic tools
from generic_pipeline import DataSource, DataProcessor


class MimicIVSource(DataSource):
    def __init__(self, config: Dict[str, Any], global_config: Dict[str, Any]):
        """
        :param config: The specific 'mimic4' section of the config
        :param global_config: The 'pipeline' section (for max_rows, etc.)
        """
        super().__init__(config)
        self.name = "mimic4"
        self.cfg = config
        self.global_cfg = global_config

        # Construct root path
        self.root = Path(self.cfg['root_dir'])

    def get_name(self) -> str:
        return self.name

    def fetch_data(self) -> pd.DataFrame:
        print(f"📂 Loading MIMIC-IV from: {self.root}")

        # 1. Read parameters from config
        files = self.cfg['files']
        cols = self.cfg.get('columns', {})  # Default to empty dict if missing

        # 2. Load Data (Using paths from config)
        # Note: We use global_cfg['max_rows'] to control data size for testing
        limit = self.global_cfg.get('max_rows', None)

        pat_df = pd.read_csv(self.root / files['patients'], nrows=limit)
        adm_df = pd.read_csv(self.root / files['admissions'], nrows=limit)
        lab_df = pd.read_csv(self.root / files['labevents'], nrows=limit)

        # 3. Processing Logic
        # (This logic remains hardcoded because it is specific to MIMIC's structure)
        context = pd.merge(adm_df, pat_df, on='subject_id', how='inner')

        # Simple encodings
        if 'gender' in context.columns:
            context['gender'] = context['gender'].astype('category').cat.codes
        if 'admission_type' in context.columns:
            context['admission_type'] = context['admission_type'].astype('category').cat.codes

        # Reward Calculation
        lab_df = lab_df.dropna(subset=['hadm_id'])
        # Use column name from config if available, else default to 'flag'
        flag_col = cols.get('label', 'flag')
        lab_df['reward'] = lab_df[flag_col].apply(lambda x: 1.0 if x == 'abnormal' else 0.0)

        # Merge
        df = pd.merge(lab_df, context, on=['subject_id', 'hadm_id'])

        # Select final columns
        return df[['subject_id', 'anchor_age', 'gender', 'admission_type', 'itemid', 'reward']]


# --- Manual Trigger with Config Loader ---
def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    # 1. Load Configuration
    full_config = load_config("config.yaml")

    # 2. Extract sections
    pipeline_cfg = full_config['pipeline']
    mimic_cfg = full_config['mimic4']

    # 3. Initialize Source with specific config
    source = MimicIVSource(config=mimic_cfg, global_config=pipeline_cfg)

    # 4. Run Processor with pipeline config
    processor = DataProcessor(config=full_config)
    processor.run(source)