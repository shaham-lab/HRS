import argparse
import logging
import sys
from typing import List, Tuple

import pyarrow.parquet as pq

from mimic4_data_loader import Mimic4DataLoader
from reward_model_config import load_and_validate_config, RewardModelConfig

logger = logging.getLogger(__name__)


def _run_assertions(parquet_file: pq.ParquetFile, config: RewardModelConfig) -> List[Tuple[str, str, str]]:
    results: List[Tuple[str, str, str]] = []

    def _skip_y2_alignment() -> None:
        raise NotImplementedError(
            "SKIPPED — full Y2 alignment requires row-level read; run Mimic4DataLoader.load() for complete validation"
        )

    loader = Mimic4DataLoader(config)
    checks = [
        ("Schema validation", lambda: loader._validate_schema(parquet_file)),
        ("y2_readmission alignment", _skip_y2_alignment),
    ]

    for name, fn in checks:
        try:
            fn()
            results.append((name, "PASS", "PASS"))
        except NotImplementedError as exc:
            results.append((name, "SKIP", str(exc)))
        except Exception as exc:  # noqa: BLE001
            results.append((name, "FAIL", str(exc)))
    return results


def main() -> int:
    """Validate dataset contract using Parquet metadata only."""
    parser = argparse.ArgumentParser(description="Validate reward model dataset contract.")
    parser.add_argument("--config", required=True, help="Path to reward_model.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    config = load_and_validate_config(args.config)
    logger.info("Opening dataset: %s", config.DATASET_PATH)
    parquet_file = pq.ParquetFile(config.DATASET_PATH)

    results = _run_assertions(parquet_file, config)
    failed = False
    for name, status, message in results:
        print(f"{status}: {name} - {message}")
        if status == "FAIL":
            failed = True

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
