from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "modeling_outputs"
DATA_PATH = OUT_DIR / "clean_model_data.csv"
FEATURE_SETS_PATH = OUT_DIR / "feature_sets.json"

RQ2_FEATURE_GROUPS_NAME = "rq2_feature_groups"


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    print("RQ2 feature group workflow scaffold is ready.")
    print(f"Input data: {DATA_PATH.relative_to(ROOT)}")
    print(f"Feature sets: {FEATURE_SETS_PATH.relative_to(ROOT)}")
    print(f"Feature group key: {RQ2_FEATURE_GROUPS_NAME}")


if __name__ == "__main__":
    main()
