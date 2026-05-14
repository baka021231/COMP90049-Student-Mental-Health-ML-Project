from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "modeling_outputs"
DATA_PATH = OUT_DIR / "clean_model_data.csv"
FEATURE_SETS_PATH = OUT_DIR / "feature_sets.json"
RESULTS_PATH = OUT_DIR / "rq1_results.csv"


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    print("RQ1 modeling workflow scaffold is ready.")
    print(f"Input data: {DATA_PATH.relative_to(ROOT)}")
    print(f"Feature sets: {FEATURE_SETS_PATH.relative_to(ROOT)}")
    print(f"Results path: {RESULTS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
