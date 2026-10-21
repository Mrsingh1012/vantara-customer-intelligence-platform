from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "online_retail_II.xlsx"

EXPECTED_SHEETS = [
    "Year 2009-2010",
    "Year 2010-2011",
]


def validate_workbook() -> None:
    """Validate the source workbook and required worksheets."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    with pd.ExcelFile(DATA_PATH) as excel_file:
        missing = set(EXPECTED_SHEETS) - set(excel_file.sheet_names)

        if missing:
            raise ValueError(f"Missing worksheets: {sorted(missing)}")

        print(f"Dataset: {DATA_PATH}")
        print(f"Worksheets: {excel_file.sheet_names}")
        print("Validation passed.")


if __name__ == "__main__":
    validate_workbook()