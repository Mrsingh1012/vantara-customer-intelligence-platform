

"""Load and combine the Online Retail II workbook."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "online_retail_II.xlsx"

SHEETS = ["Year 2009-2010", "Year 2010-2011"]


def load_transactions() -> pd.DataFrame:
    """Load both sheets and sort transactions by date."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    frames = []

    for sheet in SHEETS:
        print(f"Loading worksheet: {sheet}", flush=True)

        frame = pd.read_excel(
            DATA_PATH,
            sheet_name=sheet,
            engine="openpyxl",
        )

        frame.columns = [
            str(column).strip().replace(" ", "_")
            for column in frame.columns
        ]

        print(f"Loaded {len(frame):,} rows from {sheet}", flush=True)
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)

    required_columns = {
        "Invoice",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "Price",
        "Customer_ID",
        "Country",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], errors="coerce"
    )

    df = df.sort_values(
        "InvoiceDate", kind="stable"
    ).reset_index(drop=True)

    return df


if __name__ == "__main__":
    transactions = load_transactions()

    print("\n--- Dataset Summary ---")
    print(f"Rows: {len(transactions):,}")
    print(f"Columns: {len(transactions.columns)}")
    print(
        f"Date range: {transactions['InvoiceDate'].min()} "
        f"to {transactions['InvoiceDate'].max()}"
    )
    print("\nColumn names:")
    print(transactions.columns.tolist())
    print("\nFirst 5 rows:")
    print(transactions.head().to_string(index=False))
