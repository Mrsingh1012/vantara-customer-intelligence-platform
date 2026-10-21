from pathlib import Path
import pandas as pd

path = Path("data/interim/transactions_cleaned.csv")
df = pd.read_csv(path, parse_dates=["InvoiceDate"])

print("Rows:", len(df))
print("First transaction:", df["InvoiceDate"].min())
print("Last transaction:", df["InvoiceDate"].max())
print("Unique customers:", df["Customer_ID"].nunique())
