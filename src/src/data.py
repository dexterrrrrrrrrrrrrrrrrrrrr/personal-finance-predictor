import io
import pandas as pd

REQUIRED_COLS = ["date", "amount", "description"]


def load_sample_data() -> pd.DataFrame:
    df = pd.read_csv("data/sample_expenses.csv")
    return normalize_expenses_df(df)


def parse_uploaded_csv(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    return pd.read_csv(io.BytesIO(raw))


def validate_expenses_df(df: pd.DataFrame, require_category: bool = False) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Required: {REQUIRED_COLS}.")

    if require_category and "category" not in df.columns:
        raise ValueError("Missing 'category' column. Add it or run auto-categorize first.")

    if df["amount"].isna().any():
        raise ValueError("Column 'amount' has missing values.")
    if (pd.to_numeric(df["amount"], errors="coerce").isna()).any():
        raise ValueError("Column 'amount' must be numeric.")

    if df["date"].isna().any():
        raise ValueError("Column 'date' has missing values.")


def normalize_expenses_df(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()

    if "category" not in df2.columns:
        df2["category"] = ""

    df2["date"] = pd.to_datetime(df2["date"], errors="coerce")
    if df2["date"].isna().any():
        bad_rows = df2[df2["date"].isna()].head(5)
        raise ValueError(f"Some dates could not be parsed. Example rows:\n{bad_rows}")

    df2["amount"] = pd.to_numeric(df2["amount"], errors="coerce")
    if df2["amount"].isna().any():
        bad_rows = df2[df2["amount"].isna()].head(5)
        raise ValueError(f"Some amounts could not be parsed. Example rows:\n{bad_rows}")

    df2["description"] = df2["description"].astype(str)
    df2["category"] = df2["category"].fillna("").astype(str)

    df2 = df2.sort_values("date").reset_index(drop=True)
    return df2