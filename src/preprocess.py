import pandas as pd


def category_totals(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["category"] = d["category"].replace("", "Uncategorized")
    return d.groupby("category", as_index=False)["amount"].sum().sort_values("amount", ascending=False)


def monthly_totals(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["month"] = d["date"].dt.to_period("M").astype(str)
    return d.groupby("month", as_index=False)["amount"].sum().sort_values("month")


def spending_kpis(df: pd.DataFrame) -> dict:
    total = float(df["amount"].sum())
    ct = category_totals(df)
    top_cat = str(ct.iloc[0]["category"]) if len(ct) else "N/A"
    top_spend = float(ct.iloc[0]["amount"]) if len(ct) else 0.0
    months = df["date"].dt.to_period("M").nunique()
    return {
        "total_spend": total,
        "top_category": top_cat,
        "top_category_spend": top_spend,
        "months_covered": int(months),
    }