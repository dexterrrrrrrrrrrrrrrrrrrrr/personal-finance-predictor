from __future__ import annotations

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor


def _rule_based_category(description: str) -> str:
    d = (description or "").lower()
    rules = [
        ("rent", "Rent"),
        ("uber", "Travel"),
        ("rideshare", "Travel"),
        ("bus", "Travel"),
        ("metro", "Travel"),
        ("train", "Travel"),
        ("grocery", "Food"),
        ("cafe", "Food"),
        ("coffee", "Food"),
        ("lunch", "Food"),
        ("dinner", "Food"),
        ("takeout", "Food"),
        ("internet", "Bills"),
        ("electric", "Bills"),
        ("water bill", "Bills"),
        ("mobile", "Bills"),
        ("streaming", "Entertainment"),
        ("movie", "Entertainment"),
        ("concert", "Entertainment"),
        ("gym", "Health"),
        ("pharmacy", "Health"),
        ("clinic", "Health"),
        ("textbook", "Shopping"),
        ("notebook", "Shopping"),
        ("headphones", "Shopping"),
        ("usb", "Shopping"),
    ]
    for key, cat in rules:
        if key in d:
            return cat
    return "Other"


def get_or_train_text_classifier(df: pd.DataFrame) -> Pipeline:
    labeled = df.copy()
    labeled["category"] = labeled["category"].fillna("").astype(str)
    labeled = labeled[labeled["category"].str.strip() != ""]

    if labeled["category"].nunique() < 2 or len(labeled) < 12:
        pipe = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression(max_iter=200))])
        pipe._student_fallback_rules = True  # type: ignore[attr-defined]
        return pipe

    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
            ("clf", LogisticRegression(max_iter=200)),
        ]
    )
    pipe.fit(labeled["description"].astype(str), labeled["category"].astype(str))
    pipe._student_fallback_rules = False  # type: ignore[attr-defined]
    return pipe


def predict_categories(model: Pipeline, descriptions: list[str]) -> list[str]:
    if getattr(model, "_student_fallback_rules", False):
        return [_rule_based_category(d) for d in descriptions]
    return list(model.predict(descriptions))


def _monthly_series(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["month"] = d["date"].dt.to_period("M")
    s = d.groupby("month")["amount"].sum().sort_index()
    out = s.reset_index()
    out["month"] = out["month"].astype(str)
    out = out.rename(columns={"amount": "monthly_total"})
    return out


def _make_features(monthly_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    s = monthly_df["monthly_total"].astype(float)
    prev1 = s.shift(1)
    roll3 = s.shift(1).rolling(3).mean()
    X = pd.DataFrame({"prev1": prev1, "roll3": roll3}).bfill().fillna(0.0)
    y = s.values
    return X.values, y


def get_or_train_monthly_regressor(df: pd.DataFrame, model_name: str = "RandomForestRegressor"):
    monthly_df = _monthly_series(df)
    if len(monthly_df) < 3:
        raise ValueError("Need at least 3 months of data to train a monthly prediction model.")

    X, y = _make_features(monthly_df)

    if model_name == "LinearRegression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=250, random_state=42)

    model.fit(X[:-1], y[:-1])
    return model


def predict_next_month_spend(model, df: pd.DataFrame, model_name: str = "RandomForestRegressor") -> dict:
    monthly_df = _monthly_series(df)
    X, _ = _make_features(monthly_df)

    last_features = X[-1].reshape(1, -1)
    pred = float(model.predict(last_features)[0])

    last_month = pd.Period(monthly_df.iloc[-1]["month"])
    next_month = (last_month + 1).strftime("%Y-%m")

    return {"prediction": max(0.0, pred), "next_month": next_month, "n_months": int(len(monthly_df))}