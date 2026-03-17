import pandas as pd
from src.preprocess import category_totals

CATEGORY_TIPS = {
    "Food": "Try meal-prep 2–3 days a week and set a weekly food budget.",
    "Travel": "Use student passes, carpool, and batch errands to reduce trips.",
    "Shopping": "Wait 24 hours before buying non-essentials; use a wishlist.",
    "Bills": "Review subscriptions and switch to student plans where possible.",
    "Entertainment": "Pick one paid activity per week; use free campus events.",
    "Health": "Compare generic brands and track recurring health memberships.",
    "Rent": "If possible, renegotiate or consider a roommate split optimization.",
    "Other": "Tag unclear purchases with better descriptions to spot patterns.",
    "Uncategorized": "Add a category so the dashboard can give better insights.",
}


def generate_saving_suggestions(df: pd.DataFrame, high_share_threshold: float = 0.35) -> list[str]:
    if len(df) == 0:
        return []

    ct = category_totals(df)
    total = float(ct["amount"].sum())
    if total <= 0:
        return []

    suggestions = []
    for _, row in ct.iterrows():
        cat = str(row["category"])
        amt = float(row["amount"])
        share = amt / total
        if share >= high_share_threshold:
            reduce_by = 0.10
            suggestions.append(
                f"You spent {share*100:.0f}% on {cat} (${amt:.2f}). "
                f"Try reducing by {reduce_by*100:.0f}% next month (~${amt*reduce_by:.2f}). "
                f"{CATEGORY_TIPS.get(cat, '')}"
            )

    if not suggestions:
        suggestions.append("Great balance! Set small per-category budgets to keep it consistent.")
    return suggestions


def build_alerts(prediction: float, threshold: float) -> list[str]:
    alerts = []
    if threshold > 0 and prediction > threshold:
        alerts.append(f"⚠️ You may overspend next month: predicted ${prediction:.2f} vs budget ${threshold:.2f}.")
    return alerts