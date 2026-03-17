import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
import streamlit as st

from src.auth import ensure_default_admin_exists, login_widget, logout_button, require_login
from src.data import load_sample_data, validate_expenses_df, normalize_expenses_df, parse_uploaded_csv
from src.models import (
    get_or_train_text_classifier,
    predict_categories,
    get_or_train_monthly_regressor,
    predict_next_month_spend,
)
from src.preprocess import category_totals, monthly_totals, spending_kpis
from src.report import build_pdf_report_bytes
from src.storage import init_db, save_expenses_to_db, load_expenses_from_db
from src.suggestions import generate_saving_suggestions, build_alerts

# ---------- Currency ----------
def get_currency_symbol(curr):
    return {"INR": "₹", "USD": "$", "EUR": "€"}[curr]

def convert_amount(amount, curr):
    rates = {"INR": 1, "USD": 1/83, "EUR": 1/90}
    return amount * rates[curr]

def convert_to_inr(amount, curr):
    rates = {"INR": 1, "USD": 83, "EUR": 90}
    return amount * rates[curr]

st.set_page_config(
    page_title="Personal Finance Predictor (Student Edition)",
    page_icon="💸",
    layout="wide",
)

# ---------- Init ----------
init_db()
ensure_default_admin_exists()

# ---------- Sidebar ----------
with st.sidebar:
    st.title("🔐 Login")
    login_widget()
    logout_button()
    require_login()

    st.markdown("---")
    st.caption("Default demo account")
    st.code("username: admin\npassword: admin123", language="text")

    st.markdown("---")
    currency = st.selectbox("💱 Currency", ["INR", "USD", "EUR"])
    symbol = get_currency_symbol(currency)

# ---------- Load Data ----------
if "expenses" not in st.session_state:
    db_df = load_expenses_from_db(st.session_state["auth_user"])
    if db_df is not None and len(db_df) > 0:
        st.session_state["expenses"] = normalize_expenses_df(db_df)
    else:
        st.session_state["expenses"] = load_sample_data()

st.title("Personal Finance Predictor (Student Edition)")
st.caption(f"All values shown in {currency}. Internally stored in INR.")

tab_input, tab_dashboard, tab_report = st.tabs(["➕ Input", "📊 Dashboard", "🧾 Report"])

# ================= INPUT =================
with tab_input:
    c1, c2 = st.columns([1.1, 0.9])

    with c1:
        st.subheader("Manual expense entry")
        with st.form("manual_entry", clear_on_submit=True):
            colA, colB, colC = st.columns([1, 1, 2])
            with colA:
                d = st.date_input("Date")
            with colB:
                amt_display = st.number_input(f"Amount ({symbol})", min_value=0.0, value=10.0)
                amt = convert_to_inr(amt_display, currency)
            with colC:
                desc = st.text_input("Description", value="Lunch")

            if st.form_submit_button("Add expense"):
                new_row = pd.DataFrame([{
                    "date": pd.to_datetime(d),
                    "amount": float(amt),
                    "description": str(desc),
                    "category": ""
                }])
                st.session_state["expenses"] = normalize_expenses_df(
                    pd.concat([st.session_state["expenses"], new_row], ignore_index=True)
                )
                st.success("Added expense")

# ================= DASHBOARD =================
with tab_dashboard:
    df = normalize_expenses_df(st.session_state["expenses"])
    validate_expenses_df(df, require_category=False)

    st.subheader("Spending Analysis")

    kpis = spending_kpis(df)
    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Total spend", f"{symbol}{convert_amount(kpis['total_spend'], currency):.2f}")
    k2.metric("Top category", kpis["top_category"])
    k3.metric("Top category spend", f"{symbol}{convert_amount(kpis['top_category_spend'], currency):.2f}")
    k4.metric("Months covered", kpis["months_covered"])

    ct = category_totals(df)
    mt = monthly_totals(df)

    # 🔥 Convert charts
    ct_display = ct.copy()
    ct_display["amount"] = ct_display["amount"].apply(lambda x: convert_amount(x, currency))

    mt_display = mt.copy()
    mt_display["amount"] = mt_display["amount"].apply(lambda x: convert_amount(x, currency))

    st.bar_chart(ct_display.set_index("category")["amount"])
    st.line_chart(mt_display.set_index("month")["amount"])

    # 🔥 Prediction
    st.subheader(f"Prediction ({symbol})")

    default_val = 80000.0 if currency == "INR" else 800.0
    threshold_display = st.number_input(f"Budget ({symbol})", value=default_val)
    threshold = convert_to_inr(threshold_display, currency)

    reg = get_or_train_monthly_regressor(df)
    pred = predict_next_month_spend(reg, df)

    st.metric(
        "Next month prediction",
        f"{symbol}{convert_amount(pred['prediction'], currency):.2f}"
    )

    for a in build_alerts(pred["prediction"], threshold):
        st.warning(a)

    # 🔥 Suggestions
    st.subheader("Smart saving suggestions")
    tips = generate_saving_suggestions(df)
    for t in tips:
        st.write(f"- {t}")

# ================= REPORT =================
with tab_report:
    df = normalize_expenses_df(st.session_state["expenses"])

    default_val = 80000.0 if currency == "INR" else 800.0
    threshold_display = st.number_input(f"Report budget ({symbol})", value=default_val)
    threshold = convert_to_inr(threshold_display, currency)

    if st.button("Generate PDF"):
        reg = get_or_train_monthly_regressor(df)
        pred = predict_next_month_spend(reg, df)
        tips = generate_saving_suggestions(df)

        pdf = build_pdf_report_bytes(
            df=df,
            prediction=pred["prediction"],
            next_month=pred["next_month"],
            threshold=threshold,
            tips=tips,
            username=st.session_state["auth_user"],
        )

        st.download_button("⬇️ Download PDF", pdf, "report.pdf")