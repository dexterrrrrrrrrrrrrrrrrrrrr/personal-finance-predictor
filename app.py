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

st.set_page_config(
    page_title="Personal Finance Predictor (Student Edition)",
    page_icon="💸",
    layout="wide",
)

# ---------- Init (SQLite + default user) ----------
init_db()
ensure_default_admin_exists()

# ---------- Sidebar Auth ----------
with st.sidebar:
    st.title("🔐 Login")
    login_widget()
    logout_button()
    require_login()

    st.markdown("---")
    st.caption("Default demo account")
    st.code("username: admin\npassword: admin123", language="text")

# ---------- Load session data ----------
if "expenses" not in st.session_state:
    db_df = load_expenses_from_db(st.session_state["auth_user"])
    if db_df is not None and len(db_df) > 0:
        st.session_state["expenses"] = normalize_expenses_df(db_df)
    else:
        st.session_state["expenses"] = load_sample_data()

st.title("Personal Finance Predictor (Student Edition)")
st.caption("Track expenses, auto-categorize, analyze spending, predict next month, and get saving tips.")

tab_input, tab_dashboard, tab_report = st.tabs(["➕ Input", "📊 Dashboard", "🧾 Report"])

# =========================
# Input
# ==========================
with tab_input:
    c1, c2 = st.columns([1.1, 0.9], vertical_alignment="top")

    with c1:
        st.subheader("Manual expense entry")
        with st.form("manual_entry", clear_on_submit=True):
            colA, colB, colC = st.columns([1, 1, 2])
            with colA:
                d = st.date_input("Date")
            with colB:
                amt = st.number_input("Amount ($)", min_value=0.0, value=10.0, step=1.0)
            with colC:
                desc = st.text_input("Description", value="Lunch at campus cafe")

            submitted = st.form_submit_button("Add expense")
            if submitted:
                new_row = pd.DataFrame(
                    [{"date": pd.to_datetime(d), "amount": float(amt), "description": str(desc), "category": ""}]
                )
                df = pd.concat([st.session_state["expenses"], new_row], ignore_index=True)
                st.session_state["expenses"] = normalize_expenses_df(df)
                st.success("Added expense.")

    with c2:
        st.subheader("Upload expenses CSV")
        st.caption("Required columns: date, amount, description. Optional: category.")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded is not None:
            try:
                up_df = parse_uploaded_csv(uploaded)
                up_df = normalize_expenses_df(up_df)
                st.session_state["expenses"] = normalize_expenses_df(
                    pd.concat([st.session_state["expenses"], up_df], ignore_index=True)
                )
                st.success(f"Uploaded {len(up_df)} rows.")
            except Exception as e:
                st.error(f"Upload failed: {e}")

    st.markdown("---")
    st.subheader("Current expenses (editable)")
    st.caption("You can edit categories after auto-categorization.")
    edited = st.data_editor(
        st.session_state["expenses"],
        use_container_width=True,
        num_rows="dynamic",
        hide_index=True,
        column_config={
            "date": st.column_config.DateColumn("date"),
            "amount": st.column_config.NumberColumn("amount", format="%.2f"),
            "description": st.column_config.TextColumn("description"),
            "category": st.column_config.TextColumn("category"),
        },
        key="expenses_editor",
    )
    st.session_state["expenses"] = normalize_expenses_df(edited)

    btn1, btn2, btn3 = st.columns([1, 1, 2])
    with btn1:
        if st.button("🤖 Auto-categorize", use_container_width=True):
            try:
                df = st.session_state["expenses"]
                validate_expenses_df(df, require_category=False)

                clf = get_or_train_text_classifier(df)
                preds = predict_categories(clf, df["description"].astype(str).tolist())

                df2 = df.copy()
                mask = df2["category"].isna() | (df2["category"].astype(str).str.strip() == "")
                df2.loc[mask, "category"] = [p for i, p in enumerate(preds) if mask.iloc[i]]
                st.session_state["expenses"] = normalize_expenses_df(df2)
                st.success("Categorization complete.")
            except Exception as e:
                st.error(f"Categorization failed: {e}")

    with btn2:
        if st.button("💾 Save to SQLite", use_container_width=True):
            try:
                df = st.session_state["expenses"]
                validate_expenses_df(df, require_category=False)
                save_expenses_to_db(st.session_state["auth_user"], df)
                st.success("Saved.")
            except Exception as e:
                st.error(f"Save failed: {e}")

    with btn3:
        if st.button("🧹 Reset to sample data", use_container_width=True):
            st.session_state["expenses"] = load_sample_data()
            st.info("Reset complete.")

# =========================
# Dashboard
# =========================
with tab_dashboard:
    df = normalize_expenses_df(st.session_state["expenses"])
    try:
        validate_expenses_df(df, require_category=False)
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.subheader("Spending analysis")

    kpis = spending_kpis(df)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total spend", f"${kpis['total_spend']:.2f}")
    k2.metric("Top category", kpis["top_category"])
    k3.metric("Top category spend", f"${kpis['top_category_spend']:.2f}")
    k4.metric("Months covered", str(kpis["months_covered"]))

    ct = category_totals(df)
    mt = monthly_totals(df)

    cA, cB = st.columns([1, 1])
    with cA:
        st.markdown("**Category totals**")
        st.bar_chart(ct.set_index("category")["amount"])
    with cB:
        st.markdown("**Monthly trend**")
        st.line_chart(mt.set_index("month")["amount"])

    st.markdown("**Category share (pie)**")
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.pie(ct["amount"], labels=ct["category"], autopct="%1.1f%%")
        ax.axis("equal")
        st.pyplot(fig, use_container_width=True)
    except Exception:
        st.caption("Pie chart needs matplotlib (included in requirements.txt).")

    st.markdown("---")
    st.subheader("Next-month spending prediction")

    left, right = st.columns([1, 1])
    with left:
        threshold = st.number_input("Monthly budget threshold ($)", min_value=0.0, value=800.0, step=50.0)
        model_choice = st.selectbox("Regression model", ["RandomForestRegressor", "LinearRegression"], index=0)

    with right:
        st.caption("Prediction uses monthly totals from your history. More months = better results.")

    try:
        reg = get_or_train_monthly_regressor(df, model_name=model_choice)
        pred = predict_next_month_spend(reg, df, model_name=model_choice)

        st.metric("Predicted next month spend", f"${pred['prediction']:.2f}")
        st.caption(f"Based on {pred['n_months']} months of data. Next month: {pred['next_month']}.")

        for a in build_alerts(prediction=pred["prediction"], threshold=threshold):
            st.warning(a)

    except Exception as e:
        st.error(f"Prediction unavailable: {e}")

    st.markdown("---")
    st.subheader("Smart saving suggestions")
    tips = generate_saving_suggestions(df)
    for t in tips:
        st.write(f"- {t}")

# =========================
# Report
# =========================
with tab_report:
    st.subheader("Download report (PDF)")
    df = normalize_expenses_df(st.session_state["expenses"])

    threshold = st.number_input(
        "Include budget threshold in report ($)",
        min_value=0.0,
        value=800.0,
        step=50.0,
        key="report_threshold",
    )
    model_choice = st.selectbox(
        "Model for report",
        ["RandomForestRegressor", "LinearRegression"],
        index=0,
        key="report_model",
    )

    if st.button("Generate PDF report"):
        try:
            reg = get_or_train_monthly_regressor(df, model_name=model_choice)
            pred = predict_next_month_spend(reg, df, model_name=model_choice)
            tips = generate_saving_suggestions(df)

            pdf_bytes = build_pdf_report_bytes(
                df=df,
                prediction=pred["prediction"],
                next_month=pred["next_month"],
                threshold=threshold,
                tips=tips,
                username=st.session_state["auth_user"],
            )
            st.success("Report generated.")
            st.download_button(
                "⬇️ Download PDF",
                data=pdf_bytes,
                file_name="personal_finance_report.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.error(f"Report generation failed: {e}")

