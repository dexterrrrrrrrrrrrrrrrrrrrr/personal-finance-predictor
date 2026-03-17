import io
from datetime import datetime
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from src.preprocess import category_totals, monthly_totals, spending_kpis


def build_pdf_report_bytes(
    df: pd.DataFrame,
    prediction: float,
    next_month: str,
    threshold: float,
    tips: list[str],
    username: str,
) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    _, height = letter

    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Personal Finance Predictor (Student Edition) — Report")

    y -= 22
    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y -= 14
    c.drawString(50, y, f"User: {username}")

    k = spending_kpis(df)
    y -= 24
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Summary")
    y -= 16
    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Total spend: ${k['total_spend']:.2f}")
    y -= 14
    c.drawString(50, y, f"Top category: {k['top_category']} (${k['top_category_spend']:.2f})")
    y -= 14
    c.drawString(50, y, f"Months covered: {k['months_covered']}")

    y -= 24
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Prediction")
    y -= 16
    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Predicted spend for {next_month}: ${prediction:.2f}")
    y -= 14
    c.drawString(50, y, f"Budget threshold: ${threshold:.2f}")
    y -= 14
    c.drawString(
        50,
        y,
        "ALERT: You may overspend next month."
        if threshold > 0 and prediction > threshold
        else "No overspend alert triggered.",
    )

    y -= 24
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Category totals")
    y -= 16
    c.setFont("Helvetica", 10)
    ct = category_totals(df).head(10)
    for _, r in ct.iterrows():
        if y < 80:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 10)
        c.drawString(50, y, f"- {r['category']}: ${float(r['amount']):.2f}")
        y -= 14

    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Monthly totals")
    y -= 16
    c.setFont("Helvetica", 10)
    mt = monthly_totals(df).tail(12)
    for _, r in mt.iterrows():
        if y < 80:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 10)
        c.drawString(50, y, f"- {r['month']}: ${float(r['amount']):.2f}")
        y -= 14

    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Saving suggestions")
    y -= 16
    c.setFont("Helvetica", 10)
    for t in tips[:8]:
        if y < 80:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 10)
        text = t.strip()
        while len(text) > 95:
            c.drawString(50, y, text[:95])
            text = text[95:]
            y -= 14
        c.drawString(50, y, text)
        y -= 14

    c.showPage()
    c.save()
    return buf.getvalue()