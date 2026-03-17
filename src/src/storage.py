import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH = Path("data/app.db")


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
              username TEXT PRIMARY KEY,
              password_hash TEXT NOT NULL,
              created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS expenses (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              username TEXT NOT NULL,
              date TEXT NOT NULL,
              amount REAL NOT NULL,
              category TEXT,
              description TEXT,
              created_at TEXT DEFAULT CURRENT_TIMESTAMP,
              FOREIGN KEY(username) REFERENCES users(username)
            )
            """
        )
        conn.commit()


def save_expenses_to_db(username: str, df: pd.DataFrame) -> None:
    if not username:
        raise ValueError("No username provided.")

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM expenses WHERE username = ?", (username,))
        rows = []
        for _, r in df.iterrows():
            rows.append(
                (
                    username,
                    str(pd.to_datetime(r["date"]).date()),
                    float(r["amount"]),
                    str(r.get("category", "")),
                    str(r.get("description", "")),
                )
            )
        cur.executemany(
            "INSERT INTO expenses (username, date, amount, category, description) VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()


def load_expenses_from_db(username: str) -> pd.DataFrame | None:
    if not username:
        return None
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            "SELECT date, amount, category, description FROM expenses WHERE username = ? ORDER BY date",
            conn,
            params=(username,),
        )
    return df