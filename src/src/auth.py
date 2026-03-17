import hashlib
import hmac
import sqlite3
import streamlit as st
from src.storage import DB_PATH


def _hash_password(password: str, username: str) -> str:
    salt = username.encode("utf-8")
    pw = password.encode("utf-8")
    return hashlib.sha256(salt + b":" + pw).hexdigest()


def _verify_password(password: str, username: str, stored_hash: str) -> bool:
    candidate = _hash_password(password, username)
    return hmac.compare_digest(candidate, stored_hash)


def ensure_default_admin_exists() -> None:
    username = "admin"
    password = "admin123"
    password_hash = _hash_password(password, username)

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT username FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        if row is None:
            cur.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, password_hash),
            )
            conn.commit()


def _authenticate(username: str, password: str) -> bool:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        if row is None:
            return False
        return _verify_password(password, username, row[0])


def login_widget() -> None:
    if "auth_user" not in st.session_state:
        st.session_state["auth_user"] = ""

    if st.session_state["auth_user"]:
        st.success(f"Logged in as: {st.session_state['auth_user']}")
        return

    with st.form("login_form"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        ok = st.form_submit_button("Login")
        if ok:
            if _authenticate(u, p):
                st.session_state["auth_user"] = u
                st.success("Login successful.")
                st.rerun()
            else:
                st.error("Invalid username or password.")


def logout_button() -> None:
    if st.session_state.get("auth_user"):
        if st.button("Logout"):
            st.session_state["auth_user"] = ""
            st.rerun()


def require_login() -> None:
    if not st.session_state.get("auth_user"):
        st.info("Please log in to continue.")
        st.stop()