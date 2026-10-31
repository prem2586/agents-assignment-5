
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

DB_PATH = Path(__file__).resolve().parent / "approval_audit.db"

st.set_page_config(
    page_title="Financial Approval Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("Financial Approval Dashboard")
st.caption("LangGraph Human-in-the-Loop Approval System")


def load_data():
    if not DB_PATH.exists():
        return pd.DataFrame()

    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT * FROM audit_log ORDER BY id DESC",
            conn,
        )


@st.fragment(run_every="5s")
def dashboard():
    df = load_data()

    if df.empty:
        st.info(
            "No approval events yet. "
            "Submit a financial request to generate data."
        )
        return

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce", utc=True
    )

    total = len(df)
    approved = (df["decision"] == "approved").sum()
    rejected = (df["decision"] == "rejected").sum()
    approval_rate = (
        approved / (approved + rejected) * 100
        if approved + rejected else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Decisions", total)
    col2.metric("Approved", int(approved))
    col3.metric("Rejected", int(rejected))
    col4.metric("Approval Rate", f"{approval_rate:.1f}%")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Decision Distribution")
        counts = df["decision"].value_counts()
        st.bar_chart(counts)

    with right:
        st.subheader("Reviews by Reviewer")
        reviewers = df["reviewer"].value_counts()
        st.bar_chart(reviewers)

    st.subheader("Recent Approval Activity")

    st.dataframe(
        df[
            [
                "timestamp",
                "request_id",
                "stage",
                "reviewer",
                "decision",
                "comments",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Automatically refreshes every 5 seconds."
    )


dashboard()
