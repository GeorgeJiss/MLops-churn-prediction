"""
Monitoring dashboard for the churn model.

Visualizes:
  - Drift trend over time (from monitoring/drift_history.csv, written by drift_check.py)
  - Recent prediction volume + churn rate (from monitoring/predictions.csv, written by serve.py)

Run:
    streamlit run monitoring/dashboard.py
"""

import os

import pandas as pd
import plotly.express as px
import streamlit as st

DRIFT_HISTORY_PATH = "monitoring/drift_history.csv"
PREDICTIONS_PATH = "monitoring/predictions.csv"
DRIFT_ALERT_THRESHOLD = 0.5  # keep in sync with drift_check.py's default

st.set_page_config(page_title="Churn Model Monitoring", layout="wide")
st.title("Churn Model — Production Monitoring")


def load_csv_or_none(path: str) -> pd.DataFrame | None:
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df


drift_df = load_csv_or_none(DRIFT_HISTORY_PATH)
pred_df = load_csv_or_none(PREDICTIONS_PATH)

col1, col2, col3 = st.columns(3)

if pred_df is not None:
    pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])
    col1.metric("Total predictions logged", len(pred_df))
    col2.metric("Predicted churn rate", f"{(pred_df['prediction'] == 1).mean():.1%}")
else:
    col1.metric("Total predictions logged", 0)
    col2.metric("Predicted churn rate", "—")

if drift_df is not None:
    latest_share = drift_df["drifted_columns_share"].iloc[-1]
    col3.metric(
        "Latest drift share",
        f"{latest_share:.1%}",
        delta="ALERT" if latest_share >= DRIFT_ALERT_THRESHOLD else "OK",
        delta_color="inverse",
    )
else:
    col3.metric("Latest drift share", "—")

st.divider()

# --- Drift trend ---
st.subheader("Drift Over Time")
if drift_df is not None:
    drift_df["timestamp"] = pd.to_datetime(drift_df["timestamp"])
    fig = px.line(
        drift_df,
        x="timestamp",
        y="drifted_columns_share",
        markers=True,
        labels={"drifted_columns_share": "Share of drifted features"},
    )
    fig.add_hline(
        y=DRIFT_ALERT_THRESHOLD,
        line_dash="dash",
        line_color="red",
        annotation_text="Retrain threshold",
    )
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    st.plotly_chart(fig, width='stretch')
else:
    st.info("No drift history yet — run `python monitoring/drift_check.py` to generate the first data point.")

# --- Prediction volume + churn rate over time ---
st.subheader("Prediction Volume & Churn Rate")
if pred_df is not None:
    pred_df["date"] = pred_df["timestamp"].dt.date
    daily = pred_df.groupby("date").agg(
        volume=("prediction", "count"),
        churn_rate=("prediction", "mean"),
    ).reset_index()

    c1, c2 = st.columns(2)
    with c1:
        fig_vol = px.bar(daily, x="date", y="volume", title="Daily prediction volume")
        st.plotly_chart(fig_vol, width='stretch')
    with c2:
        fig_rate = px.line(
            daily, x="date", y="churn_rate", markers=True, title="Daily predicted churn rate"
        )
        fig_rate.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_rate, width='stretch')

    st.subheader("Recent Predictions")
    st.dataframe(
        pred_df.sort_values("timestamp", ascending=False).head(20),
        width='stretch',
    )
else:
    st.info("No predictions logged yet — call the `/predict` endpoint on the serving API to generate data.")

st.divider()
st.caption(
    "Drift is computed with Evidently's DataDriftPreset (K-S test for numeric features, "
    "chi-square for categorical). A run is flagged when the share of drifted features "
    f"crosses {DRIFT_ALERT_THRESHOLD:.0%}, matching drift_check.py's default threshold."
)
