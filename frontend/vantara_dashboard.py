from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

st.set_page_config(
    page_title="Vantara | Customer Intelligence",
    page_icon="📊",
    layout="wide",
)

st.title("Vantara")
st.subheader("Customer Intelligence Platform")
st.caption(
    "Churn risk • Model evaluation • Retention campaign • Business scenarios"
)
st.info(
    "Historical evaluation is exploratory. Business-impact figures are "
    "assumption-based scenarios, not measured campaign ROI."
)


def load_report(filename):
    path = REPORTS / filename
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        st.warning(f"Could not read {filename}: {exc}")
        return None


metrics = load_report("churn_confusion_matrix.csv")
predictions = load_report("churn_test_customer_predictions.csv")
campaign = load_report("churn_campaign_ranked_demo.csv")
capacity = load_report("churn_rf_capacity_policy_test.csv")
impact = load_report("churn_campaign_business_impact.csv")

overview, performance, campaign_tab, business_tab = st.tabs(
    ["Overview", "Model Performance", "Campaign Ranking", "Business Impact"]
)

with overview:
    st.header("Executive Overview")

    total_customers = (
        campaign["customer_id"].nunique()
        if campaign is not None and "customer_id" in campaign.columns
        else 0
    )
    selected_count = (
        int(campaign["selected_for_campaign"].astype(str).str.lower().eq("true").sum())
        if campaign is not None and "selected_for_campaign" in campaign.columns
        else 0
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Customers scored", f"{total_customers:,}")
    c2.metric("Campaign selected", f"{selected_count:,}")
    c3.metric(
        "Not selected",
        f"{max(total_customers - selected_count, 0):,}",
    )

    st.markdown("### Available reports")
    report_status = pd.DataFrame(
        {
            "Report": [
                "Model confusion matrix",
                "Customer-level test predictions",
                "Ranked demo campaign",
                "Capacity policy comparison",
                "Business impact scenarios",
            ],
            "Status": [
                "Available" if x is not None else "Missing"
                for x in [metrics, predictions, campaign, capacity, impact]
            ],
        }
    )
    st.dataframe(report_status, hide_index=True, width="stretch")

with performance:
    st.header("Model Performance")
    if metrics is None:
        st.warning("Run save_churn_test_predictions.py to generate this report.")
    else:
        st.dataframe(metrics, hide_index=True, width="stretch")

        numeric_metrics = [
            c for c in ["precision", "recall", "f1", "roc_auc"]
            if c in metrics.columns
        ]
        if "model" in metrics.columns and numeric_metrics:
            chart_data = metrics.set_index("model")[numeric_metrics]
            st.markdown("### Metric comparison")
            st.bar_chart(chart_data)

        if all(c in metrics.columns for c in ["model", "TP", "FP", "TN", "FN"]):
            st.markdown("### Confusion counts")
            st.dataframe(
                metrics[["model", "TP", "FP", "TN", "FN"]],
                hide_index=True,
                width="stretch",
            )

    st.caption(
        "These historical test results were previously inspected; treat them "
        "as exploratory rather than an untouched final evaluation."
    )

with campaign_tab:
    st.header("Churn Campaign Ranking")
    if campaign is None:
        st.warning("Ranked campaign CSV not found.")
    else:
        df = campaign.copy()
        if "customer_id" in df.columns:
            df["customer_id"] = (
                df["customer_id"].astype(str).str.replace(r"\.0+$", "", regex=True)
            )

        if "churn_probability" in df.columns:
            df["churn_probability"] = pd.to_numeric(
                df["churn_probability"], errors="coerce"
            )

        if "selected_for_campaign" in df.columns:
            selected_values = df["selected_for_campaign"].astype(str).str.lower()
            selected_df = df[selected_values.eq("true")].copy()
        else:
            selected_df = df.iloc[0:0].copy()

        c1, c2 = st.columns(2)
        c1.metric("Scored customers", f"{len(df):,}")
        c2.metric("Campaign customers", f"{len(selected_df):,}")

        if "churn_probability" in df.columns:
            min_risk = float(df["churn_probability"].min())
            max_risk = float(df["churn_probability"].max())
            risk_range = st.slider(
                "Filter by churn probability",
                min_value=0.0,
                max_value=1.0,
                value=(max(0.0, min_risk), min(1.0, max_risk)),
                step=0.01,
            )
            df = df[df["churn_probability"].between(*risk_range)]

        if "customer_id" in df.columns:
            customer_query = st.text_input("Search customer ID")
            if customer_query.strip():
                df = df[
                    df["customer_id"].str.contains(
                        customer_query.strip(), case=False, na=False
                    )
                ]

        st.markdown("### Customer results")
        st.dataframe(
            df.sort_values(
                "churn_probability", ascending=False
            ) if "churn_probability" in df.columns else df,
            hide_index=True,
            width="stretch",
        )

        st.download_button(
            "Download filtered customers CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="vantara_filtered_customers.csv",
            mime="text/csv",
        )

        st.markdown("### Selected campaign list")
        st.dataframe(
            selected_df,
            hide_index=True,
            width="stretch",
        )
        st.download_button(
            "Download campaign selection CSV",
            data=selected_df.to_csv(index=False).encode("utf-8"),
            file_name="vantara_campaign_selection.csv",
            mime="text/csv",
        )

with business_tab:
    st.header("Campaign Business Impact")
    if impact is None:
        st.warning("Business impact CSV not found.")
    else:
        st.caption(
            "All values below depend on assumed retention success rates and "
            "customer values. They are not measured results."
        )
        st.dataframe(impact, hide_index=True, width="stretch")

        required = {
            "policy",
            "assumed_retention_success_rate",
            "estimated_campaign_cost_inr",
            "estimated_retained_value_inr",
            "estimated_net_value_inr",
        }
        if required.issubset(impact.columns):
            scenario = impact.copy()
            scenario["assumed_retention_success_rate"] = pd.to_numeric(
                scenario["assumed_retention_success_rate"], errors="coerce"
            )
            rates = sorted(
                scenario["assumed_retention_success_rate"].dropna().unique()
            )
            chosen_rate = st.selectbox(
                "Assumed retention success rate",
                rates,
                index=rates.index(0.10) if 0.10 in rates else 0,
                format_func=lambda x: f"{x:.0%}",
            )
            view = scenario[
                scenario["assumed_retention_success_rate"] == chosen_rate
            ].copy()

            c1, c2 = st.columns(2)
            if not view.empty:
                c1.metric(
                    "Lowest estimated net value",
                    f"₹{view['estimated_net_value_inr'].min():,.0f}",
                )
                c2.metric(
                    "Highest estimated net value",
                    f"₹{view['estimated_net_value_inr'].max():,.0f}",
                )

            st.markdown("### Estimated net value by policy")
            if not view.empty:
                st.bar_chart(view.set_index("policy")["estimated_net_value_inr"])

            st.dataframe(view, hide_index=True, width="stretch")

    st.caption(
        "Do not interpret scenario estimates as causal lift or actual ROI. "
        "Validate campaign outcomes using a controlled experiment."
    )

st.divider()
st.caption("Vantara Customer Intelligence Platform • Local analytics dashboard")

