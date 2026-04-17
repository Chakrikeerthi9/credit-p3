import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import io

API_URL = "http://localhost:8001"

st.set_page_config(
    page_title="Credit Risk ML",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Credit Risk ML Dashboard")

tab1, tab2, tab3, tab4 = st.tabs([
    "Single Prediction",
    "Batch Scoring",
    "Audit Log",
    "Model Info"
])

with tab1:
    st.header("Single Applicant Prediction")
    col1, col2 = st.columns(2)

    with col1:
        loan_type = st.selectbox("Loan Type", ["Cash loans", "Revolving loans"])
        age_years = st.slider("Age", 18, 70, 35)
        income_total = st.number_input("Annual Income", 10000, 1000000, 135000, step=5000)
        loan_amount = st.number_input("Loan Amount", 10000, 5000000, 500000, step=10000)
        employment_years = st.slider("Years Employed", 0, 40, 4)

    with col2:
        education = st.selectbox("Education", [
        "Higher education", "Secondary", 
            "Incomplete higher", "Lower secondary"
        ])
        family_status = st.selectbox("Family Status", [
            "Married", "Single", "Separated", "Widow"
        ])
        owns_property = st.radio("Owns Property", ["Y", "N"], horizontal=True)
        owns_car = st.radio("Owns Car", ["Y", "N"], horizontal=True)
        ext_source_2 = st.slider("Credit Score (External)", 0.0, 1.0, 0.5, 0.01)

    if st.button("Predict Risk", type="primary"):
        payload = {
            "loan_type": loan_type,
            "age_years": age_years,
            "income_total": income_total,
            "loan_amount": loan_amount,
            "employment_years": employment_years,
            "education": education,
            "family_status": family_status,
            "owns_property": owns_property,
            "owns_car": owns_car,
            "ext_source_2": ext_source_2
        }
        with st.spinner("Scoring..."):
            try:
                r = requests.post(f"{API_URL}/predict", json=payload)
                result = r.json()

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Risk Score", f"{result['risk_score']:.4f}")
                with col_b:
                    color = {"APPROVE": "🟢", "REVIEW": "🟡", "DENY": "🔴"}
                    st.metric("Decision", f"{color[result['decision']]} {result['decision']}")
                with col_c:
                    st.metric("Processing Time", f"{result['processing_ms']}ms")

                st.subheader("Top Risk Factors")
                for i, reason in enumerate(result["top_reasons"], 1):
                    st.write(f"**{i}.** {reason}")

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result["risk_score"],
                    domain={"x": [0, 1], "y": [0, 1]},
                    gauge={
                        "axis": {"range": [0, 1]},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 0.3], "color": "green"},
                            {"range": [0.3, 0.6], "color": "yellow"},
                            {"range": [0.6, 1], "color": "red"}
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 4},
                            "thickness": 0.75,
                            "value": result["risk_score"]
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")

with tab2:
    st.header("Batch CSV Scoring")
    st.info("Upload a CSV with applicant data. Max 5,000 rows.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df_preview = pd.read_csv(uploaded)
        st.write(f"Rows: {len(df_preview)}")
        st.dataframe(df_preview.head())

        if st.button("Score Batch", type="primary"):
            uploaded.seek(0)
            with st.spinner(f"Scoring {len(df_preview)} applicants..."):
                try:
                    r = requests.post(
                        f"{API_URL}/predict/batch",
                        files={"file": uploaded}
                    )
                    result = r.json()

                    col_a, col_b, col_c, col_d = st.columns(4)
                    col_a.metric("Total", result["total"])
                    col_b.metric("✅ Approved", result["approved"])
                    col_c.metric("⚠️ Review", result["review"])
                    col_d.metric("❌ Denied", result["denied"])

                    fig = px.pie(
                        values=[result["approved"], result["review"], result["denied"]],
                        names=["Approved", "Review", "Denied"],
                        color_discrete_map={
                            "Approved": "green",
                    "Review": "yellow",
                            "Denied": "red"
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    results_df = pd.DataFrame(result["results"])
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "Download Results",
                        csv,
                        "batch_results.csv",
                        "text/csv"
                    )
                except Exception as e:
                    st.error(f"Error: {e}")

with tab3:
    st.header("Audit Log")
    if st.button("Refresh"):
        st.rerun()

    try:
        r = requests.get(f"{API_URL}/audit")
        data = r.json()
        if data:
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)

            r2 = requests.get(f"{API_URL}/audit/stats")
            stats = r2.json()
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Total", stats["total_predictions"])
            col_b.metric("Approved", stats["approved"])
            col_c.metric("Review", stats["review"])
            col_d.metric("Denied", stats["denied"])
        else:
            st.info("No predictions yet")
    except Exception as e:
        st.error(f"Error: {e}")

with tab4:
    st.header("Model Information")
    try:
        r = requests.get(f"{API_URL}/model/info")
        meta = r.json()

        st.success(f"Status: {meta.get('status', 'unknown')}")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("AUC-ROC", meta.get("auc_roc", "N/A"))
        col_b.metric("KS Statistic", meta.get("ks_statistic", "N/A"))
        col_c.metric("Gini", meta.get("gini", "N/A"))

        st.divider()

        col_d, col_e, col_f = st.columns(3)
        col_d.metric("XGBoost CV AUC", meta.get("xgb_cv_auc", "N/A"))
        col_e.metric("LightGBM CV AUC", meta.get("lgb_cv_auc", "N/A"))
        col_f.metric("Features", meta.get("features", "N/A"))

        st.divider()
        st.subheader("Raw metadata")
        st.json(meta)

    except Exception as e:
        st.error(f"Error: {e}")
