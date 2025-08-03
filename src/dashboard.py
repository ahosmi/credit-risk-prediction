import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

st.set_page_config(page_title="Credit Default Risk Dashboard", layout="wide")

@st.cache_data
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

def get_risk_level(prob):
    if prob < 0.25:
        return "🟢 Low", "Low"
    elif prob < 0.6:
        return "🟡 Medium", "Medium"
    else:
        return "🔴 High", "High"

def show_class_report(report_path):
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            st.sidebar.markdown("---\n**Model Classification Report:**\n")
            st.sidebar.text(f.read())

def main():
    MODEL_PATH = "models/xgb_pipeline.joblib"
    DATA_PATH = "data/Loan_Default_Processed.csv"
    REPORT_PATH = "models/classification_report.txt"

    model = load_model(MODEL_PATH)
    data = load_data(DATA_PATH)
    index_col = 'customer_id' if 'customer_id' in data.columns else data.index

    st.markdown("# :money_with_wings: Credit Default Risk Dashboard")

    # Sidebar
    st.sidebar.header("Options")
    selected_customer = st.sidebar.selectbox("Choose Customer", index_col)
    customer_row = data[data[index_col] == selected_customer] if 'customer_id' in data.columns else data.loc[[selected_customer]]

    show_class_report(REPORT_PATH)

    # Main columns
    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.subheader("Customer Summary")
        display_cols = ['Age', 'Gender', 'Monthly_Income', 'Loan_Amount']
        profile = customer_row[display_cols] if all(c in customer_row.columns for c in display_cols) else customer_row.iloc[:, :4]
        st.table(profile.T)

        X_input = customer_row.drop(columns=[col for col in [index_col, 'Label'] if col in customer_row.columns])
        prob = model.predict_proba(X_input)[0, 1]
        emoji, level = get_risk_level(prob)
        st.metric("Predicted Default Probability", f"{prob:.1%}", delta=emoji, delta_color="inverse")
        st.progress(prob)

    with col2:
        st.subheader("Feature Importance (for Current Customer)")
        feat_imp = model.named_steps['xgb'].feature_importances_
        feat_df = pd.DataFrame({
            'Feature': X_input.columns,
            'Importance': feat_imp
        }).sort_values('Importance', ascending=False)[:10]
        st.bar_chart(feat_df.set_index('Feature'))

        st.subheader("Customer Features")
        st.dataframe(X_input.T, width=600, height=400, use_container_width=True)

if __name__ == "__main__":
    main()
