import streamlit as st
import pandas as pd
from src.loader import DataLoader
from src.analyzer import DataAnalyzer
from src.visualizer import DataVisualizer
from src.recommender import InsightRecommender

st.set_page_config(
    page_title="DataVisualizer ",
    page_icon="📊",
    layout="wide"
)

st.title("DataVisualizer")
st.markdown("---")
st.sidebar.header("Data Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file:
    st.success("File uploaded successfully!")
    df = DataLoader.load_data(uploaded_file)
    analyzer = DataAnalyzer(df)
    visualizer = DataVisualizer(df)
    stats = analyzer.get_baisc_stats()
    
    st.subheader("Dataset Health Report")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", stats['rows'])
    m2.metric("Columns", stats['columns'])
    m3.metric("Missing Values", stats['missing_values'])
    m4.metric("Duplicates", stats['duplicates'])
    st.markdown("### Feature Quality")
    missing_report = analyzer.get_missing_report()
    
    col_a, col_b = st.columns(2)
    with col_a:
        if not missing_report.empty:
            st.warning("Missing Values Detected")
            st.dataframe(missing_report, use_container_width=True)
        else:
            st.success("No missing values found in features.")

    with col_b:
        types = analyzer.get_columns_type()
        st.write("**Data Composition:**")
        st.write(f"- Numeric Features: {len(types['numeric'])}")
        st.write(f"- Categorical Features: {len(types['categorical'])}")
        st.write(f"- Datetime Features: {len(types['datetime'])}")
    st.markdown("---")
    st.subheader("Class Distribution Analysis")
    target_col = st.selectbox("Select target column to check balance:", ["None"] + df.columns.tolist())
    
    imb_dist, imb_status = None, False
    if target_col != "None":
        imb_dist, imb_status = analyzer.imbalance_report(target_col)
        if imb_status:
            st.error(f"Imbalance Detected: Minority class is below 20% in '{target_col}'")
        st.write("Class Percentage Distribution:", imb_dist)

    st.markdown("---")
    if st.button("Generate Interactive Visualizations"):
        st.subheader("Data Insights")
        if len(types['numeric']) > 1:
            st.plotly_chart(visualizer.plot_correlation_matrix(types['numeric']), use_container_width=True)
        st.write("#### Feature Distributions")
        v_cols = st.columns(2)
        for i, col in enumerate(df.columns[:4]):
            with v_cols[i % 2]:
                st.plotly_chart(visualizer.plot_univariate_distribution(col), use_container_width=True)

    st.markdown("---")
    st.subheader("Technical Recommendations")
    health_summary = {
        'duplicate_count': stats['duplicates']
    }
    imb_info = {'imbalance_detected': imb_status}
    recommendations = InsightRecommender.generate_suggestions(health_summary, missing_report, imb_info)
    if recommendations:
        for rec in recommendations:
            st.info(rec)
    else:
        st.success("Data structure is healthy for baseline analysis.")

else:
    st.info("Upload a file to begin.")