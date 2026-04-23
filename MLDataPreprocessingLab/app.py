import streamlit as st
import pandas as pd
import numpy as np
import io
from utils.data_loader import DataLoader
from utils.visuals import get_detailed_audit, get_smart_recommendations
from src.processor import DataProcessor


st.set_page_config(
    page_title="ML Preprocessing Lab",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)


if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

def get_theme():
    if st.session_state.dark_mode:
        return {
            "bg":           "#0F1117",
            "card":         "#1A1F2E",
            "sidebar":      "#141820",
            "primary":      "#4DA3FF",
            "primary_dim":  "#1A3A5C",
            "text":         "#FAFAFA",
            "text_dim":     "#9AA0B4",
            "success":      "#2ECC71",
            "warning":      "#F39C12",
            "danger":       "#FF6B6B",
            "info":         "#4DA3FF",
            "border":       "#2A3040",
            "log_bg":       "#0A0D14",
            "log_text":     "#00FF88",
        }
    else:
        return {
            "bg":           "#F0F4F8",
            "card":         "#FFFFFF",
            "sidebar":      "#E8EEF4",
            "primary":      "#2E75B6",
            "primary_dim":  "#D6E4F0",
            "text":         "#1A1A2E",
            "text_dim":     "#555577",
            "success":      "#27AE60",
            "warning":      "#E67E22",
            "danger":       "#E74C3C",
            "info":         "#2980B9",
            "border":       "#D0D8E8",
            "log_bg":       "#1A1F2E",
            "log_text":     "#00FF88",
        }

def inject_css(t):
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;600;700;800&display=swap');

    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
        background-color: {t['bg']} !important;
        color: {t['text']} !important;
        font-family: 'Syne', sans-serif !important;
    }}
    [data-testid="stSidebar"] {{
        background-color: {t['sidebar']} !important;
        border-right: 1px solid {t['border']} !important;
    }}
    [data-testid="stSidebar"] * {{
        color: {t['text']} !important;
    }}
    .main .block-container {{
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px !important;
    }}
    h1, h2, h3, h4 {{
        font-family: 'Syne', sans-serif !important;
        color: {t['text']} !important;
    }}
    .stButton > button {{
        background-color: {t['primary']} !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 600 !important;
        padding: 0.45rem 1.2rem !important;
        transition: opacity 0.2s ease !important;
    }}
    .stButton > button:hover {{
        opacity: 0.85 !important;
    }}
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stNumberInput > div > div > input,
    .stSlider, .stCheckbox {{
        background-color: {t['card']} !important;
        color: {t['text']} !important;
        border-color: {t['border']} !important;
    }}
    .stDataFrame {{
        background-color: {t['card']} !important;
    }}
    div[data-testid="stMetric"] {{
        background-color: {t['card']} !important;
        border: 1px solid {t['border']} !important;
        border-radius: 10px !important;
        padding: 1rem 1.2rem !important;
    }}
    div[data-testid="stMetricValue"] {{
        color: {t['primary']} !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
    }}
    div[data-testid="stMetricLabel"] {{
        color: {t['text_dim']} !important;
    }}
    .st-expander {{
        background-color: {t['card']} !important;
        border: 1px solid {t['border']} !important;
        border-radius: 8px !important;
    }}
    .metric-card {{
        background: {t['card']};
        border: 1px solid {t['border']};
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.5rem;
    }}
    .metric-card .label {{
        font-size: 0.72rem;
        color: {t['text_dim']};
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }}
    .metric-card .value {{
        font-size: 1.6rem;
        font-weight: 800;
        color: {t['primary']};
        font-family: 'Syne', sans-serif;
        line-height: 1.1;
    }}
    .metric-card .sub {{
        font-size: 0.75rem;
        color: {t['text_dim']};
        margin-top: 0.2rem;
    }}
    .rec-card {{
        background: {t['card']};
        border-radius: 8px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.6rem;
        border-left: 4px solid;
    }}
    .rec-card.drop   {{ border-color: {t['danger']}; }}
    .rec-card.impute {{ border-color: {t['warning']}; }}
    .rec-card.skew   {{ border-color: {t['info']}; }}
    .rec-card.encode {{ border-color: {t['success']}; }}
    .rec-card.ok     {{ border-color: {t['success']}; }}
    .rec-card .rc-title {{
        font-weight: 700;
        font-size: 0.9rem;
        color: {t['text']};
    }}
    .rec-card .rc-body {{
        font-size: 0.82rem;
        color: {t['text_dim']};
        margin-top: 0.2rem;
    }}
    .badge {{
        display: inline-block;
        padding: 0.18rem 0.6rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.04em;
    }}
    .badge-healthy  {{ background: {t['success']}22; color: {t['success']}; }}
    .badge-missing  {{ background: {t['warning']}22; color: {t['warning']}; }}
    .badge-outlier  {{ background: {t['danger']}22;  color: {t['danger']}; }}
    .badge-const    {{ background: {t['text_dim']}22; color: {t['text_dim']}; }}
    .badge-cardin   {{ background: {t['info']}22;    color: {t['info']}; }}
    .badge-skewed   {{ background: {t['primary']}22; color: {t['primary']}; }}
    .log-box {{
        background: {t['log_bg']};
        border: 1px solid {t['border']};
        border-radius: 8px;
        padding: 1rem 1.2rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: {t['log_text']};
        max-height: 280px;
        overflow-y: auto;
        line-height: 1.7;
        white-space: pre-wrap;
    }}
    .log-box .log-err {{ color: #FF6B6B; }}
    .log-box .log-warn {{ color: #F39C12; }}
    .section-header {{
        font-size: 1.35rem;
        font-weight: 800;
        color: {t['text']};
        border-bottom: 2px solid {t['primary']};
        padding-bottom: 0.4rem;
        margin-bottom: 1.2rem;
        font-family: 'Syne', sans-serif;
    }}
    .hero-title {{
        font-size: 2.8rem;
        font-weight: 800;
        color: {t['text']};
        font-family: 'Syne', sans-serif;
        line-height: 1.1;
    }}
    .hero-title span {{ color: {t['primary']}; }}
    .hero-sub {{
        font-size: 1rem;
        color: {t['text_dim']};
        margin-top: 0.5rem;
        margin-bottom: 1.8rem;
    }}
    .upload-zone {{
        background: {t['card']};
        border: 2px dashed {t['primary']};
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
    }}
    .step-badge {{
        display: inline-block;
        background: {t['primary']};
        color: white;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 0.15rem 0.55rem;
        border-radius: 4px;
        margin-right: 0.5rem;
        font-family: 'JetBrains Mono', monospace;
    }}
    .divider {{
        border: none;
        border-top: 1px solid {t['border']};
        margin: 1.2rem 0;
    }}
    [data-testid="stFileUploader"] {{
        background-color: {t['card']} !important;
        border: 2px dashed {t['border']} !important;
        border-radius: 10px !important;
    }}
    </style>
    """, unsafe_allow_html=True)


for key, val in {
    "df": None,
    "file_info": None,
    "result": None,
    "logs": [],
    "pipeline_ran": False,
    "page": "Upload"
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


t = get_theme()
inject_css(t)

with st.sidebar:
    st.markdown(f"""
    <div style='padding: 0.5rem 0 1.2rem 0;'>
        <div style='font-size:1.4rem; font-weight:800; color:{t["primary"]}; font-family:Syne,sans-serif;'>⚗️ ML Prep Lab</div>
        <div style='font-size:0.72rem; color:{t["text_dim"]}; margin-top:0.2rem;'>Data Preprocessing Pipeline</div>
    </div>
    """, unsafe_allow_html=True)

    mode_label = " Light Mode" if st.session_state.dark_mode else " Dark Mode"
    if st.button(mode_label, use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    st.markdown("<hr style='border-color:#2A3040; margin:1rem 0;'>", unsafe_allow_html=True)

    pages = ["Upload", "Audit", "Recommendations", "Pipeline", "Export"]
    icons  = [" ", " ", " ", " ", " "]

    for pg, ic in zip(pages, icons):
        is_active = st.session_state.page == pg
        style = f"background:{t['primary_dim']}; border-left:3px solid {t['primary']};" if is_active else ""
        if st.button(f"{ic}  {pg}", key=f"nav_{pg}", use_container_width=True):
            st.session_state.page = pg
            st.rerun()

    st.markdown("<hr style='border-color:#2A3040; margin:1rem 0;'>", unsafe_allow_html=True)

    if st.session_state.df is not None:
        info = st.session_state.file_info
        st.markdown(f"""
        <div style='font-size:0.7rem; color:{t["text_dim"]}; line-height:2;'>
            <b style='color:{t["text"]}'>Loaded Dataset</b><br>
            Rows: <span style='color:{t["primary"]}'>{info['rows']}</span><br>
            Cols: <span style='color:{t["primary"]}'>{info['cols']}</span><br>
            Size: <span style='color:{t["primary"]}'>{info['size']}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='font-size:0.75rem; color:{t['text_dim']};'>No file loaded yet.</div>", unsafe_allow_html=True)


def page_upload():
    st.markdown(f"""
    <div class='hero-title'>ML Data <span>Preprocessing</span> Lab</div>
    <div class='hero-sub'>Upload your dataset and run a full ML preprocessing pipeline — clean, encode, scale, and export in minutes.</div>
    """, unsafe_allow_html=True)

    col1, col2, col3= st.columns(3)
    features = [
        (" ","Clean", "Handle nulls, mixed types, duplicates"),
        (" ","Encode", "One-hot, target, binary, ordinal"),
        (" ","Scale", "Standard, MinMax, Robust"),
        (" ", "RFE, Lasso, Mutual Info, Chi2"),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3], features):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size:1.6rem; margin-bottom:0.4rem;'>{icon}</div>
                <div style='font-weight:700; color:{t["text"]}; font-size:0.95rem;'>{title}</div>
                <div style='font-size:0.75rem; color:{t["text_dim"]}; margin-top:0.2rem;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Upload Dataset</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drag and drop your file here",
        type=["csv", "xlsx", "xls", "json", "parquet", "tsv"],
        help="Supported: CSV, Excel, JSON, Parquet, TSV"
    )

    if uploaded:
        loader = DataLoader()
        result = loader.load_file(uploaded)

        if isinstance(result, str):
            st.error(result)
            return

        st.session_state.df = result
        st.session_state.file_info = loader.get_basic_info(result)
        info = st.session_state.file_info

        st.success(f" **{uploaded.name}** loaded successfully.")
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Dataset Overview</div>", unsafe_allow_html=True)

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        metrics = [
                (m1, "Rows", f"{info.get('rows', 0):,}", ""),
                (m2, "Columns", f"{info.get('cols', 0):,}", ""),
                (m3, "Numeric", f"{len(info.get('numerical_cols', []))}", "features"),
                (m4, "Categorical", f"{len(info.get('categorical_cols', []))}", "features"),
                (m5, "Datetime", f"{len(info.get('datetime_cols', []))}", "features"), 
                (m6, "Missing Cols", f"{len(info.get('missing_cols', []))}", "columns"),
                ]
        for col, label, val, sub in metrics:
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='label'>{label}</div>
                    <div class='value'>{val}</div>
                    <div class='sub'>{sub}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.85rem; color:{t['text_dim']}; margin-bottom:0.5rem;'><b style='color:{t['text']}'>Preview</b> — first 5 rows</div>", unsafe_allow_html=True)
        st.dataframe(result.head(), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Proceed to Audit →", use_container_width=False):
            st.session_state.page = "Audit"
            st.rerun()

def page_audit():
    st.markdown("<div class='section-header'>Data Audit</div>", unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("No dataset loaded. Go to Upload first.")
        return

    df = st.session_state.df
    summary, health_df = get_detailed_audit(df)


    cols = st.columns(6)
    summary_items = list(summary.items())
    for col, (k, v) in zip(cols, summary_items):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='label'>{k}</div>
                <div style='font-size:1.1rem; font-weight:700; color:{t["primary"]}; font-family:Syne,sans-serif;'>{v}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)


    fc1, fc2 = st.columns([2, 2])
    with fc1:
        type_filter = st.selectbox("Filter by Type", ["All", "Numeric", "Categorical", "Issues Only"])
    with fc2:
        search = st.text_input("Search column", placeholder="Type column name...")

    filtered = health_df.copy()
    if type_filter == "Numeric":
        filtered = filtered[filtered["Type"].str.contains("int|float", case=False, na=False)]
    elif type_filter == "Categorical":
        filtered = filtered[filtered["Type"].str.contains("object|category", case=False, na=False)]
    elif type_filter == "Issues Only":
        filtered = filtered[filtered["Status"] != "Healthy"]
    if search:
        filtered = filtered[filtered["Column"].str.contains(search, case=False, na=False)]

    def color_status(val):
        if val == "Healthy":
            return f"color: {t['success']}; font-weight: 600;"
        elif "Missing" in val:
            return f"color: {t['warning']}; font-weight: 600;"
        elif "Outlier" in val:
            return f"color: {t['danger']}; font-weight: 600;"
        elif "Constant" in val:
            return f"color: {t['text_dim']}; font-weight: 600;"
        else:
            return f"color: {t['info']}; font-weight: 600;"

    styled = filtered.style.applymap(color_status, subset=["Status"])
    st.dataframe(styled, use_container_width=True, height=420)

    st.markdown(f"<div style='font-size:0.78rem; color:{t['text_dim']}; margin-top:0.4rem;'>Showing {len(filtered)} of {len(health_df)} columns</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("See Recommendations →"):
        st.session_state.page = "Recommendations"
        st.rerun()

def page_recommendations():
    st.markdown("<div class='section-header'>Smart Recommendations</div>", unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("No dataset loaded. Go to Upload first.")
        return

    recs = get_smart_recommendations(st.session_state.df)

    drop_recs   = [r for r in recs if "Drop" in r]
    impute_recs = [r for r in recs if "missing" in r.lower() or "impute" in r.lower() or "knn" in r.lower()]
    skew_recs   = [r for r in recs if "skew" in r.lower() or "transform" in r.lower()]
    encode_recs = [r for r in recs if "encod" in r.lower() or "cardinality" in r.lower()]
    other_recs  = [r for r in recs if r not in drop_recs + impute_recs + skew_recs + encode_recs]

    total_issues = len([r for r in recs if "No major issues" not in r])
    c1, c2, c3, c4 = st.columns(4)
    for col, label, count, color in [
        (c1, "Drop Columns", len(drop_recs), t["danger"]),
        (c2, "Impute", len(impute_recs), t["warning"]),
        (c3, "Transform", len(skew_recs), t["info"]),
        (c4, "Re-Encode", len(encode_recs), t["success"]),
    ]:
        with col:
            st.markdown(f"""
            <div class='metric-card' style='border-left: 3px solid {color};'>
                <div class='label'>{label}</div>
                <div class='value' style='color:{color};'>{count}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    def render_rec_cards(items, css_class, icon):
        for item in items:
            clean = item.replace("**", "").replace("*", "")
            parts = clean.split(":")
            col_name = parts[0].strip() if parts else ""
            suggestion = ":".join(parts[1:]).strip() if len(parts) > 1 else clean
            st.markdown(f"""
            <div class='rec-card {css_class}'>
                <div class='rc-title'>{icon} {col_name}</div>
                <div class='rc-body'>{suggestion}</div>
            </div>
            """, unsafe_allow_html=True)

    if drop_recs:
        st.markdown(f"<b style='color:{t['danger']}'>  Drop Candidates</b>", unsafe_allow_html=True)
        render_rec_cards(drop_recs, "drop", " ")
    if impute_recs:
        st.markdown(f"<b style='color:{t['warning']}'> Imputation Needed</b>", unsafe_allow_html=True)
        render_rec_cards(impute_recs, "impute", " ")
    if skew_recs:
        st.markdown(f"<b style='color:{t['info']}'>  Skewed Distributions</b>", unsafe_allow_html=True)
        render_rec_cards(skew_recs, "skew", " ")
    if encode_recs:
        st.markdown(f"<b style='color:{t['success']}'> Encoding Suggestions</b>", unsafe_allow_html=True)
        render_rec_cards(encode_recs, "encode", " ")
    if other_recs:
        render_rec_cards(other_recs, "ok", " ")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Go to Pipeline →"):
        st.session_state.page = "Pipeline"
        st.rerun()


def page_pipeline():
    st.markdown("<div class='section-header'>Preprocessing Pipeline</div>", unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("No dataset loaded. Go to Upload first.")
        return

    df = st.session_state.df
    all_cols = df.columns.tolist()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    config = {}

    st.markdown(f"<div style='font-size:0.85rem; color:{t['text_dim']}; margin-bottom:1rem;'>Enable the steps you want. Steps run in the fixed order shown below.</div>", unsafe_allow_html=True)

    with st.expander("STEP 1 — Remove Duplicates"):
        if st.checkbox("Enable", key="dup"):
            config["remove_duplicates"] = True

    with st.expander("STEP 2 — Impute Missing Values"):
        if st.checkbox("Enable", key="imp"):
            strategy = st.selectbox("Strategy", ["knn", "mean", "median", "most_frequent", "iterative"], key="imp_strat")
            n_neighbors = 5
            if strategy == "knn":
                n_neighbors = st.slider("KNN Neighbors", 2, 15, 5, key="knn_n")
            config["impute_params"] = {"strategy": strategy, "n_neighbors": n_neighbors}

    with st.expander("STEP 3 — Handle Outliers"):
        if st.checkbox("Enable", key="out"):
            out_cols = st.multiselect("Columns", num_cols, key="out_cols")
            oc1, oc2 = st.columns(2)
            with oc1:
                out_method = st.selectbox("Detection Method", ["iqr", "zscore", "iso_forest"], key="out_meth")
            with oc2:
                out_action = st.selectbox("Action", ["cap", "remove"], key="out_act")
            if out_cols:
                config["outlier_params"] = {"columns": out_cols, "method": out_method, "action": out_action}

    with st.expander("STEP 4 — Encode Categorical Features"):
        if st.checkbox("Enable", key="enc"):
            enc_type = st.selectbox("Encoding Type", ["onehot", "label", "ordinal", "binary", "frequency", "target"], key="enc_type")
            if enc_type == "target":
                enc_col  = st.selectbox("Column to encode", cat_cols, key="enc_col")
                enc_tgt  = st.selectbox("Target column", all_cols, key="enc_tgt")
                enc_smooth = st.slider("Smoothing", 1, 50, 10, key="enc_smooth")
                config["encode_params"] = {"type": "target", "column": enc_col, "target_column": enc_tgt, "smoothing": enc_smooth}
            else:
                enc_cols = st.multiselect("Columns to encode", cat_cols, key="enc_cols")
                if enc_cols:
                    config["encode_params"] = {"type": enc_type, "columns": enc_cols}

    with st.expander("STEP 5 — Feature Selection"):
        if st.checkbox("Enable", key="sel"):
            sel_target = st.selectbox("Target column", all_cols, key="sel_tgt")
            sel_task   = st.radio("Task type", ["classification", "regression"], horizontal=True, key="sel_task")

            use_var  = st.checkbox("Variance Filter", key="sel_var")
            use_corr = st.checkbox("Correlation Filter", key="sel_corr")

            sel_p = {"target": sel_target, "task": sel_task}
            if use_var:
                sel_p["use_variance"] = True
                sel_p["var_threshold"] = st.slider("Variance Threshold", 0.0, 0.1, 0.01, step=0.001, key="var_thresh")
            if use_corr:
                sel_p["use_correlation"] = True
                sel_p["corr_threshold"] = st.slider("Correlation Threshold", 0.7, 1.0, 0.9, step=0.01, key="corr_thresh")

            sel_method = st.selectbox("Selection Method (optional)", ["None", "rfe", "mutual_info", "lasso", "chi2"], key="sel_meth")
            if sel_method != "None":
                sel_p["method"] = sel_method
                if sel_method in ["rfe", "mutual_info", "chi2"]:
                    sel_p["k"] = st.slider("Features to keep", 2, min(20, len(num_cols)), 5, key="sel_k")
                    sel_p["n_features"] = sel_p["k"]

            config["selection_params"] = sel_p

    with st.expander("STEP 6 — Balance Classes"):
        if st.checkbox("Enable", key="bal"):
            bal_target = st.selectbox("Target column", all_cols, key="bal_tgt")
            bal_method = st.selectbox("Method", ["smote", "adasyn", "undersample"], key="bal_meth")
            config["balance_params"] = {"target": bal_target, "method": bal_method}

    with st.expander("STEP 7 — Scale Features"):
        if st.checkbox("Enable", key="scl"):
            scl_method = st.selectbox("Method", ["standard", "minmax", "robust"], key="scl_meth")
            config["scale_params"] = {"method": scl_method}

    with st.expander("STEP 8 — Transform Distributions"):
        if st.checkbox("Enable", key="trn"):
            trn_type = st.selectbox("Type", ["log", "power"], key="trn_type")
            trn_cols = st.multiselect("Columns", num_cols, key="trn_cols")
            trn_p = {"type": trn_type, "columns": trn_cols}
            if trn_type == "power":
                trn_p["method"] = st.selectbox("Power Method", ["yeo-johnson", "box-cox"], key="trn_pow")
            if trn_cols:
                config["transform_params"] = trn_p

    with st.expander("STEP 9 — Train / Test Split"):
        if st.checkbox("Enable", key="spl"):
            spl_target   = st.selectbox("Target column", all_cols, key="spl_tgt")
            spl_size     = st.slider("Test size", 0.1, 0.4, 0.2, step=0.05, key="spl_size")
            spl_stratify = st.checkbox("Stratify split", key="spl_strat")
            config["split_params"] = {"target": spl_target, "test_size": spl_size, "stratify": spl_stratify}

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    if st.button("▶  Run Pipeline", use_container_width=True):
        with st.spinner("Running pipeline..."):
            processor = DataProcessor()
            result, logs = processor.Pipeline(df, config)

        st.session_state.result = result
        st.session_state.logs = logs
        st.session_state.pipeline_ran = True

    if st.session_state.pipeline_ran:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-weight:700; color:{t['text']}; margin-bottom:0.5rem;'>Pipeline Log</div>", unsafe_allow_html=True)
        log_html = ""
        for line in st.session_state.logs:
            css = "log-err" if "Error" in line else ("log-warn" if "Warning" in line else "")
            log_html += f'<div class="{css}">&gt; {line}</div>'
        st.markdown(f"<div class='log-box'>{log_html}</div>", unsafe_allow_html=True)

        if st.session_state.result is not None:
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

            result = st.session_state.result
            if isinstance(result, tuple):
                train_df, test_df = result
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"<b style='color:{t['success']}'>Train Set</b> — {train_df.shape[0]} rows × {train_df.shape[1]} cols", unsafe_allow_html=True)
                    st.dataframe(train_df.head(), use_container_width=True)
                with c2:
                    st.markdown(f"<b style='color:{t['info']}'>Test Set</b> — {test_df.shape[0]} rows × {test_df.shape[1]} cols", unsafe_allow_html=True)
                    st.dataframe(test_df.head(), use_container_width=True)
            else:
                st.markdown(f"<b style='color:{t['success']}'>Processed Dataset</b> — {result.shape[0]} rows × {result.shape[1]} cols", unsafe_allow_html=True)
                st.dataframe(result.head(), use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Go to Export →"):
                st.session_state.page = "Export"
                st.rerun()


def page_export():
    st.markdown("<div class='section-header'>Export & Download</div>", unsafe_allow_html=True)

    if not st.session_state.pipeline_ran or st.session_state.result is None:
        st.warning("Pipeline has not been run yet. Go to Pipeline first.")
        return

    result = st.session_state.result
    logs   = st.session_state.logs

    def df_to_csv(df):
        return df.to_csv(index=False).encode("utf-8")

    def logs_to_txt(logs):
        return "\n".join(logs).encode("utf-8")

    if isinstance(result, tuple):
        train_df, test_df = result
        tab1, tab2 = st.tabs(["🟢  Train Set", "🔵  Test Set"])

        with tab1:
            r1c1, r1c2, r1c3 = st.columns(3)
            with r1c1:
                st.markdown(f"<div class='metric-card'><div class='label'>Rows</div><div class='value'>{train_df.shape[0]:,}</div></div>", unsafe_allow_html=True)
            with r1c2:
                st.markdown(f"<div class='metric-card'><div class='label'>Columns</div><div class='value'>{train_df.shape[1]:,}</div></div>", unsafe_allow_html=True)
            with r1c3:
                size_mb = train_df.memory_usage(deep=True).sum() / 1024**2
                st.markdown(f"<div class='metric-card'><div class='label'>Memory</div><div class='value'>{size_mb:.2f} MB</div></div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(train_df, use_container_width=True, height=320)
            st.download_button("⬇ Download Train CSV", df_to_csv(train_df), "train_processed.csv", "text/csv", use_container_width=True)

        with tab2:
            r2c1, r2c2, r2c3 = st.columns(3)
            with r2c1:
                st.markdown(f"<div class='metric-card'><div class='label'>Rows</div><div class='value'>{test_df.shape[0]:,}</div></div>", unsafe_allow_html=True)
            with r2c2:
                st.markdown(f"<div class='metric-card'><div class='label'>Columns</div><div class='value'>{test_df.shape[1]:,}</div></div>", unsafe_allow_html=True)
            with r2c3:
                size_mb = test_df.memory_usage(deep=True).sum() / 1024**2
                st.markdown(f"<div class='metric-card'><div class='label'>Memory</div><div class='value'>{size_mb:.2f} MB</div></div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(test_df, use_container_width=True, height=320)
            st.download_button("⬇ Download Test CSV", df_to_csv(test_df), "test_processed.csv", "text/csv", use_container_width=True)

    else:
        ec1, ec2, ec3 = st.columns(3)
        with ec1:
            st.markdown(f"<div class='metric-card'><div class='label'>Rows</div><div class='value'>{result.shape[0]:,}</div></div>", unsafe_allow_html=True)
        with ec2:
            st.markdown(f"<div class='metric-card'><div class='label'>Columns</div><div class='value'>{result.shape[1]:,}</div></div>", unsafe_allow_html=True)
        with ec3:
            size_mb = result.memory_usage(deep=True).sum() / 1024**2
            st.markdown(f"<div class='metric-card'><div class='label'>Memory</div><div class='value'>{size_mb:.2f} MB</div></div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(result, use_container_width=True, height=380)
        st.download_button("⬇ Download Processed CSV", df_to_csv(result), "processed.csv", "text/csv", use_container_width=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-weight:700; color:{t['text']}; margin-bottom:0.5rem;'>Pipeline Log</div>", unsafe_allow_html=True)

    log_html = ""
    for line in logs:
        css = "log-err" if "Error" in line else ("log-warn" if "Warning" in line else "")
        log_html += f'<div class="{css}">&gt; {line}</div>'
    st.markdown(f"<div class='log-box'>{log_html}</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button("⬇ Download Pipeline Log", logs_to_txt(logs), "pipeline_log.txt", "text/plain", use_container_width=True)


page = st.session_state.page
if page == "Upload":
    page_upload()
elif page == "Audit":
    page_audit()
elif page == "Recommendations":
    page_recommendations()
elif page == "Pipeline":
    page_pipeline()
elif page == "Export":
    page_export()