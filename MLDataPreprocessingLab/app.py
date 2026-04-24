import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import DataLoader
from utils.visuals import get_detailed_audit, get_smart_recommendations
from src.processor import DataProcessor

st.set_page_config(page_title="ML Preprocessing Lab", page_icon=" ", layout="wide", initial_sidebar_state="expanded")

T = {
    "bg": "#0B0F1A", "card": "#131929", "sidebar": "#0E1320",
    "primary": "#4DA3FF", "primary_dim": "#1A3A5C",
    "text": "#E8EDF5", "text_dim": "#8A93A8",
    "success": "#2ECC71", "warning": "#F39C12", "danger": "#FF6B6B",
    "info": "#4DA3FF", "border": "#1E2840",
    "log_bg": "#070B14", "log_text": "#00FF88", "accent": "#7B61FF",
}

def inject_css():
    st.markdown(f"""<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');
    html,body,[data-testid="stAppViewContainer"],[data-testid="stApp"]{{background:{T['bg']}!important;color:{T['text']}!important;font-family:'IBM Plex Sans',sans-serif!important;}}
    [data-testid="stSidebar"]{{background:{T['sidebar']}!important;border-right:1px solid {T['border']}!important;}}
    [data-testid="stSidebar"] *{{color:{T['text']}!important;}}
    .main .block-container{{padding-top:1.5rem!important;padding-bottom:3rem!important;max-width:1280px!important;}}
    h1,h2,h3,h4{{font-family:'IBM Plex Sans',sans-serif!important;color:{T['text']}!important;font-weight:700!important;}}
    .stButton>button{{background:{T['primary']}!important;color:#fff!important;border:none!important;border-radius:5px!important;font-family:'IBM Plex Sans',sans-serif!important;font-weight:600!important;transition:opacity 0.2s!important;}}
    .stButton>button:hover{{opacity:0.82!important;}}
    .stSelectbox label,.stMultiSelect label,.stSlider label,.stCheckbox label,.stRadio label,.stTextInput label{{color:{T['text_dim']}!important;font-size:0.78rem!important;font-weight:500!important;letter-spacing:0.05em!important;text-transform:uppercase!important;}}
    .stSelectbox>div>div,.stMultiSelect>div>div,.stTextInput>div>div>input,.stNumberInput>div>div>input{{background:{T['card']}!important;color:{T['text']}!important;border:1px solid {T['border']}!important;border-radius:5px!important;font-family:'IBM Plex Sans',sans-serif!important;}}
    div[data-testid="stMetric"]{{background:{T['card']}!important;border:1px solid {T['border']}!important;border-radius:8px!important;padding:1rem 1.2rem!important;}}
    div[data-testid="stMetricValue"]{{color:{T['primary']}!important;font-weight:700!important;}}
    .st-expander{{background:{T['card']}!important;border:1px solid {T['border']}!important;border-radius:8px!important;}}
    [data-testid="stExpander"] summary{{font-family:'IBM Plex Mono',monospace!important;font-size:0.8rem!important;font-weight:600!important;color:{T['primary']}!important;letter-spacing:0.04em!important;}}
    [data-testid="stFileUploader"]{{background:{T['card']}!important;border:1px dashed {T['border']}!important;border-radius:8px!important;}}
    [data-testid="stDownloadButton"]>button{{background:{T['success']}!important;font-weight:700!important;}}
    ::-webkit-scrollbar{{width:5px;height:5px;}}
    ::-webkit-scrollbar-thumb{{background:{T['primary_dim']};border-radius:3px;}}
    #MainMenu,footer,header{{visibility:hidden;}}
    .mcard{{background:{T['card']};border:1px solid {T['border']};border-radius:8px;padding:1rem 1.2rem;margin-bottom:0.5rem;}}
    .mlabel{{font-size:0.67rem;color:{T['text_dim']};font-weight:600;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.3rem;font-family:'IBM Plex Mono',monospace;}}
    .mvalue{{font-size:1.65rem;font-weight:700;color:{T['primary']};font-family:'IBM Plex Sans',sans-serif;line-height:1.1;}}
    .msub{{font-size:0.71rem;color:{T['text_dim']};margin-top:0.2rem;}}
    .sec-header{{font-size:1.1rem;font-weight:700;color:{T['text']};border-bottom:1px solid {T['border']};padding-bottom:0.45rem;margin:1.4rem 0 1rem 0;font-family:'IBM Plex Sans',sans-serif;letter-spacing:0.01em;}}
    .hero-title{{font-size:2.5rem;font-weight:700;color:{T['text']};font-family:'IBM Plex Sans',sans-serif;line-height:1.1;letter-spacing:-0.02em;}}
    .hero-title span{{color:{T['primary']};}}
    .hero-sub{{font-size:0.92rem;color:{T['text_dim']};margin-top:0.6rem;margin-bottom:2rem;max-width:640px;}}
    .rec-card{{background:{T['card']};border-radius:6px;padding:0.8rem 1rem;margin-bottom:0.5rem;border-left:3px solid;}}
    .rec-card.drop{{border-color:{T['danger']};}} .rec-card.impute{{border-color:{T['warning']};}}
    .rec-card.skew{{border-color:{T['accent']};}} .rec-card.encode{{border-color:{T['success']};}}
    .rec-card.ok{{border-color:{T['text_dim']};}} .rec-card.outlier{{border-color:{T['danger']};}}
    .rc-col{{font-weight:700;font-size:0.83rem;color:{T['text']};font-family:'IBM Plex Mono',monospace;}}
    .rc-stat{{font-size:0.74rem;color:{T['text_dim']};margin-top:0.15rem;}}
    .rc-sugg{{font-size:0.79rem;color:{T['primary']};margin-top:0.3rem;font-weight:500;}}
    .log-box{{background:{T['log_bg']};border:1px solid {T['border']};border-radius:6px;padding:1rem 1.2rem;font-family:'IBM Plex Mono',monospace;font-size:0.75rem;color:{T['log_text']};max-height:250px;overflow-y:auto;line-height:1.8;}}
    .log-err{{color:{T['danger']};}} .log-warn{{color:{T['warning']};}}
    .divider{{border:none;border-top:1px solid {T['border']};margin:1rem 0;}}
    .feat-eng-row{{background:{T['card']};border:1px solid {T['border']};border-radius:6px;padding:0.7rem 1rem;margin-bottom:0.4rem;}}
    </style>""", unsafe_allow_html=True)

DEFAULTS = {"df":None,"working_df":None,"file_info":None,"result":None,"logs":[],"pipeline_ran":False,"page":"Upload","fe_rows":[]}
for k,v in DEFAULTS.items():
    if k not in st.session_state: st.session_state[k] = v

inject_css()

def mcard(label, value, sub=""):
    return f"<div class='mcard'><div class='mlabel'>{label}</div><div class='mvalue'>{value}</div>{'<div class=msub>'+sub+'</div>' if sub else ''}</div>"

def sec(title): st.markdown(f"<div class='sec-header'>{title}</div>", unsafe_allow_html=True)
def divider(): st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

def render_logs(logs):
    html = "".join(f'<div class="{"log-err" if "Error" in l else "log-warn" if "Warning" in l else ""}">&gt; {l}</div>' for l in logs)
    st.markdown(f"<div class='log-box'>{html}</div>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f"<div style='padding:0.4rem 0 1.2rem 0;'><div style='font-size:1.1rem;font-weight:700;color:{T['primary']};font-family:IBM Plex Mono,monospace;letter-spacing:0.05em;'> ML PREP LAB</div><div style='font-size:0.67rem;color:{T['text_dim']};margin-top:0.2rem;font-family:IBM Plex Mono,monospace;'>preprocessing pipeline </div></div>", unsafe_allow_html=True)
    st.markdown(f"<hr style='border-color:{T['border']};margin:0 0 0.8rem 0;'>", unsafe_allow_html=True)
    for pg, ic in [("Upload"," "),("Audit"," "),("Recommendations"," "),("Pipeline"," "),("Export"," ")]:
        if st.button(f"{ic}  {pg}", key=f"nav_{pg}", use_container_width=True):
            st.session_state.page = pg; st.rerun()
    st.markdown(f"<hr style='border-color:{T['border']};margin:0.8rem 0;'>", unsafe_allow_html=True)
    if st.session_state.df is not None:
        info = st.session_state.file_info
        wdf  = st.session_state.working_df
        st.markdown(f"<div style='font-size:0.69rem;color:{T['text_dim']};line-height:2.1;font-family:IBM Plex Mono,monospace;'><b style='color:{T['text']}'>ORIGINAL</b><br>Rows&nbsp;: <span style='color:{T['primary']}'>{info['rows']:,}</span><br>Cols&nbsp;: <span style='color:{T['primary']}'>{info['cols']}</span><br>Size&nbsp;: <span style='color:{T['primary']}'>{info['size']}</span></div>", unsafe_allow_html=True)
        if wdf is not None and wdf.shape != st.session_state.df.shape:
            st.markdown(f"<div style='font-size:0.69rem;color:{T['text_dim']};line-height:2.1;font-family:IBM Plex Mono,monospace;margin-top:0.6rem;'><b style='color:{T['success']}'>WORKING</b><br>Rows&nbsp;: <span style='color:{T['success']}'>{wdf.shape[0]:,}</span><br>Cols&nbsp;: <span style='color:{T['success']}'>{wdf.shape[1]}</span></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='font-size:0.72rem;color:{T['text_dim']};font-family:IBM Plex Mono,monospace;'>No file loaded.</div>", unsafe_allow_html=True)

def page_upload():
    st.markdown(f"<div class='hero-title'>ML Data <span>Preprocessing</span> Lab</div><div class='hero-sub'>Load any dataset and apply a fully configurable preprocessing pipeline — drop, engineer features, impute per-column, encode per-column, scale, and export.</div>", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    for col,icon,title,desc in [
    (c1," ","Clean","Nulls, dtypes, duplicates, drops"),
    (c2,"  ","Engineer","Build new columns from existing ones"),
    (c3," ","Encode","Per-column: one-hot, target, binary…"),
    (c4," ","Scale & Select","RFE, Lasso, Standard, Robust")]:
        with col: st.markdown(f"<div class='mcard'><div style='font-size:1.3rem;margin-bottom:0.3rem;'>{icon}</div><div style='font-weight:700;color:{T['text']};font-size:0.88rem;'>{title}</div><div style='font-size:0.72rem;color:{T['text_dim']};margin-top:0.15rem;'>{desc}</div></div>", unsafe_allow_html=True)
    divider(); sec("Upload Dataset")
    uploaded = st.file_uploader("Drag and drop your file — CSV, Excel, JSON, Parquet, TSV", type=["csv","xlsx","xls","json","parquet","tsv"])
    if uploaded:
        loader = DataLoader()
        result = loader.load_file(uploaded)
        if isinstance(result, str): st.error(result); return
        st.session_state.df = result
        st.session_state.working_df = result.copy()
        st.session_state.file_info  = loader.get_basic_info(result)
        st.session_state.pipeline_ran = False
        st.session_state.result = None
        info = st.session_state.file_info
        st.success(f"✅ **{uploaded.name}** loaded — {info['rows']:,} rows × {info['cols']} columns")
        divider(); sec("Dataset Overview")
        cols = st.columns(6)
        for col,label,val,sub in [(cols[0],"Rows",f"{info['rows']:,}",""),(cols[1],"Columns",f"{info['cols']}",""),(cols[2],"Numeric",f"{len(info['numerical_cols'])}","features"),(cols[3],"Categorical",f"{len(info['categorical_cols'])}","features"),(cols[4],"Datetime",f"{len(info['datetime_cols'])}","features"),(cols[5],"Missing Cols",f"{len(info['missing_cols'])}","columns")]:
            with col: st.markdown(mcard(label,val,sub), unsafe_allow_html=True)
        divider(); sec("Full Data Preview")
        st.dataframe(result, use_container_width=True, height=420)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Proceed to Audit →"): st.session_state.page="Audit"; st.rerun()

def page_audit():
    sec("Data Audit")
    if st.session_state.df is None: st.warning("No dataset loaded."); return
    df = st.session_state.working_df
    summary, health_df = get_detailed_audit(df)
    cols = st.columns(6)
    for col,(k,v) in zip(cols, list(summary.items())):
        with col: st.markdown(f"<div class='mcard'><div class='mlabel'>{k}</div><div style='font-size:1rem;font-weight:700;color:{T['primary']};'>{v}</div></div>", unsafe_allow_html=True)
    divider()
    fc1,fc2,fc3 = st.columns([2,2,2])
    with fc1: type_filter = st.selectbox("Filter by Type",["All","Numeric","Categorical","Issues Only"])
    with fc2: search = st.text_input("Search column", placeholder="column name…")
    with fc3: st.markdown("<br>",unsafe_allow_html=True); show_only_issues = st.checkbox("Issues only")
    fdf = health_df.copy()
    if type_filter=="Numeric": fdf=fdf[fdf["Type"].str.contains("int|float",case=False,na=False)]
    elif type_filter=="Categorical": fdf=fdf[fdf["Type"].str.contains("object|category",case=False,na=False)]
    elif type_filter=="Issues Only": fdf=fdf[fdf["Status"]!="Healthy"]
    if show_only_issues: fdf=fdf[fdf["Status"]!="Healthy"]
    if search: fdf=fdf[fdf["Column"].str.contains(search,case=False,na=False)]
    def color_status(val):
        if val=="Healthy": return f"color:{T['success']};font-weight:600;"
        if "Missing" in val: return f"color:{T['warning']};font-weight:600;"
        if "Outlier" in val: return f"color:{T['danger']};font-weight:600;"
        if "Constant" in val: return f"color:{T['text_dim']};font-weight:600;"
        if "Skewed" in val: return f"color:{T['accent']};font-weight:600;"
        return f"color:{T['info']};font-weight:600;"
    st.dataframe(fdf.style.applymap(color_status,subset=["Status"]), use_container_width=True, height=480)
    st.markdown(f"<div style='font-size:0.74rem;color:{T['text_dim']};'>Showing {len(fdf)} of {len(health_df)} columns</div>", unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    if st.button("See Recommendations →"): st.session_state.page="Recommendations"; st.rerun()
def page_recommendations():
    sec("Smart Recommendations")
    if st.session_state.df is None: st.warning("No dataset loaded."); return
    df = st.session_state.working_df
    _, health_df = get_detailed_audit(df)
    drop_recs,impute_recs,skew_recs,encode_recs,outlier_recs = [],[],[],[],[]
    for _,row in health_df.iterrows():
        col   = row["Column"]; dtype = row["Type"]
        miss  = float(str(row["Missing %"]).replace("%",""))
        uniq  = row["Unique Values"]
        outs  = int(row["Outliers"]) if str(row["Outliers"]).isdigit() else 0
        total = df.shape[0]
        if miss>60: drop_recs.append({"col":col,"stat":f"{miss:.1f}% missing","sugg":"Drop — too many missing values"})
        elif uniq==1: drop_recs.append({"col":col,"stat":"Constant column","sugg":"Drop — no information for ML"})
        elif uniq==total and "object" in dtype: drop_recs.append({"col":col,"stat":f"All {total:,} values unique","sugg":"Likely ID column — drop before training"})
        if 0<miss<=60:
            m = "KNN Imputer" if total<50000 else "Median/Mode (too large for KNN)"
            impute_recs.append({"col":col,"stat":f"{miss:.1f}% missing","sugg":f"Impute with {m}"})
        if outs>0:
            pct=outs/total*100
            outlier_recs.append({"col":col,"stat":f"{outs:,} outliers ({pct:.1f}%)","sugg":"Cap with IQR" if pct<5 else "High outlier rate — investigate"})
        try:
            sk=float(row["Skewness"])
            if abs(sk)>2:
                t_s = "Log Transform" if df[col].min()>0 else "Yeo-Johnson Power Transform"
                skew_recs.append({"col":col,"stat":f"Skewness = {sk:.2f} (high)","sugg":t_s})
            elif abs(sk)>1:
                skew_recs.append({"col":col,"stat":f"Skewness = {sk:.2f} (moderate)","sugg":"Consider Power Transform"})
        except: pass
        if "object" in dtype or "category" in dtype:
            if 2<=uniq<=10: encode_recs.append({"col":col,"stat":f"{uniq} unique labels","sugg":"One-Hot Encoding"})
            elif uniq<=50: encode_recs.append({"col":col,"stat":f"{uniq} unique labels","sugg":"Binary or Frequency Encoding"})
            elif uniq>50: encode_recs.append({"col":col,"stat":f"{uniq} unique labels (high cardinality)","sugg":"Target Encoding — one-hot would explode dimensions"})
    c1,c2,c3,c4,c5 = st.columns(5)
    for col,label,items,color in [(c1,"Drop",drop_recs,T["danger"]),(c2,"Impute",impute_recs,T["warning"]),(c3,"Outliers",outlier_recs,T["danger"]),(c4,"Transform",skew_recs,T["accent"]),(c5,"Encode",encode_recs,T["success"])]:
        with col: st.markdown(f"<div class='mcard' style='border-left:3px solid {color};'><div class='mlabel'>{label}</div><div class='mvalue' style='color:{color};font-size:1.4rem;'>{len(items)}</div></div>", unsafe_allow_html=True)
    divider()
    def render_cards(items,css,icon):
        for r in items: st.markdown(f"<div class='rec-card {css}'><div class='rc-col'>{icon} {r['col']}</div><div class='rc-stat'>{r['stat']}</div><div class='rc-sugg'>→ {r['sugg']}</div></div>", unsafe_allow_html=True)
    tabs = st.tabs([f" Drop ({len(drop_recs)})",f" Impute ({len(impute_recs)})",f" Outliers ({len(outlier_recs)})",f" Transform ({len(skew_recs)})",f" Encode ({len(encode_recs)})"])
    with tabs[0]: render_cards(drop_recs,"drop"," ") if drop_recs else st.info("No drop candidates.")
    with tabs[1]: render_cards(impute_recs,"impute"," ") if impute_recs else st.info("No missing values.")
    with tabs[2]: render_cards(outlier_recs,"outlier"," ") if outlier_recs else st.info("No significant outliers.")
    with tabs[3]: render_cards(skew_recs,"skew"," ") if skew_recs else st.info("No skewed columns.")
    with tabs[4]: render_cards(encode_recs,"encode"," ") if encode_recs else st.info("No categorical columns.")
    st.markdown("<br>",unsafe_allow_html=True)
    if st.button("Go to Pipeline →"): st.session_state.page="Pipeline"; st.rerun()

def page_pipeline():
    sec("Preprocessing Pipeline")
    if st.session_state.df is None: st.warning("No dataset loaded."); return
    df       = st.session_state.df.copy()
    all_cols = df.columns.tolist()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
    config   = {}
    st.markdown(f"<div style='font-size:0.78rem;color:{T['text_dim']};margin-bottom:1.2rem;font-family:IBM Plex Mono,monospace;'>All enabled steps apply to the same dataset in order. Per-column steps let you configure each column independently.</div>", unsafe_allow_html=True)

    with st.expander("STEP 0 — Drop Columns"):
        if st.checkbox("Enable", key="drop_en"):
            drop_cols = st.multiselect("Columns to drop", all_cols, key="drop_cols")
            if drop_cols: config["drop_columns"] = drop_cols

    with st.expander("STEP 0b — Data Type Conversion"):
        if st.checkbox("Enable", key="dtype_en"):
            dtype_map = {}
            dtype_cols = st.multiselect("Columns to convert", all_cols, key="dtype_cols")
            for dc in dtype_cols:
                cur = str(df[dc].dtype)
                new_t = st.selectbox(f"{dc}  (now: {cur})", ["datetime","numeric","category","string"], key=f"dtype_{dc}")
                dtype_map[dc] = new_t
            if dtype_map: config["dtype_conversions"] = dtype_map

    with st.expander("STEP 0c — Feature Engineering "):
        if st.checkbox("Enable", key="fe_en"):
            st.markdown(f"<div style='font-size:0.77rem;color:{T['text_dim']};margin-bottom:0.7rem;'>Combine numeric columns to create new features. Missing values filled with 0.</div>", unsafe_allow_html=True)
            with st.form("fe_form", clear_on_submit=True):
                fc1,fc2,fc3 = st.columns([2,3,1])
                with fc1: new_name = st.text_input("New column name", placeholder="e.g. total_guests")
                with fc2: fe_src   = st.multiselect("Source columns", num_cols, key="fe_src")
                with fc3: fe_op    = st.selectbox("Op", ["+","-","*","/"], key="fe_op")
                if st.form_submit_button("Add") and new_name and len(fe_src)>=2:
                    st.session_state.fe_rows.append({"name":new_name,"columns":fe_src,"op":fe_op})
            to_rm = []
            for i,row in enumerate(st.session_state.fe_rows):
                expr = f" {row['op']} ".join(row["columns"])
                cc1,cc2 = st.columns([5,1])
                with cc1: st.markdown(f"<div class='feat-eng-row'><span style='color:{T['primary']};font-family:IBM Plex Mono,monospace;font-size:0.8rem;font-weight:600;'>{row['name']}</span><span style='color:{T['text_dim']};font-size:0.76rem;margin-left:0.5rem;'>= {expr}</span></div>", unsafe_allow_html=True)
                with cc2:
                    if st.button("✕", key=f"fe_del_{i}"): to_rm.append(i)
            for i in reversed(to_rm): st.session_state.fe_rows.pop(i)
            if st.session_state.fe_rows: config["feature_engineering"] = st.session_state.fe_rows

    with st.expander("STEP 1 — Remove Duplicates"):
        if st.checkbox("Enable", key="dup_en"):
            config["remove_duplicates"] = True

    with st.expander("STEP 2 — Impute Missing Values (per column)"):
        if st.checkbox("Enable", key="imp_en"):
            missing_cols = [c for c in all_cols if df[c].isnull().any()]
            if not missing_cols: st.info("No missing values detected.")
            else:
                st.markdown(f"<div style='font-size:0.77rem;color:{T['text_dim']};margin-bottom:0.6rem;'>Set strategy per column independently.</div>", unsafe_allow_html=True)
                imp_cfg = {}
                for mc in missing_cols:
                    pct = df[mc].isnull().mean()*100
                    ic1,ic2,ic3 = st.columns([2,2,1])
                    with ic1: st.markdown(f"<div style='font-size:0.78rem;padding-top:0.5rem;font-family:IBM Plex Mono,monospace;color:{T['text']};'>{mc}</div>", unsafe_allow_html=True)
                    with ic2: strat = st.selectbox("", ["knn","mean","median","most_frequent","iterative"], key=f"imp_{mc}")
                    with ic3: st.markdown(f"<div style='font-size:0.71rem;color:{T['warning']};padding-top:0.5rem;'>{pct:.1f}% null</div>", unsafe_allow_html=True)
                    imp_cfg[mc] = {"strategy":strat,"n_neighbors":5}
                config["impute_params"] = imp_cfg

    with st.expander("STEP 3 — Handle Outliers"):
        if st.checkbox("Enable", key="out_en"):
            out_cols = st.multiselect("Columns", num_cols, key="out_cols")
            oc1,oc2 = st.columns(2)
            with oc1: out_method = st.selectbox("Detection", ["iqr","zscore","iso_forest"], key="out_meth")
            with oc2: out_action = st.selectbox("Action", ["cap","remove"], key="out_act")
            if out_cols: config["outlier_params"] = {"columns":out_cols,"method":out_method,"action":out_action}

    with st.expander("STEP 4 — Encode Categorical Features (per column)"):
        if st.checkbox("Enable", key="enc_en"):
            if not cat_cols: st.info("No categorical columns.")
            else:
                enc_select = st.multiselect("Columns to encode", cat_cols, key="enc_sel")
                enc_cfg = {}
                for ec in enc_select:
                    uniq_c = df[ec].nunique()
                    ec1,ec2 = st.columns([2,3])
                    with ec1: 
                            st.markdown(f"<div style='font-size:0.78rem;padding-top:0.5rem;font-family:IBM Plex Mono,monospace;color:{T['text']};'>{ec} <span style='color:{T['text_dim']};font-size:0.68rem;'>({uniq_c} unique)</span></div>", unsafe_allow_html=True)
                    with ec2: enc_type = st.selectbox("", ["onehot","label","ordinal","binary","frequency","target"], key=f"enc_{ec}")
                    cfg = {"type":enc_type}
                    if enc_type=="target":
                        cfg["target_column"] = st.selectbox(f"Target for {ec}", all_cols, key=f"enc_tgt_{ec}")
                        cfg["smoothing"] = 10
                    enc_cfg[ec] = cfg
                if enc_cfg: config["encode_params"] = enc_cfg

    with st.expander("STEP 5 — Feature Selection"):
        if st.checkbox("Enable", key="sel_en"):
            sel_target = st.selectbox("Target column", all_cols, key="sel_tgt")
            sel_task   = st.radio("Task", ["classification","regression"], horizontal=True, key="sel_task")
            use_var    = st.checkbox("Variance Filter", key="sel_var")
            use_corr   = st.checkbox("Correlation Filter", key="sel_corr")
            sel_p = {"target":sel_target,"task":sel_task}
            if use_var:  sel_p["use_variance"]=True;  sel_p["var_threshold"]=st.slider("Variance Threshold",0.0,0.1,0.01,0.001,key="var_th")
            if use_corr: sel_p["use_correlation"]=True; sel_p["corr_threshold"]=st.slider("Correlation Threshold",0.7,1.0,0.9,0.01,key="corr_th")
            sel_method = st.selectbox("Method (optional)",["None","rfe","mutual_info","lasso","chi2"],key="sel_meth")
            if sel_method!="None":
                sel_p["method"]=sel_method
                if sel_method in ["rfe","mutual_info","chi2"]:
                    sel_p["n_features"]=st.slider("Features to keep",2,min(20,max(len(num_cols),2)),5,key="sel_k")
                    sel_p["k"]=sel_p["n_features"]
            config["selection_params"] = sel_p

    with st.expander("STEP 6 — Balance Classes"):
        if st.checkbox("Enable", key="bal_en"):
            config["balance_params"] = {"target":st.selectbox("Target",all_cols,key="bal_tgt"),"method":st.selectbox("Method",["smote","adasyn","undersample"],key="bal_meth")}

    with st.expander("STEP 7 — Scale Features"):
        if st.checkbox("Enable", key="scl_en"):
            config["scale_params"] = {"method":st.selectbox("Method",["standard","minmax","robust"],key="scl_meth")}

    with st.expander("STEP 8 — Transform Distributions (per column)"):
        if st.checkbox("Enable", key="trn_en"):
            if not num_cols: st.info("No numeric columns.")
            else:
                trn_select = st.multiselect("Columns to transform", num_cols, key="trn_sel")
                trn_cfg = {}
                for tc in trn_select:
                    tc1,tc2 = st.columns([2,3])
                    with tc1: st.markdown(f"<div style='font-size:0.78rem;padding-top:0.5rem;font-family:IBM Plex Mono,monospace;color:{T['text']};'>{tc}</div>", unsafe_allow_html=True)
                    with tc2: trn_type = st.selectbox("", ["log","power"], key=f"trn_{tc}")
                    tcfg = {"type":trn_type}
                    if trn_type=="power": tcfg["method"]=st.selectbox(f"Power for {tc}",["yeo-johnson","box-cox"],key=f"trn_pow_{tc}")
                    trn_cfg[tc] = tcfg
                if trn_cfg: config["transform_params"] = trn_cfg

    with st.expander("STEP 9 — Train / Test Split"):
        if st.checkbox("Enable", key="spl_en"):
            config["split_params"] = {"target":st.selectbox("Target",all_cols,key="spl_tgt"),"test_size":st.slider("Test size",0.1,0.4,0.2,0.05,key="spl_size"),"stratify":st.checkbox("Stratify",key="spl_strat")}

    divider()
    if st.button("▶  Run Pipeline", use_container_width=True):
        with st.spinner("Running pipeline…"):
            processor = DataProcessor()
            result,logs = processor.Pipeline(df, config)
        st.session_state.result = result
        st.session_state.logs   = logs
        st.session_state.pipeline_ran = True
        if result is not None:
            st.session_state.working_df = result[0] if isinstance(result,tuple) else result
        st.rerun()

    if st.session_state.pipeline_ran:
        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.78rem;font-weight:600;color:{T['text']};margin-bottom:0.4rem;font-family:IBM Plex Mono,monospace;'>PIPELINE LOG</div>", unsafe_allow_html=True)
        render_logs(st.session_state.logs)
        result = st.session_state.result
        if result is not None:
            divider()
            if isinstance(result,tuple):
                train_df,test_df = result
                tc1,tc2 = st.columns(2)
                with tc1:
                    st.markdown(f"<div style='font-size:0.78rem;font-weight:600;color:{T['success']};margin-bottom:0.3rem;'>TRAIN — {train_df.shape[0]:,} × {train_df.shape[1]}</div>", unsafe_allow_html=True)
                    st.dataframe(train_df, use_container_width=True, height=320)
                with tc2:
                    st.markdown(f"<div style='font-size:0.78rem;font-weight:600;color:{T['info']};margin-bottom:0.3rem;'>TEST — {test_df.shape[0]:,} × {test_df.shape[1]}</div>", unsafe_allow_html=True)
                    st.dataframe(test_df, use_container_width=True, height=320)
            else:
                st.markdown(f"<div style='font-size:0.78rem;font-weight:600;color:{T['success']};margin-bottom:0.3rem;'>PROCESSED — {result.shape[0]:,} × {result.shape[1]}</div>", unsafe_allow_html=True)
                st.dataframe(result, use_container_width=True, height=420)
            st.markdown("<br>",unsafe_allow_html=True)
            if st.button("Go to Export →"): st.session_state.page="Export"; st.rerun()

def page_export():
    sec("Export & Download")
    if not st.session_state.pipeline_ran or st.session_state.result is None:
        st.warning("Pipeline has not been run yet."); return
    result = st.session_state.result
    logs   = st.session_state.logs
    to_csv = lambda d: d.to_csv(index=False).encode("utf-8")
    to_txt = lambda l: "\n".join(l).encode("utf-8")
    if isinstance(result,tuple):
        train_df,test_df = result
        tab1,tab2 = st.tabs(["🟢  Train Set","🔵  Test Set"])
        for tab,ddf,fname,color in [(tab1,train_df,"train_processed.csv",T["success"]),(tab2,test_df,"test_processed.csv",T["info"])]:
            with tab:
                c1,c2,c3 = st.columns(3)
                with c1: st.markdown(mcard("Rows",f"{ddf.shape[0]:,}"), unsafe_allow_html=True)
                with c2: st.markdown(mcard("Columns",f"{ddf.shape[1]}"), unsafe_allow_html=True)
                with c3: st.markdown(mcard("Memory",f"{ddf.memory_usage(deep=True).sum()/1024**2:.2f} MB"), unsafe_allow_html=True)
                st.markdown("<br>",unsafe_allow_html=True)
                st.dataframe(ddf, use_container_width=True, height=400)
                st.download_button(f"⬇ Download {fname}", to_csv(ddf), fname, "text/csv", use_container_width=True)
    else:
        c1,c2,c3 = st.columns(3)
        with c1: st.markdown(mcard("Rows",f"{result.shape[0]:,}"), unsafe_allow_html=True)
        with c2: st.markdown(mcard("Columns",f"{result.shape[1]}"), unsafe_allow_html=True)
        with c3: st.markdown(mcard("Memory",f"{result.memory_usage(deep=True).sum()/1024**2:.2f} MB"), unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        st.dataframe(result, use_container_width=True, height=450)
        st.download_button("⬇ Download Processed CSV", to_csv(result), "processed.csv", "text/csv", use_container_width=True)
    divider()
    st.markdown(f"<div style='font-size:0.78rem;font-weight:600;color:{T['text']};margin-bottom:0.4rem;font-family:IBM Plex Mono,monospace;'>PIPELINE LOG</div>", unsafe_allow_html=True)
    render_logs(logs)
    st.markdown("<br>",unsafe_allow_html=True)
    st.download_button("⬇ Download Log", to_txt(logs), "pipeline_log.txt", "text/plain", use_container_width=True)

{
    "Upload":page_upload,
    "Audit":page_audit,
    "Recommendations":page_recommendations,
    "Pipeline":page_pipeline,
    "Export":page_export
}[st.session_state.page]()