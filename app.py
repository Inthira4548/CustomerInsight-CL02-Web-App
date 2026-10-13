import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram

st.set_page_config(page_title="CustomerInsight | CL02", page_icon="💗", layout="wide",
                   initial_sidebar_state="expanded")

# ---------- Theme ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Prompt', sans-serif; }
.stApp { background: linear-gradient(135deg, #fbfaff 0%, #f5f7ff 55%, #fff8fc 100%); }
section[data-testid="stSidebar"] { background: linear-gradient(180deg,#f0edff 0%,#fff2fa 100%); border-right: 1px solid #e4ddff; }
.hero { padding: 28px 30px; border-radius: 24px; color: #1e1b4b;
  background: linear-gradient(115deg,#f5dfff 0%,#e2e9ff 52%,#d9fff1 100%);
  border: 1px solid #e4ddff; margin-bottom: 18px; }
.hero h1 { font-size: 2.15rem; line-height: 1.25; margin: 0 0 8px 0; color:#19145b; }
.hero p { font-size: 1rem; margin: 0; color:#484276; }
.kpi { padding: 18px 20px; border-radius: 20px; border: 1px solid #e4e5f7;
  background: rgba(255,255,255,.88); box-shadow: 0 5px 18px rgba(72,62,130,.05); min-height: 115px; }
.kpi-label { color:#68648b; font-size:.88rem; }
.kpi-value { color:#211b63; font-size:1.75rem; font-weight:700; margin-top:3px; }
.kpi-note { color:#7a76a0; font-size:.78rem; margin-top:4px; }
.panel-title { font-weight:600; color:#27216a; font-size:1.1rem; margin: 8px 0 10px; }
.box { padding:16px 20px; border-radius:18px; border:1px solid #e6e3f7; background:#ffffffd9; }
.tag { display:inline-block; padding:3px 10px; border-radius:999px; font-size:.8rem; margin:2px 4px 2px 0;
  background:#f3f0ff; border:1px solid #e0dafc; color:#3b3480; }
div[data-testid="stMetric"] { background:#ffffffd9; padding:16px; border-radius:18px; border:1px solid #e6e3f7; }
.stButton > button, .stDownloadButton > button { border-radius: 12px; border: 0;
  background: linear-gradient(90deg,#7760db,#b55ec6); color:white; font-weight:600; }
div[data-testid="stDataFrame"] { border-radius: 14px; overflow:hidden; }
.small-note { color:#777393; font-size:.82rem; }
</style>
""", unsafe_allow_html=True)

# ---------- Constants ----------
KNOWN_FEATURES = ["age", "monthly_spend", "visits_month", "avg_items",
                  "discount_usage_percent", "online_orders_month"]
TH_NAME = {
    "age": "อายุ", "monthly_spend": "ยอดใช้จ่าย/เดือน", "visits_month": "เข้าร้าน/เดือน",
    "avg_items": "สินค้าต่อครั้ง", "discount_usage_percent": "ใช้ส่วนลด (%)",
    "online_orders_month": "Online Orders/เดือน",
}
ACTIONS = {
    "monthly_spend": "สิทธิพิเศษ/สะสมแต้มเพื่อเพิ่มยอดซื้อ",
    "discount_usage_percent": "โปรโมชันส่วนลด/คูปอง",
    "online_orders_month": "แคมเปญช่องทางออนไลน์ (ส่งฟรี/แอป)",
    "visits_month": "โปรโมชันหน้าร้าน/สะสมแต้มต่อครั้ง",
    "avg_items": "ชุดสินค้าแนะนำ (bundle)",
    "age": "สื่อสารให้ตรงช่วงวัย",
}
ID_CANDIDATES = ["customer_id", "customerid", "id", "รหัสลูกค้า"]
COLORS = ["#7c83fd", "#f47db5", "#56cfa5", "#ffc66d", "#a78bfa", "#60a5fa", "#fb7185", "#34d399"]
K_RANGE = list(range(2, 9))


def label(c):
    return TH_NAME.get(c, c)


def style(fig):
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def kpi(col, title, value, note):
    col.markdown(f'<div class="kpi"><div class="kpi-label">{title}</div>'
                 f'<div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>',
                 unsafe_allow_html=True)


# ---------- Data helpers ----------
def find_id_column(columns):
    lookup = {str(c).strip().lower(): c for c in columns}
    for key in ID_CANDIDATES:
        if key in lookup:
            return lookup[key]
    return None


def load_excel(src):
    try:
        xls = pd.ExcelFile(src)
        sheet = "Raw_Data" if "Raw_Data" in xls.sheet_names else xls.sheet_names[0]
        return pd.read_excel(xls, sheet_name=sheet), sheet
    except Exception as e:
        st.error(f"อ่านไฟล์ไม่สำเร็จ: {e}")
        return None, None


def iqr_outliers(s):
    q1, q3 = s.quantile(.25), s.quantile(.75)
    i = q3 - q1
    return int(((s < q1 - 1.5 * i) | (s > q3 + 1.5 * i)).sum())


def clean_frame(raw):
    """Clean data and return (clean_df, report). Every step is counted so it can be explained."""
    data = raw.copy()
    data.columns = [str(c).strip() for c in data.columns]
    n_raw = len(data)
    dup = int(data.duplicated().sum())
    data = data.drop_duplicates().reset_index(drop=True)
    id_col = find_id_column(data.columns)
    feats = [c for c in KNOWN_FEATURES if c in data.columns]
    if not feats:
        feats = [c for c in data.select_dtypes(include=np.number).columns if c != id_col]
    missing, negative = {}, {}
    for c in feats:
        data[c] = pd.to_numeric(data[c], errors="coerce")
        missing[c] = int(data[c].isna().sum())
        neg = data[c] < 0  # spend, counts and percentages cannot be negative
        negative[c] = int(neg.sum())
        data.loc[neg, c] = np.nan
        data[c] = data[c].fillna(data[c].median())
    outliers = {c: iqr_outliers(data[c]) for c in feats}
    report = dict(n_raw=n_raw, dup=dup, n_clean=len(data), n_cols=raw.shape[1], id_col=id_col,
                  feats=feats, missing=missing, negative=negative, outliers=outliers,
                  dtypes={str(c).strip(): str(t) for c, t in raw.dtypes.items()})
    return data, report


def cluster_means(Z, labels, k):
    return np.array([Z[labels == q].mean(axis=0) for q in range(k)])


@st.cache_data(show_spinner="กำลังประเมินโมเดล…")
def evaluate_models(Xs):
    """K-Means vs Hierarchical (Ward) for K=2..8 + stability of K-Means."""
    out = {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
        hc = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Xs)
        stab = np.mean([adjusted_rand_score(
            km.labels_, KMeans(n_clusters=k, random_state=s, n_init=10).fit_predict(Xs)) for s in (1, 2)])
        out[k] = dict(inertia=km.inertia_, labels=km.labels_, hc_labels=hc,
                      sil=silhouette_score(Xs, km.labels_), hc_sil=silhouette_score(Xs, hc),
                      db=davies_bouldin_score(Xs, km.labels_), stability=stab,
                      ari=adjusted_rand_score(km.labels_, hc))
    return out


@st.cache_data(show_spinner=False)
def holdout_test(X, k):
    """80/20 split. Scaler and K-Means are fitted on train only; test points go to the nearest centre."""
    Xtr, Xte = train_test_split(X, test_size=0.2, random_state=42)
    sc = StandardScaler().fit(Xtr)
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Ztr)
    return dict(n_train=len(Xtr), n_test=len(Xte),
                sil_train=silhouette_score(Ztr, km.labels_),
                sil_test=silhouette_score(Zte, km.predict(Zte)))


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 💗 CustomerInsight")
    st.caption("CL02 · Customer Behavior Clustering")
    st.markdown("---")
    st.markdown("### ⚙️ ตั้งค่าการวิเคราะห์")
    k_value = st.slider("จำนวนกลุ่ม (K)", min_value=2, max_value=8, value=3, step=1)
    method = st.radio("วิธีที่ใช้แสดงโปรไฟล์กลุ่ม", ["K-Means", "Hierarchical (Ward)"])
    st.caption("เลือก K จากกราฟ Elbow/Silhouette ในแท็บ “โมเดล & ผลลัพธ์”")
    st.markdown("---")
    st.caption("Made with Python · Streamlit · scikit-learn")

st.markdown("""
<div class="hero">
  <h1>Customer Segmentation ✨</h1>
  <p>แดชบอร์ดวิเคราะห์พฤติกรรมลูกค้าด้วย K-Means และ Hierarchical Clustering</p>
  <p style="margin-top:8px">เปลี่ยนข้อมูลให้เป็น Insight ที่เข้าใจง่าย สำหรับโปรเจกต์ CL02</p>
</div>
""", unsafe_allow_html=True)

# ---------- Data input ----------
raw_df, sheet_name, source_name = None, None, None
with st.expander("📂 เลือก Dataset", expanded=False):
    uploaded = st.file_uploader("อัปโหลดไฟล์ Excel (.xlsx)", type=["xlsx"])
    default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "CL02_customer_behavior_clustering.xlsx")
    if uploaded is not None:
        raw_df, sheet_name = load_excel(uploaded)
        source_name = uploaded.name
    elif os.path.exists(default_path):
        raw_df, sheet_name = load_excel(default_path)
        source_name = os.path.basename(default_path)
    else:
        st.info("อัปโหลดไฟล์ CL02 เพื่อเริ่มวิเคราะห์")

if raw_df is None:
    st.warning("กรุณาอัปโหลดไฟล์ CL02_customer_behavior_clustering.xlsx ในส่วน “เลือก Dataset”")
    st.stop()

clean_df, rep = clean_frame(raw_df)
features, id_col = rep["feats"], rep["id_col"]
if len(features) < 2:
    st.error("ต้องมีคอลัมน์ตัวเลขอย่างน้อย 2 คอลัมน์เพื่อทำ Clustering")
    st.write("คอลัมน์ที่พบ:", list(clean_df.columns))
    st.stop()

X = clean_df[features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

ev = evaluate_models(X_scaled)
cur = ev[k_value]
labels = cur["labels"] if method == "K-Means" else cur["hc_labels"]
result_df = clean_df.copy()
result_df["Cluster"] = labels
sil_score = cur["sil"] if method == "K-Means" else cur["hc_sil"]
best_k = max(K_RANGE, key=lambda k: ev[k]["sil"])
ho = holdout_test(X.values, k_value)

# ---------- Tabs ----------
t_prob, t_prep, t_eda, t_model, t_prof, t_dl = st.tabs([
    "🎯 โจทย์ & ข้อมูล", "🧹 เตรียมข้อมูล", "📊 EDA", "🤖 โมเดล & ผลลัพธ์", "🔎 โปรไฟล์กลุ่ม", "⬇️ ดาวน์โหลด & Deploy"])

# ===== 1. Problem & data understanding =====
with t_prob:
    st.markdown("### 🎯 วิเคราะห์เพื่ออะไร")
    st.markdown(
        f"""<div class="box">
<b>โจทย์:</b> แบ่งลูกค้า {rep['n_raw']:,} ราย เป็นกลุ่มตามพฤติกรรมการซื้อ (ยอดใช้จ่าย ความถี่เข้าร้าน การใช้ส่วนลด การสั่งออนไลน์)
เพื่อวางโปรโมชันและแคมเปญให้ตรงกลุ่ม แทนการทำการตลาดแบบหว่านแห<br><br>
<b>ทำไมเป็น Clustering:</b> ไม่มีคอลัมน์เฉลย (Target) จึงเป็น Unsupervised Learning ให้ข้อมูลบอกเองว่าลูกค้าคล้ายกันแค่ไหน<br><br>
<b>ผลที่คาดหวัง:</b> ได้กลุ่มลูกค้าที่อธิบายลักษณะได้ และนำไปกำหนดกลยุทธ์ต่อกลุ่มได้จริง</div>""",
        unsafe_allow_html=True)
    st.markdown("")
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "แถวทั้งหมด", f"{rep['n_raw']:,}", f"ไฟล์ {source_name} · ชีต {sheet_name}")
    kpi(c2, "คอลัมน์", rep["n_cols"], "รวมรหัสลูกค้า")
    kpi(c3, "Features", len(features), "ใช้จัดกลุ่ม")
    kpi(c4, "Target", "ไม่มี", "Unsupervised")
    st.markdown('<div class="panel-title">📖 คำอธิบายตัวแปร</div>', unsafe_allow_html=True)
    rows = []
    for c in raw_df.columns:
        c = str(c).strip()
        is_f = c in features
        rows.append({"คอลัมน์": c,
                     "ความหมาย": label(c) if is_f else ("รหัสลูกค้า" if c == id_col else "—"),
                     "บทบาท": "Feature (เข้าโมเดล)" if is_f else "Identifier (ตัดออก)" if c == id_col else "ไม่ใช้",
                     "ชนิดข้อมูลในไฟล์": rep["dtypes"].get(c, "")})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.info("หน่วยของตัวแปรต่างกันมาก (ยอดใช้จ่ายหลักหมื่น แต่เข้าร้านหลักสิบ) จึงต้อง Scale ก่อนวัดระยะห่าง "
            "ไม่เช่นนั้นยอดใช้จ่ายจะครอบงำการจัดกลุ่ม")

# ===== 2. Data preparation =====
with t_prep:
    n_miss, n_neg, n_out = sum(rep["missing"].values()), sum(rep["negative"].values()), sum(rep["outliers"].values())
    st.markdown("### 🧹 ขั้นตอนเตรียมข้อมูล พร้อมเหตุผล")
    steps = pd.DataFrame([
        ["1 ตรวจ Shape/Type", f"{rep['n_raw']:,} แถว × {rep['n_cols']} คอลัมน์", "แปลงเป็นตัวเลข", "ให้คำนวณระยะห่างได้"],
        ["2 แถวซ้ำ", f"{rep['dup']:,}", "ลบ", "ลูกค้าคนเดียวไม่ควรถูกนับซ้ำ จะดึงศูนย์กลางกลุ่มเอนเอียง"],
        ["3 ค่าว่าง", f"{n_miss:,}", "เติมด้วย Median", "ไม่เสียแถวข้อมูล และ Median ไม่โดน Outlier ดึง"],
        ["4 ค่าติดลบที่เป็นไปไม่ได้", f"{n_neg:,}", "ถือเป็นค่าผิดพลาด เติม Median",
         "ยอดใช้จ่าย จำนวนครั้ง และ % ส่วนลด ติดลบไม่มีความหมายทางธุรกิจ"],
        ["5 Outlier (IQR)", f"{n_out:,}", "ไม่ลบ ตรวจดูแล้วเก็บไว้",
         "อาจเป็นลูกค้าจริง เช่น กลุ่มใช้จ่ายสูง การลบจะทำให้เสียกลุ่มสำคัญ"],
        ["6 เลือก Feature", "ตัดรหัสลูกค้า", f"ใช้ {len(features)} ตัวแปรพฤติกรรม", "รหัสไม่ได้บอกพฤติกรรม"],
        ["7 Scaling", "mean 0, sd 1", "StandardScaler", "ให้ทุกตัวแปรมีน้ำหนักเท่ากัน"],
        ["8 Train/Test split", "80% / 20%", "ใช้ทดสอบในแท็บโมเดล", "ดูว่าใช้กับลูกค้าที่ไม่เคยเห็นได้หรือไม่"],
    ], columns=["ขั้นตอน", "ที่พบ", "วิธีจัดการ", "เหตุผล"])
    st.dataframe(steps, use_container_width=True, hide_index=True)
    a, b, c = st.columns(3)
    a.metric("แถวก่อน Clean", f"{rep['n_raw']:,}")
    b.metric("แถวหลัง Clean", f"{rep['n_clean']:,}", f"-{rep['dup']:,} ซ้ำ", delta_color="off")
    c.metric("Missing ใน Feature หลัง Clean", f"{int(clean_df[features].isna().sum().sum()):,}")
    st.markdown('<div class="panel-title">สรุปรายตัวแปร</div>', unsafe_allow_html=True)
    per = pd.DataFrame({"ตัวแปร": [label(c) for c in features],
                        "ค่าว่าง": [rep["missing"][c] for c in features],
                        "ติดลบ": [rep["negative"][c] for c in features],
                        "Outlier (IQR)": [rep["outliers"][c] for c in features]})
    st.dataframe(per, use_container_width=True, hide_index=True)
    st.markdown('<div class="panel-title">สถิติเบื้องต้นหลัง Clean</div>', unsafe_allow_html=True)
    st.dataframe(clean_df[features].describe().T.round(2), use_container_width=True)

# ===== 3. EDA =====
with t_eda:
    st.markdown("### 📊 EDA และการแปลผล")
    st.markdown("#### 1) การกระจายของข้อมูล")
    f_sel = st.selectbox("เลือก Feature", features, format_func=label)
    fig = px.histogram(clean_df, x=f_sel, nbins=30, color_discrete_sequence=["#8b7cf6"],
                       title=f"Histogram: {label(f_sel)}")
    st.plotly_chart(style(fig), use_container_width=True)
    st.caption("ใช้ดูรูปร่างการกระจายและหางของข้อมูล หางยาวแปลว่ามีลูกค้ากลุ่มค่าสูงจำนวนน้อย")
    fig = px.box(clean_df, y=f_sel, points="outliers", title=f"Boxplot: {label(f_sel)}",
                 color_discrete_sequence=["#8b7cf6"])
    st.plotly_chart(style(fig), use_container_width=True)
    st.markdown("#### 2) ความสัมพันธ์ระหว่างตัวแปร")
    corr = clean_df[features].corr().round(2)
    corr.index = corr.columns = [label(c) for c in features]
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    st.plotly_chart(style(fig), use_container_width=True)
    pairs = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack()
    top = pairs.abs().idxmax()
    st.caption(f"คู่ที่สัมพันธ์กันสูงสุดคือ {top[0]} กับ {top[1]} (r={pairs[top]:.2f}) "
               + ("— ตัวแปรแต่ละตัวให้ข้อมูลต่างกัน เหมาะนำมาใช้ร่วมกัน" if abs(pairs[top]) < .3
                  else "— มีความสัมพันธ์กันพอสมควร"))
    st.markdown("#### 3) ปัญหาคุณภาพข้อมูลรายตัวแปร")
    q = pd.DataFrame({"ตัวแปร": [label(c) for c in features], "ค่าว่าง": [rep["missing"][c] for c in features],
                      "ติดลบ": [rep["negative"][c] for c in features],
                      "Outlier": [rep["outliers"][c] for c in features]}).melt("ตัวแปร", var_name="ประเภท", value_name="จำนวน")
    fig = px.bar(q, x="ตัวแปร", y="จำนวน", color="ประเภท", barmode="group",
                 color_discrete_sequence=["#7c83fd", "#f47db5", "#ffc66d"])
    st.plotly_chart(style(fig), use_container_width=True)

# ===== 4. Model & evaluation =====
with t_model:
    st.markdown("### 🤖 โมเดลและการประเมินผล")
    st.markdown('<div class="box"><b>K-Means</b> แบ่งตามระยะห่างจากศูนย์กลางกลุ่ม · '
                '<b>Hierarchical (Ward)</b> ไล่รวมลูกค้าที่ใกล้กันทีละคู่จนเหลือ K กลุ่ม · '
                'ประเมินด้วย Silhouette (สูงดี), Davies-Bouldin (ต่ำดี), ความเสถียร และ Train/Test</div>',
                unsafe_allow_html=True)
    st.markdown("")
    c = st.columns(5)
    kpi(c[0], "Silhouette (K-Means)", f"{cur['sil']:.3f}", f"K={k_value} · Hierarchical {cur['hc_sil']:.3f}")
    kpi(c[1], "Davies-Bouldin", f"{cur['db']:.2f}", "ยิ่งต่ำยิ่งแยกกลุ่มดี")
    kpi(c[2], "ความเสถียร (ARI)", f"{cur['stability']:.2f}", "สุ่มเริ่มใหม่แล้วได้กลุ่มเดิมแค่ไหน")
    kpi(c[3], "Test Silhouette", f"{ho['sil_test']:.3f}", f"Train {ho['sil_train']:.3f} ({ho['n_train']}/{ho['n_test']} แถว)")
    kpi(c[4], "K-Means vs Hier. (ARI)", f"{cur['ari']:.2f}", "1 = ตรงกันทุกคน")
    gap = abs(ho["sil_train"] - ho["sil_test"])
    s = cur["sil"]
    level = "ดี" if s > .5 else "พอใช้" if s > .25 else "ค่อนข้างต่ำ"
    st.markdown("")
    st.markdown("#### 🧐 แม่นยำหรือไม่ เพราะเหตุใด")
    st.caption("งานนี้ไม่มี Target จึงไม่มีค่า Accuracy แบบ Classification ต้องวัดคุณภาพกลุ่มจากข้อมูลเอง")
    st.markdown(f"""
- **การแยกกลุ่ม:** Silhouette {s:.3f} ({level}) — {"ลูกค้าส่วนใหญ่อยู่ใกล้กลุ่มตัวเองมากกว่ากลุ่มอื่น" if s > .25 else "กลุ่มซ้อนทับกันพอสมควร เพราะพฤติกรรมลูกค้าเป็นค่าต่อเนื่อง ไม่มีเส้นแบ่งชัด ผลจึงเหมาะใช้แบ่งกลุ่มเพื่อทำการตลาด มากกว่าเป็นกลุ่มตามธรรมชาติ"}
- **ความเสถียร:** ARI {cur['stability']:.2f} — {"ผลคงที่แม้สุ่มจุดเริ่มต้นใหม่" if cur['stability'] >= .8 else "ผลเปลี่ยนตามจุดเริ่มต้น ควรระวังการตีความ"}
- **ใช้กับข้อมูลใหม่:** Silhouette ชุดทดสอบต่างจากชุดเทรน {gap:.3f} — {"กลุ่มใช้กับลูกค้าที่ไม่เคยเห็นได้ใกล้เคียงกัน" if gap < .05 else "ผลค่อนข้างผันผวนเมื่อเจอข้อมูลใหม่"}
- **เทียบสองวิธี:** ARI {cur['ari']:.2f} — {"สองวิธีให้ผลสอดคล้อง ช่วยยืนยันว่าโครงสร้างกลุ่มไม่ได้เกิดจากวิธีใดวิธีหนึ่ง" if cur['ari'] >= .6 else "สองวิธีแบ่งต่างกันพอสมควร แปลว่าขอบเขตกลุ่มไม่แน่นอน"} · วิธีที่ Silhouette สูงกว่าที่ K นี้คือ **{"Hierarchical" if cur['hc_sil'] > cur['sil'] else "K-Means"}**
""")
    st.markdown("#### เลือก K ที่เหมาะสม")
    left, right = st.columns(2)
    with left:
        fig = go.Figure(go.Scatter(x=K_RANGE, y=[ev[k]["inertia"] for k in K_RANGE], mode="lines+markers",
                                   line=dict(color="#7c83fd", width=3), marker=dict(size=8)))
        fig.update_layout(title="Elbow Method (K-Means)", xaxis_title="จำนวนกลุ่ม (K)", yaxis_title="Inertia")
        st.plotly_chart(style(fig), use_container_width=True)
    with right:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=K_RANGE, y=[ev[k]["sil"] for k in K_RANGE], name="K-Means",
                                 mode="lines+markers", line=dict(color="#e879b9", width=3)))
        fig.add_trace(go.Scatter(x=K_RANGE, y=[ev[k]["hc_sil"] for k in K_RANGE], name="Hierarchical",
                                 mode="lines+markers", line=dict(color="#56cfa5", width=3)))
        fig.update_layout(title="Silhouette Score", xaxis_title="จำนวนกลุ่ม (K)", yaxis_title="Silhouette")
        st.plotly_chart(style(fig), use_container_width=True)
    table = pd.DataFrame({"K": K_RANGE, "Inertia": [ev[k]["inertia"] for k in K_RANGE],
                          "Silhouette K-Means": [ev[k]["sil"] for k in K_RANGE],
                          "Silhouette Hierarchical": [ev[k]["hc_sil"] for k in K_RANGE],
                          "Davies-Bouldin": [ev[k]["db"] for k in K_RANGE],
                          "Stability (ARI)": [ev[k]["stability"] for k in K_RANGE],
                          "ARI KM vs HC": [ev[k]["ari"] for k in K_RANGE]})
    st.dataframe(table.round(4), use_container_width=True, hide_index=True)
    st.success(f"K ที่มี Silhouette (K-Means) สูงสุดในช่วงที่ทดสอบคือ {best_k} "
               "(ควรพิจารณาร่วมกับ Elbow และความหมายทางธุรกิจ)")
    st.markdown("#### 🌳 Dendrogram (Hierarchical, Ward)")
    Z = linkage(X_scaled, method="ward")
    fig_d, ax = plt.subplots(figsize=(10, 3.6))
    dendrogram(Z, truncate_mode="lastp", p=30, ax=ax, color_threshold=Z[-(k_value - 1), 2] - 1e-9)
    ax.set_xlabel("กลุ่มย่อย (ตัวเลขในวงเล็บ = จำนวนลูกค้า)")
    ax.set_ylabel("ระยะ Ward")
    st.pyplot(fig_d)
    plt.close(fig_d)
    st.caption("เส้นสีแยกกลุ่มตามค่า K ที่เลือกในแถบด้านข้าง")

# ===== 5. Cluster profile =====
with t_prof:
    st.markdown(f"### 🔎 โปรไฟล์ลูกค้าแต่ละกลุ่ม ({method}, K={k_value})")
    left, right = st.columns([1, 1])
    with left:
        counts = result_df["Cluster"].value_counts().sort_index().rename_axis("Cluster").reset_index(name="จำนวนลูกค้า")
        counts["Cluster"] = counts["Cluster"].map(lambda i: f"กลุ่ม {chr(65 + i)}")
        fig = px.pie(counts, names="Cluster", values="จำนวนลูกค้า", hole=.55, color_discrete_sequence=COLORS,
                     title="ขนาดของแต่ละกลุ่ม")
        st.plotly_chart(style(fig), use_container_width=True)
    with right:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_scaled)
        pdf = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1],
                            "Cluster": [f"กลุ่ม {chr(65 + i)}" for i in labels]})
        fig = px.scatter(pdf, x="PC1", y="PC2", color="Cluster", opacity=.72, color_discrete_sequence=COLORS,
                         title="แผนภาพลูกค้า (PCA 2D)")
        st.plotly_chart(style(fig), use_container_width=True)
        st.caption(f"PCA 2D อธิบายความแปรปรวนรวม {pca.explained_variance_ratio_.sum() * 100:.1f}% ของข้อมูล")
    zmeans = cluster_means(X_scaled, labels, k_value)
    st.markdown('<div class="panel-title">ลักษณะและแนวทางใช้งานของแต่ละกลุ่ม</div>', unsafe_allow_html=True)
    cols = st.columns(min(k_value, 4))
    for q in range(k_value):
        size = int((labels == q).sum())
        z = zmeans[q]
        order = np.argsort(-np.abs(z))[:3]
        tags = "".join(f'<span class="tag">{label(features[j])} {"⬆️ สูง" if z[j] > 0 else "⬇️ ต่ำ"}</span>'
                       for j in order if abs(z[j]) >= .4) or '<span class="tag">ค่ากลางๆ ทุกด้าน</span>'
        ups = [j for j in order if z[j] >= .4 and features[j] in ACTIONS]
        tip = f'<div class="small-note">💡 แนวทาง: {ACTIONS[features[ups[0]]]}</div>' if ups else ""
        cols[q % len(cols)].markdown(
            f'<div class="kpi" style="margin-bottom:12px"><b style="color:{COLORS[q]}">●</b> '
            f'<b>กลุ่ม {chr(65 + q)}</b> <span class="small-note">· {size:,} คน ({size / len(labels) * 100:.1f}%)</span>'
            f'<div style="margin-top:8px">{tags}</div>{tip}</div>', unsafe_allow_html=True)
    zdf = pd.DataFrame(zmeans, columns=[label(c) for c in features])
    zdf.insert(0, "กลุ่ม", [f"กลุ่ม {chr(65 + i)}" for i in range(k_value)])
    fig = go.Figure()
    for i in range(k_value):
        vals = list(zmeans[i]) + [zmeans[i][0]]
        fig.add_trace(go.Scatterpolar(r=vals, theta=[label(c) for c in features] + [label(features[0])],
                                      name=f"กลุ่ม {chr(65 + i)}", line=dict(color=COLORS[i]), fill="toself", opacity=.55))
    fig.update_layout(title="เปรียบเทียบทุกกลุ่ม (ค่ามาตรฐาน)", height=450)
    st.plotly_chart(style(fig), use_container_width=True)
    profile = result_df.groupby("Cluster")[features].mean().round(2)
    profile.index = [f"กลุ่ม {chr(65 + i)}" for i in profile.index]
    profile.columns = [label(c) for c in profile.columns]
    st.markdown('<div class="panel-title">ค่าเฉลี่ยรายกลุ่ม</div>', unsafe_allow_html=True)
    st.dataframe(profile, use_container_width=True)
    st.markdown("#### ตัวอย่างข้อมูลพร้อม Cluster")
    st.dataframe(result_df.head(30), use_container_width=True, hide_index=True)

# ===== 6. Download & deploy =====
with t_dl:
    st.markdown("### ⬇️ ดาวน์โหลดผลการวิเคราะห์")
    out = io.BytesIO()
    summary = pd.DataFrame({"ตัวแปร": [label(c) for c in features],
                            "ค่าว่าง": [rep["missing"][c] for c in features],
                            "ติดลบ": [rep["negative"][c] for c in features],
                            "Outlier (IQR)": [rep["outliers"][c] for c in features]})
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        result_df.to_excel(writer, index=False, sheet_name="Cluster_Result")
        result_df.groupby("Cluster")[features].mean().round(2).to_excel(writer, sheet_name="Cluster_Profile")
        table.round(4).to_excel(writer, index=False, sheet_name="Model_Evaluation")
        summary.to_excel(writer, index=False, sheet_name="Data_Prep_Summary")
    st.download_button("💜 ดาวน์โหลด Excel ผลลัพธ์", data=out.getvalue(),
                       file_name="CL02_customer_behavior_clustering_result.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.markdown("#### Preview")
    st.dataframe(result_df.head(20), use_container_width=True, hide_index=True)
    st.markdown("### 🚀 การ Deploy")
    st.markdown("""
1. สร้าง Repository บน GitHub แล้วอัปโหลด `app.py`, `requirements.txt` และไฟล์ Excel (ถ้าต้องการข้อมูลเริ่มต้น)
2. เข้า https://share.streamlit.io/ เชื่อมบัญชี GitHub
3. เลือก Repository และตั้ง Main file path เป็น `app.py` แล้วกด Deploy
4. ส่ง URL ที่ได้ให้อาจารย์เปิดดู

อย่าอัปโหลดข้อมูลส่วนบุคคลหรือข้อมูลลับขึ้น Repository สาธารณะ
""")

st.markdown("---")
st.markdown('<p class="small-note">💗 CustomerInsight · CL02 Customer Behavior Clustering · '
            'ผลลัพธ์ขึ้นอยู่กับ Dataset ที่อัปโหลดและค่า K ที่เลือก</p>', unsafe_allow_html=True)
