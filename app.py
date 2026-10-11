import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

st.set_page_config(page_title="CustomerInsight CL02", page_icon="💜", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] {font-family:'Prompt',sans-serif;}
.stApp {background:linear-gradient(135deg,#fbf8ff 0%,#f4f3ff 55%,#fff7fc 100%);}
section[data-testid="stSidebar"] {background:linear-gradient(180deg,#eee9ff,#fff0f8);}
.hero {padding:26px 30px;border-radius:24px;background:linear-gradient(115deg,#f6dcff,#e1e8ff 55%,#dcfff1);border:1px solid #e7ddff;margin-bottom:18px;}
.hero h1 {color:#29205f;font-size:2.1rem;margin:0;}
.hero p {color:#5b5383;margin:.4rem 0 0;}
div[data-testid="stMetric"] {background:#ffffffd9;border:1px solid #e9e2fa;padding:16px;border-radius:18px;}
h2,h3 {color:#33246f;}
.stButton>button,.stDownloadButton>button {border:0;border-radius:12px;background:linear-gradient(90deg,#8065dc,#d16bb5);color:white;font-weight:600;}
</style>
""", unsafe_allow_html=True)

FEATURES_DEFAULT = [
    "age", "monthly_spend", "visits_month", "avg_items",
    "discount_usage_percent", "online_orders_month"
]
DEFAULT_FILE = os.path.join(os.path.dirname(__file__), "CL02_customer_behavior_clustering.xlsx")

st.sidebar.markdown("## 💜 CustomerInsight")
st.sidebar.caption("CL02 · Customer Behavior Clustering")
page = st.sidebar.radio("เมนู", ["🏠 ภาพรวม", "🧹 Data Cleaning", "📊 Clustering", "🔎 Cluster Profile", "⬇️ ดาวน์โหลดผลลัพธ์"])
st.sidebar.markdown("---")
st.sidebar.caption("Pastel purple · pink · mint")

st.markdown("""
<div class="hero">
  <h1>CustomerInsight ✨</h1>
  <p>Customer Behavior Clustering Dashboard</p>
  <p>สำรวจคุณภาพข้อมูล แบ่งกลุ่มลูกค้า และดู Insight จากข้อมูลจริง</p>
</div>
""", unsafe_allow_html=True)

with st.expander("📂 เลือก Dataset (Excel)", expanded=not os.path.exists(DEFAULT_FILE)):
    uploaded = st.file_uploader("อัปโหลดไฟล์ CL02 (.xlsx)", type=["xlsx"])
    if uploaded:
        source_bytes = uploaded.getvalue()
        try:
            excel = pd.ExcelFile(io.BytesIO(source_bytes))
            sheet = "Raw_Data" if "Raw_Data" in excel.sheet_names else excel.sheet_names[0]
            raw = pd.read_excel(io.BytesIO(source_bytes), sheet_name=sheet)
            source_name = uploaded.name
        except Exception as e:
            st.error(f"อ่านไฟล์ไม่สำเร็จ: {e}")
            st.stop()
    elif os.path.exists(DEFAULT_FILE):
        source_bytes = open(DEFAULT_FILE, "rb").read()
        excel = pd.ExcelFile(io.BytesIO(source_bytes))
        sheet = "Raw_Data" if "Raw_Data" in excel.sheet_names else excel.sheet_names[0]
        raw = pd.read_excel(io.BytesIO(source_bytes), sheet_name=sheet)
        source_name = os.path.basename(DEFAULT_FILE)
    else:
        raw = None

if raw is None:
    st.info("อัปโหลดไฟล์ CL02_customer_behavior_clustering.xlsx เพื่อเริ่มใช้งาน")
    st.stop()

raw.columns = [str(c).strip() for c in raw.columns]
id_col = "customer_id" if "customer_id" in raw.columns else None
features = [c for c in FEATURES_DEFAULT if c in raw.columns]
if len(features) < 2:
    features = [c for c in raw.select_dtypes(include=np.number).columns if c != id_col]
if len(features) < 2:
    st.error("Dataset ต้องมี Feature ตัวเลขอย่างน้อย 2 คอลัมน์")
    st.stop()

rows_before = len(raw)
duplicates = int(raw.duplicated().sum())
missing_before = int(raw[features].isna().sum().sum())

clean = raw.drop_duplicates().copy()
for col in features:
    clean[col] = pd.to_numeric(clean[col], errors="coerce")
    clean[col] = clean[col].fillna(clean[col].median())

# Use a stable scaler and cluster on selected numeric behavior features.
with st.sidebar:
    st.markdown("### ⚙️ ตั้งค่ากลุ่ม")
    k = st.slider("จำนวน Cluster (K)", min_value=2, max_value=8, value=3)
    selected_features = st.multiselect("Features ที่ใช้", features, default=features)
if len(selected_features) < 2:
    st.warning("เลือก Feature อย่างน้อย 2 ตัวเพื่อทำ Clustering")
    st.stop()

X = clean[selected_features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = model.fit_predict(X_scaled)
clean["Cluster"] = labels
sil = silhouette_score(X_scaled, labels) if len(clean) > k and len(np.unique(labels)) > 1 else float("nan")

# K evaluation
k_values = list(range(2, min(8, len(clean)-1) + 1))
eval_rows = []
for kk in k_values:
    km = KMeans(n_clusters=kk, random_state=42, n_init=10)
    labs = km.fit_predict(X_scaled)
    eval_rows.append({
        "K": kk,
        "Inertia": float(km.inertia_),
        "Silhouette Score": float(silhouette_score(X_scaled, labs))
    })
evaluation = pd.DataFrame(eval_rows)
profile = clean.groupby("Cluster")[selected_features].mean().round(2)
counts = clean["Cluster"].value_counts().sort_index()

if page == "🏠 ภาพรวม":
    st.subheader("สวัสดี ยินดีต้อนรับสู่ CustomerInsight 💗")
    st.write("แดชบอร์ดสำหรับวิเคราะห์พฤติกรรมลูกค้าด้วย K-Means เลือกค่า K และ Feature ได้จากแถบด้านซ้าย")
    a,b,c,d = st.columns(4)
    a.metric("👥 Records หลัง Clean", f"{len(clean):,}")
    b.metric("🧩 Features", len(selected_features))
    c.metric("🎯 จำนวนกลุ่ม K", k)
    d.metric("📐 Silhouette Score", f"{sil:.3f}")
    left,right = st.columns([1.2,1])
    with left:
        st.markdown("### จำนวนลูกค้าในแต่ละ Cluster")
        plot_df = counts.rename_axis("Cluster").reset_index(name="จำนวนลูกค้า")
        fig = px.bar(plot_df, x="Cluster", y="จำนวนลูกค้า", color="Cluster",
                     color_discrete_sequence=["#8b7cf6","#f28ac2","#5bcfa8","#ffc66d","#9b8af7","#70a7f9","#ed79a7","#55c5c0"])
        fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("### ข้อมูลที่ตรวจพบ")
        st.write(f"**ไฟล์:** {source_name}")
        st.write(f"**ชีต:** {sheet}")
        st.write(f"**จำนวนแถวก่อน Clean:** {rows_before:,}")
        st.write(f"**แถวซ้ำทั้งแถว:** {duplicates:,}")
        st.write(f"**Missing ใน Features ก่อน Clean:** {missing_before:,}")
        st.info("ระบบลบแถวซ้ำทั้งแถว และเติม Missing ใน Features ด้วย Median ส่วน Outlier จะแสดงให้ตรวจสอบ ไม่ลบอัตโนมัติ")

elif page == "🧹 Data Cleaning":
    st.subheader("ตรวจสอบและเตรียมข้อมูล 🧹")
    a,b,c = st.columns(3)
    a.metric("Rows ก่อน Clean", f"{rows_before:,}")
    b.metric("Duplicate rows", f"{duplicates:,}")
    c.metric("Missing ใน Features ก่อน Clean", f"{missing_before:,}")
    st.markdown("#### Missing Values รายคอลัมน์")
    st.dataframe(raw.isna().sum().rename("Missing Values").rename_axis("Column").reset_index(), use_container_width=True, hide_index=True)
    st.markdown("#### สถิติหลัง Clean")
    st.dataframe(clean[selected_features].describe().T.round(2), use_container_width=True)
    chosen = st.selectbox("เลือก Feature เพื่อดู Outlier", selected_features)
    fig = px.box(clean, y=chosen, points="outliers", title=f"Boxplot: {chosen}",
                 color_discrete_sequence=["#9b7cf6"])
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("จุดนอก whiskers เป็นค่าที่ควรตรวจสอบ ไม่ได้หมายความว่าต้องลบทิ้งเสมอไป")

elif page == "📊 Clustering":
    st.subheader("เลือกจำนวนกลุ่มด้วย Elbow และ Silhouette 📊")
    l,r = st.columns(2)
    with l:
        fig = px.line(evaluation, x="K", y="Inertia", markers=True, title="Elbow Method",
                      color_discrete_sequence=["#8370e8"])
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with r:
        fig = px.line(evaluation, x="K", y="Silhouette Score", markers=True, title="Silhouette Score",
                      color_discrete_sequence=["#e879b9"])
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(evaluation.round(4), use_container_width=True, hide_index=True)
    best_k = int(evaluation.loc[evaluation["Silhouette Score"].idxmax(), "K"])
    st.success(f"K ที่ได้ Silhouette Score สูงสุดในช่วงที่ทดลองคือ {best_k} — ควรพิจารณาร่วมกับ Elbow และความหมายของกลุ่มด้วย")
    st.markdown("#### แสดงกลุ่มใน 2 มิติด้วย PCA")
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame({"PC1": coords[:,0], "PC2": coords[:,1], "Cluster": labels.astype(str)})
    fig = px.scatter(pca_df, x="PC1", y="PC2", color="Cluster", opacity=.72,
                     color_discrete_sequence=["#8370e8","#ef83bf","#55cfa5","#ffc66d","#8aa8ff","#f28b82","#6fc8df","#ba8cf4"])
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"PCA 2D แสดงความแปรปรวนรวม {pca.explained_variance_ratio_.sum()*100:.1f}% ของข้อมูล")

elif page == "🔎 Cluster Profile":
    st.subheader("พฤติกรรมเฉลี่ยของแต่ละกลุ่ม 🔎")
    st.dataframe(profile, use_container_width=True)
    feature = st.selectbox("เลือก Feature เพื่อเปรียบเทียบค่าเฉลี่ย", selected_features)
    prof = profile[feature].rename("ค่าเฉลี่ย").reset_index()
    fig = px.bar(prof, x="Cluster", y="ค่าเฉลี่ย", color="Cluster",
                 color_discrete_sequence=["#8370e8","#ef83bf","#55cfa5","#ffc66d","#8aa8ff","#f28b82","#6fc8df","#ba8cf4"])
    fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    st.info("ตั้งชื่อกลุ่มจากค่าจริงในตาราง เช่น กลุ่มใช้จ่ายสูง หรือกลุ่มซื้อออนไลน์บ่อย หลีกเลี่ยงการสรุปก่อนดูผล")
    st.markdown("#### ตัวอย่างแถวข้อมูลพร้อม Cluster")
    st.dataframe(clean.head(30), use_container_width=True, hide_index=True)

elif page == "⬇️ ดาวน์โหลดผลลัพธ์":
    st.subheader("ดาวน์โหลดผลลัพธ์ 📥")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        clean.to_excel(writer, index=False, sheet_name="Cluster_Result")
        profile.to_excel(writer, sheet_name="Cluster_Profile")
        evaluation.round(4).to_excel(writer, index=False, sheet_name="K_Evaluation")
    st.download_button("💜 ดาวน์โหลด Excel ผลการวิเคราะห์", data=buffer.getvalue(),
        file_name="CL02_customer_behavior_clustering_result.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.dataframe(clean.head(20), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("💜 CustomerInsight CL02 · ผลลัพธ์คำนวณจาก Dataset ที่เลือก · ตรวจสอบค่า K และวิธี Clean ก่อนนำเสนอ")
