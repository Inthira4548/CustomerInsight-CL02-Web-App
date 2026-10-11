import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="CustomerInsight | CL02",
    page_icon="💗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Theme ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Prompt', sans-serif; }
.stApp { background: linear-gradient(135deg, #fbfaff 0%, #f5f7ff 55%, #fff8fc 100%); }
section[data-testid="stSidebar"] { background: linear-gradient(180deg,#f0edff 0%,#fff2fa 100%); border-right: 1px solid #e4ddff; }
.hero {
  padding: 28px 30px; border-radius: 24px; color: #1e1b4b;
  background: linear-gradient(115deg,#f5dfff 0%,#e2e9ff 52%,#d9fff1 100%);
  border: 1px solid #e4ddff; margin-bottom: 18px;
}
.hero h1 { font-size: 2.15rem; line-height: 1.25; margin: 0 0 8px 0; color:#19145b; }
.hero p { font-size: 1rem; margin: 0; color:#484276; }
.kpi {
  padding: 18px 20px; border-radius: 20px; border: 1px solid #e4e5f7;
  background: rgba(255,255,255,.88); box-shadow: 0 5px 18px rgba(72,62,130,.05);
  min-height: 115px;
}
.kpi-label { color:#68648b; font-size:.88rem; }
.kpi-value { color:#211b63; font-size:1.75rem; font-weight:700; margin-top:3px; }
.kpi-note { color:#7a76a0; font-size:.78rem; margin-top:4px; }
.panel-title { font-weight:600; color:#27216a; font-size:1.1rem; margin: 8px 0 10px; }
div[data-testid="stMetric"] { background:#ffffffd9; padding:16px; border-radius:18px; border:1px solid #e6e3f7; }
.stButton > button, .stDownloadButton > button {
  border-radius: 12px; border: 0; background: linear-gradient(90deg,#7760db,#b55ec6);
  color:white; font-weight:600;
}
div[data-testid="stDataFrame"] { border-radius: 14px; overflow:hidden; }
.small-note { color:#777393; font-size:.82rem; }
</style>
""", unsafe_allow_html=True)

# ---------- Helpers ----------
KNOWN_FEATURES = [
    "age", "monthly_spend", "visits_month", "avg_items",
    "discount_usage_percent", "online_orders_month"
]
ID_CANDIDATES = ["customer_id", "customerid", "id", "รหัสลูกค้า"]

def find_id_column(columns):
    lookup = {str(c).strip().lower(): c for c in columns}
    for key in ID_CANDIDATES:
        if key in lookup:
            return lookup[key]
    for c in columns:
        if "id" in str(c).strip().lower() and pd.api.types.is_numeric_dtype(pd.Series([1])):
            return c
    return None

def load_excel(uploaded_file):
    try:
        xls = pd.ExcelFile(uploaded_file)
        sheet = "Raw_Data" if "Raw_Data" in xls.sheet_names else xls.sheet_names[0]
        return pd.read_excel(xls, sheet_name=sheet), sheet
    except Exception as e:
        st.error(f"อ่านไฟล์ไม่สำเร็จ: {e}")
        return None, None

def clean_frame(raw):
    data = raw.copy()
    data.columns = [str(c).strip() for c in data.columns]
    before_rows = len(data)
    duplicate_count = int(data.duplicated().sum())
    data = data.drop_duplicates().copy()
    id_col = find_id_column(data.columns)
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    feature_cols = [c for c in KNOWN_FEATURES if c in data.columns]
    if not feature_cols:
        feature_cols = [c for c in numeric_cols if c != id_col]
    # Only impute numeric analysis fields; retain identifiers as-is.
    for c in feature_cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")
        if data[c].isna().any():
            median = data[c].median()
            data[c] = data[c].fillna(median)
    return data, before_rows, duplicate_count, feature_cols, id_col

def fmt_num(v):
    return f"{v:,.2f}" if pd.notna(v) else "—"

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 💗 CustomerInsight")
    st.caption("CL02 · Customer Behavior Clustering")
    page = st.radio(
        "เมนูหลัก",
        ["🏠 ภาพรวม", "🧹 ตรวจและ Clean Data", "📊 วิเคราะห์ Clustering", "🔎 โปรไฟล์ลูกค้า", "⬇️ ดาวน์โหลดผลลัพธ์"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("### ✨ โปรเจกต์ของเรา")
    st.markdown("แดชบอร์ดจัดกลุ่มพฤติกรรมลูกค้า เพื่อช่วยมองเห็นรูปแบบการซื้อและสนับสนุนการวางแผนธุรกิจ")
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
with st.expander("📂 เลือก Dataset", expanded=False):
    uploaded = st.file_uploader("อัปโหลดไฟล์ Excel (.xlsx)", type=["xlsx"])
    default_path = os.path.join(os.path.dirname(__file__), "CL02_customer_behavior_clustering.xlsx")
    if uploaded is not None:
        raw_df, sheet_name = load_excel(uploaded)
        source_name = uploaded.name
    elif os.path.exists(default_path):
        raw_df, sheet_name = load_excel(default_path)
        source_name = os.path.basename(default_path)
    else:
        raw_df, sheet_name, source_name = None, None, None
        st.info("อัปโหลดไฟล์ CL02 เพื่อเริ่มวิเคราะห์")

if "raw_df" not in locals() or raw_df is None:
    st.warning("กรุณาอัปโหลดไฟล์ CL02_customer_behavior_clustering.xlsx ในส่วน “เลือก Dataset”")
    st.stop()

clean_df, original_rows, duplicate_count, features, id_col = clean_frame(raw_df)
if len(features) < 2:
    st.error("ต้องมีคอลัมน์ตัวเลขอย่างน้อย 2 คอลัมน์เพื่อทำ Clustering")
    st.write("คอลัมน์ที่พบ:", list(clean_df.columns))
    st.stop()

missing_before = int(raw_df.isna().sum().sum())
missing_after = int(clean_df[features].isna().sum().sum())
X = clean_df[features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Select K based on user control and calculate scores
with st.sidebar:
    st.markdown("### ⚙️ ตั้งค่าการวิเคราะห์")
    k_value = st.slider("จำนวนกลุ่ม (K)", min_value=2, max_value=8, value=3, step=1)
    st.caption("K=3 เป็นค่าเริ่มต้น ปรับได้ตามผล Elbow และ Silhouette")

kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
sil_score = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 and len(X_scaled) > k_value else np.nan
result_df = clean_df.copy()
result_df["Cluster"] = labels

# For K-selection chart
k_candidates = list(range(2, min(10, len(result_df)-1) + 1))
inertias, silhouettes = [], []
for k in k_candidates:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    lab = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, lab))

# ---------- Pages ----------
if page == "🏠 ภาพรวม":
    st.markdown("### 👋 ยินดีต้อนรับสู่ CustomerInsight")
    st.write("สำรวจข้อมูลลูกค้า ตรวจสอบคุณภาพข้อมูล และทดลองแบ่งกลุ่มลูกค้าได้จากเมนูด้านซ้าย")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="kpi"><div class="kpi-label">👥 จำนวน Records</div><div class="kpi-value">{len(result_df):,}</div><div class="kpi-note">หลังลบแถวซ้ำ</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi"><div class="kpi-label">🧩 จำนวน Features</div><div class="kpi-value">{len(features)}</div><div class="kpi-note">ตัวแปรที่ใช้วิเคราะห์</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="kpi"><div class="kpi-label">🎯 จำนวน Cluster</div><div class="kpi-value">{k_value}</div><div class="kpi-note">ปรับได้จากแถบด้านข้าง</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="kpi"><div class="kpi-label">📐 Silhouette Score</div><div class="kpi-value">{sil_score:.3f}</div><div class="kpi-note">ยิ่งสูงโดยทั่วไปยิ่งแยกกลุ่มได้ดี</div></div>', unsafe_allow_html=True)
    st.markdown("")
    left, right = st.columns([1.25, 1])
    with left:
        st.markdown('<div class="panel-title">📈 ขนาดของแต่ละกลุ่ม</div>', unsafe_allow_html=True)
        counts = result_df["Cluster"].value_counts().sort_index().rename_axis("Cluster").reset_index(name="จำนวนลูกค้า")
        fig = px.bar(counts, x="Cluster", y="จำนวนลูกค้า", color="Cluster",
                     color_discrete_sequence=["#7c83fd","#f47db5","#56cfa5","#ffc66d","#a78bfa","#60a5fa","#fb7185","#34d399"])
        fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown('<div class="panel-title">🧠 ภาพรวมการเตรียมข้อมูล</div>', unsafe_allow_html=True)
        st.write(f"**ไฟล์:** `{source_name}`")
        st.write(f"**ชีตที่อ่าน:** `{sheet_name}`")
        st.write(f"**แถวก่อน Clean:** {original_rows:,}")
        st.write(f"**แถวซ้ำที่พบ:** {duplicate_count:,}")
        st.write(f"**Missing ก่อน Clean:** {missing_before:,}")
        st.write(f"**Features:** {', '.join(features)}")
        st.info("ระบบนี้ลบแถวที่ซ้ำทั้งแถว และเติม Missing ใน Features เชิงตัวเลขด้วย Median โดยไม่ลบ Outlier อัตโนมัติ")

elif page == "🧹 ตรวจและ Clean Data":
    st.markdown("### 🧹 ตรวจสอบคุณภาพข้อมูล")
    a, b, c = st.columns(3)
    a.metric("Rows ก่อน Clean", f"{original_rows:,}")
    b.metric("Duplicate rows", f"{duplicate_count:,}")
    c.metric("Missing ใน Feature หลัง Clean", f"{missing_after:,}")
    st.markdown("#### Missing Values ก่อน Clean")
    miss = raw_df.isna().sum().rename("Missing Values").reset_index().rename(columns={"index":"Column"})
    st.dataframe(miss, use_container_width=True, hide_index=True)
    st.markdown("#### สถิติเบื้องต้นหลัง Clean")
    st.dataframe(clean_df[features].describe().T.round(2), use_container_width=True)
    st.markdown("#### Boxplot สำหรับตรวจ Outlier")
    selected_feature = st.selectbox("เลือก Feature", features)
    fig = px.box(clean_df, y=selected_feature, points="outliers", title=f"Boxplot: {selected_feature}",
                 color_discrete_sequence=["#8b7cf6"])
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("หมายเหตุ: จุดที่อยู่นอก whiskers เป็นค่าที่ควรตรวจสอบ ไม่ได้แปลว่าต้องลบทิ้งเสมอไป")

elif page == "📊 วิเคราะห์ Clustering":
    st.markdown("### 📊 หา K ที่เหมาะสม")
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=k_candidates, y=inertias, mode="lines+markers", line=dict(color="#7c83fd", width=3), marker=dict(size=8)))
        fig.update_layout(title="Elbow Method", xaxis_title="จำนวนกลุ่ม (K)", yaxis_title="Inertia", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=k_candidates, y=silhouettes, mode="lines+markers", line=dict(color="#e879b9", width=3), marker=dict(size=8)))
        fig.update_layout(title="Silhouette Score", xaxis_title="จำนวนกลุ่ม (K)", yaxis_title="Silhouette Score", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    score_table = pd.DataFrame({"K": k_candidates, "Inertia": inertias, "Silhouette Score": silhouettes})
    st.dataframe(score_table.round(4), use_container_width=True, hide_index=True)
    best_k = int(score_table.loc[score_table["Silhouette Score"].idxmax(), "K"])
    st.success(f"K ที่มี Silhouette Score สูงสุดในช่วงที่ทดสอบคือ {best_k} (ควรพิจารณาร่วมกับ Elbow และความหมายทางธุรกิจ)")
    st.markdown("#### 🫧 แผนภาพกลุ่มลูกค้า (PCA 2D)")
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame({"PC1": coords[:,0], "PC2": coords[:,1], "Cluster": labels.astype(str)})
    fig = px.scatter(pca_df, x="PC1", y="PC2", color="Cluster", opacity=0.72,
                     color_discrete_sequence=["#7c83fd","#f47db5","#56cfa5","#ffc66d","#a78bfa","#60a5fa","#fb7185","#34d399"],
                     title="K-Means clusters projected with PCA")
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"PCA 2D อธิบายความแปรปรวนรวม {pca.explained_variance_ratio_.sum()*100:.1f}% ของข้อมูลที่แสดงบนกราฟนี้")

elif page == "🔎 โปรไฟล์ลูกค้า":
    st.markdown("### 🔎 วิเคราะห์ลักษณะของแต่ละ Cluster")
    profile = result_df.groupby("Cluster")[features].mean().round(2)
    st.dataframe(profile, use_container_width=True)
    st.markdown("#### เปรียบเทียบค่าเฉลี่ยราย Feature")
    feature_choice = st.selectbox("เลือก Feature เพื่อเปรียบเทียบ", features)
    profile_plot = profile[feature_choice].rename("ค่าเฉลี่ย").reset_index()
    fig = px.bar(profile_plot, x="Cluster", y="ค่าเฉลี่ย", color="Cluster",
                 color_discrete_sequence=["#7c83fd","#f47db5","#56cfa5","#ffc66d","#a78bfa","#60a5fa","#fb7185","#34d399"])
    fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### 💡 แนวทางตีความ")
    st.write("นำค่าเฉลี่ยของแต่ละกลุ่มมาเปรียบเทียบกัน เช่น กลุ่มที่มี `monthly_spend` และ `visits_month` สูง อาจตีความได้ว่าเป็นกลุ่มลูกค้าที่ใช้จ่ายและมาใช้บริการบ่อย ทั้งนี้ต้องอ้างอิงค่าที่แสดงจริง ไม่ควรตั้งชื่อกลุ่มล่วงหน้า")
    st.markdown("#### ตัวอย่างข้อมูลพร้อม Cluster")
    st.dataframe(result_df.head(30), use_container_width=True, hide_index=True)

elif page == "⬇️ ดาวน์โหลดผลลัพธ์":
    st.markdown("### ⬇️ ดาวน์โหลดผลการวิเคราะห์")
    st.write("ดาวน์โหลดข้อมูลที่ผ่านการ Clean พร้อมหมายเลข Cluster เพื่อนำไปใช้ทำรายงานหรือเปิดใน Excel")
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        result_df.to_excel(writer, index=False, sheet_name="Cluster_Result")
        result_df.groupby("Cluster")[features].mean().round(2).to_excel(writer, sheet_name="Cluster_Profile")
        pd.DataFrame({"K": k_candidates, "Inertia": inertias, "Silhouette Score": silhouettes}).round(4).to_excel(writer, index=False, sheet_name="K_Evaluation")
    st.download_button(
        "💜 ดาวน์โหลด Excel ผลลัพธ์",
        data=out.getvalue(),
        file_name="CL02_customer_behavior_clustering_result.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    st.markdown("#### Preview")
    st.dataframe(result_df.head(20), use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown('<p class="small-note">💗 CustomerInsight · CL02 Customer Behavior Clustering · ผลลัพธ์ขึ้นอยู่กับ Dataset ที่อัปโหลดและค่า K ที่เลือก</p>', unsafe_allow_html=True)
