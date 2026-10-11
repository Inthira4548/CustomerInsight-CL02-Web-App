
import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# -------------------------------
# ตั้งค่าหน้าเว็บไซต์
# -------------------------------
st.set_page_config(
    page_title="CustomerInsight CL02",
    page_icon="📊",
    layout="wide"
)

# -------------------------------
# ธีมน้ำเงิน-ขาว
# -------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Prompt', sans-serif;
}

.stApp {
    background: #F5F8FC;
    color: #26384A;
}

section[data-testid="stSidebar"] {
    background: #EAF1F9;
    border-right: 1px solid #D8E3F0;
}

.header-box {
    background: white;
    border: 1px solid #DCE6F1;
    border-left: 6px solid #245B91;
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 20px;
}

.header-box h1 {
    color: #173B63;
    margin: 0;
    font-size: 2rem;
}

.header-box p {
    color: #60758C;
    margin: 8px 0 0 0;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #DCE6F1;
    border-radius: 12px;
    padding: 16px;
}

div[data-testid="stMetricValue"] {
    color: #245B91;
    font-weight: 700;
}

h1, h2, h3 {
    color: #173B63;
}

.stButton > button,
.stDownloadButton > button {
    background: #245B91;
    color: white;
    border-radius: 8px;
    border: 1px solid #245B91;
    font-weight: 500;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: #174675;
    color: white;
    border-color: #174675;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #DCE6F1;
    border-radius: 10px;
    overflow: hidden;
}

hr {
    border-color: #DCE6F1;
}
</style>
""", unsafe_allow_html=True)


BLUE_PALETTE = [
    "#245B91",
    "#4F83B8",
    "#77A5D0",
    "#9DBDDA",
    "#173B63",
    "#6C99C2",
    "#8BAFCF",
    "#B5CCE2"
]

DEFAULT_FILE = (
    Path(__file__).parent
    / "CL02_customer_behavior_clustering.xlsx"
)

PREFERRED_FEATURES = [
    "age",
    "monthly_spend",
    "visits_month",
    "avg_items",
    "discount_usage_percent",
    "online_orders_month"
]


# -------------------------------
# เมนูด้านข้าง
# -------------------------------
st.sidebar.markdown("## 📊 CustomerInsight")
st.sidebar.caption("CL02 | Customer Behavior Clustering")

page = st.sidebar.radio(
    "เมนูหลัก",
    [
        "ภาพรวม",
        "ตรวจสอบข้อมูล",
        "วิเคราะห์ Clustering",
        "เปรียบเทียบกลุ่มลูกค้า",
        "ดาวน์โหลดผลลัพธ์"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Customer Behavior Analysis")


# -------------------------------
# ส่วนหัวเว็บไซต์
# -------------------------------
st.markdown("""
<div class="header-box">
    <h1>CustomerInsight</h1>
    <p>
        CL02 | วิเคราะห์พฤติกรรมลูกค้า
        เพื่อช่วยให้เข้าใจรูปแบบการซื้อและวางแผนธุรกิจ
    </p>
</div>
""", unsafe_allow_html=True)


# -------------------------------
# โหลด Dataset
# -------------------------------
with st.expander("📁 เลือก Dataset", expanded=not DEFAULT_FILE.exists()):

    uploaded_file = st.file_uploader(
        "อัปโหลดไฟล์ Excel (.xlsx)",
        type=["xlsx"]
    )

    raw = None
    source_name = ""

    try:
        if uploaded_file is not None:

            excel = pd.ExcelFile(
                io.BytesIO(uploaded_file.getvalue())
            )

            sheet_name = (
                "Raw_Data"
                if "Raw_Data" in excel.sheet_names
                else excel.sheet_names[0]
            )

            raw = pd.read_excel(
                io.BytesIO(uploaded_file.getvalue()),
                sheet_name=sheet_name
            )

            source_name = uploaded_file.name

        elif DEFAULT_FILE.exists():

            excel = pd.ExcelFile(DEFAULT_FILE)

            sheet_name = (
                "Raw_Data"
                if "Raw_Data" in excel.sheet_names
                else excel.sheet_names[0]
            )

            raw = pd.read_excel(
                DEFAULT_FILE,
                sheet_name=sheet_name
            )

            source_name = DEFAULT_FILE.name

    except Exception as error:
        st.error(f"ไม่สามารถอ่านไฟล์ได้: {error}")


if raw is None:
    st.info("กรุณาอัปโหลดไฟล์ Dataset เพื่อเริ่มวิเคราะห์")
    st.stop()


# -------------------------------
# ตรวจสอบคอลัมน์
# -------------------------------
raw.columns = [str(c).strip() for c in raw.columns]

id_col = next(
    (
        c for c in raw.columns
        if c.lower() in ["customer_id", "customerid", "id"]
    ),
    None
)

features = [
    c for c in PREFERRED_FEATURES
    if c in raw.columns
]

if len(features) < 2:
    features = [
        c for c in raw.select_dtypes(include=np.number).columns
        if c != id_col
    ]

if len(features) < 2:
    st.error(
        "Dataset ต้องมีคอลัมน์ตัวเลขอย่างน้อย 2 คอลัมน์"
    )
    st.write("คอลัมน์ที่พบ:", raw.columns.tolist())
    st.stop()


# -------------------------------
# Data Cleaning
# -------------------------------
rows_before = len(raw)
duplicate_count = int(raw.duplicated().sum())

missing_count = int(
    raw[features].isna().sum().sum()
)

clean = raw.drop_duplicates().copy()

for col in features:

    clean[col] = pd.to_numeric(
        clean[col],
        errors="coerce"
    )

    median_value = clean[col].median()

    clean[col] = clean[col].fillna(median_value)


if len(clean) < 3:
    st.error("ข้อมูลมีน้อยเกินไปสำหรับการแบ่งกลุ่ม")
    st.stop()


# -------------------------------
# ตั้งค่าการวิเคราะห์
# -------------------------------
st.sidebar.markdown("### ⚙️ ตั้งค่าการวิเคราะห์")

max_k = min(8, len(clean) - 1)

k = st.sidebar.slider(
    "จำนวนกลุ่มลูกค้า (K)",
    min_value=2,
    max_value=max_k,
    value=min(3, max_k)
)

selected_features = st.sidebar.multiselect(
    "ตัวแปรที่ใช้วิเคราะห์",
    features,
    default=features
)

if len(selected_features) < 2:
    st.warning("กรุณาเลือกตัวแปรอย่างน้อย 2 รายการ")
    st.stop()


# -------------------------------
# เตรียมข้อมูลสำหรับ K-Means
# -------------------------------
X = clean[selected_features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

labels = model.fit_predict(X_scaled)
clean["Cluster"] = labels

if len(np.unique(labels)) > 1 and len(clean) > k:
    score = silhouette_score(X_scaled, labels)
else:
    score = float("nan")


# -------------------------------
# ประเมินค่า K
# -------------------------------
evaluation_rows = []

for candidate_k in range(2, max_k + 1):

    candidate_model = KMeans(
        n_clusters=candidate_k,
        random_state=42,
        n_init=10
    )

    candidate_labels = candidate_model.fit_predict(
        X_scaled
    )

    evaluation_rows.append({
        "K": candidate_k,
        "Inertia": candidate_model.inertia_,
        "Silhouette Score": silhouette_score(
            X_scaled,
            candidate_labels
        )
    })

evaluation = pd.DataFrame(evaluation_rows)

cluster_counts = (
    clean["Cluster"]
    .value_counts()
    .sort_index()
)

cluster_profile = (
    clean.groupby("Cluster")[selected_features]
    .mean()
    .round(2)
)


# -------------------------------
# หน้า 1: ภาพรวม
# -------------------------------
if page == "ภาพรวม":

    st.subheader("ภาพรวมข้อมูลลูกค้า")

    st.write(
        "สรุปข้อมูลเบื้องต้นและจำนวนลูกค้าในแต่ละกลุ่ม"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "ข้อมูลหลัง Clean",
        f"{len(clean):,} แถว"
    )

    col2.metric(
        "จำนวน Features",
        len(selected_features)
    )

    col3.metric(
        "จำนวน Cluster",
        k
    )

    col4.metric(
        "Silhouette Score",
        f"{score:.3f}" if np.isfinite(score) else "N/A"
    )

    left, right = st.columns([1.2, 1])

    with left:

        st.markdown("### จำนวนลูกค้าแต่ละกลุ่ม")

        count_df = (
            cluster_counts
            .rename_axis("Cluster")
            .reset_index(name="จำนวนลูกค้า")
        )

        fig = px.bar(
            count_df,
            x="Cluster",
            y="จำนวนลูกค้า",
            color="Cluster",
            color_discrete_sequence=BLUE_PALETTE,
            labels={
                "Cluster": "กลุ่มลูกค้า",
                "จำนวนลูกค้า": "จำนวนคน"
            }
        )

        fig.update_layout(
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        st.markdown("### ข้อมูลต้นทาง")

        st.write(f"**ไฟล์:** {source_name}")
        st.write(f"**ชีต:** {sheet_name}")
        st.write(f"**แถวก่อน Clean:** {rows_before:,}")
        st.write(f"**แถวซ้ำ:** {duplicate_count:,}")
        st.write(f"**ค่าว่างใน Features:** {missing_count:,}")

        st.info(
            "ระบบลบแถวที่ซ้ำกันทุกคอลัมน์ "
            "และเติมค่าว่างของตัวแปรตัวเลขด้วย Median "
            "ส่วน Outlier จะตรวจสอบแยกต่างหาก"
        )

    st.markdown("### ตัวอย่างข้อมูล")

    st.dataframe(
        clean.head(10),
        use_container_width=True,
        hide_index=True
    )


# -------------------------------
# หน้า 2: ตรวจสอบข้อมูล
# -------------------------------
elif page == "ตรวจสอบข้อมูล":

    st.subheader("ตรวจสอบและทำความสะอาดข้อมูล")

    col1, col2, col3 = st.columns(3)

    col1.metric("จำนวนแถวเดิม", f"{rows_before:,}")
    col2.metric("แถวซ้ำ", f"{duplicate_count:,}")
    col3.metric("ค่าว่าง", f"{missing_count:,}")

    st.markdown("### ค่าว่างแยกตามคอลัมน์")

    missing_table = (
        raw.isna()
        .sum()
        .rename("จำนวนค่าว่าง")
        .rename_axis("คอลัมน์")
        .reset_index()
    )

    st.dataframe(
        missing_table,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### สถิติหลัง Clean")

    st.dataframe(
        clean[selected_features].describe().T.round(2),
        use_container_width=True
    )

    selected_box = st.selectbox(
        "เลือกตัวแปรเพื่อตรวจสอบ Outlier",
        selected_features
    )

    fig = px.box(
        clean,
        y=selected_box,
        points="outliers",
        title=f"การกระจายของ {selected_box}",
        color_discrete_sequence=["#3979B8"]
    )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "จุดที่อยู่นอกช่วงกล่องเป็นค่าที่ควรตรวจสอบ "
        "แต่ไม่จำเป็นต้องลบทิ้งทุกกรณี"
    )


# -------------------------------
# หน้า 3: Clustering
# -------------------------------
elif page == "วิเคราะห์ Clustering":

    st.subheader("วิเคราะห์และเลือกจำนวนกลุ่ม")

    st.write(
        "ใช้กราฟ Elbow และ Silhouette Score "
        "ช่วยพิจารณาจำนวนกลุ่มที่เหมาะสม"
    )

    left, right = st.columns(2)

    with left:

        fig = px.line(
            evaluation,
            x="K",
            y="Inertia",
            markers=True,
            title="Elbow Method",
            color_discrete_sequence=["#245B91"]
        )

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        fig = px.line(
            evaluation,
            x="K",
            y="Silhouette Score",
            markers=True,
            title="Silhouette Score",
            color_discrete_sequence=["#4F83B8"]
        )

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("### ตารางเปรียบเทียบค่า K")

    st.dataframe(
        evaluation.round(4),
        use_container_width=True,
        hide_index=True
    )

    if not evaluation.empty:

        best_row = evaluation.loc[
            evaluation["Silhouette Score"].idxmax()
        ]

        st.info(
            f"ค่า Silhouette Score สูงสุดในช่วงที่ทดลองคือ "
            f"K = {int(best_row['K'])} "
            f"(คะแนน {best_row['Silhouette Score']:.3f}) "
            "ควรดูกราฟ Elbow และความหมายของแต่ละกลุ่มประกอบ"
        )

    st.markdown("### ภาพรวมกลุ่มลูกค้า")

    pca = PCA(n_components=2, random_state=42)
    coordinates = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame({
        "แกนที่ 1": coordinates[:, 0],
        "แกนที่ 2": coordinates[:, 1],
        "Cluster": labels.astype(str)
    })

    fig = px.scatter(
        pca_df,
        x="แกนที่ 1",
        y="แกนที่ 2",
        color="Cluster",
        opacity=0.75,
        color_discrete_sequence=BLUE_PALETTE
    )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "PCA ย่อข้อมูลหลายตัวแปรให้เหลือ 2 มิติ "
        "เพื่อช่วยมองภาพรวมของกลุ่ม"
    )


# -------------------------------
# หน้า 4: เปรียบเทียบกลุ่ม
# -------------------------------
elif page == "เปรียบเทียบกลุ่มลูกค้า":

    st.subheader("เปรียบเทียบพฤติกรรมลูกค้า")

    st.write(
        "ใช้ค่าเฉลี่ยของแต่ละตัวแปรเพื่ออธิบายความแตกต่าง "
        "ระหว่างกลุ่มลูกค้า"
    )

    st.dataframe(
        cluster_profile,
        use_container_width=True
    )

    selected_feature = st.selectbox(
        "เลือกตัวแปรที่ต้องการเปรียบเทียบ",
        selected_features
    )

    profile_df = (
        cluster_profile[selected_feature]
        .rename("ค่าเฉลี่ย")
        .reset_index()
    )

    fig = px.bar(
        profile_df,
        x="Cluster",
        y="ค่าเฉลี่ย",
        color="Cluster",
        color_discrete_sequence=BLUE_PALETTE
    )

    fig.update_layout(
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("### ตัวอย่างข้อมูลพร้อม Cluster")

    st.dataframe(
        clean.head(30),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "หมายเลข Cluster เป็นเพียงป้ายกำกับกลุ่ม "
        "ไม่ได้บอกว่ากลุ่มใดดีกว่ากลุ่มอื่น"
    )


# -------------------------------
# หน้า 5: ดาวน์โหลดผลลัพธ์
# -------------------------------
elif page == "ดาวน์โหลดผลลัพธ์":

    st.subheader("ดาวน์โหลดผลการวิเคราะห์")

    st.write(
        "ดาวน์โหลดข้อมูลที่ติดป้าย Cluster แล้ว "
        "พร้อมตารางสรุปและผลการทดลองค่า K"
    )

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        clean.to_excel(
            writer,
            index=False,
            sheet_name="Cluster_Result"
        )

        cluster_profile.to_excel(
            writer,
            sheet_name="Cluster_Profile"
        )

        evaluation.round(4).to_excel(
            writer,
            index=False,
            sheet_name="K_Evaluation"
        )

    st.download_button(
        label="📥 ดาวน์โหลดผลลัพธ์ Excel",
        data=output.getvalue(),
        file_name="CL02_customer_behavior_clustering_result.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        )
    )

    st.markdown("### ตัวอย่างผลลัพธ์")

    st.dataframe(
        clean.head(20),
        use_container_width=True,
        hide_index=True
    )


# -------------------------------
# ท้ายเว็บไซต์
# -------------------------------
st.divider()

st.caption(
    "CustomerInsight CL02 | เครื่องมือประกอบการเรียนรู้ "
    "และวิเคราะห์พฤติกรรมลูกค้าด้วย K-Means"
)
