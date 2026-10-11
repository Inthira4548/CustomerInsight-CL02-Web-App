# CustomerInsight — CL02 Customer Behavior Clustering

เว็บแดชบอร์ดภาษาไทยสำหรับสำรวจและจัดกลุ่มพฤติกรรมลูกค้าด้วย K-Means พร้อมกราฟ Elbow, Silhouette, PCA และ Cluster Profile

## รันบนเครื่อง
1. ติดตั้ง Python 3.10 ขึ้นไป
2. เปิด Terminal ในโฟลเดอร์นี้
3. ติดตั้งไลบรารี:
   ```bash
   pip install -r requirements.txt
   ```
4. รันเว็บ:
   ```bash
   streamlit run app.py
   ```
5. เว็บจะเปิดในเบราว์เซอร์โดยปกติที่ `http://localhost:8501`

## Deploy ขึ้นออนไลน์ด้วย Streamlit Community Cloud
1. สร้าง Repository ใหม่บน GitHub
2. อัปโหลด `app.py`, `requirements.txt` และไฟล์ Excel `CL02_customer_behavior_clustering.xlsx` (ถ้าต้องการให้เว็บมีข้อมูลเริ่มต้น)
3. เข้า https://share.streamlit.io/ แล้วเชื่อมบัญชี GitHub
4. เลือก Repository และกำหนด Main file path เป็น `app.py`
5. กด Deploy
6. เมื่อ deploy สำเร็จ จะได้ URL สำหรับแชร์ให้อาจารย์เปิดดู

## ข้อควรทราบ
- เว็บจะใช้ไฟล์ Excel ที่แนบมาเป็นข้อมูลเริ่มต้น หากไม่มีไฟล์ดังกล่าว ผู้ใช้สามารถอัปโหลด `.xlsx` ผ่านหน้าเว็บได้
- ถ้าไฟล์มีชีตชื่อ `Raw_Data` ระบบจะอ่านชีตนั้นก่อน มิฉะนั้นจะอ่านชีตแรก
- ค่า K เริ่มต้นเป็น 3 และปรับได้ใน Sidebar ควรเลือก K จากกราฟ Elbow และ Silhouette ของข้อมูลจริงก่อนนำเสนอ
- ระบบลบแถวซ้ำทั้งแถว และเติม Missing ของ Feature เชิงตัวเลขด้วย Median เพื่อให้เริ่มทดลองได้ แต่ควรตรวจสอบวิธีจัดการ Missing/Outlier ให้สอดคล้องกับข้อกำหนดของรายวิชาก่อนส่งงาน
- อย่าอัปโหลดข้อมูลลูกค้าที่เป็นข้อมูลส่วนบุคคลหรือเป็นความลับขึ้น Repository สาธารณะ
