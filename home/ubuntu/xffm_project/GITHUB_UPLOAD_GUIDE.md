# คู่มือการอัพโหลดโปรเจค X-FFM ขึ้น GitHub

เอกสารนี้จะแนะนำขั้นตอนการนำโปรเจค X-FFM ของคุณขึ้นไปแชร์บน GitHub อย่างมืออาชีพ

---

### **ขั้นตอนที่ 1: เตรียมโปรเจค**

ก่อนจะอัพโหลด เราได้สร้างไฟล์ที่จำเป็น 3 ไฟล์ไว้ให้คุณแล้ว:

1.  **`README_for_GitHub.md`**: ไฟล์ README ที่สวยงามและครบถ้วนสำหรับหน้าแรกของ Repository (คุณจะต้องเปลี่ยนชื่อเป็น `README.md`)
2.  **`.gitignore`**: ไฟล์ที่บอก Git ว่าไม่ต้องสนใจไฟล์อะไรบ้าง (เช่น Virtual Environment, ไฟล์ชั่วคราว)
3.  **`LICENSE`**: ไฟล์ลิขสิทธิ์แบบ MIT License ซึ่งเป็นที่นิยมและเปิดกว้าง

**สิ่งที่ต้องทำ:**

*   เข้าไปที่โฟลเดอร์ `xffm_project` แล้วเปลี่ยนชื่อไฟล์ `README_for_GitHub.md` เป็น `README.md` (ทับไฟล์เก่าได้เลย)

### **ขั้นตอนที่ 2: สร้าง Repository บน GitHub**

1.  ไปที่เว็บไซต์ [GitHub](https://github.com) และล็อคอิน
2.  คลิกที่เครื่องหมาย `+` ที่มุมขวาบน แล้วเลือก **New repository**
3.  ตั้งค่า Repository ดังนี้:
    *   **Repository name:** `X-FFM-Biosignal-Explainability` (หรือชื่ออื่นที่สื่อความหมายชัดเจน)
    *   **Description:** `Official implementation of "X-FFM: Cross-Modal Interpretability and Explainability for Clinical Decision Support". A framework for building inherently interpretable AI for multimodal biosignal analysis.`
    *   **Public/Private:** เลือกเป็น **Public** เพื่อให้คนอื่นเห็นและนำไปใช้ต่อได้
    *   **Initialize this repository with:** **ไม่ต้องติ๊ก** ทั้ง `Add a README file`, `Add .gitignore`, และ `Choose a license` เพราะเราสร้างไฟล์เหล่านี้ไว้แล้ว
4.  คลิก **Create repository**

### **ขั้นตอนที่ 3: อัพโหลดไฟล์ด้วย Git**

หลังจากสร้าง Repository แล้ว GitHub จะแสดงหน้าคำสั่งสำหรับ Push โค้ดขึ้นไป ให้คุณทำตามคำสั่งในส่วน `…or push an existing repository from the command line`

เปิด Terminal ขึ้นมา แล้วทำตามขั้นตอนต่อไปนี้:

```bash
# 1. เข้าไปที่โฟลเดอร์โปรเจคของคุณ
cd /path/to/xffm_project

# 2. เริ่มต้น Git ในโปรเจค
git init

# 3. เพิ่มไฟล์ทั้งหมดเข้าสู่ Git (ยกเว้นไฟล์ใน .gitignore)
git add .

# 4. Commit ไฟล์ทั้งหมด พร้อมใส่ข้อความอธิบาย
git commit -m "Initial commit: X-FFM project setup and implementation"

# 5. เปลี่ยนชื่อ Branch หลักเป็น main (เป็นมาตรฐานใหม่)
git branch -M main

# 6. เชื่อมต่อโปรเจคของคุณกับ Repository บน GitHub
# (คัดลอก URL จากหน้า GitHub ของคุณมาใส่แทน)
git remote add origin https://github.com/your-username/X-FFM-Biosignal-Explainability.git

# 7. Push โค้ดทั้งหมดขึ้นไปบน GitHub
git push -u origin main
```

**หมายเหตุ:** อย่าลืมเปลี่ยน `your-username` และ `X-FFM-Biosignal-Explainability.git` ให้ตรงกับของคุณ

---

เพียงเท่านี้ โปรเจค X-FFM ของคุณก็จะปรากฏบน GitHub อย่างสวยงาม พร้อมให้คนทั่วโลกได้เห็นและนำไปต่อยอดแล้วครับ!
