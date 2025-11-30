# X-FFM Quick Start Guide: สาธิตการรันโค้ด Step-by-Step

เอกสารนี้จะสาธิตวิธีการรันโปรเจค X-FFM ตั้งแต่ต้นจนจบอย่างละเอียดครับ

---

### **ขั้นตอนที่ 1: โหลดข้อมูล (Data Loading)**

**เป้าหมาย:** สร้างชุดข้อมูลตัวอย่าง (Synthetic Dataset) เพื่อใช้ในการทดลอง

**วิธีการ:**

1.  เปิด Terminal
2.  เข้าไปที่โฟลเดอร์โปรเจค `xffm_project`
3.  รันสคริปต์ `create_sample_data.py`

**คำสั่ง:**

```bash
cd /home/ubuntu/xffm_project
python3.11 create_sample_data.py
```

**ผลลัพธ์ที่คาดหวัง:**

คุณจะเห็นข้อความแสดงการสร้างไฟล์ข้อมูลต่างๆ และสรุปไฟล์ที่สร้างขึ้น ดังภาพ:

```text
================================================================================
กำลังสร้างข้อมูลตัวอย่าง 1000 samples...
================================================================================
[1/4] สร้างสัญญาณ ECG...
  ✓ สร้าง ECG เสร็จ: (1000, 1000)
[2/4] สร้างสัญญาณ PPG...
  ✓ สร้าง PPG เสร็จ: (1000, 1000)
[3/4] สร้าง Labels...
  ✓ สร้าง Labels เสร็จ: (1000,)
    - Class 0 (Normal): 500 samples
    - Class 1 (Arrhythmia): 500 samples
[4/4] สร้าง Concept Annotations...
  ✓ สร้าง Concepts เสร็จ: (1000, 5)
[บันทึกไฟล์] กำลังบันทึกไปที่ /home/ubuntu/xffm_project/data/synthetic...
================================================================================
✓ สร้างข้อมูลตัวอย่างเสร็จสมบูรณ์!
================================================================================
```

หลังจากขั้นตอนนี้ คุณจะมีข้อมูลพร้อมใช้งานในโฟลเดอร์ `/home/ubuntu/xffm_project/data/synthetic/` ซึ่งประกอบด้วยไฟล์ `.npy` 4 ไฟล์ ได้แก่ `ecg_signals.npy`, `ppg_signals.npy`, `labels.npy`, และ `concepts.npy`

---

### **ขั้นตอนที่ 2: รันโค้ดเพื่อดูผลลัพธ์ (Run the Demo)**

**เป้าหมาย:** รันสคริปต์สาธิตเพื่อดูการทำงานของ XAI และสร้าง Visualization

**วิธีการ:**

1.  อยู่ใน Terminal ที่โฟลเดอร์ `xffm_project`
2.  รันสคริปต์ `demo_visualization_only.py` (เวอร์ชันนี้ไม่ต้องใช้ PyTorch ทำให้รันได้ทันที)

**คำสั่ง:**

```bash
python3.11 experiments/demo_visualization_only.py
```

**ผลลัพธ์ที่คาดหวัง:**

สคริปต์จะทำงานและสร้างไฟล์รูปภาพผลลัพธ์ 4 ภาพ เก็บไว้ในโฟลเดอร์ `results/` คุณจะเห็นข้อความยืนยันการสร้างไฟล์แต่ละไฟล์ ดังภาพ:

```text
================================================================================
X-FFM: Explainability Visualization Demo
================================================================================
[1/5] Generating synthetic biosignals...
  ✓ Generated biosignals
[2/5] Simulating concept activations...
  ✓ Concept activations simulated
[3/5] Creating concept activation visualization...
  ✓ Saved to results/concept_activations.png
[4/5] Creating counterfactual comparison...
  ✓ Saved to results/counterfactual_ecg.png
[5/5] Creating concept comparison...
  ✓ Saved to results/concept_comparison.png
[6/6] Creating explainability dashboard...
  ✓ Saved to results/explainability_dashboard.png
================================================================================
✓ Visualization Demo Completed Successfully!
================================================================================
```

---

### **ขั้นตอนที่ 3: ตรวจสอบผลลัพธ์ (Check the Results)**

**เป้าหมาย:** ดูไฟล์รูปภาพ Visualization ที่สคริปต์ได้สร้างขึ้น

**วิธีการ:**

เข้าไปที่โฟลเดอร์ `/home/ubuntu/xffm_project/results/` คุณจะพบไฟล์รูปภาพ 4 ไฟล์:

1.  **`concept_activations.png`**: แสดงค่า Concept ของข้อมูลตัวอย่าง
2.  **`counterfactual_ecg.png`**: เปรียบเทียบสัญญาณ ECG ก่อนและหลังการทำ Counterfactual
3.  **`concept_comparison.png`**: เปรียบเทียบค่า Concept ก่อนและหลัง
4.  **`explainability_dashboard.png`**: Dashboard สรุปผลทั้งหมดในภาพเดียว

**ตัวอย่างผลลัพธ์ (Dashboard):**

![Dashboard สรุปผล](/home/ubuntu/xffm_project/results/explainability_dashboard.png)

---

### **สรุป**

เพียง 2 คำสั่งง่ายๆ คุณก็สามารถสร้างข้อมูลและเห็นผลลัพธ์การทำงานของโปรเจค X-FFM ได้แล้วครับ:

1.  `python3.11 create_sample_data.py`
2.  `python3.11 experiments/demo_visualization_only.py`

สำหรับขั้นตอนการฝึกโมเดลด้วยข้อมูลจริง (Training) ให้ทำตาม `USER_GUIDE.md` ซึ่งจะต้องมีการติดตั้ง PyTorch ก่อนครับ
