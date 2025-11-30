
# X-FFM Project: คู่มือการใช้งานฉบับสมบูรณ์

**Author:** Sumetee Jirapattarasakul

---

## 1. ภาพรวมโครงการ

โครงการ **X-FFM (Cross-Modal Interpretability and Explainability for Clinical Decision Support)** นี้ถูกสร้างขึ้นเพื่อเป็นเฟรมเวิร์กสำหรับการพัฒนางานวิจัยด้าน AI ที่สามารถอธิบายผลได้สำหรับการวิเคราะห์สัญญาณชีวภาพทางการแพทย์

เอกสารนี้จะแนะนำขั้นตอนทั้งหมด ตั้งแต่การติดตั้ง, การเตรียมข้อมูล, การฝึกโมเดล, ไปจนถึงการรันสคริปต์เพื่อดูผลลัพธ์การอธิบายของโมเดล

## 2. การติดตั้ง (Installation)

โปรเจคนี้ต้องการ Python 3.9+ และ PyTorch 1.12+ แนะนำให้สร้าง Virtual Environment เพื่อจัดการ Dependencies

```bash
# 1. Clone หรือแตกไฟล์โปรเจค
cd /path/to/xffm_project

# 2. สร้าง Virtual Environment
python3 -m venv venv

# 3. Activate Virtual Environment
source venv/bin/activate

# 4. ติดตั้ง Dependencies
pip install -r requirements.txt
```

**หมายเหตุ:** หากคุณใช้เครื่องที่ไม่มี GPU, PyTorch จะทำงานในโหมด CPU โดยอัตโนมัติ

## 3. การเตรียมข้อมูล (Data Preparation)

โมเดล X-FFM ถูกออกแบบมาให้ทำงานกับข้อมูลสัญญาณชีวภาพแบบ Multimodal (เช่น ECG, PPG) พร้อมกับข้อมูลประกอบ (Labels และ Concepts)

### 3.1. โครงสร้างข้อมูลที่ต้องการ

คุณต้องจัดเตรียมข้อมูลของคุณให้อยู่ในรูปแบบไฟล์ `.npy` และมีโครงสร้างดังนี้:

```
data/
└── your_dataset_name/
    ├── ecg_signals.npy     # ข้อมูลสัญญาณ ECG (N_samples, signal_length)
    ├── ppg_signals.npy     # ข้อมูลสัญญาณ PPG (N_samples, signal_length)
    ├── labels.npy          # Label ของแต่ละ sample (N_samples,)
    └── concepts.npy        # Concept annotations ของแต่ละ sample (N_samples, N_concepts)
```

*   **`*_signals.npy`**: ไฟล์ข้อมูลสัญญาณดิบ ควรเป็น NumPy array ที่มี 2 มิติ
*   **`labels.npy`**: Label ของคลาส (เช่น 0=Normal, 1=Arrhythmia)
*   **`concepts.npy`**: ค่าของ Clinical Concepts ที่สอดคล้องกับแต่ละ sample (ค่าระหว่าง 0-1) หากไม่มีข้อมูลส่วนนี้ สคริปต์จะสร้างค่าจำลองให้ แต่เพื่อประสิทธิภาพสูงสุด ควรมีข้อมูลส่วนนี้

### 3.2. การใช้ข้อมูลตัวอย่าง (Synthetic Data)

เพื่อความสะดวกในการทดสอบ ผมได้สร้างสคริปต์สำหรับสร้างข้อมูลตัวอย่างขึ้นมาใช้งาน คุณสามารถรันสคริปต์นี้เพื่อสร้างชุดข้อมูลเริ่มต้นได้:

```bash
python data/data_loader.py
```

คำสั่งนี้จะสร้างชุดข้อมูลตัวอย่างไว้ที่ `/home/ubuntu/xffm_project/data/synthetic` ซึ่งพร้อมสำหรับนำไปใช้ในการฝึกโมเดลทันที

## 4. การฝึกโมเดล (Model Training)

หลังจากเตรียมข้อมูลเรียบร้อยแล้ว คุณสามารถเริ่มฝึกโมเดล X-FFM ได้โดยใช้สคริปต์ `train_xffm.py`

### 4.1. การตั้งค่าการฝึก

คุณสามารถปรับแต่งค่าพารามิเตอร์ต่างๆ ได้โดยตรงในฟังก์ชัน `main()` ของไฟล์ `experiments/train_xffm.py`:

```python
CONFIG = {
    'signal_length': 1000,
    'num_classes': 2,
    'concept_names': [...],
    'modalities': ['ecg', 'ppg'],
    'batch_size': 32,
    'num_epochs': 10,
    'learning_rate': 1e-3,
    'lambda_concept': 0.5, # น้ำหนักของ Concept Loss
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}
```

### 4.2. การรันสคริปต์ฝึก

รันคำสั่งต่อไปนี้ใน Terminal:

```bash
python experiments/train_xffm.py
```

สคริปต์จะเริ่มกระบวนการฝึกโมเดล โดยจะแสดงผล Loss และ Accuracy ของทั้ง Training และ Validation set ในแต่ละ Epoch

โมเดลที่มี Validation Accuracy ดีที่สุดจะถูกบันทึกไว้ที่ `checkpoints/best_model.pth` โดยอัตโนมัติ

## 5. การดูผลลัพธ์การอธิบาย (Explainability Demo)

หัวใจสำคัญของโปรเจคนี้คือการสร้างคำอธิบายที่เข้าใจง่าย ผมได้สร้างสคริปต์ `demo_explainability.py` เพื่อสาธิตความสามารถนี้

### 5.1. การทำงานของสคริปต์

สคริปต์นี้จะ:
1.  โหลดโมเดล X-FFM ที่ยังไม่ผ่านการฝึก (เพื่อความรวดเร็วในการสาธิต)
2.  สร้างข้อมูลสัญญาณตัวอย่างขึ้นมา 1 sample
3.  ทำการทำนายผล (Prediction) และแสดงค่า Concept Activations
4.  สร้างคำอธิบายแบบ Counterfactual (หาการเปลี่ยนแปลงที่น้อยที่สุดที่ทำให้ผลการทำนายเปลี่ยนไป)
5.  สร้าง Visualization ทั้งหมดบันทึกไว้ในโฟลเดอร์ `results/`

### 5.2. การรันสคริปต์

รันคำสั่งต่อไปนี้ใน Terminal:

```bash
python experiments/demo_explainability.py
```

หลังจากรันเสร็จ คุณจะพบไฟล์รูปภาพผลลัพธ์ต่างๆ ในโฟลเดอร์ `results/` ได้แก่:

*   `concept_activations.png`: กราฟแสดงค่า Concept ที่โมเดลเรียนรู้
*   `counterfactual_ecg.png`: กราฟเปรียบเทียบสัญญาณดั้งเดิมและสัญญาณ Counterfactual
*   `concept_comparison.png`: กราฟเปรียบเทียบค่า Concept ก่อนและหลังการทำ Counterfactual
*   `explainability_dashboard.png`: Dashboard สรุปผลลัพธ์ทั้งหมดในภาพเดียว

## 6. การนำไปใช้งานต่อ

*   **ฝึกด้วยข้อมูลจริง:** ทำตามขั้นตอนในข้อ 3 เพื่อเตรียมข้อมูลของคุณ และรันการฝึกตามข้อ 4
*   **ทดสอบกับโมเดลที่ฝึกแล้ว:** แก้ไขสคริปต์ `demo_explainability.py` ให้โหลดโมเดลที่ฝึกแล้วจาก `checkpoints/best_model.pth` เพื่อดูคำอธิบายที่แม่นยำยิ่งขึ้น
*   **ปรับเปลี่ยน Concepts:** คุณสามารถทดลองเปลี่ยนชุดของ Clinical Concepts ได้ในไฟล์ `train_xffm.py` และ `demo_explainability.py` เพื่อให้เข้ากับโจทย์วิจัยของคุณ

---

หวังว่าคู่มือนี้จะเป็นประโยชน์ในการเริ่มต้นใช้งานและต่อยอดโครงการ X-FFM ของคุณนะครับ หากมีคำถามเพิ่มเติม สามารถสอบถามได้ตลอดเวลาครับ
