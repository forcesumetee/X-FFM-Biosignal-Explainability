"""
สคริปต์สร้างข้อมูลตัวอย่างสำหรับ X-FFM
ไม่ต้องใช้ PyTorch - ใช้แค่ NumPy
"""

import numpy as np
import os


def create_synthetic_dataset(
    num_samples: int = 1000,
    signal_length: int = 1000,
    num_classes: int = 2,
    num_concepts: int = 5,
    save_dir: str = './data/synthetic'
):
    """
    สร้างชุดข้อมูลตัวอย่างสำหรับทดสอบ
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 80)
    print(f"กำลังสร้างข้อมูลตัวอย่าง {num_samples} samples...")
    print("=" * 80)
    
    # สร้างสัญญาณ ECG
    print("\n[1/4] สร้างสัญญาณ ECG...")
    ecg_signals = []
    for i in range(num_samples):
        t = np.linspace(0, 4 * np.pi, signal_length)
        if i % 2 == 0:  # Normal
            signal = np.sin(1.0 * t) + 0.3 * np.sin(2.0 * t)
        else:  # Arrhythmia
            signal = np.sin(1.5 * t) + 0.5 * np.sin(3.0 * t) + 0.2 * np.sin(5.0 * t)
        signal += np.random.randn(signal_length) * 0.1
        ecg_signals.append(signal)
    ecg_signals = np.array(ecg_signals, dtype=np.float32)
    print(f"  ✓ สร้าง ECG เสร็จ: {ecg_signals.shape}")
    
    # สร้างสัญญาณ PPG
    print("\n[2/4] สร้างสัญญาณ PPG...")
    ppg_signals = []
    for i in range(num_samples):
        t = np.linspace(0, 4 * np.pi, signal_length)
        if i % 2 == 0:  # Normal
            signal = np.sin(0.8 * t) + 0.2 * np.sin(1.6 * t)
        else:  # Arrhythmia
            signal = np.sin(1.2 * t) + 0.4 * np.sin(2.4 * t)
        signal += np.random.randn(signal_length) * 0.1
        ppg_signals.append(signal)
    ppg_signals = np.array(ppg_signals, dtype=np.float32)
    print(f"  ✓ สร้าง PPG เสร็จ: {ppg_signals.shape}")
    
    # สร้าง Labels
    print("\n[3/4] สร้าง Labels...")
    labels = np.array([i % num_classes for i in range(num_samples)], dtype=np.int64)
    print(f"  ✓ สร้าง Labels เสร็จ: {labels.shape}")
    print(f"    - Class 0 (Normal): {np.sum(labels == 0)} samples")
    print(f"    - Class 1 (Arrhythmia): {np.sum(labels == 1)} samples")
    
    # สร้าง Concept annotations
    print("\n[4/4] สร้าง Concept Annotations...")
    concepts = np.zeros((num_samples, num_concepts), dtype=np.float32)
    for i in range(num_samples):
        if labels[i] == 0:  # Normal
            concepts[i] = np.array([0.85, 0.92, 0.78, 0.88, 0.95]) + np.random.randn(num_concepts) * 0.05
        else:  # Arrhythmia
            concepts[i] = np.array([0.45, 0.52, 0.38, 0.48, 0.55]) + np.random.randn(num_concepts) * 0.05
        concepts[i] = np.clip(concepts[i], 0, 1)
    print(f"  ✓ สร้าง Concepts เสร็จ: {concepts.shape}")
    
    # บันทึกไฟล์
    print(f"\n[บันทึกไฟล์] กำลังบันทึกไปที่ {save_dir}...")
    np.save(os.path.join(save_dir, 'ecg_signals.npy'), ecg_signals)
    np.save(os.path.join(save_dir, 'ppg_signals.npy'), ppg_signals)
    np.save(os.path.join(save_dir, 'labels.npy'), labels)
    np.save(os.path.join(save_dir, 'concepts.npy'), concepts)
    
    print("\n" + "=" * 80)
    print("✓ สร้างข้อมูลตัวอย่างเสร็จสมบูรณ์!")
    print("=" * 80)
    print(f"\nไฟล์ที่สร้าง:")
    print(f"  1. {save_dir}/ecg_signals.npy - สัญญาณ ECG ({ecg_signals.shape})")
    print(f"  2. {save_dir}/ppg_signals.npy - สัญญาณ PPG ({ppg_signals.shape})")
    print(f"  3. {save_dir}/labels.npy - Labels ({labels.shape})")
    print(f"  4. {save_dir}/concepts.npy - Concepts ({concepts.shape})")
    print("\nคุณสามารถใช้ข้อมูลนี้ในการฝึกโมเดลได้เลย!")
    print("=" * 80)


if __name__ == "__main__":
    np.random.seed(42)
    create_synthetic_dataset(
        num_samples=1000,
        signal_length=1000,
        save_dir='/home/ubuntu/xffm_project/data/synthetic'
    )
