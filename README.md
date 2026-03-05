# HybridDDS: Hybrid Driver Drowsiness Detection System

## Overview

HybridDDS is a **dual-branch spatiotemporal deep learning framework** designed for real-time driver drowsiness detection. The system combines **Swin Transformer visual features** with **physiological-proxy handcrafted signals** such as Eye Aspect Ratio (EAR), PERCLOS, Mouth Aspect Ratio (MAR), and head pose estimation. Temporal patterns of drowsiness are modeled using an **LSTM module**, enabling accurate detection across video sequences.

The model is evaluated across **five public benchmark datasets**, demonstrating strong cross-dataset generalization and robustness under diverse driving conditions.

---

## Key Features

* Dual-branch architecture combining **deep learning and handcrafted physiological indicators**
* **Swin Transformer Tiny** backbone for visual feature extraction
* **Handcrafted feature branch** including:

  * Eye Aspect Ratio (EAR)
  * Mouth Aspect Ratio (MAR)
  * PERCLOS
  * Head pose (pitch, yaw, roll)
* **Temporal modeling with LSTM**
* **Cross-dataset evaluation across five benchmarks**
* Real-time capable (~30 FPS)

---

## System Architecture

HybridDDS consists of five major components:

1. **Face Detection and Landmark Extraction**

   * 68-point facial landmark detection using dlib

2. **Branch A – Deep Visual Features**

   * Eye region cropped and processed by **Swin Transformer Tiny**
   * Output feature vector: **768-dimensional**

3. **Branch B – Physiological-Proxy Features**

   * Extracted from facial landmarks:

     * EAR
     * MAR
     * PERCLOS
     * Head pose (pitch, yaw, roll)
     * Head-down indicator
   * Processed using a small **MLP network**

4. **Feature Fusion**

   * Concatenation of deep and handcrafted features
   * Linear projection to fused representation

5. **Temporal Modeling**

   * LSTM processes sequences of **10 frames**
   * Final classifier outputs **drowsiness probability**

---

## Datasets Used

The model is evaluated on **five widely used driver drowsiness detection datasets**:

| Dataset             | Type                   | Size                     | Key Characteristics       |
| ------------------- | ---------------------- | ------------------------ | ------------------------- |
| **MRL Eye Dataset** | Eye images             | 84,898 images            | Open vs closed eye states |
| **NTHU-DDD**        | Driving frames         | 18,000 images            | Real driving conditions   |
| **YawDD**           | In-car videos          | 349 videos (~19K frames) | Yawning behaviors         |
| **UTA-RLDD**        | Webcam videos          | ~30,000 frames           | Diverse demographics      |
| **3MDAD**           | Multiview driving data | ~25,000 frames           | RGB + depth + IR          |

Total data used: **~250,000 labeled frames across ~290 subjects**

---

## Training Strategy

HybridDDS uses a **progressive transfer learning pipeline**:

### Phase 1 – Eye State Pretraining

* Dataset: MRL Eye
* Task: Open vs Closed eye classification

### Phase 2 – Dual-Branch Fine-Tuning

* Dataset: NTHU-DDD
* Training both branches and fusion layer

### Phase 3 – Video Dataset Processing

* Frame extraction from:

  * YawDD
  * UTA-RLDD
  * 3MDAD

### Phase 4 – Temporal Training

* LSTM trained on sequences from all datasets

### Phase 5 – Cross-Dataset Evaluation

* Leave-one-dataset-out validation

---

## Results

| Model Configuration      | Accuracy   | F1 Score   | AUC        |
| ------------------------ | ---------- | ---------- | ---------- |
| Swin + Head (MRL Eye)    | 99.64%     | 0.9964     | 0.9999     |
| Dual-Branch (Deep + HC)  | 94.33%     | 0.9451     | 0.9873     |
| **HybridDDS Full Model** | **97.30%** | **0.9697** | **0.9959** |

The full HybridDDS architecture achieves **state-of-the-art performance** across multiple datasets.

---

## Model Efficiency

| Component               | Parameters |
| ----------------------- | ---------- |
| Swin Transformer Tiny   | ~27.5M     |
| Handcrafted Feature MLP | ~9K        |
| Fusion Layer            | ~230K      |
| LSTM                    | ~82K       |
| Classifier              | ~2K        |

Total parameters: **27.85M**

Trainable during final phase: **~323K (1.2%)**

---

## Evaluation Protocol

Three evaluation strategies were used:

1. **Combined Evaluation**

   * Train and test across pooled datasets

2. **Per-Dataset Evaluation**

   * Evaluate performance on each dataset individually

3. **Leave-One-Dataset-Out (LODO)**

   * Train on four datasets and test on the fifth

This protocol ensures **robust cross-domain generalization**.

---

## Requirements

Python 3.8+

Main libraries:

* PyTorch
* OpenCV
* dlib
* NumPy
* scikit-learn
* torchvision
* matplotlib

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Clone the repository:

```bash
git clone https://github.com/yourusername/HybridDDS.git
cd HybridDDS
```

Run training:

```bash
python train.py
```

Run inference:

```bash
python detect_drowsiness.py
```

---

## Future Work

* Deployment on **embedded automotive hardware**
* Integration with **IR night-vision systems**
* Multi-sensor fusion (vehicle telemetry + facial analysis)

---

## Citation

If you use this work, please cite:

```
HybridDDS: A Dual-Branch Spatiotemporal Framework for Real-Time Driver Drowsiness Detection Using Swin Transformer and Physiological-Proxy Feature Fusion
```

---

## License

This project is released for **research and academic purposes**.
