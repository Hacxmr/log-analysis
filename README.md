# 🔐 Intrusion Detection Systems with Deep Learning

This repository contains four deep learning-based projects for detecting cyberattacks in network traffic using various public datasets:

1. **UNSW-NB15** – Deep Learning-Based Anomaly Detection  
2. **KDD Cup 1999** – Multiclass Neural Network Classifier  
3. **CIC-IDS 2017** – Binary Classifier for Attack vs Benign Traffic  
4. **Kyoto 2006+** – Three-Class Fully Connected Neural Network Classifier

---

## 📊 Combined Results Overview

| Dataset         | Model Type         | Accuracy | Precision | Recall | F1-Score | Macro F1 | Notes                                  |
|------------------|--------------------|----------|-----------|--------|----------|----------|----------------------------------------|
| UNSW-NB15         | Autoencoder (Unsupervised) | **0.98**   | 1.00 (Normal)<br>0.96 (Anomaly) | 0.95 (Normal)<br>1.00 (Anomaly) | 0.97 / 0.98 | —        | Based on MSE thresholding (0.0131)     |
| KDD Cup 1999      | Deep Neural Network | **0.97**   | High (Major classes)<br>Low (Rare) | High (Major classes)<br>Low (Rare) | 0.97 (Weighted) | 0.59     | 34 classes, imbalanced                 |
| CIC-IDS 2017      | Deep Binary Classifier | **~0.999** | 1.00 (Benign)<br>0.99 (Attack) | 1.00 (Benign)<br>0.99 (Attack) | 1.00 / 0.99 | 0.995    | Strong performance after SMOTE + MI    |
| **Kyoto 2006+**   | Fully Connected NN (3-Class) | **0.9809** | High for all 3 classes | High for all 3 classes | ~0.98 | ~0.98 | Normal, Known Attack, Unknown Behavior |

---

## 🔍 UNSW-NB15: Autoencoder-Based Anomaly Detection

### 🔧 Threshold Used:
- **0.0131** (95th percentile of MSE on training normal data)

### 📈 Evaluation:

| Metric      | Normal (0) | Anomaly (1) | Overall |
|-------------|------------|-------------|---------|
| Precision   | 1.00       | 0.96        | —       |
| Recall      | 0.95       | 1.00        | —       |
| F1-Score    | 0.97       | 0.98        | —       |
| Accuracy    | —          | —           | **0.98** |

### ✅ Conclusion

- Excellent anomaly detection with no false negatives
- Trained only on normal data
- Ideal for real-world, unsupervised IDS setups

---

## 🧠 KDD Cup 1999: Deep Learning IDS

### 🗃 Dataset Details:

- ~30,000 cleaned test samples
- 34 attacks + normal class
- One-hot encoding + SMOTE for imbalance

### 🧬 Model Architecture:

- Dense(256, ReLU) → BN → Dropout(0.5)  
- Dense(128, ReLU) → BN → Dropout(0.4)  
- Dense(64, ReLU) → Dropout(0.3)  
- Output: Dense(35, Softmax)

### 📈 Evaluation Results:

- **Test Accuracy**: **97%**
- **Weighted F1 Score**: **0.97**
- **Macro F1 Score**: **0.59**
- **Issue**: Poor detection for rare classes

---

## 🛰 CIC-IDS 2017: Deep Learning Classifier

### 📁 Dataset:
[CIC-IDS 2017](https://www.unb.ca/cic/datasets/ids-2017.html)

### ⚙️ Pipeline:

1. Combined CSVs → Parquet
2. Cleaned: removed `inf`, NaNs, duplicates
3. Label binarized: `BENIGN` vs `ATTACK`
4. Feature selection via Mutual Information (Top 30)
5. Quantile scaling
6. Balanced with SMOTE
7. Trained binary classifier with dropout + early stopping

### 🧬 Model Architecture:

```
Input (30 features)
↓
Dense(64, ReLU) → Dropout(0.3)
↓
Dense(32, ReLU) → Dropout(0.2)
↓
Dense(1, Sigmoid)
```

---

## 🏯 Kyoto 2006+: Deep Learning IDS (3-Class Classification)

### 📁 Dataset Overview

- Source: [Kyoto 2006+](https://www.takakura.com/Kyoto_data/)
- Format: 24-column real-world network logs
- Mapped Labels:
  - `-2`: Unknown → 0  
  - `-1`: Known Attack → 1  
  - `1`: Normal → 2  

### 🔄 Preprocessing

- Filtered only valid 24-column `.txt` files
- Dropped missing/invalid rows
- Mapped labels to 3 categories
- Scaled features using `StandardScaler`
- One-hot encoded labels for multiclass classification

### 🧬 Model Architecture

```python
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(3, activation='softmax')
])
```

- Loss: `categorical_crossentropy`
- Optimizer: `adam`
- Accuracy metric used for evaluation

### 📈 Training Results

| Metric              | Value     |
|---------------------|-----------|
| Test Accuracy       | **98.09%** |
| Train Accuracy      | 98.03% |
| Validation Accuracy | 98.21% |
| Best Val Loss       | 0.0503 |

### ✅ Conclusion

- Excellent accuracy across all three classes
- Stable convergence and no signs of overfitting
- Simple architecture effective for real-world intrusion detection using Kyoto logs

---

## 📦 Future Work & Deployment

Planned extensions:

- Export models using `model.save()` for deployment
- Convert models to **ONNX** or **TF Lite** for edge deployment
- Add **confusion matrices**, **ROC curves**, and **visual analytics**
- Integrate into a real-time traffic pipeline using **Kafka**, **Wireshark**, or **Zeek**

---
## 📊 Intrusion detection systems with Supervised Learning
This repository implements and evaluates supervised learning methods for network-based intrusion detection using four benchmark datasets.

--- 
## 1. CIC IDS‑2017
Model Used: Multiclass classifier (RandomForest/XGBoost).

## Results:

Excellent performance: overall accuracy ≈ 99.83%.

High precision and recall across all attack classes, though some rare classes (like XSS) had ill-defined precision and were set to zero (warning from sklearn.metrics) due to zero predictions.

Balanced classification report with macro avg ≈ 0.83 and weighted avg ≈ 1.00.

--- 
## 2. NSL‑KDD Dataset Analysis
🔍 Preprocessing:
Low-variance features were removed using VarianceThreshold(threshold=0.01).

 Models Used:
1. XGBoost Classifier
Accuracy: 0.89

Macro Avg F1-score: 0.68

Weighted Avg F1-score: 0.89

2. Random Forest Classifier
Accuracy: 0.8932
Macro Avg F1-score: Similar (as seen in the range of 0.39 to 0.98 across classes).

---
## 3. UNSW‑NB15
Approach 1: Direct classification (RandomForest/XGBoost).

Accuracy only ≈ 54%.

Class ‘0’ (benign) yielded precision ≈ 0.49, recall ≈ 0.66; class ‘1’ (attack) had precision ≈ 0.62, recall ≈ 0.45.

Approach 2: SMOTE oversampling → Logistic Regression.

SMOTE balanced both classes to ~119K samples each.

Balanced performance: accuracy ≈ 64%, class ‘0’: precision 0.58/re 0.76, class ‘1’: precision 0.73/recall 0.55.

Shows improvement for attack detection using oversampling and logistic regression.

---
## 4. Kyoto2006+
Model Used: RandomForest.

Results:

Perfect performance on this dataset: accuracy = 1.00.

Single-label problem (likely only benign or only attack), resulting in trivial 100% classification.

---


## 🧾 License

MIT License

---

## 🙋‍♂️ Author

Feel free to raise issues, suggest improvements, or contribute!
