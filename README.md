## Results and Conclusion 

### Threshold Used:
- `0.0131` (95th percentile of MSE on normal samples)

### Classification Report:

| Metric      | Normal (0) | Anomaly (1) | Overall |
|-------------|------------|-------------|---------|
| Precision   | 1.00       | 0.96        |         |
| Recall      | 0.95       | 1.00        |         |
| F1-Score    | 0.97       | 0.98        |         |
| Accuracy    |            |             | **0.98** |

### Conclusion

The autoencoder-based anomaly detection model achieves **98% accuracy** on the UNSW-NB15 test set, with:

- Near-perfect **precision (1.00)** for normal traffic (low false positives)
- Perfect **recall (1.00)** for detecting anomalies (no false negatives)
- Overall **F1-score of 0.98**, indicating balanced and effective performance

These results demonstrate that an unsupervised autoencoder trained only on normal traffic is highly effective at identifying anomalous network behavior in the UNSW-NB15 dataset. The reconstruction error threshold successfully separates attack patterns from benign traffic, making this approach suitable for real-world intrusion detection systems (IDS).

---

# KDD Deep Learning Intrusion Detection System

A deep learning-based Intrusion Detection System (IDS) built using the KDD Cup 1999 dataset. The project aims to accurately detect and classify different types of network intrusions using a neural network.

---

## Dataset

- **Source**: [KDD Cup 1999 Dataset](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)
- **Total Samples (post-cleaning)**: ~30,000 test samples
- **Classes**: 34 attack types + normal traffic
- **Challenge**: Highly imbalanced dataset with rare and frequent classes

---

## Preprocessing Steps

1. Dropped unnecessary columns (`Unnamed: 0`, `difficulty`)
2. One-hot encoded categorical features (`protocol_type`, `service`, `flag`)
3. Standardized numerical features using `StandardScaler`
4. Label encoded target variable and converted it to one-hot vectors
5. Filtered out extremely rare classes (≤3 samples)
6. Applied **SMOTE** to balance the training set

---

## Model Architecture

A custom Keras Sequential model with the following layers:

- Dense(256, ReLU) + BatchNormalization + Dropout(0.5)
- Dense(128, ReLU) + BatchNormalization + Dropout(0.4)
- Dense(64, ReLU) + Dropout(0.3)
- Output: Dense(num_classes, Softmax)

**Loss**: `categorical_crossentropy`  
**Optimizer**: `adam`  
**Callback**: `EarlyStopping` with patience of 5 epochs

---

## 📈 Evaluation Results

- **Test Accuracy**: **97%**
- **Weighted F1 Score**: **0.97**
- **Macro F1 Score**: **0.59**
- **Precision & Recall**: High for majority classes, low for rare attacks


