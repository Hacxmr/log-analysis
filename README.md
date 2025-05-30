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

For improved generalization or production readiness, consider:

- Using **Variational Autoencoders (VAEs)** for probabilistic modeling  
- Applying **LSTM-based Autoencoders** for sequence-sensitive logs  
- Performing **online anomaly detection** on streaming data

