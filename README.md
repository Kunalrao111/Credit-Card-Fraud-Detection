#  Credit Card Fraud Detection using Isolation Forest

This project applies **unsupervised anomaly detection** to detect fraudulent transactions in the well-known [Credit Card Fraud dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud).  
We use **Isolation Forest** as the main model, along with **PCA** for visualization and threshold tuning using the **precision–recall curve**.

---

## 📌 Project Workflow
1. **Understand the Dataset**
   - 284,807 transactions, 492 frauds (~0.17%)
   - Features: V1–V28 (PCA components), Amount, Time
2. **Clean the Data**
   - Removed duplicates
   - Scaled `Amount` and `Time` with `RobustScaler`
3. **Exploratory Data Analysis**
   - Distribution of transaction amounts
   - Time-of-day fraud patterns
   - Fraud rate per hour
4. **Anomaly Detection Models**
   - Isolation Forest (baseline & tuned)
   - PCA (2D projection of anomalies)
   - DBSCAN considered but not used (dataset too large for efficient DBSCAN)
5. **Model Evaluation**
   - Metrics: Confusion Matrix, Precision, Recall, F1, ROC AUC
   - Contamination parameter tuning
   - Threshold tuning via precision–recall curve
6. **Present Results**
   - PCA visualization of anomalies vs majority
   - Temporal fraud patterns (fraud rate peaks at night)
   - Business impact analysis (higher recall → fewer missed frauds, at cost of false positives)

---

## 📊 Results

| Setting                        | Precision | Recall | F1   | ROC AUC |
|--------------------------------|-----------|--------|------|---------|
| Baseline (contamination=0.0017)| 0.24      | 0.24   | 0.24 | 0.947   |
| Tuned contamination=0.005      | 0.15      | 0.45   | 0.23 | 0.947   |
| Threshold (recall ≈ 0.50)      | 0.13      | 0.50   | 0.20 | 0.947   |

- **Trade-off**: Increasing recall improves fraud detection but reduces precision (more false positives).
- In fraud detection, recall is often prioritized to minimize missed frauds.

---

## 📈 Visualizations
- 📊 **Class imbalance** (fraud vs normal)
- ⏰ **Fraud rate by hour of day** (spikes in early morning)
- ⚖️ **Precision–Recall curve** (threshold selection)
- 🧩 **PCA plot** (fraudulent transactions scattered outside dense cluster)

*(add plots from your notebook here as images)*

---

## 🛠️ Tech Stack
- **Python** (pandas, numpy, matplotlib, seaborn, scikit-learn)
- **Jupyter Notebook**
- Dataset: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

---

## 🚀 How to Run
1. Clone this repo:
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3.Place creditcard.csv in the project folder (download from Kaggle).

4. Run the notebook:
   ```bash
   jupyter notebook AnomalyDetection.ipynb
