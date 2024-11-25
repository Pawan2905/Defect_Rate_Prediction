# 🔧 **Defect Rate Prediction**

## 📝 **Overview**

Predict manufacturing defect rates 7 days in advance to enhance quality control, minimize production defects, and reduce operational costs. This project uses machine learning and time-series forecasting techniques to identify critical factors impacting defect rates.

---

## 🎯 **Objectives**

- **Proactively predict defect rates** to address issues early.
- **Optimize manufacturing processes** and reduce waste.
- **Enhance product quality** and customer satisfaction.

---

## 📂 **Data**

The project uses datasets representing various steps in the manufacturing process, located in the `Masked_Renamed_Factory_Sample` folder:

| 📄 Dataset Name                    | 📝 Description                                   |
|------------------------------------|-------------------------------------------------|
| `Step1_Mount_Terminals.csv`        | Data from the terminal mounting process.        |
| `Step1_Mount_Terminal_Resin.csv`   | Data from resin application to terminals.       |
| `Step2_Wind_Wire.csv`              | Data from the wire winding process.             |
| `Step3_Peel_Wire.csv`              | Data from the wire peeling process.             |
| `Step4_Check_Alignment.csv`        | Data from the alignment checking process.       |
| `Defect Rates.csv`                 | Historical defect rates.                        |

### Processed Data:

- `combined_processed_data.csv`  
- `defect_rate_preprocessed.csv`

---

## 🛠 **Tools and Libraries**

- **Languages and Frameworks:** Python, PyTorch  
- **Libraries:** NumPy, pandas, seaborn, matplotlib, scikit-learn, pmdarima  

---

## ⚙️ **Methodology**

### 🧹 **1. Data Preprocessing**
- Cleaned and transformed datasets for consistency.
- Combined datasets and engineered time-based features.
- Removed low-variance columns to reduce noise.
- **Output:**  
  - `correlationplot.png`

---

### 📊 **2. Exploratory Data Analysis (EDA)**
- Visualized feature relationships with defect rates.
- Analyzed trends and seasonality.

---

### 🔍 **3. Feature Selection**
- Used Recursive Feature Elimination (RFE) to identify top features affecting defect rates.

---

### 🤖 **4. Model Development**
- **Approaches:**
  - **Time Series Models:** LSTM, Auto ARIMA
  - **Regression Models:** Random Forest, XGBoost, Linear Regression
- **Performance Metrics:**  
  - R², MSE, RMSE, MAPE
- **Hyperparameter Tuning:**  
  Optimized Random Forest and XGBoost models using randomized search.  
- **Outputs:**  
  - `tuned_model_metrics.xlsx`  
  - Saved models in `.pkl` format.

---

### 📈 **5. Explainability**
- Performed SHAP analysis for feature importance.
- **Output:**  
  - `shap_feature_importance.png`

---

## 📦 **Outputs**

| Output Type                     | Description                                 |
|---------------------------------|---------------------------------------------|
| 📉 `correlationplot.png`         | Visual correlation analysis.               |
| 🔍 `shap_feature_importance.png` | Feature importance analysis using SHAP.    |
| 📂 `model_metrics.xlsx`          | Metrics for all trained models.            |
| 📂 `tuned_model_metrics.xlsx`    | Metrics for tuned models.                  |
| 🗂️ Saved Models                  | `.pkl` files for Random Forest and XGBoost.|

---

## 🏗️ **Directory Structure**

```plaintext
Defect Rate Prediction/
│
├── Masked_Renamed_Factory_Sample/
│   ├── Step1_Mount_Terminals.csv
│   ├── Step1_Mount_Terminal_Resin.csv
│   ├── Step2_Wind_Wire.csv
│   ├── Step3_Peel_Wire.csv
│   ├── Step4_Check_Alignment.csv
│   ├── Defect Rates.csv
│
├── preprocessed_data/
│   ├── combined_processed_data.csv
│   └── defect_rate_preprocessed.csv
│
├── saved_model/
│   ├── best_rf_model.pkl
│   ├── best_xg_model.pkl
│   ├── best_xgb_model_tuned.pkl
│   └── best_rf_model_tuned.pkl
│
├── src/
│   ├── config.py
│   ├── main.py
│   ├── data_processing.py
│   ├── feature_selection.py
│   ├── model_training.py
│   ├── hyperparameter_tuning.py
│   ├── train_time_series_model.py
│   └── shap_analysis.py
│
├── Outputs/
│   ├── correlationplot.png
│   ├── shap_feature_importance.png
│   ├── model_metrics.xlsx
│   └── tuned_model_metrics.xlsx
│___|_Document/
|    |__Defect Rate Prediction
└── README.md
```

---

## 📊 **Results and Insights**

- **Key Influential Features:**  
  - `S1X`  
  - `UpperLeft_ProductCenter_IrradiationDistanceX`  
  - `Trg1Terminal4CoreCenterDistance`  

- **Model Performance:**  
  - Tuned Random Forest and XGBoost models achieved the highest accuracy.  
  - LSTM and ARIMA effectively captured trends but require optimization.

---

## 🚀 **Limitations and Enhancements**

### **Limitations**
- Hyperparameter tuning used randomized search, which may miss global optima.
- Feature selection was linear regression-based; tree-based methods may reveal additional insights.

### **Enhancements**
- Explore advanced hyperparameter optimization (e.g., Bayesian optimization).  
- Incorporate cross-validation for robust evaluation.  
- Enhance LSTM architecture (e.g., BiLSTM, stacked LSTMs).

---

## 🏁 **Conclusion**

This pipeline offers a robust framework for defect rate prediction, combining machine learning and explainability. Its modularity and interpretability make it suitable for iterative enhancements and real-world deployment.

--- 
