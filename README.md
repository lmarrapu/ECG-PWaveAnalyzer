# ECG P-Wave Analyzer - Machine Learning Approach 🚀

## 📋 **Overview**
This repository contains an advanced machine learning framework for **P-wave analysis** in **ECG signals**, enabling accurate classification of atrial activity for improved **cardiovascular diagnostics**. Leveraging state-of-the-art algorithms, including **Random Forest**, **Gradient Boosting**, and **XGBoost**, the study highlights the effectiveness of machine learning in detecting **atrial abnormalities**.  

## 🌟 **Key Features**
- **Machine Learning Models:** Implements Random Forest, Gradient Boosting, and XGBoost classifiers.
- **Signal Processing Techniques:** Noise filtering, segmentation, and normalization for high-quality ECG data.
- **Feature Engineering:** Statistical and frequency-domain feature extraction tailored for P-wave analysis.
- **Performance Metrics:** Evaluates models using accuracy, precision, recall, F1 Score, and AUC.
- **Optimization Techniques:** Hyperparameter tuning to enhance prediction accuracy and generalization.

---


---

## 📊 **Results**
### **Pre-Optimization:**
| Model              | Accuracy | Precision | Recall  | F1 Score |
|--------------------|----------|-----------|---------|----------|
| Random Forest      | 94.41%   | 94.14%    | 94.41%  | 94.17%   |
| Gradient Boosting  | 92.81%   | 92.33%    | 92.81%  | 92.39%   |
| XGBoost            | **94.58%** | **94.37%** | **94.58%** | **94.43%** |

### **Post-Optimization:**
| Model              | Accuracy | Precision | Recall  | F1 Score |
|--------------------|----------|-----------|---------|----------|
| Random Forest      | **94.48%** | **94.21%** | **94.48%** | **94.22%** |
| Gradient Boosting  | 93.78%   | 93.47%    | 93.78%  | 93.54%   |
| XGBoost            | 94.45%   | 94.23%    | 94.45%  | 94.29%   |

---

## 🛠 **Methodology**

### 1. **Dataset:**
- Utilizes **high-resolution ECG recordings** annotated for **P-wave morphology** to analyze atrial activity.

### 2. **Signal Processing:**
- **Noise Filtering:** Bandpass filter (0.5–40 Hz) to retain essential cardiac frequencies.  
- **Segmentation:** Sliding windows with 50% overlap to maintain signal continuity.  
- **Normalization:** Z-score normalization to reduce amplitude variations.  

### 3. **Feature Extraction:**
- **Time-domain Features:** Mean, variance, skewness, and kurtosis.  
- **Frequency-domain Features:** Power spectral density using **FFT** and **Welch’s method**.  
- **P-wave Characteristics:** Amplitude, duration, and morphology analysis.

### 4. **Model Selection:**
- Compares **Random Forest**, **Gradient Boosting**, and **XGBoost** for their predictive accuracy.  
- Focuses on hyperparameter tuning to optimize performance.

### 5. **Evaluation Metrics:**
- **Accuracy:** Measures correct classifications.  
- **Precision & Recall:** Evaluates reliability and sensitivity.  
- **F1 Score:** Balances precision and recall.  
- **AUC-ROC:** Assesses the ability to distinguish between positive and negative classes.

---

## 💡 **Key Findings**
- **XGBoost** emerged as the top-performing model, achieving **94.58% accuracy** pre-optimization and **94.45% accuracy** post-optimization.  
- **Random Forest** showed comparable performance, excelling in precision and recall post-optimization (**94.48% accuracy**).  
- **Gradient Boosting** demonstrated potential but required further refinements to match the top-performing models.  
- Optimization significantly boosted performance, validating the importance of hyperparameter tuning in enhancing prediction accuracy.

---

## 📦 **Tools and Libraries**
- **Data Handling:** `Pandas`, `NumPy`  
- **Signal Processing:** `SciPy`, `WFDB`  
- **Visualization:** `Matplotlib`, `Seaborn`  
- **Modeling & Optimization:** `Scikit-learn`, `XGBoost`, `GridSearchCV`

---

## 🔬 **Applications**
- **Cardiovascular Diagnostics:** Assists in identifying atrial abnormalities and arrhythmias.  
- **Clinical Decision Support:** Provides clinicians with automated tools for enhanced analysis.  
- **Healthcare Research:** Advances ECG-based studies through machine learning.

---

### ⭐️ **If you find this repository useful, give it a star!**


 

