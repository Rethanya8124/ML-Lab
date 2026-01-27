# UCS2612---Machine Learning Lab
# EXPERIMENT 01
# Exploratory Data Analysis Pipeline

This experiment implements a complete end-to-end Exploratory data analysis workflow for five diverse datasets. The analysis covers data loading, exploratory data analysis (EDA), preprocessing, feature selection, and dataset splitting, designed to prepare data for machine learning modeling.

## 📂 Datasets Analyzed
The pipeline was applied to the following datasets, each representing a unique data type and challenge:

1.  **Iris Dataset** (`Iris.csv`): Classical multi-class classification of floral species.
2.  **Loan Approval Dataset** (`loan_approval_dataset.csv`): Binary classification for financial risk assessment.
3.  **Diabetes Prediction Dataset** (`diabetes_prediction_dataset.csv`): Medical diagnostic data for disease prediction.
4.  **Email Spam Dataset** (`email.csv`): Natural Language Processing (NLP) task distinguishing Spam from Ham.
5.  **English Characters Dataset** (`english.csv`): Computer Vision metadata analysis for handwritten character recognition.

---

## 🛠️ Methodology

For each dataset, the following five-step protocol was strictly followed:

### 1. Data Loading
* Importing data using `pandas`.
* Initial inspection of schema, data types, and value counts.
* Cleaning column names (e.g., stripping whitespace) to ensure consistency.

### 2. Exploratory Data Analysis (EDA) & Visualization
Visualizations were generated in **grid formats** with `width=1` bar settings and saved as `.eps` files for high-quality publication.
* **Univariate Analysis:**
    * **Histograms:** To visualize data distribution and skewness.
    * **Box Plots:** To identify outliers and interquartile ranges.
    * **Bar Charts:** To analyze class balance in categorical variables.
* **Bivariate Analysis:**
    * **Correlation Heatmaps:** To detect multicollinearity among numerical features.
    * **Scatter Plots:** To observe relationships between features and class separation.

### 3. Data Preprocessing
* **Handling Missing Values:** Dropping null rows to ensure data integrity.
* **Feature Engineering (Special Cases):**
    * *Email Dataset:* Extracted `message_len`, `word_count`, `digit_count`, and `special_char_count` from raw text.
    * *English Dataset:* Extracted `class_id` and `sample_id` from image file paths.
* **Encoding:** Converting categorical variables (e.g., 'Gender', 'Education') into numeric format using `LabelEncoder`.
* **Scaling:**
    * **StandardScaler:** Used for ANOVA and Data Splitting (Centers mean at 0, unit variance).
    * **MinMaxScaler:** Used strictly for Chi-Square testing (requires non-negative values).

### 4. Feature Selection
Statistical tests were conducted to identify the most relevant features:
* **Chi-Square Test:** Measures dependence between non-negative features and the target class.
* **ANOVA (F-test):** Analyzes variance to determine if feature means differ significantly across target classes.

### 5. Data Splitting
The data was stratified and split into three sets to prevent data leakage and ensure robust model evaluation:
* **Training Set:** 70%
* **Validation Set:** 15%
* **Testing Set:** 15%

---

## 📊 Key Findings

| Dataset | Primary Predictors (Top Features) | Key Insight |
| :--- | :--- | :--- |
| **Iris** | `PetalLengthCm`, `PetalWidthCm` | Petal dimensions provide near-perfect separation of species. |
| **Loan Approval** | `cibil_score`, `loan_term` | Credit score is the overwhelming factor in approval decisions; income is secondary. |
| **Diabetes** | `blood_glucose_level`, `HbA1c_level` | Direct blood markers are far more predictive than age or BMI. |
| **Email** | `digit_count`, `message_len` | Spam messages contain significantly more digits (dates, phone numbers) than normal emails. |
| **English** | `class_id` (from filename) | The filename structure perfectly maps to the character label, confirming the dataset organization. |

---

## 💻 Requirements

To reproduce this analysis, the following Python libraries are required:

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
re  # Standard library for Regex (used in English/Email datasets)