# ✈️ Flight Price Prediction System - Streamlit Dashboard

An interactive web application for **Exploratory Data Analysis (EDA)** and **Machine Learning regression modeling** of flight ticket prices, built with **Streamlit**, **Plotly**, and **scikit-learn**.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Dataset Information](#dataset-information)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage & Dashboard Pages](#usage--dashboard-pages)
- [Feature Engineering Framework](#feature-engineering-framework)
- [Machine Learning Models & Metrics](#machine-learning-models--metrics)
- [Requirements](#requirements)
- [License & Acknowledgments](#license--acknowledgments)

---

## 🎯 Overview

This system provides a full end-to-end Machine Learning solution to analyze dynamic flight price variations and predict ticket prices (in Indian Rupees - ₹). It features a multi-page **Streamlit** web application equipped with interactive **Plotly** visualizations, comprehensive diagnostic analytics, and real-time price inference engines.

Key highlights:
- **Interactive EDA**: Filterable Plotly distribution charts, city pricing dynamics, top routes, temporal trends, outlier detection, and correlation heatmaps.
- **Multiple ML Regression Models**: OLS Linear Regression, L2 Regularized Ridge Regression, and L1 Regularized Lasso Regression.
- **Preprocessing Pipeline**: Flexible Label / One-Hot Encoding toggles, `StandardScaler` feature scaling, and robust string parsing.
- **Diagnostic Analytics**: Variance Inflation Factor (VIF) multicollinearity checks, Actual vs. Predicted scatter plots, and Residual Distribution histograms.
- **Real-Time Prediction Engine**: Form interface with dropdowns and sliders for instant flight price estimation.
- **Data Exporting**: Download buttons for filtered raw data and trained model metrics as CSV files.

---

## ✨ Key Features

### 1. 📊 Overview Page
- Total records, feature count, target variable (`Price`), and data shape summary.
- Data quality metrics: duplicate row detection, missing value counts, and complete row statistics.
- Interactive missing values bar chart.
- Descriptive statistics for numeric columns.
- Raw sample data viewer with slider control and CSV download capability.
- **Raw vs. Processed Data Toggle**: Side-by-side comparison of original Excel data vs. feature-engineered dataset.

### 2. 🔍 Exploratory Data Analysis (EDA) Page
- **Price Distribution**: Histogram with box-plot marginal and configurable bin counts.
- **Price Statistics Grid**: Neatly arranged 2-column layout showing Mean, Median, Std Dev, Min, Max, **Skewness (+1.81)**, and **Kurtosis (+13.25)**.
- **Airline Price Analysis**: Horizontal bar chart with selectable aggregation methods (Mean, Median, Min, Max).
- **Source & Destination Analysis**: City-wise pricing breakdowns and flight counts.
- **Top Routes Analysis**: Side-by-side comparison of the Top 10 Most Frequent and Top 10 Most Expensive routes.
- **Temporal Analysis**: Monthly trend lines with flight count overlays and day-of-month price variations.
- **Outlier Detection**: Interquartile Range (IQR) method calculations, lower/upper boundaries, and interactive box plots.
- **Correlation Heatmap**: Full-feature correlation matrix visualization.
- **Sidebar Filters**: Multi-select filters for Airline, Source City, Destination City, and Price Range sliders with CSV export for filtered data.

### 3. 🤖 Model Training & Prediction Page
- **Models Benchmarked**: Linear Regression (OLS), Ridge Regression, Lasso Regression.
- **Configurable Settings**: Test set size slider (10%–40%), Random State seed, Label Encoding vs. One-Hot Encoding toggle, and `StandardScaler` toggle.
- **Interactive Feature Selection**: Checkbox/multi-select list to test specific feature subsets.
- **Performance Benchmarking Table & Charts**: Multi-model metrics table and dual bar plots ($R^2$, Adjusted $R^2$, RMSE, MAE).
- **Multicollinearity Diagnostics**: Variance Inflation Factor (VIF) table with color-coded status badges (`Low < 5`, `Moderate < 10`, `High > 10`).
- **Feature Coefficients**: Colored bar chart displaying positive and negative regression feature weights.
- **Diagnostic Scatter Plots**: Actual vs. Predicted scatter plots (Train & Test) with 45-degree reference line and Residual Distribution histograms.
- **Model Summary Card**: Automated text summary highlighting metrics and predictive quality rating.
- **Exporting**: Download model evaluation metrics as a CSV file.

### 4. 🎯 Predict Price Page
- Interactive input form with dropdowns for Airline, Source, Destination, and Total Stops.
- Input controls for Travel Date (Day, Month), Departure Time (Hour, Minute), Arrival Time (Hour, Minute), and Flight Duration (Hours, Minutes).
- Instant price prediction calculated by the underlying trained regression model.
- Gradient-styled result display card summarizing flight details and estimated ticket cost.

---

## 📊 Dataset Information

**Source:** [Kaggle - Flight Price Prediction Dataset](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction)

The application processes `flight_price.xlsx` located in the project root.

### Raw Data Dictionary

| Column Name | Type | Domain Category | Description & Example Values |
| :--- | :--- | :--- | :--- |
| **Airline** | Categorical (`object`) | Categorical | Carrier name (e.g., *IndiGo, Air India, Jet Airways*) |
| **Date_of_Journey** | String (`object`) | Temporal | Travel date formatted `DD/MM/YYYY` (e.g., `"24/03/2019"`) |
| **Source** | Categorical (`object`) | Geographical | Departure city (e.g., *Banglore, Kolkata, Delhi*) |
| **Destination** | Categorical (`object`) | Geographical | Arrival city (e.g., *New Delhi, Cochin, Banglore*) |
| **Route** | String (`object`) | Flight Path | Layover airport sequence (e.g., `"BLR → DEL"`) |
| **Dep_Time** | String (`object`) | Temporal | Departure time in HH:MM (e.g., `"22:20"`) |
| **Arrival_Time** | String (`object`) | Temporal | Arrival time (e.g., `"01:10 22 Mar"`) |
| **Duration** | String (`object`) | Temporal | Duration string (e.g., `"2h 50m"`, `"19h"`, `"45m"`) |
| **Total_Stops** | Categorical (`object`) | Logistics | Intermediate stops (e.g., `"non-stop"`, `"1 stop"`, `"2 stops"`) |
| **Additional_Info** | Categorical (`object`) | Metadata | Extra notes (e.g., `"No info"`, `"In-flight meal not included"`) |
| **Price** | Numerical (`int64`) | Target | Ticket price in INR ₹ (**Target Variable**) |

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10 or higher
- `pip` package manager

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd "Flight Price Prediction System"
```

### Step 2: Create a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Dashboard
```bash
streamlit run app.py
```

The web dashboard will launch automatically at `http://localhost:8501`.

---

## 📁 Project Structure

```
Flight Price Prediction System/
├── app.py                         # Streamlit Multi-Page Dashboard UI (4 pages)
├── data_utils.py                  # Data loading, feature engineering & encoding utilities
├── eda_utils.py                   # Matplotlib/Seaborn EDA plotting routines
├── ml_model.py                    # FlightPriceModel class (Linear, Ridge, Lasso)
├── Flight Price Prediction.ipynb # Jupyter Notebook for exploratory analysis & modeling
├── PROJECT_REPORT.md              # Comprehensive Technical Project Report & Interview Guide
├── flight_price.xlsx              # Raw Excel Flight Dataset
├── requirements.txt               # Python dependencies manifest
└── README.md                      # Project documentation
```

### File Responsibilities

| File | Primary Responsibility |
| :--- | :--- |
| `app.py` | Main Streamlit interface managing layout, sidebar navigation, user inputs, and Plotly rendering. |
| `data_utils.py` | Handles raw data loading, date/time/duration string parsing, total stops mapping, and feature encoding. |
| `eda_utils.py` | Utility functions for Matplotlib and Seaborn statistical plotting. |
| `ml_model.py` | Object-oriented `FlightPriceModel` class encapsulating train/test splits, model fitting, metric evaluation, and diagnostic plotting routines. |
| `PROJECT_REPORT.md` | Comprehensive 15-section technical report detailing end-to-end architecture, mathematical intuition, empirical benchmarks, and an interview preparation masterclass. |
| `Flight Price Prediction.ipynb` | Standalone Jupyter Notebook containing interactive data exploration, feature extraction, and model benchmarking. |

---

## 🔧 Feature Engineering Framework

Automated feature processing pipeline implemented in `data_utils.py` and `app.py`:

| Raw Feature | Transformed Features | Method & Domain Rationale |
| :--- | :--- | :--- |
| `Date_of_Journey` | `Date`, `Month`, `Year` | String split on `/`; converts date text into integer features. |
| `Dep_Time` | `Departure_hour`, `Departure_min` | String split on `:`; captures time-of-day pricing dynamics. |
| `Arrival_Time` | `Arrival_hour`, `Arrival_min` | Strips date info and splits on `:`; captures arrival time windows. |
| `Duration` | `Duration_hour`, `Duration_min` | Parses `'h'` and `'m'` tokens safely; converts text into numeric minutes/hours. |
| `Total_Stops` | `Total_Stops` | Ordinal mapping: `'non-stop'`: 0, `'1 stop'`: 1, `'2 stops'`: 2, `'3 stops'`: 3, `'4 stops'`: 4. |
| `Route` & `Additional_Info` | *[DROPPED]* | Dropped due to high cardinality redundancy (`Route`) and 80%+ `"No info"` values (`Additional_Info`). |
| Categorical Encodings | Dummy / Integer columns | Supports both Label Encoding and One-Hot Encoding (`pd.get_dummies(drop_first=True)`). |

---

## 🤖 Machine Learning Models & Metrics

### Supported Algorithms
1. **Ordinary Least Squares (OLS) Linear Regression**: Unregularized linear baseline.
2. **Ridge Regression (L2 Regularization)**: Adds squared L2 penalty penalty ($\alpha \sum \beta_j^2$) to control multicollinearity.
3. **Lasso Regression (L1 Regularization)**: Adds absolute L1 penalty penalty ($\alpha \sum |\beta_j|$) for embedded feature selection.

### Empirical Benchmarking Results

| Model Algorithm | Training $R^2$ | Testing $R^2$ | Adjusted $R^2$ | Testing RMSE (₹) | Testing MAE (₹) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear Regression (OLS)** | 0.6870 | **0.6870** | **0.6812** | **₹2,574** | **₹1,787** | Top Baseline |
| **Lasso Regression ($\alpha=0.001$)** | 0.6870 | **0.6870** | **0.6812** | **₹2,574** | **₹1,787** | Sparse Linear |
| **Ridge Regression ($\alpha=1.0$)** | 0.6868 | 0.6868 | 0.6810 | ₹2,575 | ₹1,787 | Regularized Linear |

### Key Metrics Formulations
- **Coefficient of Determination ($R^2$)**: $R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$
- **Adjusted $R^2$**: $R^2_{\text{adj}} = 1 - \left[ (1 - R^2) \frac{n - 1}{n - p - 1} \right]$
- **Root Mean Squared Error (RMSE)**: $RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$
- **Mean Absolute Error (MAE)**: $MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$

---

## 📦 Requirements Manifest

```
streamlit>=1.28.0
numpy
pandas
plotly
matplotlib
seaborn
scikit-learn
openpyxl
statsmodels
scipy
```

| Package | Purpose |
| :--- | :--- |
| `streamlit` | Multi-page web dashboard framework |
| `plotly` | Interactive chart generation engine |
| `pandas` / `numpy` | Data manipulation, array operations, and feature transformation |
| `matplotlib` / `seaborn` | Static statistical plotting helper utilities |
| `scikit-learn` | Regression estimators, encoders, scaling transformers, train/test split, and metrics |
| `openpyxl` | Excel file reader for `.xlsx` dataset |
| `statsmodels` | Variance Inflation Factor (VIF) multicollinearity computation |
| `scipy` | Statistical quantile calculations |

---

## 📄 License & Acknowledgments

- **Dataset**: Kaggle Flight Price Prediction Dataset
- **Tools**: Streamlit, scikit-learn, Plotly, Pandas
- **License**: Open for educational and personal research use.
