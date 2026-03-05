# ✈️ Flight Price Prediction - Streamlit Application

An interactive web application for **Exploratory Data Analysis (EDA)** and **Machine Learning modeling** of flight prices, built with **Streamlit** and **Plotly**. Supports multiple regression models, advanced diagnostics, and interactive price prediction.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset Information](#dataset-information)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Pages](#pages)
- [Feature Engineering](#feature-engineering)
- [Model Details](#model-details)
- [Requirements](#requirements)

---

## 🎯 Overview

This project provides an interactive dashboard to explore flight pricing data and build a predictive regression model. The app is powered by **Streamlit** with **Plotly** for interactive charts and **scikit-learn** for machine learning.

Key highlights:
- Filterable, interactive EDA visualizations (Plotly)
- Multiple ML models: Linear, Ridge, Lasso Regression & Random Forest
- One-Hot / Label Encoding toggle, StandardScaler, log-transform Price
- K-Fold Cross-Validation, Adjusted R², VIF, QQ-Plot diagnostics
- Interactive price prediction page with dropdowns & sliders
- Download buttons for filtered data and model results
- Temporal analysis, outlier detection, violin plots, top routes
- Custom-styled sidebar with branded navigation

---

## ✨ Features

### 1. Data Overview
- Dataset shape, record count, and column types
- Duplicate row count and data quality metrics
- Missing-value bar chart and null-count summary
- Descriptive statistics for numeric columns
- Adjustable raw-data sample viewer with CSV download
- Raw vs feature-engineered data toggle comparison

### 2. Exploratory Data Analysis (EDA)
- Price distribution histogram with box-plot marginal
- Price statistics (mean, median, std dev, min, max, **skewness, kurtosis**)
- Average price by airline (with selectable aggregation: mean / median / min / max)
- **Violin plots** for price distribution by Airline, Source, Destination, or Stops
- Price by source city and destination city
- **Top routes analysis** (most frequent & most expensive Source→Destination)
- **Temporal price analysis** — average price by month and day of month
- Price by additional info category and number of stops
- **Outlier detection** with IQR method, box plots, and metrics
- Price vs duration scatter (colored by airline)
- Full-feature correlation heatmap
- Sidebar filters for airline, source, destination, and price range
- **Download filtered data** as CSV

### 3. Model Training & Prediction
- **Multiple models:** Linear Regression, Ridge, Lasso, Random Forest
- **Encoding toggle:** Label Encoding or One-Hot Encoding
- **StandardScaler** toggle for feature scaling
- **Log-transform Price** toggle
- Interactive feature selection from sidebar
- Configurable test-size and random state
- **Model comparison table & bar charts** (when multiple models selected)
- Train / test performance metrics (R², **Adjusted R²**, RMSE, MAE)
- **K-Fold Cross-Validation** scores
- **VIF table** for multicollinearity analysis
- Feature coefficient bar chart and table (linear models) / feature importance (tree models)
- Actual vs Predicted scatter plots (train & test)
- Residuals distribution histograms (train & test)
- Residuals vs Predicted diagnostic scatter
- **QQ-Plot** for residual normality check
- Auto-generated model summary with quality assessment
- **Download model metrics** as CSV

### 4. Predict Price
- Interactive prediction form with dropdowns and sliders
- Select Airline, Source, Destination, Stops, date/time, and duration
- Instant price prediction using a pre-trained Linear Regression model
- Styled result card with flight details

---

## 📊 Dataset Information

**Source:** [Kaggle - Flight Price Prediction](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction)

The app expects an Excel file named `flight_price.xlsx` in the project root.

### Dataset Features

| Feature | Type | Description |
|---------|------|-------------|
| **Airline** | Categorical | Airline company name |
| **Flight** | Categorical | Flight code |
| **Date_of_Journey** | String | Travel date (DD/MM/YYYY) |
| **Source** | Categorical | Origin city |
| **Destination** | Categorical | Destination city |
| **Route** | Categorical | Flight route details |
| **Dep_Time** | Time | Departure time (HH:MM) |
| **Arrival_Time** | Time | Arrival time (HH:MM) |
| **Duration** | String | Total flight duration (e.g. "2h 30m") |
| **Total_Stops** | Categorical | Number of stops (non-stop, 1 stop, 2 stops, etc.) |
| **Additional_Info** | Categorical | Extra ticket information |
| **Price** | Numerical | **Target variable** — ticket price in ₹ (INR) |

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip

### Step 1: Clone or Download
```bash
git clone <repository-url>
cd "Flight EDA"
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Add the Dataset
Place `flight_price.xlsx` in the project root directory. The dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction).

### Step 5: Run the Application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📁 Project Structure

```
Flight EDA/
├── app.py                 # Main Streamlit application (4 pages)
├── data_utils.py          # Data loading, feature engineering & encoding utilities
├── eda_utils.py           # Matplotlib/Seaborn-based EDA helper functions
├── ml_model.py            # FlightPriceModel class (Linear Regression)
├── EDA And FE Flight Price.ipynb  # Jupyter notebook for standalone EDA & feature engineering
├── requirements.txt       # Python dependencies
├── flight_price.xlsx      # Flight dataset (Excel)
└── README.md              # This file
```

| File | Description |
|------|-------------|
| `app.py` | Streamlit dashboard with Overview, EDA, and Model Training pages. Uses Plotly for all interactive charts. |
| `data_utils.py` | `load_flight_data()`, `feature_engineering()`, `encode_categorical_features()`, `get_data_statistics()`. |
| `eda_utils.py` | Matplotlib/Seaborn plotting helpers (price distribution, airline analysis, correlation heatmap, etc.). |
| `ml_model.py` | `FlightPriceModel` class — data preparation, training, metrics, coefficient analysis, and diagnostic plots. |
| `EDA And FE Flight Price.ipynb` | Jupyter notebook covering the same EDA and feature engineering workflow interactively. |

---

## 📖 Usage

```bash
streamlit run app.py
```

Navigate using the sidebar:

1. **📊 Overview** — Dataset info, shape, null counts, descriptive stats, raw-data preview, raw vs processed toggle.
2. **🔍 Exploratory Data Analysis** — Interactive Plotly charts with sidebar filters, violin plots, temporal analysis, outlier detection, top routes.
3. **🤖 Model Training & Prediction** — Train multiple ML models with encoding/scaling/log-transform toggles, cross-validation, VIF, QQ-plot.
4. **🎯 Predict Price** — Enter flight details and get an instant price prediction.

---

## 🔧 Feature Engineering

Performed automatically in `app.py` (and available as standalone functions in `data_utils.py`):

| Step | Transformation |
|------|---------------|
| Date | Extract `Date`, `Month`, `Year` from `Date_of_Journey`; drop original column |
| Arrival Time | Extract `Arrival_hour`, `Arrival_min`; drop original column |
| Departure Time | Extract `Departure_hour`, `Departure_min` from `Dep_Time`; drop original column |
| Duration | Extract `Duration_hour`, `Duration_min` (handles "2h 30m", "5m", "2h"); drop original column |
| Stops | Map `Total_Stops` text → numeric (non-stop=0 … 4 stops=4); fill NaN with 1 |
| Cleanup | Drop `Route` and `Additional_Info` columns |
| Encoding | Label-encode all remaining categorical columns for modeling |

---

## 🤖 Model Details

### Available Models

| Model | Description |
|-------|-------------|
| **Linear Regression** | Standard OLS regression (always available) |
| **Ridge Regression** | L2 regularization with configurable alpha |
| **Lasso Regression** | L1 regularization with configurable alpha |
| **Random Forest** | Ensemble tree model with configurable n_estimators |

### Configurable Parameters (via sidebar)

| Parameter | Default | Range |
|-----------|---------|-------|
| Test size | 20 % | 10–40 % |
| Random state | 42 | 0–1000 |
| Encoding | Label Encoding | Label / One-Hot |
| Feature Scaling | Off | StandardScaler toggle |
| Log-Transform Price | Off | log1p toggle |
| K-Fold CV | Off | 3–10 folds |
| Features | All | User-selectable subset |

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **R²** | Proportion of variance explained (higher is better) |
| **Adjusted R²** | R² adjusted for number of predictors |
| **RMSE** | Root Mean Squared Error in ₹ (lower is better) |
| **MAE** | Mean Absolute Error in ₹ (lower is better) |
| **CV R²** | Mean R² from K-Fold Cross-Validation |

### Diagnostic Visualizations
- Model comparison table & bar charts (multi-model)
- Feature coefficient bar chart / feature importance (tree models)
- VIF table for multicollinearity analysis
- Actual vs Predicted scatter (train & test, with perfect-fit reference line)
- Residuals distribution histograms (train & test)
- Residuals vs Predicted scatter (test set)
- QQ-Plot for residual normality check
- Auto-generated model summary with quality rating (good / moderate / poor)

---

## 📦 Requirements

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
|---------|---------|
| streamlit | Web application framework |
| plotly | Interactive visualizations in the dashboard |
| pandas / numpy | Data manipulation and numerical computing |
| matplotlib / seaborn | Static plots (utility modules & notebook) |
| scikit-learn | Regression models, encoding, train/test split, cross-validation, metrics |
| openpyxl | Reading the `.xlsx` dataset |
| statsmodels | Variance Inflation Factor (VIF) analysis |
| scipy | QQ-plot and statistical tests |

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `FileNotFoundError: flight_price.xlsx` | Place the Excel file in the project root |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Port already in use | `streamlit run app.py --server.port 8502` |

---

## 📄 License

This project is open for educational and personal use.

## 🙏 Acknowledgments

- Kaggle for the flight price dataset
- Streamlit for the web framework
- scikit-learn for ML tools
- Pandas, NumPy, Matplotlib, Seaborn for data processing and visualization

---

## 📧 Support

For issues or questions:
1. Check the Troubleshooting section
2. Review the Feature Engineering section
3. Verify dataset structure
4. Check requirements installation

---

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ Exploratory Data Analysis (EDA)
- ✅ Feature Engineering
- ✅ Data Preprocessing
- ✅ Machine Learning Model Training
- ✅ Model Evaluation and Metrics
- ✅ Data Visualization
- ✅ Web Application Development
- ✅ Python Programming Best Practices
- ✅ Data Science Workflow

---

**Last Updated**: March 2026  
**Version**: 1.0.0
