import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Flight Price Prediction",
    # page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 10px;
    }

    /* ── Sidebar Styling ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1b2d 0%, #1a2942 40%, #1e3a5f 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e0e8f0 !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.15) !important;
    }

    /* Sidebar brand area */
    .sidebar-brand {
        text-align: center;
        padding: 0.8rem 0 0.4rem 0;
    }
    .sidebar-brand .logo {
        font-size: 2.6rem;
        line-height: 1;
    }
    .sidebar-brand .title {
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        color: #ffffff !important;
        margin-top: 0.15rem;
    }
    .sidebar-brand .subtitle {
        font-size: 0.78rem;
        color: #8ba3c0 !important;
        margin-top: 0.1rem;
    }

    /* Sidebar section labels */
    .sidebar-section {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #6d8db5 !important;
        margin: 1rem 0 0.4rem 0;
        padding-bottom: 0.2rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }

    /* Radio buttons styling */
    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 0.15rem !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 0.55rem 0.85rem !important;
        transition: all 0.2s ease;
        cursor: pointer;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(255,255,255,0.10);
        border-color: rgba(91,155,213,0.5);
    }
    [data-testid="stSidebar"] [role="radiogroup"] [data-checked="true"] {
        background: rgba(91,155,213,0.22) !important;
        border-color: #5B9BD5 !important;
    }

    /* Multiselect, sliders, inputs in sidebar */
    [data-testid="stSidebar"] .stMultiSelect > div > div {
        background: rgba(255,255,255,0.06) !important;
        border-color: rgba(255,255,255,0.15) !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] .stSlider > div > div > div {
        color: #5B9BD5 !important;
    }
    [data-testid="stSidebar"] input {
        background: rgba(255,255,255,0.06) !important;
        border-color: rgba(255,255,255,0.15) !important;
        border-radius: 8px !important;
    }

    /* Sidebar footer */
    .sidebar-footer {
        text-align: center;
        font-size: 0.75rem;
        color: #5a7a9a !important;
        padding: 1rem 0 0.5rem 0;
        border-top: 1px solid rgba(255,255,255,0.08);
        margin-top: 1.5rem;
    }
    .sidebar-footer a {
        color: #5B9BD5 !important;
        text-decoration: none;
    }

    /* Metric card enhancements */
    [data-testid="stMetric"] {
        background: rgba(91,155,213,0.06);
        border: 1px solid rgba(91,155,213,0.15);
        border-radius: 10px;
        padding: 0.8rem 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">✈️ Flight Price Prediction Analysis</h1>', unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    path = 'flight_price.xlsx'
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(__file__), '..', 'flight_price.xlsx')
    df = pd.read_excel(path)
    return df

@st.cache_data
def preprocess_data(df):
    """Preprocess the data with feature engineering"""
    df_processed = df.copy()
    
    # Feature Engineering - Date
    df_processed['Date'] = df_processed['Date_of_Journey'].str.split('/').str[0].astype(int)
    df_processed['Month'] = df_processed['Date_of_Journey'].str.split('/').str[1].astype(int)
    df_processed['Year'] = df_processed['Date_of_Journey'].str.split('/').str[2].astype(int)
    df_processed.drop('Date_of_Journey', axis=1, inplace=True)
    
    # Feature Engineering - Arrival Time
    df_processed['Arrival_Time'] = df_processed['Arrival_Time'].apply(lambda x: x.split(' ')[0])
    df_processed['Arrival_hour'] = df_processed['Arrival_Time'].str.split(':').str[0].astype(int)
    df_processed['Arrival_min'] = df_processed['Arrival_Time'].str.split(':').str[1].astype(int)
    df_processed.drop('Arrival_Time', axis=1, inplace=True)
    
    # Feature Engineering - Departure Time (Column name is Dep_Time)
    df_processed['Departure_hour'] = df_processed['Dep_Time'].str.split(':').str[0].astype(int)
    df_processed['Departure_min'] = df_processed['Dep_Time'].str.split(':').str[1].astype(int)
    df_processed.drop('Dep_Time', axis=1, inplace=True)
    
    # Feature Engineering - Duration (handles "2h 30m", "5m", "2h" formats)
    def parse_hours(val):
        if pd.isna(val): return 0
        return int(str(val).split('h')[0].strip()) if 'h' in str(val) else 0

    def parse_mins(val):
        if pd.isna(val): return 0
        val = str(val)
        if 'h' in val and 'm' in val:
            return int(val.split('h')[1].split('m')[0].strip())
        elif 'm' in val:
            return int(val.split('m')[0].strip())
        return 0

    df_processed['Duration_hour'] = df_processed['Duration'].apply(parse_hours)
    df_processed['Duration_min'] = df_processed['Duration'].apply(parse_mins)
    df_processed.drop('Duration', axis=1, inplace=True)
    
    # Convert Total_Stops from text to numeric
    df_processed['Total_Stops'] = df_processed['Total_Stops'].map({
        'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4
    })
    df_processed['Total_Stops'] = df_processed['Total_Stops'].fillna(1).astype(int)
    
    # Drop unnecessary columns
    columns_to_drop = ['Route', 'Additional_Info']
    df_processed = df_processed.drop(columns=[col for col in columns_to_drop if col in df_processed.columns])
    
    return df_processed

# Load data
df = load_data()
df_processed = preprocess_data(df)

# Sidebar navigation
st.sidebar.markdown("""
<div class="sidebar-brand">
    <div class="logo">✈️</div>
    <div class="title">Flight Price Predictor</div>
    <div class="subtitle">EDA &amp; Regression Dashboard</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

st.sidebar.markdown('<div class="sidebar-section">Navigation</div>', unsafe_allow_html=True)
page = st.sidebar.radio(
    "Go to",
    ["📊  Overview", "🔍  Exploratory Data Analysis", "🤖  Model Training & Prediction", "🎯  Predict Price"],
    label_visibility="collapsed"
)

# ============= OVERVIEW PAGE =============
if page == "📊  Overview":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<h2 class="section-header">Dataset Information</h2>', unsafe_allow_html=True)
        st.write(f"**Total Records:** {len(df):,}")
        st.write(f"**Total Features:** {len(df.columns)}")
        st.write(f"**Target Variable:** Price")
    
    with col2:
        st.markdown('<h2 class="section-header">Data Shape</h2>', unsafe_allow_html=True)
        st.info(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    with col3:
        st.markdown('<h2 class="section-header">Data Quality</h2>', unsafe_allow_html=True)
        st.write(f"**Duplicate Rows:** {df.duplicated().sum():,}")
        st.write(f"**Total Missing Values:** {df.isnull().sum().sum():,}")
        st.write(f"**Complete Rows:** {df.dropna().shape[0]:,} of {len(df):,}")
    
    st.markdown('<h2 class="section-header">Raw Data Sample</h2>', unsafe_allow_html=True)
    num_rows = st.slider("Number of rows to display", 5, 100, 10, key="overview_rows")
    st.dataframe(df.head(num_rows), use_container_width=True)
    
    # Download raw data
    csv_raw = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Raw Data as CSV", csv_raw, "flight_data_raw.csv", "text/csv", key="dl_raw")
    
    st.markdown('<h2 class="section-header">Data Statistics</h2>', unsafe_allow_html=True)
    st.dataframe(df.describe(), use_container_width=True)
    
    # Missing values bar chart
    st.markdown('<h2 class="section-header">Missing Values Analysis</h2>', unsafe_allow_html=True)
    null_counts = df.isnull().sum()
    null_df = pd.DataFrame({'Column': null_counts.index, 'Missing Count': null_counts.values})
    null_df = null_df[null_df['Missing Count'] > 0].sort_values('Missing Count', ascending=True)
    if len(null_df) > 0:
        fig = px.bar(null_df, y='Column', x='Missing Count', orientation='h',
                     color='Missing Count', color_continuous_scale='Reds',
                     title='Missing Values per Column')
        fig.update_traces(hovertemplate='<b>%{y}</b><br>Missing: %{x:,}<extra></extra>')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No missing values found in the dataset!")
    
    st.markdown('<h2 class="section-header">Data Info</h2>', unsafe_allow_html=True)
    buffer = pd.DataFrame({
        'Column': df.columns,
        'Dtype': df.dtypes.values,
        'Non-Null Count': df.count().values,
        'Null Count': df.isnull().sum().values,
        'Unique Values': [df[col].nunique() for col in df.columns]
    })
    st.dataframe(buffer, use_container_width=True)
    
    # Toggle: Raw vs Processed data comparison
    st.markdown('<h2 class="section-header">Raw vs Processed Data</h2>', unsafe_allow_html=True)
    show_processed = st.toggle("Show Feature-Engineered Data", value=False, key="show_processed")
    if show_processed:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Raw Data")
            st.caption(f"Shape: {df.shape}")
            st.dataframe(df.head(10), use_container_width=True)
        with col2:
            st.subheader("Processed Data")
            st.caption(f"Shape: {df_processed.shape}")
            st.dataframe(df_processed.head(10), use_container_width=True)

# ============= EDA PAGE =============
elif page == "🔍  Exploratory Data Analysis":

    # --- Sidebar Filters for EDA ---
    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="sidebar-section">🎯 Filters</div>', unsafe_allow_html=True)
    
    # Airline filter
    all_airlines = sorted(df['Airline'].unique())
    selected_airlines = st.sidebar.multiselect("Select Airlines", all_airlines, default=all_airlines, key="eda_airlines")
    
    # Source filter
    all_sources = sorted(df['Source'].unique())
    selected_sources = st.sidebar.multiselect("Select Source Cities", all_sources, default=all_sources, key="eda_sources")
    
    # Destination filter
    all_destinations = sorted(df['Destination'].unique())
    selected_destinations = st.sidebar.multiselect("Select Destinations", all_destinations, default=all_destinations, key="eda_dest")
    
    # Price range filter
    price_min, price_max = int(df['Price'].min()), int(df['Price'].max())
    selected_price = st.sidebar.slider("Price Range (₹)", price_min, price_max, (price_min, price_max), key="eda_price")
    
    # Apply filters
    df_filtered = df[
        (df['Airline'].isin(selected_airlines)) &
        (df['Source'].isin(selected_sources)) &
        (df['Destination'].isin(selected_destinations)) &
        (df['Price'] >= selected_price[0]) &
        (df['Price'] <= selected_price[1])
    ]
    
    st.caption(f"Showing {len(df_filtered):,} of {len(df):,} records after filtering")
    
    # Download filtered data
    csv_filt = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Filtered Data as CSV", csv_filt, "flight_data_filtered.csv", "text/csv", key="dl_filtered")
    
    # --- Price Distribution ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h2 class="section-header">Price Distribution</h2>', unsafe_allow_html=True)
        num_bins = st.slider("Number of bins", 10, 100, 30, key="price_bins")
        fig = px.histogram(
            df_filtered, x='Price', nbins=num_bins, marginal='box',
            color_discrete_sequence=['#5B9BD5'],
            labels={'Price': 'Price (₹)', 'count': 'Frequency'},
            title='Distribution of Flight Prices'
        )
        fig.update_traces(hovertemplate='Price Range: ₹%{x}<br>Count: %{y}')
        fig.update_layout(bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<h2 class="section-header">Price Statistics</h2>', unsafe_allow_html=True)
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("Mean Price", f"₹{df_filtered['Price'].mean():,.0f}")
            st.metric("Median Price", f"₹{df_filtered['Price'].median():,.0f}")
            st.metric("Std Dev", f"₹{df_filtered['Price'].std():,.0f}")
            st.metric("Min Price", f"₹{df_filtered['Price'].min():,.0f}")
        with m_col2:
            st.metric("Max Price", f"₹{df_filtered['Price'].max():,.0f}")
            skewness = df_filtered['Price'].skew()
            kurtosis = df_filtered['Price'].kurtosis()
            st.metric("Skewness", f"{skewness:.3f}")
            st.metric("Kurtosis", f"{kurtosis:.3f}")
    
    # --- Airline Analysis ---
    st.markdown('<h2 class="section-header">Price by Airline</h2>', unsafe_allow_html=True)

    agg_method = st.radio("Aggregation", ["Mean", "Median", "Min", "Max"], horizontal=True, key="airline_agg")
    airline_agg = df_filtered.groupby('Airline')['Price'].agg(agg_method.lower()).sort_values(ascending=True).reset_index()
    airline_agg.columns = ['Airline', 'Price']
    
    fig = px.bar(
        airline_agg, y='Airline', x='Price', orientation='h',
        color='Price', color_continuous_scale='Sunset',
        labels={'Price': f'{agg_method} Price (₹)'},
        title=f'{agg_method} Flight Price by Airline'
    )
    fig.update_traces(hovertemplate='<b>%{y}</b><br>' + agg_method + ' Price: ₹%{x:,.0f}<extra></extra>')
    st.plotly_chart(fig, use_container_width=True)
    
    # --- Violin Plots: Airline, Source, Destination ---
    # st.markdown('<h2 class="section-header">Price Distribution by Category (Violin Plots)</h2>', unsafe_allow_html=True)
    # violin_cat = st.radio("Category", ["Airline", "Source", "Destination", "Total_Stops"], horizontal=True, key="violin_cat")
    # fig = px.violin(
    #     df_filtered, x=violin_cat, y='Price', color=violin_cat, box=True, points='outliers',
    #     title=f'Price Distribution by {violin_cat}',
    #     labels={'Price': 'Price (₹)'}
    # )
    # fig.update_layout(showlegend=False, xaxis_tickangle=-45)
    # st.plotly_chart(fig, use_container_width=True)
    
    # --- Source and Destination Cities ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h2 class="section-header">Price by Source City</h2>', unsafe_allow_html=True)
        source_price = df_filtered.groupby('Source')['Price'].agg(['mean', 'median', 'count']).reset_index()
        source_price.columns = ['Source', 'Mean Price', 'Median Price', 'Flight Count']
        fig = px.bar(
            source_price, x='Source', y='Mean Price',
            color='Mean Price', color_continuous_scale='Blues',
            hover_data={'Median Price': ':,.0f', 'Flight Count': ':,'},
            title='Average Price by Source City'
        )
        fig.update_traces(hovertemplate='<b>%{x}</b><br>Mean: ₹%{y:,.0f}<br>Median: ₹%{customdata[0]:,.0f}<br>Flights: %{customdata[1]:,}<extra></extra>')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<h2 class="section-header">Price by Destination City</h2>', unsafe_allow_html=True)
        dest_price = df_filtered.groupby('Destination')['Price'].agg(['mean', 'median', 'count']).reset_index()
        dest_price.columns = ['Destination', 'Mean Price', 'Median Price', 'Flight Count']
        fig = px.bar(
            dest_price, x='Destination', y='Mean Price',
            color='Mean Price', color_continuous_scale='Greens',
            hover_data={'Median Price': ':,.0f', 'Flight Count': ':,'},
            title='Average Price by Destination City'
        )
        fig.update_traces(hovertemplate='<b>%{x}</b><br>Mean: ₹%{y:,.0f}<br>Median: ₹%{customdata[0]:,.0f}<br>Flights: %{customdata[1]:,}<extra></extra>')
        st.plotly_chart(fig, use_container_width=True)
    
    # --- Top Routes Analysis ---
    st.markdown('<h2 class="section-header">Top Routes Analysis</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    df_filtered_routes = df_filtered.copy()
    df_filtered_routes['Route_Name'] = df_filtered_routes['Source'] + ' → ' + df_filtered_routes['Destination']
    route_stats = df_filtered_routes.groupby('Route_Name')['Price'].agg(['mean', 'count']).reset_index()
    route_stats.columns = ['Route', 'Mean Price', 'Flight Count']
    
    with col1:
        top_freq = route_stats.nlargest(10, 'Flight Count')
        fig = px.bar(top_freq, y='Route', x='Flight Count', orientation='h',
                     color='Flight Count', color_continuous_scale='Blues',
                     title='Top 10 Most Frequent Routes')
        fig.update_traces(hovertemplate='<b>%{y}</b><br>Flights: %{x:,}<extra></extra>')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        top_exp = route_stats.nlargest(10, 'Mean Price')
        fig = px.bar(top_exp, y='Route', x='Mean Price', orientation='h',
                     color='Mean Price', color_continuous_scale='Reds',
                     title='Top 10 Most Expensive Routes (Avg Price)')
        fig.update_traces(hovertemplate='<b>%{y}</b><br>Avg Price: ₹%{x:,.0f}<extra></extra>')
        st.plotly_chart(fig, use_container_width=True)
    
    # --- Temporal Analysis ---
    st.markdown('<h2 class="section-header">Temporal Price Analysis</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    df_filt_proc = df_processed.loc[df_filtered.index]
    
    with col1:
        month_price = df_filt_proc.groupby('Month')['Price'].agg(['mean', 'median', 'count']).reset_index()
        month_price.columns = ['Month', 'Mean Price', 'Median Price', 'Count']
        month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
        month_price['Month_Name'] = month_price['Month'].map(month_names)
        fig = px.line(month_price, x='Month_Name', y='Mean Price', markers=True,
                      title='Average Price by Month',
                      labels={'Mean Price': 'Mean Price (₹)', 'Month_Name': 'Month'})
        fig.add_bar(x=month_price['Month_Name'], y=month_price['Count'], name='Flight Count',
                    yaxis='y2', opacity=0.3, marker_color='#5B9BD5')
        fig.update_layout(yaxis2=dict(title='Flight Count', overlaying='y', side='right'),
                          hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        day_price = df_filt_proc.groupby('Date')['Price'].agg(['mean', 'count']).reset_index()
        day_price.columns = ['Day', 'Mean Price', 'Count']
        fig = px.bar(day_price, x='Day', y='Mean Price',
                     color='Mean Price', color_continuous_scale='Viridis',
                     title='Average Price by Day of Month',
                     labels={'Mean Price': 'Mean Price (₹)', 'Day': 'Day of Month'})
        fig.update_traces(hovertemplate='Day %{x}<br>Avg Price: ₹%{y:,.0f}<extra></extra>')
        st.plotly_chart(fig, use_container_width=True)
    
    # --- Additional Info and Stops Analysis ---
    # col1, col2 = st.columns(2)
    # 
    # with col1:
    #     st.markdown('<h2 class="section-header">Price by Additional Info</h2>', unsafe_allow_html=True)
    #     info_price = df_filtered.groupby('Additional_Info')['Price'].agg(['mean', 'count']).reset_index()
    #     info_price.columns = ['Additional_Info', 'Mean Price', 'Count']
    #     info_price = info_price.sort_values('Mean Price', ascending=True)
    #     fig = px.bar(
    #         info_price, y='Additional_Info', x='Mean Price', orientation='h',
    #         color='Mean Price', color_continuous_scale='Teal',
    #         hover_data={'Count': ':,'},
    #         title='Average Price by Additional Info'
    #     )
    #     fig.update_traces(hovertemplate='<b>%{y}</b><br>Avg Price: ₹%{x:,.0f}<br>Count: %{customdata[0]:,}<extra></extra>')
    #     st.plotly_chart(fig, use_container_width=True)
    # 
    # with col2:
    #     st.markdown('<h2 class="section-header">Price by Number of Stops</h2>', unsafe_allow_html=True)
    #     stops_agg = df_filtered.groupby('Total_Stops')['Price'].agg(['mean', 'median', 'count']).reset_index()
    #     stops_agg.columns = ['Total_Stops', 'Mean Price', 'Median Price', 'Count']
    #     fig = px.bar(
    #         stops_agg, x='Total_Stops', y='Mean Price',
    #         color='Mean Price', color_continuous_scale='Purples',
    #         hover_data={'Median Price': ':,.0f', 'Count': ':,'},
    #         title='Average Price by Number of Stops'
    #     )
    #     fig.update_traces(hovertemplate='<b>Stops: %{x}</b><br>Mean: ₹%{y:,.0f}<br>Median: ₹%{customdata[0]:,.0f}<br>Flights: %{customdata[1]:,}<extra></extra>')
    #     st.plotly_chart(fig, use_container_width=True)
    
    # --- Outlier Detection ---
    st.markdown('<h2 class="section-header">Outlier Detection (IQR Method)</h2>', unsafe_allow_html=True)
    outlier_col = st.selectbox("Select column for outlier analysis", ["Price", "Airline", "Source", "Destination"], key="outlier_col")
    
    Q1 = df_filtered['Price'].quantile(0.25)
    Q3 = df_filtered['Price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_filtered[(df_filtered['Price'] < lower_bound) | (df_filtered['Price'] > upper_bound)]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Lower Bound", f"₹{lower_bound:,.0f}")
    col2.metric("Upper Bound", f"₹{upper_bound:,.0f}")
    col3.metric("Outliers Found", f"{len(outliers):,} ({len(outliers)/len(df_filtered)*100:.1f}%)")
    
    if outlier_col == "Price":
        fig = px.box(df_filtered, y='Price', title='Price Outlier Box Plot',
                     labels={'Price': 'Price (₹)'}, color_discrete_sequence=['#5B9BD5'])
    else:
        fig = px.box(df_filtered, x=outlier_col, y='Price',
                     title=f'Price Outliers by {outlier_col}', color=outlier_col,
                     labels={'Price': 'Price (₹)'})
        fig.update_layout(showlegend=False, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # --- Price vs Duration Scatter ---
    # st.markdown('<h2 class="section-header">Price vs Duration</h2>', unsafe_allow_html=True)
    # df_scatter = df_filtered.copy()
    # df_scatter['Duration_total'] = df_processed.loc[df_filtered.index, 'Duration_hour'] + df_processed.loc[df_filtered.index, 'Duration_min'] / 60
    # fig = px.scatter(
    #     df_scatter, x='Duration_total', y='Price',
    #     color='Airline', size='Price', size_max=12, opacity=0.6,
    #     labels={'Duration_total': 'Total Duration (hours)', 'Price': 'Price (₹)'},
    #     title='Flight Price vs Duration (colored by Airline)',
    #     hover_data={'Airline': True, 'Source': True, 'Destination': True}
    # )
    # fig.update_traces(hovertemplate='<b>%{customdata[0]}</b><br>%{customdata[1]} → %{customdata[2]}<br>Duration: %{x:.1f}h<br>Price: ₹%{y:,.0f}<extra></extra>')
    # st.plotly_chart(fig, use_container_width=True)
    
    # --- Correlation Heatmap ---
    st.markdown('<h2 class="section-header">Correlation Analysis</h2>', unsafe_allow_html=True)
    
    df_corr = df_processed.copy()
    for col in df_corr.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df_corr[col] = le.fit_transform(df_corr[col])
    
    correlation = df_corr.corr()
    fig = px.imshow(
        correlation, text_auto='.2f', color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1, aspect='auto',
        title='Correlation Heatmap of All Features'
    )
    fig.update_traces(hovertemplate='%{x} ↔ %{y}<br>Correlation: %{z:.3f}<extra></extra>')
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)

# ============= MODEL PAGE =============
elif page == "🤖  Model Training & Prediction":
    st.markdown('<h2 class="section-header">Machine Learning Models</h2>', unsafe_allow_html=True)
    
    # --- User controls for model ---
    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="sidebar-section">⚙️ Model Settings</div>', unsafe_allow_html=True)
    test_size = st.sidebar.slider("Test Set Size (%)", 10, 40, 20, key="test_size") / 100
    random_state = st.sidebar.number_input("Random State", 0, 1000, 42, key="random_state")
    
    st.sidebar.markdown('<div class="sidebar-section">🔧 Preprocessing</div>', unsafe_allow_html=True)
    encoding_method = st.sidebar.radio("Encoding", ["Label Encoding", "One-Hot Encoding"], key="enc_method")
    use_scaling = st.sidebar.toggle("Feature Scaling (StandardScaler)", value=False, key="use_scaling")
    
    st.sidebar.markdown('<div class="sidebar-section">🧠 Models to Train</div>', unsafe_allow_html=True)
    run_linear = st.sidebar.checkbox("Linear Regression", value=True, key="run_lr")
    run_ridge = st.sidebar.checkbox("Ridge Regression", value=False, key="run_ridge")
    run_lasso = st.sidebar.checkbox("Lasso Regression", value=False, key="run_lasso")
    
    if run_ridge:
        ridge_alpha = st.sidebar.slider("Ridge Alpha", 0.01, 100.0, 1.0, key="ridge_alpha")
    if run_lasso:
        lasso_alpha = st.sidebar.slider("Lasso Alpha", 0.01, 100.0, 1.0, key="lasso_alpha")
    
    # Prepare data for modeling
    df_model = df_processed.copy()
    
    if encoding_method == "Label Encoding":
        le_dict = {}
        for col in df_model.select_dtypes(include='object').columns:
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col])
            le_dict[col] = le
    else:
        cat_cols = df_model.select_dtypes(include='object').columns.tolist()
        df_model = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)
        for col in df_model.columns:
            if df_model[col].dtype == 'bool':
                df_model[col] = df_model[col].astype(int)
    
    X = df_model.drop('Price', axis=1)
    X = X.fillna(X.median(numeric_only=True))
    y = df_model['Price'].copy()
    
    # User can select features to include
    st.sidebar.markdown('<div class="sidebar-section">📋 Feature Selection</div>', unsafe_allow_html=True)
    available_features = X.columns.tolist()
    selected_features = st.sidebar.multiselect(
        "Include Features", available_features, default=available_features, key="model_features"
    )
    
    if not selected_features:
        st.error("Please select at least one feature.")
        st.stop()
    
    X = X[selected_features]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    scaler = None
    if use_scaling:
        scaler = StandardScaler()
        X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test_sc = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    else:
        X_train_sc = X_train
        X_test_sc = X_test
    
    # --- Train all selected models ---
    models_dict = {}
    if run_linear:
        models_dict['Linear Regression'] = LinearRegression()
    if run_ridge:
        models_dict['Ridge Regression'] = Ridge(alpha=ridge_alpha)
    if run_lasso:
        models_dict['Lasso Regression'] = Lasso(alpha=lasso_alpha)
    
    if not models_dict:
        st.error("Please select at least one model.")
        st.stop()
    
    results = {}
    for name, mdl in models_dict.items():
        mdl.fit(X_train_sc, y_train)
        pred_train = mdl.predict(X_train_sc)
        pred_test = mdl.predict(X_test_sc)
        
        y_train_real = y_train
        y_test_real = y_test
        pred_train_real = pred_train
        pred_test_real = pred_test
        
        n_test = len(y_test)
        p = len(selected_features)
        test_r2 = r2_score(y_test_real, pred_test_real)
        adj_r2 = 1 - (1 - test_r2) * (n_test - 1) / (n_test - p - 1) if n_test - p - 1 > 0 else test_r2
        
        results[name] = {
            'model': mdl,
            'pred_train': pred_train_real,
            'pred_test': pred_test_real,
            'y_train_real': y_train_real,
            'y_test_real': y_test_real,
            'train_r2': r2_score(y_train_real, pred_train_real),
            'test_r2': test_r2,
            'adj_r2': adj_r2,
            'test_rmse': np.sqrt(mean_squared_error(y_test_real, pred_test_real)),
            'test_mae': mean_absolute_error(y_test_real, pred_test_real),
        }
    
    # --- Model Comparison Table ---
    if len(results) > 1:
        st.markdown('<h2 class="section-header">Model Comparison</h2>', unsafe_allow_html=True)
        comp_rows = []
        for name, r in results.items():
            row = {
                'Model': name,
                'Train R²': f"{r['train_r2']:.4f}",
                'Test R²': f"{r['test_r2']:.4f}",
                'Adjusted R²': f"{r['adj_r2']:.4f}",
                'RMSE (₹)': f"{r['test_rmse']:,.0f}",
                'MAE (₹)': f"{r['test_mae']:,.0f}",
            }
            comp_rows.append(row)
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)
        
        # Bar chart comparison
        comp_df = pd.DataFrame([{'Model': n, 'Test R²': r['test_r2'], 'RMSE': r['test_rmse']} for n, r in results.items()])
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(comp_df, x='Model', y='Test R²', color='Model', title='Test R² Comparison',
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(comp_df, x='Model', y='RMSE', color='Model', title='Test RMSE Comparison (₹)',
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # --- Per-model detailed analysis ---
    model_names = list(results.keys())
    selected_model_name = st.selectbox("Select model for detailed analysis", model_names, key="detail_model")
    r = results[selected_model_name]
    model = r['model']
    y_pred_train = r['pred_train']
    y_pred_test = r['pred_test']
    y_train_real = r['y_train_real']
    y_test_real = r['y_test_real']
    train_r2 = r['train_r2']
    test_r2 = r['test_r2']
    adj_r2 = r['adj_r2']
    test_rmse = r['test_rmse']
    test_mae = r['test_mae']
    
    st.markdown(f'<h2 class="section-header">{selected_model_name} — Detailed Results</h2>', unsafe_allow_html=True)
    
    # --- Performance Metrics ---
    metric_cols = st.columns(5)
    metric_cols[0].metric("Train R²", f"{train_r2:.4f}")
    metric_cols[1].metric("Test R²", f"{test_r2:.4f}")
    metric_cols[2].metric("Adjusted R²", f"{adj_r2:.4f}")
    metric_cols[3].metric("Test RMSE", f"₹{test_rmse:,.0f}")
    metric_cols[4].metric("Test MAE", f"₹{test_mae:,.0f}")
    
    # --- Feature Coefficients (for linear models) ---
    if hasattr(model, 'coef_'):
        st.markdown('<h2 class="section-header">Feature Coefficients</h2>', unsafe_allow_html=True)
        
        coefficients = pd.DataFrame({
            'Feature': selected_features,
            'Coefficient': model.coef_
        }).sort_values('Coefficient', ascending=True)
        
        fig = px.bar(
            coefficients, y='Feature', x='Coefficient', orientation='h',
            color='Coefficient', color_continuous_scale='RdYlGn', color_continuous_midpoint=0,
            title=f'Feature Coefficients — {selected_model_name}'
        )
        fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=1)
        fig.update_traces(hovertemplate='<b>%{y}</b><br>Coefficient: %{x:,.2f}<extra></extra>')
        fig.update_layout(height=max(400, len(selected_features) * 30))
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(coefficients.sort_values('Coefficient', ascending=False).reset_index(drop=True), use_container_width=True)
    
    # --- VIF Analysis ---
    st.markdown('<h2 class="section-header">Variance Inflation Factor (VIF)</h2>', unsafe_allow_html=True)
    with st.expander("Show VIF Table (checks multicollinearity)", expanded=False):
        try:
            X_vif = X_train_sc.copy()
            X_vif = X_vif.select_dtypes(include=[np.number])
            if len(X_vif.columns) > 1:
                vif_data = pd.DataFrame({
                    'Feature': X_vif.columns,
                    'VIF': [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
                }).sort_values('VIF', ascending=False)
                vif_data['Status'] = vif_data['VIF'].apply(lambda v: '🟢 Low' if v < 5 else ('🟡 Moderate' if v < 10 else '🔴 High'))
                st.dataframe(vif_data, use_container_width=True, hide_index=True)
                st.caption("VIF > 10 indicates high multicollinearity. VIF > 5 is moderate.")
            else:
                st.info("Need at least 2 features to calculate VIF.")
        except Exception as e:
            st.warning(f"Could not compute VIF: {e}")
    
    # --- Actual vs Predicted ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h2 class="section-header">Training: Actual vs Predicted</h2>', unsafe_allow_html=True)
        train_df = pd.DataFrame({'Actual': y_train_real, 'Predicted': y_pred_train})
        fig = px.scatter(
            train_df, x='Actual', y='Predicted', opacity=0.4,
            color_discrete_sequence=['#636EFA'],
            title='Training Set: Actual vs Predicted Prices'
        )
        fig.add_trace(go.Scatter(
            x=[y_train_real.min(), y_train_real.max()], y=[y_train_real.min(), y_train_real.max()],
            mode='lines', line=dict(color='red', dash='dash', width=2), name='Perfect Fit'
        ))
        fig.update_traces(hovertemplate='Actual: ₹%{x:,.0f}<br>Predicted: ₹%{y:,.0f}<extra></extra>', selector=dict(mode='markers'))
        fig.update_layout(xaxis_title='Actual Price (₹)', yaxis_title='Predicted Price (₹)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<h2 class="section-header">Testing: Actual vs Predicted</h2>', unsafe_allow_html=True)
        test_df = pd.DataFrame({'Actual': y_test_real, 'Predicted': y_pred_test})
        fig = px.scatter(
            test_df, x='Actual', y='Predicted', opacity=0.4,
            color_discrete_sequence=['#00CC96'],
            title='Testing Set: Actual vs Predicted Prices'
        )
        fig.add_trace(go.Scatter(
            x=[y_test_real.min(), y_test_real.max()], y=[y_test_real.min(), y_test_real.max()],
            mode='lines', line=dict(color='red', dash='dash', width=2), name='Perfect Fit'
        ))
        fig.update_traces(hovertemplate='Actual: ₹%{x:,.0f}<br>Predicted: ₹%{y:,.0f}<extra></extra>', selector=dict(mode='markers'))
        fig.update_layout(xaxis_title='Actual Price (₹)', yaxis_title='Predicted Price (₹)')
        st.plotly_chart(fig, use_container_width=True)
    
    # --- Residuals Analysis ---
    residuals_train = y_train_real - y_pred_train
    residuals_test = y_test_real - y_pred_test
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h2 class="section-header">Residuals Distribution (Train)</h2>', unsafe_allow_html=True)
        fig = px.histogram(
            x=residuals_train, nbins=40, marginal='box',
            color_discrete_sequence=['#5B9BD5'],
            labels={'x': 'Residuals (₹)', 'count': 'Frequency'},
            title='Training Residuals Distribution'
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red", line_width=2)
        fig.update_traces(hovertemplate='Residual: ₹%{x:,.0f}<br>Count: %{y}<extra></extra>')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<h2 class="section-header">Residuals Distribution (Test)</h2>', unsafe_allow_html=True)
        fig = px.histogram(
            x=residuals_test, nbins=40, marginal='box',
            color_discrete_sequence=['#00CC96'],
            labels={'x': 'Residuals (₹)', 'count': 'Frequency'},
            title='Testing Residuals Distribution'
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red", line_width=2)
        fig.update_traces(hovertemplate='Residual: ₹%{x:,.0f}<br>Count: %{y}<extra></extra>')
        st.plotly_chart(fig, use_container_width=True)
    
    # --- Residuals vs Predicted ---
    st.markdown('<h2 class="section-header">Residuals vs Predicted (Test)</h2>', unsafe_allow_html=True)
    resid_df = pd.DataFrame({'Predicted': y_pred_test, 'Residual': np.array(residuals_test)})
    fig = px.scatter(
        resid_df, x='Predicted', y='Residual', opacity=0.4,
        color_discrete_sequence=['#EF553B'],
        title='Residuals vs Predicted Values (Test Set)'
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
    fig.update_traces(hovertemplate='Predicted: ₹%{x:,.0f}<br>Residual: ₹%{y:,.0f}<extra></extra>')
    fig.update_layout(xaxis_title='Predicted Price (₹)', yaxis_title='Residual (₹)')
    st.plotly_chart(fig, use_container_width=True)
    
    # --- QQ Plot ---
    # st.markdown('<h2 class="section-header">QQ Plot — Residual Normality Check</h2>', unsafe_allow_html=True)
    # residuals_arr = np.array(residuals_test).flatten()
    # sorted_resid = np.sort(residuals_arr)
    # n = len(sorted_resid)
    # theoretical_q = np.array([stats.norm.ppf((i - 0.5) / n) for i in range(1, n + 1)])
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=theoretical_q, y=sorted_resid, mode='markers',
    #                          marker=dict(color='#636EFA', size=4, opacity=0.5), name='Residuals'))
    # # Reference line
    # slope, intercept = np.polyfit(theoretical_q, sorted_resid, 1)
    # fig.add_trace(go.Scatter(x=[theoretical_q.min(), theoretical_q.max()],
    #                          y=[slope*theoretical_q.min()+intercept, slope*theoretical_q.max()+intercept],
    #                          mode='lines', line=dict(color='red', dash='dash', width=2), name='Normal Line'))
    # fig.update_layout(title='QQ Plot of Test Residuals', xaxis_title='Theoretical Quantiles',
    #                   yaxis_title='Sample Quantiles (₹)', height=500)
    # st.plotly_chart(fig, use_container_width=True)
    # st.caption("If residuals are normally distributed, points closely follow the red dashed line.")
    
    # --- Model Summary ---
    st.markdown('<h2 class="section-header">Model Summary</h2>', unsafe_allow_html=True)
    
    summary_parts = [
        f"**Model:** {selected_model_name}",
    ]
    if hasattr(model, 'intercept_'):
        intercept_val = model.intercept_ if np.isscalar(model.intercept_) else model.intercept_
        summary_parts.append(f"**Intercept:** {intercept_val:,.2f}")
    
    summary_parts += [
        f"**Encoding:** {encoding_method}",
        f"**Feature Scaling:** {'StandardScaler' if use_scaling else 'None'}",
        f"**Features Used:** {len(selected_features)} of {len(available_features)}",
        f"**Test Size:** {test_size*100:.0f}% | **Random State:** {random_state}",
        "",
        f"**Training R²:** {train_r2:.4f} (Explains {train_r2*100:.2f}% of variance)",
        f"**Testing R²:** {test_r2:.4f} (Explains {test_r2*100:.2f}% of variance)",
        f"**Adjusted R²:** {adj_r2:.4f}",
        f"**Test RMSE:** ₹{test_rmse:,.0f} | **Test MAE:** ₹{test_mae:,.0f}",
    ]
    
    quality = 'good' if test_r2 > 0.7 else 'moderate' if test_r2 > 0.5 else 'poor'
    summary_parts.append(f"\n**Model Quality:** R² of {test_r2:.4f} on test data → **{quality}** predictive performance.")
    
    st.info("\n\n".join(summary_parts))
    
    # Download model results
    results_csv = pd.DataFrame([{
        'Model': selected_model_name, 'Train_R2': train_r2, 'Test_R2': test_r2,
        'Adjusted_R2': adj_r2, 'Test_RMSE': test_rmse, 'Test_MAE': test_mae
    }]).to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Model Metrics", results_csv, "model_metrics.csv", "text/csv", key="dl_metrics")

# ============= PREDICTION PAGE =============
elif page == "🎯  Predict Price":
    st.markdown('<h2 class="section-header">Predict Flight Price</h2>', unsafe_allow_html=True)
    st.write("Enter flight details below to get an estimated price.")
    
    # Train a model behind the scenes using current data
    @st.cache_resource
    def get_prediction_model():
        df_m = df_processed.copy()
        le_d = {}
        cat_cols_list = df_m.select_dtypes(include='object').columns.tolist()
        for col in cat_cols_list:
            le = LabelEncoder()
            df_m[col] = le.fit_transform(df_m[col])
            le_d[col] = le
        X_all = df_m.drop('Price', axis=1)
        y_all = df_m['Price']
        mdl = LinearRegression()
        mdl.fit(X_all, y_all)
        return mdl, le_d, X_all.columns.tolist()
    
    pred_model, pred_le_dict, pred_features = get_prediction_model()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="sidebar-section">✈️ Flight Details</div>', unsafe_allow_html=True)
    
    # Gather unique categorical values from raw data
    airlines_list = sorted(df['Airline'].unique())
    sources_list = sorted(df['Source'].unique())
    dest_list = sorted(df['Destination'].unique())
    
    col1, col2 = st.columns(2)
    
    with col1:
        p_airline = st.selectbox("Airline", airlines_list, key="p_airline")
        p_source = st.selectbox("Source City", sources_list, key="p_source")
        p_destination = st.selectbox("Destination City", dest_list, key="p_dest")
        p_stops = st.selectbox("Total Stops", [0, 1, 2, 3, 4], key="p_stops")
    
    with col2:
        p_date = st.number_input("Day of Month", 1, 31, 15, key="p_date")
        p_month = st.selectbox("Month", list(range(1, 13)),
                               format_func=lambda m: {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}[m],
                               key="p_month")
        p_dep_hour = st.slider("Departure Hour", 0, 23, 10, key="p_dep_h")
        p_dep_min = st.slider("Departure Minute", 0, 59, 0, key="p_dep_m")
    
    col3, col4 = st.columns(2)
    with col3:
        p_arr_hour = st.slider("Arrival Hour", 0, 23, 14, key="p_arr_h")
        p_arr_min = st.slider("Arrival Minute", 0, 59, 0, key="p_arr_m")
    with col4:
        p_dur_hour = st.number_input("Duration Hours", 0, 50, 2, key="p_dur_h")
        p_dur_min = st.number_input("Duration Minutes", 0, 59, 30, key="p_dur_m")
    
    # Year - use the most common year in the dataset
    most_common_year = int(df_processed['Year'].mode()[0])
    
    # Build feature vector
    input_dict = {
        'Total_Stops': p_stops,
        'Date': p_date,
        'Month': p_month,
        'Year': most_common_year,
        'Arrival_hour': p_arr_hour,
        'Arrival_min': p_arr_min,
        'Departure_hour': p_dep_hour,
        'Departure_min': p_dep_min,
        'Duration_hour': p_dur_hour,
        'Duration_min': p_dur_min,
    }
    
    # Encode categoricals using the fitted label encoders
    cat_input_map = {'Airline': p_airline, 'Source': p_source, 'Destination': p_destination}
    # Only add categorical features that exist in pred_features (i.e., were not dropped)
    for col_name, col_val in cat_input_map.items():
        if col_name in pred_le_dict and col_name in pred_features:
            le = pred_le_dict[col_name]
            if col_val in le.classes_:
                input_dict[col_name] = le.transform([col_val])[0]
            else:
                input_dict[col_name] = 0
    
    # Also handle 'Flight' if it exists
    if 'Flight' in pred_features and 'Flight' in pred_le_dict:
        input_dict['Flight'] = 0  # default
    
    # Build dataframe in correct column order
    input_row = {}
    for feat in pred_features:
        input_row[feat] = input_dict.get(feat, 0)
    
    input_df = pd.DataFrame([input_row])
    
    st.markdown("---")
    if st.button("🚀 Predict Price", type="primary", use_container_width=True):
        predicted_price = pred_model.predict(input_df)[0]
        predicted_price = max(0, predicted_price)
        
        st.markdown(f"""
        <div style="text-align:center; padding:2rem; background: linear-gradient(135deg, #0f1b2d 0%, #1e3a5f 100%);
                    border-radius:15px; margin:1rem 0;">
            <p style="color:#8ba3c0; font-size:1.1rem; margin-bottom:0.5rem;">Estimated Flight Price</p>
            <p style="color:#5B9BD5; font-size:3.5rem; font-weight:800; margin:0;">₹{predicted_price:,.0f}</p>
            <p style="color:#6d8db5; font-size:0.85rem; margin-top:0.5rem;">
                {p_airline} &nbsp;|&nbsp; {p_source} → {p_destination} &nbsp;|&nbsp;
                {p_stops} stop{'s' if p_stops != 1 else ''} &nbsp;|&nbsp; {p_dur_hour}h {p_dur_min}m
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.caption("Prediction is based on a Linear Regression model trained on the full dataset.")