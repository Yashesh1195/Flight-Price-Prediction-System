"""
EDA and Visualization Module
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

def plot_price_distribution(df):
    """Create price distribution plot"""
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(df['Price'], bins=30, kde=True, ax=ax, color='skyblue')
    ax.set_xlabel('Price (₹)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Flight Prices', fontsize=14, fontweight='bold')
    return fig

def plot_airline_analysis(df):
    """Create airline price analysis plot"""
    fig, ax = plt.subplots(figsize=(12, 6))
    airline_price = df.groupby('Airline')['Price'].mean().sort_values(ascending=False)
    airline_price.plot(kind='barh', ax=ax, color='coral')
    ax.set_xlabel('Average Price (₹)', fontsize=12)
    ax.set_ylabel('Airline', fontsize=12)
    ax.set_title('Average Flight Price by Airline', fontsize=14, fontweight='bold')
    return fig

def plot_source_destination_analysis(df):
    """Create source and destination city analysis"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Source City
    source_price = df.groupby('Source_City')['Price'].mean().sort_values(ascending=False)
    source_price.plot(kind='bar', ax=ax1, color='lightblue')
    ax1.set_xlabel('Source City', fontsize=12)
    ax1.set_ylabel('Average Price (₹)', fontsize=12)
    ax1.set_title('Average Price by Source City', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    
    # Destination City
    dest_price = df.groupby('Destination_City')['Price'].mean().sort_values(ascending=False)
    dest_price.plot(kind='bar', ax=ax2, color='lightgreen')
    ax2.set_xlabel('Destination City', fontsize=12)
    ax2.set_ylabel('Average Price (₹)', fontsize=12)
    ax2.set_title('Average Price by Destination City', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

def plot_class_stops_analysis(df):
    """Create seat class and stops analysis"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Class Analysis
    class_price = df.groupby('Class')['Price'].mean()
    class_price.plot(kind='bar', ax=ax1, color=['#1f77b4', '#ff7f0e'])
    ax1.set_xlabel('Seat Class', fontsize=12)
    ax1.set_ylabel('Average Price (₹)', fontsize=12)
    ax1.set_title('Average Price by Seat Class', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=0)
    
    # Stops Analysis
    stops_price = df.groupby('Stops')['Price'].mean().sort_index()
    stops_price.plot(kind='bar', ax=ax2, color='mediumpurple')
    ax2.set_xlabel('Number of Stops', fontsize=12)
    ax2.set_ylabel('Average Price (₹)', fontsize=12)
    ax2.set_title('Average Price by Number of Stops', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    return fig

def plot_correlation_heatmap(df):
    """Create correlation heatmap for all features"""
    df_corr = df.copy()
    
    # Encode categorical variables
    for col in df_corr.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df_corr[col] = le.fit_transform(df_corr[col])
    
    fig, ax = plt.subplots(figsize=(16, 12))
    correlation = df_corr.corr()
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title('Correlation Heatmap of All Features', fontsize=14, fontweight='bold')
    
    return fig

def get_price_statistics(df):
    """Get comprehensive price statistics"""
    stats = {
        'mean': df['Price'].mean(),
        'median': df['Price'].median(),
        'std_dev': df['Price'].std(),
        'min': df['Price'].min(),
        'max': df['Price'].max(),
        'q25': df['Price'].quantile(0.25),
        'q75': df['Price'].quantile(0.75)
    }
    return stats
