"""
Data Loading and Preprocessing Module
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_flight_data(filepath):
    """
    Load flight price data from Excel file
    
    Parameters:
    -----------
    filepath : str
        Path to the Excel file containing flight data
    
    Returns:
    --------
    pd.DataFrame
        Loaded dataframe
    """
    df = pd.read_excel(filepath)
    return df

def feature_engineering(df):
    """
    Perform feature engineering on flight data
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with raw flight data
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with engineered features
    """
    df_processed = df.copy()
    
    # Date Feature Engineering
    if 'Date_of_Journey' in df_processed.columns:
        df_processed['Date'] = df_processed['Date_of_Journey'].str.split('/').str[0].astype(int)
        df_processed['Month'] = df_processed['Date_of_Journey'].str.split('/').str[1].astype(int)
        df_processed['Year'] = df_processed['Date_of_Journey'].str.split('/').str[2].astype(int)
        df_processed.drop('Date_of_Journey', axis=1, inplace=True)
    
    # Arrival Time Feature Engineering
    if 'Arrival_Time' in df_processed.columns:
        df_processed['Arrival_Time'] = df_processed['Arrival_Time'].apply(lambda x: str(x).split(' ')[0])
        df_processed['Arrival_hour'] = df_processed['Arrival_Time'].str.split(':').str[0].astype(int)
        df_processed['Arrival_min'] = df_processed['Arrival_Time'].str.split(':').str[1].astype(int)
        df_processed.drop('Arrival_Time', axis=1, inplace=True)
    
    # Departure Time Feature Engineering (raw Excel column is Dep_Time)
    dep_col = 'Dep_Time' if 'Dep_Time' in df_processed.columns else ('Departure_Time' if 'Departure_Time' in df_processed.columns else None)
    if dep_col:
        df_processed['Departure_hour'] = df_processed[dep_col].str.split(':').str[0].astype(int)
        df_processed['Departure_min'] = df_processed[dep_col].str.split(':').str[1].astype(int)
        df_processed.drop(dep_col, axis=1, inplace=True)
    
    # Duration Feature Engineering
    if 'Duration' in df_processed.columns:
        def parse_hours(val):
            if pd.isna(val): return 0
            val_str = str(val)
            return int(val_str.split('h')[0].strip()) if 'h' in val_str else 0

        def parse_mins(val):
            if pd.isna(val): return 0
            val_str = str(val)
            if 'h' in val_str and 'm' in val_str:
                return int(val_str.split('h')[1].split('m')[0].strip())
            elif 'm' in val_str:
                return int(val_str.split('m')[0].strip())
            return 0

        df_processed['Duration_hour'] = df_processed['Duration'].apply(parse_hours)
        df_processed['Duration_min'] = df_processed['Duration'].apply(parse_mins)
        df_processed.drop('Duration', axis=1, inplace=True)
    
    # Total Stops Feature Engineering
    if 'Total_Stops' in df_processed.columns:
        df_processed['Total_Stops'] = df_processed['Total_Stops'].map({
            'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4
        })
        df_processed['Total_Stops'] = df_processed['Total_Stops'].fillna(1).astype(int)
    
    # Drop Route and Additional_Info if present
    columns_to_drop = ['Route', 'Additional_Info']
    df_processed.drop(columns=[c for c in columns_to_drop if c in df_processed.columns], inplace=True)
    
    return df_processed

def encode_categorical_features(df):
    """
    Encode categorical features for modeling
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with categorical features
    
    Returns:
    --------
    tuple
        (encoded_dataframe, label_encoder_dict)
    """
    df_encoded = df.copy()
    le_dict = {}
    
    for col in df_encoded.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        le_dict[col] = le
    
    return df_encoded, le_dict

def get_data_statistics(df):
    """
    Get comprehensive statistics about the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    
    Returns:
    --------
    dict
        Dictionary containing dataset statistics
    """
    stats = {
        'total_records': len(df),
        'total_features': len(df.columns),
        'numeric_features': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_features': df.select_dtypes(include=['object']).columns.tolist(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum()
    }
    return stats
