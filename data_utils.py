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
    df_processed['Date'] = df_processed['Date_of_Journey'].str.split('/').str[0].astype(int)
    df_processed['Month'] = df_processed['Date_of_Journey'].str.split('/').str[1].astype(int)
    df_processed['Year'] = df_processed['Date_of_Journey'].str.split('/').str[2].astype(int)
    df_processed.drop('Date_of_Journey', axis=1, inplace=True)
    
    # Arrival Time Feature Engineering
    df_processed['Arrival_Time'] = df_processed['Arrival_Time'].apply(lambda x: x.split(' ')[0])
    df_processed['Arrival_hour'] = df_processed['Arrival_Time'].str.split(':').str[0].astype(int)
    df_processed['Arrival_min'] = df_processed['Arrival_Time'].str.split(':').str[1].astype(int)
    df_processed.drop('Arrival_Time', axis=1, inplace=True)
    
    # Departure Time Feature Engineering
    df_processed['Departure_hour'] = df_processed['Departure_Time'].str.split(':').str[0].astype(int)
    df_processed['Departure_min'] = df_processed['Departure_Time'].str.split(':').str[1].astype(int)
    df_processed.drop('Departure_Time', axis=1, inplace=True)
    
    # Duration Feature Engineering
    df_processed['Duration_hours'] = df_processed['Duration'].apply(
        lambda x: int(x.split('h')[0].strip()) if 'h' in x else 0
    )
    df_processed['Duration_mins'] = df_processed['Duration'].apply(
        lambda x: int(x.split('h')[1].split('m')[0].strip()) if 'h' in x and 'm' in x 
        else (int(x.split('m')[0].strip()) if 'm' in x else 0)
    )
    df_processed.drop('Duration', axis=1, inplace=True)
    
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
