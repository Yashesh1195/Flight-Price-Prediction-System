"""
Machine Learning Model Module
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class FlightPriceModel:
    """Multiple Linear Regression Model for Flight Price Prediction"""
    
    def __init__(self, test_size=0.2, random_state=42):
        """
        Initialize the model
        
        Parameters:
        -----------
        test_size : float
            Proportion of data to use for testing (default: 0.2)
        random_state : int
            Random seed for reproducibility (default: 42)
        """
        self.test_size = test_size
        self.random_state = random_state
        self.model = LinearRegression()
        self.le_dict = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred_train = None
        self.y_pred_test = None
        self.feature_names = None
    
    def prepare_data(self, df):
        """
        Prepare data for modeling
        
        Parameters:
        -----------
        df : pd.DataFrame
            Processed dataframe with all features
        
        Returns:
        --------
        tuple
            (X, y) - Features and target variable
        """
        df_model = df.copy()
        
        # Encode categorical variables
        for col in df_model.select_dtypes(include='object').columns:
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col])
            self.le_dict[col] = le
        
        # Separate features and target
        X = df_model.drop('Price', axis=1)
        y = df_model['Price']
        
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def train(self, X, y):
        """
        Train the linear regression model
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target variable
        """
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Train model
        self.model.fit(self.X_train, self.y_train)
        
        # Make predictions
        self.y_pred_train = self.model.predict(self.X_train)
        self.y_pred_test = self.model.predict(self.X_test)
    
    def get_metrics(self):
        """
        Get model performance metrics
        
        Returns:
        --------
        dict
            Dictionary containing all performance metrics
        """
        metrics = {
            'train_r2': r2_score(self.y_train, self.y_pred_train),
            'test_r2': r2_score(self.y_test, self.y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(self.y_train, self.y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(self.y_test, self.y_pred_test)),
            'train_mae': mean_absolute_error(self.y_train, self.y_pred_train),
            'test_mae': mean_absolute_error(self.y_test, self.y_pred_test)
        }
        return metrics
    
    def get_coefficients(self):
        """
        Get feature coefficients
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with feature coefficients
        """
        coefficients = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': self.model.coef_
        }).sort_values('Coefficient', ascending=False)
        
        return coefficients
    
    def plot_coefficients(self):
        """Create feature coefficients plot"""
        coefficients = self.get_coefficients()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ['green' if x > 0 else 'red' for x in coefficients['Coefficient']]
        ax.barh(coefficients['Feature'], coefficients['Coefficient'], color=colors)
        ax.set_xlabel('Coefficient Value', fontsize=12)
        ax.set_title('Feature Coefficients in Linear Regression Model', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        
        return fig
    
    def plot_actual_vs_predicted(self):
        """Create actual vs predicted plots"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Training set
        ax1.scatter(self.y_train, self.y_pred_train, alpha=0.5, color='blue')
        ax1.plot([self.y_train.min(), self.y_train.max()], 
                [self.y_train.min(), self.y_train.max()], 
                'r--', lw=2)
        ax1.set_xlabel('Actual Price (₹)', fontsize=12)
        ax1.set_ylabel('Predicted Price (₹)', fontsize=12)
        ax1.set_title('Training Set: Actual vs Predicted Prices', fontsize=14, fontweight='bold')
        
        # Testing set
        ax2.scatter(self.y_test, self.y_pred_test, alpha=0.5, color='green')
        ax2.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 
                'r--', lw=2)
        ax2.set_xlabel('Actual Price (₹)', fontsize=12)
        ax2.set_ylabel('Predicted Price (₹)', fontsize=12)
        ax2.set_title('Testing Set: Actual vs Predicted Prices', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_residuals(self):
        """Create residuals distribution plots"""
        residuals_train = self.y_train - self.y_pred_train
        residuals_test = self.y_test - self.y_pred_test
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Training residuals
        ax1.hist(residuals_train, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('Residuals (₹)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Training Residuals Distribution', fontsize=14, fontweight='bold')
        
        # Testing residuals
        ax2.hist(residuals_test, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Residuals (₹)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Testing Residuals Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def predict(self, X_new):
        """
        Make predictions on new data
        
        Parameters:
        -----------
        X_new : pd.DataFrame
            New data for prediction
        
        Returns:
        --------
        np.array
            Predicted prices
        """
        return self.model.predict(X_new)
    
    def get_model_info(self):
        """Get model information"""
        metrics = self.get_metrics()
        
        info = f"""
        MODEL INFORMATION
        {'='*50}
        
        Intercept: ₹{self.model.intercept_:,.2f}
        Number of Features: {len(self.feature_names)}
        
        PERFORMANCE METRICS
        {'='*50}
        
        Training R² Score: {metrics['train_r2']:.4f}
        Testing R² Score: {metrics['test_r2']:.4f}
        
        Training RMSE: ₹{metrics['train_rmse']:,.2f}
        Testing RMSE: ₹{metrics['test_rmse']:,.2f}
        
        Training MAE: ₹{metrics['train_mae']:,.2f}
        Testing MAE: ₹{metrics['test_mae']:,.2f}
        
        {'='*50}
        """
        
        return info
