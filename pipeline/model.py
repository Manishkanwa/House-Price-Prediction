# model.py
import joblib
from sklearn.linear_model import Ridge
from pipeline.preprocessing import FeatureScalerSelector
import pandas as pd

from pipeline.cleaning import clean_data
from pipeline.feature_engineering import feature_engineering # type: ignore

import os

MODEL_PATH = os.path.join("models", "ridge_model.pkl")
SELECTOR_PATH = os.path.join("models", "feature_selector.pkl")


path = os.path.join('data', 'train_data.csv')
def train_ridge(data = pd.read_csv(path)) -> Ridge:
    """
    Train Ridge Regression with GridSearchCV to find best hyperparameters.
    Returns the best model."""
    
    data = clean_data(data.copy())
    data = feature_engineering(data)
    # Ensure all expected columns exist \
    print(data.columns)
    X_train = data.drop(columns=['SalePrice'])
    y_train = data['SalePrice']  
    preprocessor = FeatureScalerSelector(k = 170)
    selected = preprocessor.fit_transform(X_train, y_train)
    print(f"Selected features: {preprocessor.selected_columns}")
    X_train = selected
    joblib.dump(preprocessor, SELECTOR_PATH)
    
    ridge = Ridge(alpha = 100, max_iter = 1000, solver = 'lsqr')
    ridge.fit(X_train, y_train) 
    joblib.dump(ridge, MODEL_PATH)

def load_model(path=MODEL_PATH):
    """Load the trained model from disk."""
    train_ridge()
    return joblib.load(path)
