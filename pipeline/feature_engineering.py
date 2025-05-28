# feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def feature_engineering(df:pd.DataFrame) -> pd.DataFrame:
    """
    Performs feature engineering on the housing dataset.
    - Log-transforms SalePrice
    - Creates new features (age, total baths, total square footage)
    - Encodes categorical variables
    - One-hot encodes nominal features
    """
    df = df.copy()
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    df['GarageAge'] = df['YrSold'] - df['GarageYrBlt']
    df['TotalBath'] = df['FullBath'] + 0.5 * df['HalfBath'] + \
                      df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    print(df.columns)  # Display columns for debugging
    if 'SalePrice' in df.columns:
        df['SalePrice'] = df['SalePrice'].apply(lambda x: max(x, 1))  # Avoid log(0)
        df['SalePrice'] = np.log(df['SalePrice'])   # Log-transform SalePrice
    
    # Handle missing values in categorical features
    none_fill_cols = [
        'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
        'BsmtFinType2', 'MasVnrType'
    ]
    for col in none_fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna('None')  # Fill with 'None' for categorical features
    zero_fill_cols = [
        'MasVnrArea', 'GarageCars', 'GarageArea',
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
        'BsmtFullBath', 'BsmtHalfBath'
    ]
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])
    for col in zero_fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)  # Fill with 0 for numerical features
    if 'LotFrontage' in df.columns and 'Neighborhood' in df.columns:
        df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )
    if 'Electrical' in df.columns: # Ensure 'Electrical' exists before dropping
        df.dropna(subset=['Electrical'], inplace=True)
    
    
    
    # Encode ordinal categorical features
    ordinal_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
                    'HeatingQC', 'KitchenQual', 'FireplaceQu',
                    'GarageQual', 'GarageCond', 'PoolQC']
    
    for col in ordinal_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))  # ensure string input
            
    df = pd.get_dummies(df, drop_first=True)
    print(df.head() ) # Display the first few rows for debugging
    return df