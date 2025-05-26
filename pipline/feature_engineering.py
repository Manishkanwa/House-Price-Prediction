# feature_engineering.py
import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add custom features to the dataset for improved regression performance.
    Features include age of house, remodeling, garage age, total area, and total bathrooms.
    """
    if all(col in df.columns for col in ['YrSold', 'YearBuilt']):
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']

    if all(col in df.columns for col in ['YrSold', 'YearRemodAdd']):
        df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']

    if all(col in df.columns for col in ['YrSold', 'GarageYrBlt']):
        df['GarageAge'] = df['YrSold'] - df['GarageYrBlt']

    if all(col in df.columns for col in ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']):
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

    if all(col in df.columns for col in ['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath']):
        df['TotalBath'] = df['BsmtFullBath'] + df['FullBath'] + \
                          0.5 * (df['BsmtHalfBath'] + df['HalfBath'])

    return df
