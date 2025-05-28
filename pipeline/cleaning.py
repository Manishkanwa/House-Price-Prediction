# cleaning.py
import pandas as pd

def clean_data(df : pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the housing dataframe by handling missing values.
    Categorical features with missing values are filled with 'None',
    numerical ones with 0. Special case handling for LotFrontage.
    """
    none_fill_cols = [
        'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
        'MasVnrType'
    ]

    for col in none_fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna('None')
    zero_fill_cols = [
        'MasVnrArea', 'GarageCars', 'GarageArea',
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
        'BsmtFullBath', 'BsmtHalfBath'
    ]
    
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])
    for col in zero_fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    if 'LotFrontage' in df.columns and 'Neighborhood' in df.columns:
        df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )

    if 'Electrical' in df.columns:
        df.dropna(subset=['Electrical'], inplace=True)

    return df
