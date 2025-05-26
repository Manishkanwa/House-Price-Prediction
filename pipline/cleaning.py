# cleaning.py
import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the housing dataframe by handling missing values.
    Categorical features with missing values are filled with 'None',
    numerical ones with 0. Special case handling for LotFrontage and GarageYrBlt.
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

    for col in zero_fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Special handling for GarageYrBlt
    if 'GarageYrBlt' in df.columns and 'YearBuilt' in df.columns:
        df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])

    # Neighborhood-wise median imputation for LotFrontage
    if 'LotFrontage' in df.columns and 'Neighborhood' in df.columns:
        df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

    # Drop rows with missing Electrical (only 1 row)
    if 'Electrical' in df.columns:
        df.dropna(subset=['Electrical'], inplace=True)

    return df
