# preprocessing.py
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


class FeatureScalerSelector:
    def __init__(self, k=50):
        self.k = k
        self.scaler = StandardScaler()
        self.selector = SelectKBest(score_func=f_regression, k=k)
        self.selected_columns = None

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        # Scale training data
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns, index=X.index)

        # Select top k features
        X_selected = self.selector.fit_transform(X_scaled, y)
        self.selected_columns = X.columns[self.selector.get_support()]
        
        return pd.DataFrame(X_selected, columns=self.selected_columns, index=X.index)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Scale and select columns from test/validation data
        X_scaled = pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)
        X_selected = X_scaled[self.selected_columns]
        return X_selected