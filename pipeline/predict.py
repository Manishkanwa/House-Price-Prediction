# predict.py
import pandas as pd
import joblib
from pipeline.cleaning import clean_data
from pipeline.feature_engineering import feature_engineering # type: ignore
from pipeline.preprocessing import FeatureScalerSelector
from pipeline.model import load_model


def predict_price(house_df, all_features_df,  selector_path):
    """
    Given a single-row house_df and the full features dataframe structure,
    preprocess and predict its price.
    """
    
    model = load_model()
    # Clean and engineer features
    data = clean_data(house_df.copy())
    data = feature_engineering(data)

    # Load model components
    preprocessor = joblib.load(selector_path)
    if not isinstance(preprocessor, FeatureScalerSelector):
        raise ValueError("Loaded preprocessor is not of type FeatureScalerSelector")
    # Ensure the preprocessor has the correct columns
    missing_columns = set(preprocessor.selected_columns) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Preprocessor selected columns  do not match house_df columns. Missing")
    extra_columns = set(data.columns) - set(preprocessor.selected_columns)
    if extra_columns:
        raise ValueError(f"House_df contains extra columns not in preprocessor: {extra_columns}")

    # Preprocess and select features
    selected = preprocessor.transform(house_df)

    prediction = model.predict(selected)[0]
    return prediction
