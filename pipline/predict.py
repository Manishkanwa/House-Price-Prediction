# predict.py
import pandas as pd
import joblib
from pipeline.cleaning import clean_data
from pipeline.feature_engineering import add_features # type: ignore
from pipeline.preprocessing import get_preprocessor, apply_selector
from pipeline.model import load_model


def predict_price(house_df, all_features_df, numeric_features, categorical_features, selector_path):
    """
    Given a single-row house_df and the full features dataframe structure,
    preprocess and predict its price.
    """
    # Clean and engineer features
    house_df = clean_data(house_df.copy())
    house_df = add_features(house_df)

    # Ensure all expected columns exist
    missing_cols = [col for col in all_features_df.columns if col not in house_df.columns]
    for col in missing_cols:
        house_df[col] = 0
    house_df = house_df[all_features_df.columns]  # reorder to match

    # Load model components
    preprocessor = get_preprocessor(numeric_features, categorical_features)
    selector = joblib.load(selector_path)
    model = load_model()

    # Preprocess and select features
    processed = preprocessor.fit_transform(house_df)
    selected = apply_selector(processed, selector)

    # Predict
    prediction = model.predict(selected)[0]
    return prediction
