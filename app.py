# app.py
from flask import Flask, render_template, request
import pandas as pd
import os
from pipeline.predict import predict_price
from pipeline.logger import log_prediction

# === Config ===
app = Flask(__name__)
TEST_DATA_PATH = os.path.join("data", "test_data.csv")
SELECTOR_PATH = os.path.join("models", "feature_selector.pkl")

# Define these based on training
NUMERIC_FEATURES = ['LotArea', 'GrLivArea', 'TotalBsmtSF']  # example
CATEGORICAL_FEATURES = ['Neighborhood', 'HouseStyle', 'Exterior1st']  # example

# Load test data (must exist)
test_df = pd.read_csv(TEST_DATA_PATH)

@app.route("/")
def index():
    return render_template("index.html", houses=test_df[['Id']])

@app.route("/predict", methods=["POST"])
def predict():
    house_id = int(request.form["house_id"])
    selected = test_df[test_df['Id'] == house_id]

    if selected.empty:
        return "Invalid House ID"

    pred_price = predict_price(
        house_df=selected,
        all_features_df=test_df,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        selector_path=SELECTOR_PATH
    )

    log_prediction(house_id, pred_price)

    return render_template("result.html", house_id=house_id, price=round(pred_price, 2))

if __name__ == '__main__':
    app.run(debug=True)