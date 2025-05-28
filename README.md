# 🏠 House Price Prediction Web App

This project is an end-to-end machine learning regression pipeline deployed as a Flask web app. Users can select a house ID from the test dataset to view its predicted sale price.

---

## 📁 Project Structure
```
project-root/
├── app.py                  # Flask web app
├── requirements.txt        # Dependencies
├── data/
│   └── test_data.csv       # Test dataset with house IDs
├── models/
│   ├── ridge_model.pkl     # Trained Ridge Regression model
│   └── feature_selector.pkl# SelectKBest feature selector
├── logs/
│   └── pipeline.log        # Logs for predictions
├── templates/
│   ├── index.html          # Form to choose House ID
│   └── result.html         # Displays prediction
└── pipeline/
    ├── cleaning.py         # Missing value imputation
    ├── feature_engineering.py # Derived features
    ├── preprocessing.py    # Encoding, scaling, feature selection
    ├── model.py            # Train, save, load model
    ├── predict.py          # Pipeline for making predictions
    └── logger.py           # Logs predictions using Loguru
```

---

## 🚀 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Manishkanwa/house-price-predictor.git
cd house-price-predictor
```

### 2. Create Virtual Environment (optional)
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Add Required Files
- Place `test_data.csv` inside the `data/` folder
- Place `ridge_model.pkl` and `feature_selector.pkl` in `models/`

### 5. Run the App
```bash
python app.py
```
Then open your browser at [http://localhost:5000](http://localhost:5000)

---

## 🧠 ML Pipeline Overview
- **Cleaning**: Handles missing values
- **Feature Engineering**: Adds features like `HouseAge`, `TotalSF`, `TotalBath`
- **Preprocessing**: Encoding and scaling
- **Model**: Ridge Regression with GridSearchCV
- **Prediction**: Predicts based on selected row
- **Logging**: Every prediction logged to `logs/pipeline.log`

---

## ✨ Example Usage
1. Select a House ID from the dropdown
2. Submit the form
3. View predicted price on results page

---

## 📌 Author
Created by **Manish** — Data Scientist & Analyst
