# model.py
import joblib
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import os

MODEL_PATH = os.path.join("models", "ridge_model.pkl")


def train_ridge(X_train, y_train):
    """
    Train Ridge Regression with GridSearchCV to find best hyperparameters.
    Returns the best model.
    """
    ridge = Ridge()
    params = {
        'alpha': [0.1, 1.0, 10.0, 100.0],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr'],
        'max_iter': [None, 1000, 5000]
    }
    grid = GridSearchCV(ridge, params, cv=5, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_


def save_model(model, path=MODEL_PATH):
    """Save the trained model to disk."""
    joblib.dump(model, path)


def load_model(path=MODEL_PATH):
    """Load the trained model from disk."""
    return joblib.load(path)
