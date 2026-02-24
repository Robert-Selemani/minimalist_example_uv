"""Train a simple linear regression model for disease prediction."""

import argparse

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor 


def train(train_data_path, model_path):
    """Train the model on the provided data.

    Parameters
    ----------
    train_data_path : str
        Path to the training data CSV file.
    model_path : str
        Path where the trained model will be saved.
    """
    df = pd.read_csv(train_data_path)
    # Use the same feature columns that we'll use at prediction time.
    # The future data includes both rainfall and mean_temperature, so
    # include both here to ensure feature names match between fit/predict.
    features = df[["rainfall", "mean_temperature"]].fillna(0)
    target = df["disease_cases"].fillna(0)

    # Defensive check: CHAP may pass temporary CSVs that are empty for some
    # evaluation folds (if the dataset is too small for the requested
    # prediction horizon). Detect that early and raise a clear error so the
    # evaluation log explains the real problem instead of a cryptic sklearn
    # failure about 0 samples.
    if features.shape[0] == 0 or target.shape[0] == 0:
        raise ValueError(
            "No training samples were provided to the trainer. "
            "This usually means the dataset is too small (too few time periods) "
            "for the evaluation windows CHAP is trying to use. "
            "Provide more historical data, or update your model template's "
            "min_prediction_length/max_prediction_length so CHAP doesn't create empty folds."
        )

    #model = LinearRegression()
    model = RandomForestRegressor()
    model.fit(features, target)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a disease prediction model")
    parser.add_argument("train_data", help="Path to training data CSV file")
    parser.add_argument("model", help="Path to save the trained model")
    args = parser.parse_args()

    train(args.train_data, args.model)
