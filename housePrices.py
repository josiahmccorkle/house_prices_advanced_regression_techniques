import argparse
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from preprocessing import clean_data, preprocess


# ---------------------------------
#  Argument parsing
# ---------------------------------
# Initialize the parser
parser = argparse.ArgumentParser(description="passing in test flag to run predictions")
# Add a flag (boolean)
parser.add_argument("--test", action="store_true", help="This will run the test and predict house prices")
# Parse the arguments
args = parser.parse_args()


# --------------------------------
#  Load Data
# --------------------------------
# List all CSV files in a folder
filePath = "./data/"
files = [f for f in os.listdir(filePath) if f.endswith('.csv')]
print(files)
# # # Load each file into a dictionary of DataFrames
dfs = {file: pd.read_csv(f"{filePath}{file}") for file in files}

if args.test:
    df = dfs["test.csv"]
else:
    df = dfs["train.csv"]


# --------------------------------
# 5. Clean Data
# --------------------------------
df = clean_data(df)


# --------------------------------
# 6. Prepare Features and Target
# --------------------------------
if not args.test:
    y = df["SalePrice"]
    X = df.drop(["SalePrice", "Id"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
else:
    X = df.drop(["Id"], axis=1)


# --------------------------------
# 7. Build Model Pipeline
# --------------------------------
preprocessor = preprocess(X)
model_pipeline = Pipeline(steps=[("preprocess", preprocessor),("model", Ridge())])


# --------------------------------
# 8. Train or Predict
# --------------------------------
# Ensure outputs directory exists
output_dir = "./outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model_path = os.path.join(output_dir, "model.pkl")

if not args.test:
    model_pipeline.fit(X_train,y_train)
    joblib.dump(model_pipeline, model_path)  # Save the trained model
    print(f"✅ Model trained and saved to {model_path}")

    # After training
    y_train_pred = model_pipeline.predict(X_train)
    train_score = model_pipeline.score(X_train, y_train)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)

    print(f"Train R² Score: {train_score:.4f}")
    print(f"Train RMSE: {train_rmse:.2f}")
else:
    if not os.path.exists(model_path):
        raise FileNotFoundError("🚨 model.pkl not found. Please run training first to create the model.")

    model_pipeline = joblib.load(model_path)
    y_pred = model_pipeline.predict(X)
    submission = pd.DataFrame({
        "Id": df["Id"],
        "SalePrice": y_pred
    })
    submission_path = os.path.join(output_dir, "submission.csv")
    submission.to_csv(submission_path, index=False)
    print("Submission saved to submission.csv")

