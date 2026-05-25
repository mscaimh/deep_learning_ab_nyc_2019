"""
Classical Regression Model
Features: room_type, availability_365, neighbourhood
"""

import pandas as pd
import numpy as np
import pickle, os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Load data
TRAIN_PATH = r"C:\Users\mahah\git\deep_learning_ab_nyc_2019\split_data\AB_NYC_2019_training.csv"

df = pd.read_csv(TRAIN_PATH)
print(f"Training set loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Data cleanup

# ..drop columns that carry no predictive signal (free-text, surrogate IDs, date string)
df.drop(columns=['id', 'name', 'host_id', 'host_name', 'last_review'], inplace=True)

# ..remove listings with price == 0
before = len(df)
df = df[df['price'] > 0].copy()
print(f"Removed {before - len(df)} rows with price = 0  ->  {len(df)} rows remain")

# ..fill missing reviews_per_month with 0
df['reviews_per_month'] = df['reviews_per_month'].fillna(0)

# ..one-hot encode categorical columns (only selected features)
df = pd.get_dummies(df, columns=['neighbourhood', 'room_type'], drop_first=False)
print(f"After one-hot encoding: {df.shape[1]} features")

# Features
TARGET = 'price'

# ..use only the top 3 predictive features
SELECTED_BASE = ['availability_365']
SELECTED_OHE  = [c for c in df.columns if (c.startswith('neighbourhood_') and c != 'neighbourhood_group') or c.startswith('room_type_')]
FEATURE_COLS  = SELECTED_BASE + SELECTED_OHE

X = df[FEATURE_COLS].values.astype(np.float32)
y = df[TARGET].values.astype(np.float32)

# ..scale features (zero mean, unit variance)
scaler = StandardScaler()
X = scaler.fit_transform(X)

print(f"Feature matrix shape : {X.shape}")
print(f"Target vector shape  : {y.shape}")
print(f"Price  ->  min={y.min():.0f}  max={y.max():.0f}  mean={y.mean():.1f}")

# Training
model = LinearRegression()

print("\n-- Training ------------------------------------------")
model.fit(X, y)
y_pred = model.predict(X)
mae  = mean_absolute_error(y, y_pred)
rmse = mean_squared_error(y, y_pred) ** 0.5
r2   = r2_score(y, y_pred)
print(f"  Linear Regression  |  MAE: {mae:>7.2f}  |  RMSE: {rmse:>7.2f}  |  R2: {r2:.4f}")

# Print summary
print("\n-- Training-set metrics ------------------------------")
print(f"  MAE  : {mae:.2f}")
print(f"  RMSE : {rmse:.2f}")
print(f"  R2   : {r2:.4f}")

# Let's save the model for later use
MODEL_DIR = r"C:\Users\mahah\git\deep_learning_ab_nyc_2019"

with open(os.path.join(MODEL_DIR, 'model_linear.pkl'), 'wb') as f:
    pickle.dump(model, f)
print("\nModel saved  ->  model_linear.pkl")

with open(os.path.join(MODEL_DIR, 'scaler_classical.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler saved ->  scaler_classical.pkl")

with open(os.path.join(MODEL_DIR, 'classical_feature_cols.pkl'), 'wb') as f:
    pickle.dump(FEATURE_COLS, f)
print("Features saved ->  classical_feature_cols.pkl")
