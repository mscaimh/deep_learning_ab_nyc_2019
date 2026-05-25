"""
3-Layer Deep Neural Network to predict Airbnb NYC 2019 listing price.
Architecture: Input -> Hidden Layer 1 (128, ReLU) -> Hidden Layer 2 (64, ReLU) -> Hidden Layer 3 (32, ReLU) -> Output (1)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Data
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

# ..one-hot encode categorical columns
df = pd.get_dummies(df, columns=['neighbourhood_group', 'neighbourhood', 'room_type'], drop_first=False)
print(f"After one-hot encoding: {df.shape[1]} features")

# Note down features and target
TARGET = 'price'
FEATURE_COLS = [c for c in df.columns if c != TARGET]

X = df[FEATURE_COLS].values.astype(np.float32)
y = df[TARGET].values.astype(np.float32).reshape(-1, 1)

# ..scale features (zero mean, unit variance)
scaler = StandardScaler()
X = scaler.fit_transform(X)

print(f"Feature matrix shape : {X.shape}")
print(f"Target vector shape  : {y.shape}")
print(f"Price  ->  min={y.min():.0f}  max={y.max():.0f}  mean={y.mean():.1f}")

# Prep dataset
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

dataset    = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# Modelling bits
class ThreeLayerDNN(nn.Module):
    def __init__(self, input_dim):
        super(ThreeLayerDNN, self).__init__()

        # Hidden Layer 1
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        # Hidden Layer 2
        self.layer2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        # Hidden Layer 3
        self.layer3 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        # Output Layer (regression - no activation)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.output(x)
        return x

input_dim = X.shape[1]
model     = ThreeLayerDNN(input_dim)
print(f"\nModel architecture:\n{model}")
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {total_params:,}")

# Training
EPOCHS = 100
LR     = 1e-3

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

print("\n-- Training ------------------------------------------")
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0

    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss        = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(X_batch)

    scheduler.step()

    avg_loss = epoch_loss / len(dataset)
    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoch {epoch:>3}/{EPOCHS}  |  MSE Loss: {avg_loss:>10.2f}  |  RMSE: {avg_loss**0.5:>7.2f}")

# Output evaluation metrics
model.eval()
with torch.no_grad():
    y_pred = model(X_tensor).numpy().flatten()

y_true = y.flatten()
mae  = mean_absolute_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred) ** 0.5
r2   = r2_score(y_true, y_pred)

print("\n-- Training-set metrics ------------------------------")
print(f"  MAE  : {mae:.2f}")
print(f"  RMSE : {rmse:.2f}")
print(f"  R2   : {r2:.4f}")

# Let's save the model and scaler for future use.
import pickle, os

MODEL_DIR = r"C:\Users\mahah\git\deep_learning_ab_nyc_2019"

torch.save({
    'model_state_dict' : model.state_dict(),
    'input_dim'        : input_dim,
    'feature_cols'     : FEATURE_COLS,
}, os.path.join(MODEL_DIR, 'model_3layer.pth'))

with open(os.path.join(MODEL_DIR, 'scaler_3layer.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

print("\nModel saved  ->  model_3layer.pth")
print("Scaler saved ->  scaler_3layer.pkl")
