import pandas as pd, numpy as np, torch, torch.nn as nn, pickle
from sklearn.metrics import mean_squared_error

TEST_PATH = r'C:\Users\mahah\git\deep_learning_ab_nyc_2019\split_data\AB_NYC_2019_testing.csv'
MODEL_DIR = r'C:\Users\mahah\git\deep_learning_ab_nyc_2019'

# Pre-process
df = pd.read_csv(TEST_PATH)
df.drop(columns=['id','name','host_id','host_name','last_review'], inplace=True)
df = df[df['price'] > 0].copy()
df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
df = pd.get_dummies(df, columns=['neighbourhood_group','neighbourhood','room_type'], drop_first=False)

# Load model
class TwoLayerDNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2))
        self.layer2 = nn.Sequential(nn.Linear(128,64),        nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2))
        self.output = nn.Linear(64, 1)
    def forward(self, x):
        return self.output(self.layer2(self.layer1(x)))

ckpt         = torch.load(MODEL_DIR + r'\model_2layer.pth', weights_only=False)
feature_cols = ckpt['feature_cols']
input_dim    = ckpt['input_dim']

with open(MODEL_DIR + r'\scaler_2layer.pkl', 'rb') as f:
    scaler = pickle.load(f)

y_test = df['price'].values.astype(np.float32)
X_test = df.reindex(columns=feature_cols, fill_value=0).values.astype(np.float32)
X_test = scaler.transform(X_test)

model = TwoLayerDNN(input_dim)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

def predict(X):
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy().flatten()

baseline_rmse = mean_squared_error(y_test, predict(X_test)) ** 0.5

# Permutation importance - group OHE columns back to original feature
# Map each feature_col back to its original variable name
def original_name(col):
    for prefix in ['neighbourhood_group_', 'neighbourhood_', 'room_type_']:
        if col.startswith(prefix):
            return prefix.rstrip('_')
    return col

groups = {}
for i, col in enumerate(feature_cols):
    key = original_name(col)
    groups.setdefault(key, []).append(i)

np.random.seed(42)
importances = {}
for group_name, indices in groups.items():
    X_permuted = X_test.copy()
    # shuffle all columns in the group together (preserves within-row correlation for OHE)
    shuffled_idx = np.random.permutation(len(X_permuted))
    X_permuted[:, indices] = X_permuted[shuffled_idx][:, indices]
    permuted_rmse = mean_squared_error(y_test, predict(X_permuted)) ** 0.5
    importances[group_name] = permuted_rmse - baseline_rmse

# Sort descending
sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)

print(f'Baseline RMSE: {baseline_rmse:.2f}')
print()
print(f'{\"Rank\":<6} {\"Feature\":<38} {\"RMSE increase when shuffled\":>26} {\"Importance\":>12}')
print('-' * 86)
max_imp = sorted_imp[0][1]
for rank, (feat, imp) in enumerate(sorted_imp, 1):
    bar = '#' * int((imp / max_imp) * 30) if imp > 0 else ''
    print(f'{rank:<6} {feat:<38} {imp:>+26.4f}   {bar}')
    