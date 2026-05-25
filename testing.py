import pandas as pd, numpy as np, torch, torch.nn as nn, pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TEST_PATH = r'C:\Users\mahah\git\deep_learning_ab_nyc_2019\split_data\AB_NYC_2019_testing.csv'
MODEL_DIR = r'C:\Users\mahah\git\deep_learning_ab_nyc_2019'

# Pre-processing
def preprocess(path):
    df = pd.read_csv(path)
    df.drop(columns=['id','name','host_id','host_name','last_review'], inplace=True)
    df = df[df['price'] > 0].copy()
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
    df = pd.get_dummies(df, columns=['neighbourhood_group','neighbourhood','room_type'], drop_first=False)
    return df

df_test = preprocess(TEST_PATH)
print('Test rows after cleanup:', len(df_test))

# model definitions
class TwoLayerDNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2))
        self.layer2 = nn.Sequential(nn.Linear(128,64),        nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2))
        self.output = nn.Linear(64, 1)
    def forward(self, x):
        return self.output(self.layer2(self.layer1(x)))

class ThreeLayerDNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2))
        self.layer2 = nn.Sequential(nn.Linear(128,64),        nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2))
        self.layer3 = nn.Sequential(nn.Linear(64,32),         nn.BatchNorm1d(32),  nn.ReLU(), nn.Dropout(0.2))
        self.output = nn.Linear(32, 1)
    def forward(self, x):
        return self.output(self.layer3(self.layer2(self.layer1(x))))

# evaluation
def evaluate(model_path, scaler_path, model_cls, df_test):
    ckpt = torch.load(model_path, weights_only=False)
    feature_cols = ckpt['feature_cols']
    input_dim    = ckpt['input_dim']

    y_test = df_test['price'].values.astype(np.float32)
    X_test = df_test.reindex(columns=feature_cols, fill_value=0).values.astype(np.float32)

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    X_test = scaler.transform(X_test)

    model = model_cls(input_dim)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    with torch.no_grad():
        y_pred = model(torch.tensor(X_test, dtype=torch.float32)).numpy().flatten()

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2   = r2_score(y_test, y_pred)
    return mae, rmse, r2

mae2, rmse2, r2_2 = evaluate(
    MODEL_DIR + r'\model_2layer.pth',
    MODEL_DIR + r'\scaler_2layer.pkl',
    TwoLayerDNN, df_test)

mae3, rmse3, r2_3 = evaluate(
    MODEL_DIR + r'\model_3layer.pth',
    MODEL_DIR + r'\scaler_3layer.pkl',
    ThreeLayerDNN, df_test)

print()
print('=' * 52)
print(f'  Metric        2-Layer DNN     3-Layer DNN')
print('=' * 52)
print(f'  MAE           {mae2:>10.2f}      {mae3:>10.2f}')
print(f'  RMSE          {rmse2:>10.2f}      {rmse3:>10.2f}')
print(f'  R2            {r2_2:>10.4f}      {r2_3:>10.4f}')
print('=' * 52)
