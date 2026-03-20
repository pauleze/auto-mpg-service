"""
generate_model.py
-----------------
Run this ONCE in your Google Colab notebook (Module 4) to produce the
model weights file (auto_mpg_model.pth) that the microservice loads.

Paste the entire cell into Colab, run it, then download auto_mpg_model.pth
and place it in the same folder as main.py before deploying.
"""

# ── Paste into Colab after your Module 4 training cells ─────────────────────

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load and clean Auto MPG
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
cols = ["mpg","cylinders","displacement","horsepower","weight",
        "acceleration","model_year","origin","car_name"]
df = pd.read_csv(url, names=cols, sep=r'\s+', na_values="?")
df.dropna(inplace=True)
df.drop(columns=["car_name"], inplace=True)

# 2. Split
X = df.drop(columns=["mpg"]).values.astype(np.float32)
y = df["mpg"].values.astype(np.float32).reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train).astype(np.float32)
X_test_s  = scaler.transform(X_test).astype(np.float32)

# IMPORTANT: print scaler statistics — copy these into main.py
print("Feature means:", scaler.mean_.tolist())
print("Feature stds: ", scaler.scale_.tolist())

# 4. Define model (must match main.py exactly)
class MPGRegressorDropout(nn.Module):
    def __init__(self, n_features=7, dropout_p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(), nn.Dropout(dropout_p),
            nn.Linear(64, 32),         nn.ReLU(), nn.Dropout(dropout_p),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.net(x)

# 5. Train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = MPGRegressorDropout().to(device)
opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

X_t = torch.tensor(X_train_s).to(device)
y_t = torch.tensor(y_train).to(device)

for epoch in range(500):
    model.train()
    opt.zero_grad()
    loss = loss_fn(model(X_t), y_t)
    loss.backward()
    opt.step()

# 6. Evaluate
model.eval()
with torch.no_grad():
    preds = model(torch.tensor(X_test_s).to(device)).cpu().numpy().flatten()
from sklearn.metrics import mean_squared_error, r2_score
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2   = r2_score(y_test, preds)
print(f"Test RMSE: {rmse:.3f} mpg  |  Test R²: {r2:.4f}")

# 7. Save weights
torch.save(model.state_dict(), "auto_mpg_model.pth")
print("Saved: auto_mpg_model.pth")
print("Download this file and place it next to main.py before deploying.")
