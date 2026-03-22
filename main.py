import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split


colorscheme = "viridis"
df = pd.read_csv("case1Data.csv")

# print("Top Absolute Correlations for y compared to others")
# c = df.corr()['y'].abs().sort_values(ascending=False)
# print(c[1:21])
#
#
#
# best_feature = c.index[1]
# print(f"Best feature: {best_feature} | Correlation: {c.iloc[1]:.4f}")




####################################################
######################## CV ########################
####################################################
print("\nCROSS VALIDATION")
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

fold_rmse, fold_r2 = [], []
for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = model.score(X_test, y_test)
    fold_rmse.append(rmse)
    fold_r2.append(r2)
    print(f"Fold {fold}: RMSE={rmse:.4f}  R²={r2:.4f}")

print(f"Mean RMSE: {np.mean(fold_rmse):.4f} ± {np.std(fold_rmse):.4f}")
print(f"Mean R^2:   {np.mean(fold_r2):.4f} ± {np.std(fold_r2):.4f}")



#####################################################
###################### XGBoost ######################
#####################################################
print("\nXGBOOST")

data = df.dropna(subset=['y'])

feature_cols = [c for c in data.columns if c != 'y']
X = data[feature_cols].values
y = data['y'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Test RMSE: {rmse:.4f}")
print(f"Test R^2:   {model.score(X_test, y_test):.4f}")

