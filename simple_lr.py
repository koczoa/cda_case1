import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

df = pd.read_csv("case1Data.csv")

print("\nLINEAR REGRESSION")
print("Top absolute correlations for y compared to others")
c = df.corr()['y'].abs().sort_values(ascending=False)
print(c[1:21])
best_feature = c.index[1]
print(f"Best feature: {best_feature} | Correlation: {c.iloc[1]:.4f}")
data = df[['y', best_feature]].copy()
data[best_feature] = data[best_feature].fillna(data[best_feature].mean())
data['y'] = data['y'].fillna(data['y'].mean())

X = data[[best_feature]].values
y = data['y'].values

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print(f"Model: y = {model.coef_[0]:.4f} * {best_feature} + {model.intercept_:.4f}")
print(f"R^2:   {model.score(X, y):.4f}")
print(f"RMSE: {rmse:.4f}")
