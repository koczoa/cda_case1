import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

df = pd.read_csv("case1Data.csv")
data = df.dropna(subset=['y'])

print(f"Dataset shape: {data.shape}")
print(f"y range: {data['y'].min():.2f} - {data['y'].max():.2f}")

feature_cols = [c for c in data.columns if c != 'y']
cat_cols = [c for c in feature_cols if c.startswith('C')]
num_cols = [c for c in feature_cols if not c.startswith('C')]

print(f"Numerical cols: {len(num_cols)}, Categorical cols: {len(cat_cols)}")
print(f"Total features: {len(feature_cols)}, Samples: {len(data)}")

X = data[feature_cols]
y = data['y'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_transformer = Pipeline([
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', RobustScaler())
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])

l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]
en_cv = ElasticNetCV(
    l1_ratio=l1_ratios,
    alphas=np.logspace(-3, 2, 60),
    cv=80,
    # cv=5,
    max_iter=20000,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('elasticnet', en_cv)
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = pipeline.score(X_test, y_test)

best_alpha = pipeline.named_steps['elasticnet'].alpha_
best_l1 = pipeline.named_steps['elasticnet'].l1_ratio_
n_nonzero = np.sum(pipeline.named_steps['elasticnet'].coef_ != 0)

print(f"\nBest alpha:    {best_alpha:.6f}")
print(f"Best l1_ratio: {best_l1:.4f}")
print(f"Non-zero coefficients: {n_nonzero}")
print(f"Test RMSE:  {rmse:.4f}")
print(f"Test R^2:    {r2:.4f}")
