import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import RFE

df = pd.read_csv("case1Data.csv")
data = df.dropna(subset=['y']).copy()

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
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])

print("\nPreprocessing the data...")
X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()
print(f"Total features after preprocessing: {len(feature_names)}")

print("\nFinding optimal Elastic Net parameters...")
l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]
en_cv = ElasticNetCV(
    l1_ratio=l1_ratios,
    alphas=np.logspace(-3, 2, 60),
    cv=5,
    max_iter=20000,
    random_state=42,
    n_jobs=-1
)
en_cv.fit(X_train_prep, y_train)

best_alpha = en_cv.alpha_
best_l1_ratio = en_cv.l1_ratio_
print(f"Optimal Alpha: {best_alpha:.4f}, Optimal L1 Ratio: {best_l1_ratio:.2f}")

print("\nPerforming Recursive Feature Elimination...")

best_en_model = ElasticNet(
    alpha=best_alpha,
    l1_ratio=best_l1_ratio,
    max_iter=10000,
    random_state=42
)

rfe = RFE(
    estimator=best_en_model,
    n_features_to_select=10,
    step=0.1
)

rfe.fit(X_train_prep, y_train)

selected_mask = rfe.support_
selected_features = feature_names[selected_mask]

print(f"\n--- RFE Completed ---")
print(f"Selected {len(selected_features)} most important features:")
for feature in selected_features:
    clean_name = feature.replace('num__', '').replace('cat__', '')
    print(f" - {clean_name}")

X_train_final = rfe.transform(X_train_prep)
X_test_final = rfe.transform(X_test_prep)

best_en_model.fit(X_train_final, y_train)
y_pred = best_en_model.predict(X_test_final)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance on Test Set (using {len(selected_features)} features):")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R^2:  {r2:.4f}")
