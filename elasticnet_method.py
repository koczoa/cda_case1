import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

df = pd.read_csv("case1Data.csv")
data = df.dropna(subset=['y'])

feature_cols = [c for c in data.columns if c != 'y']
cat_cols = [c for c in feature_cols if c.startswith('C')]
num_cols = [c for c in feature_cols if not c.startswith('C')]

X = data[feature_cols]
y = data['y'].values


def create_pipeline(num_features, cat_features, inner_cv=5):
    num_transformer = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', RobustScaler())
    ])

    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])

    en_cv = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
        alphas=np.logspace(-3, 2, 60),
        cv=inner_cv,
        max_iter=20000,
        random_state=42,
        n_jobs=-1
    )

    return Pipeline([
        ('preprocessor', preprocessor),
        ('elasticnet', en_cv)
    ])

outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = []

print("Starting Nested CV for robust RMSE estimation...")

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
    X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
    y_train_fold, y_test_fold = y[train_idx], y[test_idx]

    pipeline = create_pipeline(num_cols, cat_cols, inner_cv=5)
    pipeline.fit(X_train_fold, y_train_fold)

    y_pred_fold = pipeline.predict(X_test_fold)
    fold_rmse = np.sqrt(mean_squared_error(y_test_fold, y_pred_fold))
    rmse_scores.append(fold_rmse)
    
    print(f"Fold {fold+1} RMSE: {fold_rmse:.4f}")

estimated_rmse = np.mean(rmse_scores)
print(f"\nFinal Estimated RMSE: {estimated_rmse:.4f}")

print("\nTraining final model on 100% of the data...")
# final_en_cv = ElasticNetCV(
#     l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
#     alphas=np.logspace(-3, 2, 60),
#     cv=10,
#     max_iter=20000,
#     random_state=42,
#     n_jobs=-1
# )

final_pipeline = create_pipeline(num_cols, cat_cols, inner_cv=10)
final_pipeline.fit(X, y)

final_pipeline.fit(X, y)

best_alpha = final_pipeline.named_steps['elasticnet'].alpha_
best_l1 = final_pipeline.named_steps['elasticnet'].l1_ratio_
n_nonzero = np.sum(final_pipeline.named_steps['elasticnet'].coef_ != 0)

print(f"Final Model - Best alpha: {best_alpha:.6f}")
print(f"Final Model - Best l1_ratio: {best_l1:.4f}")
print(f"Final Model - Non-zero coefs: {n_nonzero}")

eval_ds = pd.read_csv("case1Data_Xnew.csv")

predictions = final_pipeline.predict(eval_ds)

pd.Series(predictions).to_csv(f"sample_predictions_YourStudentNo.csv", index=False, header=False)

pd.Series([estimated_rmse]).to_csv(f"sample_estimatedRMSE_YourStudentNo.csv", index=False, header=False)
