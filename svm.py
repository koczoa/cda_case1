import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 1. Load the dataset
df = pd.read_csv('case1Data.csv')

# Separate features (X) and target (y)
X = df.drop(columns=['y'])
y = df['y']

# Identify categorical ('c') and numerical ('x') columns
c_cols = [c for c in X.columns if c.lower().startswith('c')]
x_cols = [x for x in X.columns if not x.lower().startswith('c')]

# 2. Build the Preprocessing Pipeline
# Numerical: Fill missing with median, scale to mean=0, var=1
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical: Fill missing with most frequent, scale to 0-1 range
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', MinMaxScaler())
])

# Combine into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, x_cols),
        ('cat', categorical_transformer, c_cols)
    ])

# 3. Build the full Model Pipeline (Preprocessor + SVM)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', SVR(kernel='linear'))
])

# 4. Hyperparameter Tuning using Cross-Validation
# We test different values of C to find the one that generalizes best
param_grid = {
    'regressor__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X, y)

# Get the best model from the grid search
best_model = grid_search.best_estimator_

# 5. Evaluate the model on the training data
y_pred = best_model.predict(X)
print(f"Optimal C parameter: {grid_search.best_params_['regressor__C']}")
mse = mean_squared_error(y, y_pred)
print(f"Mean Squared Error: {mse}")
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2_score(y, y_pred):.4f}")

# 6. Save the trained pipeline to disk
joblib.dump(best_model, 'svm_linear_model.pkl')
print("Model saved successfully as 'svm_linear_model.pkl'")