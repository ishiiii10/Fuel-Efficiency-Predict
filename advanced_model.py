import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("car_data.csv")

# Create target variable (MPG - using combination_mpg)
df['mpg'] = df['combination_mpg']

# Feature engineering
print("Performing feature engineering...")

# Engine efficiency metrics
df['power_per_cylinder'] = df['displacement'] / df['cylinders']

# Interaction terms
drive_map = {'fwd': 1, 'rwd': 2, '4wd': 3, 'awd': 4}
df['drive_numeric'] = df['drive'].map(drive_map)
df['cylinders_drive'] = df['cylinders'] * df['drive_numeric']

# Fuel type as numeric
fuel_map = {'gas': 1, 'diesel': 2, 'electricity': 3}
df['fuel_numeric'] = df['fuel_type'].map(fuel_map)

# Transmission as numeric
trans_map = {'a': 1, 'm': 2}
df['trans_numeric'] = df['transmission'].map(trans_map)

# Create city-highway difference
df['mpg_diff'] = df['highway_mpg'] - df['city_mpg']

# Create mpg to displacement ratio (efficiency)
df['mpg_per_liter'] = df['combination_mpg'] / df['displacement']

# Polynomial features
df['cylinders_squared'] = df['cylinders'] ** 2
df['displacement_squared'] = df['displacement'] ** 2

# Car age feature
df['car_age'] = 2025 - df['year']

# Car make feature (simplified to reduce cardinality)
top_makes = df['make'].value_counts().head(10).index
df['make_top'] = df['make'].apply(lambda x: x if x in top_makes else 'other')

# Efficiency category based on quantiles
df['efficiency_category'] = pd.qcut(df['combination_mpg'], 4, labels=False)

# Clean data - drop rows with missing values
required_cols = ['class', 'cylinders', 'drive', 'fuel_type', 'transmission', 
                 'combination_mpg', 'displacement', 'power_per_cylinder', 
                 'cylinders_drive', 'car_age', 'city_mpg', 'highway_mpg']
df = df.dropna(subset=required_cols)

# Select features and target variable
features = ["class", "cylinders", "drive", "fuel_type", "transmission", 
            "displacement", "power_per_cylinder", "cylinders_drive", 
            "cylinders_squared", "displacement_squared", "car_age",
            "mpg_diff", "mpg_per_liter", "make_top", "city_mpg", "highway_mpg",
            "fuel_numeric", "trans_numeric"]
target = "mpg"

X = df[features]
y = df[target]

# Handle categorical features - make_top and class, etc.
categorical_cols = ["class", "drive", "fuel_type", "transmission", "make_top"]
X_cat = X[categorical_cols]

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoder_fitted = encoder.fit(X_cat)
X_encoded = encoder_fitted.transform(X_cat)

# Get numerical features and handle any missing values
numerical_cols = [col for col in features if col not in categorical_cols]
numerical_features = X[numerical_cols].values
imputer = SimpleImputer(strategy='mean')
numerical_features = imputer.fit_transform(numerical_features)

# Combine with numerical features
X_combined = np.hstack((numerical_features, X_encoded))

# Create and fit scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert y_train to numpy array if it's a pandas Series
y_train_np = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train
y_test_np = y_test.to_numpy() if isinstance(y_test, pd.Series) else y_test

# Data augmentation
def augment_data(X, y, n_samples=1000):
    # Get indices of samples to augment (randomly select)
    indices = np.random.choice(len(X), size=n_samples, replace=True)
    
    # Create copies with slight perturbations
    X_aug = X[indices].copy()
    y_aug = y[indices].copy()
    
    # Add random noise to numerical features
    for i in range(X_aug.shape[1]):
        X_aug[:, i] = X_aug[:, i] + np.random.normal(0, 0.03, size=n_samples) * X_aug[:, i]
    
    # Add slight variations to target
    y_aug = y_aug + np.random.normal(0, 0.01, size=n_samples) * y_aug
    
    # Combine with original data
    X_combined = np.vstack([X, X_aug])
    y_combined = np.concatenate([y, y_aug])
    
    return X_combined, y_combined

# Augment the training data
print("Augmenting training data...")
X_train_aug, y_train_aug = augment_data(X_train, y_train_np, n_samples=len(X_train) * 2)

# Feature selection using Random Forest
print("Performing feature selection...")
rf_selector = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
selector = SelectFromModel(rf_selector, threshold='median')
selector.fit(X_train_aug, y_train_aug)
X_train_selected = selector.transform(X_train_aug)
X_test_selected = selector.transform(X_test)

# Create more powerful base models for the stacked model
base_models = [
    ('gb', GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=7, random_state=42)),
    ('rf', RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)),
    ('et', ExtraTreesRegressor(n_estimators=200, max_depth=20, random_state=42)),
    ('ada', AdaBoostRegressor(n_estimators=100, random_state=42)),
    ('ridge', Ridge(alpha=1.0, random_state=42)),
    ('mlp', MLPRegressor(hidden_layer_sizes=(150, 100, 50), max_iter=2000, early_stopping=True, random_state=42)),
    ('svr', SVR(kernel='rbf', C=10, epsilon=0.1))
]

# Define the stacked model with a more powerful final estimator
stacked_model = StackingRegressor(
    estimators=base_models,
    final_estimator=GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
    cv=5
)

# Train the stacked model
print("Training stacked model...")
stacked_model.fit(X_train_selected, y_train_aug)

# Evaluate model
y_pred = stacked_model.predict(X_test_selected)
r2 = r2_score(y_test_np, y_pred)
print(f"Stacked Model R² Score: {r2}")
print(f"Accuracy: {r2 * 100:.2f}%")

# Save model and preprocessors
print("Saving model and preprocessors...")
with open("models/stacked_model.pkl", "wb") as f:
    pickle.dump(stacked_model, f)

with open("models/encoder_advanced.pkl", "wb") as f:
    pickle.dump(encoder_fitted, f)

with open("models/scaler_advanced.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("models/imputer_advanced.pkl", "wb") as f:
    pickle.dump(imputer, f)

with open("models/selector.pkl", "wb") as f:
    pickle.dump(selector, f)

print("Advanced model and preprocessors saved successfully!") 