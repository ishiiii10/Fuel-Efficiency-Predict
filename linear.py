import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv("car_data.csv")

# Create target variable (MPG - using combination_mpg)
df['mpg'] = df['combination_mpg']

# Clean data - drop rows with missing values
df = df.dropna(subset=['class', 'cylinders', 'drive', 'fuel_type', 'transmission', 'combination_mpg'])

# Select features and target variable
features = ["class", "cylinders", "drive", "fuel_type", "transmission"]
target = "mpg"

X = df[features]
y = df[target]

# One-Hot Encoding for categorical features
categorical_cols = ["class", "drive", "fuel_type", "transmission"]
X_cat = X[categorical_cols]

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoder_fitted = encoder.fit(X_cat)
X_encoded = encoder_fitted.transform(X_cat)

# Get numerical features and handle any missing values
numerical_features = X[["cylinders"]].values
imputer = SimpleImputer(strategy='mean')
numerical_features = imputer.fit_transform(numerical_features)

# Combine with numerical features
X_combined = np.hstack((numerical_features, X_encoded))

# Create and fit scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Evaluate model
y_pred = lr_model.predict(X_test)
print(f"Linear Regression R² Score: {r2_score(y_test, y_pred)}")

# Save model and preprocessors
with open("models/lr_model.pkl", "wb") as f:
    pickle.dump(lr_model, f)

with open("models/encoder.pkl", "wb") as f:
    pickle.dump(encoder_fitted, f)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("models/imputer.pkl", "wb") as f:
    pickle.dump(imputer, f)

print("Linear Regression model and preprocessors saved successfully!")
