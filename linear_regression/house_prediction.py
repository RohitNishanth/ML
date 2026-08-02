import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)

# Load dataset
df = pd.read_csv('house_prices.csv')
# print(df.head())

# Features and target variable
x = df[['SizeSqFt', 'Bedrooms', 'YearBuilt']]AgeAtSaleYears
# print(x)
y = df['SalePriceUSD']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# Initialize & fit model
model = LinearRegression()
model.fit(X_train, y_train)

# print("Coefficients:", model.coef_)
# print("Intercept:", model.intercept_)

# Predict price for a single test house
new_house = pd.DataFrame({
    'SizeSqFt': [1800],
    'Bedrooms': [3],
    'YearBuilt': [2005]
})
predicted_house = model.predict(new_house)
print(f"Predicted price for sample house: ${predicted_house[0]:,.2f}\n")

# Model predictions on test set
predictions = model.predict(X_test)
print("Actual targets (y_test):\n", y_test)
print("Predictions:\n", predictions, "\n")

# Evaluation metrics
mae = mean_absolute_error(y_test, predictions)
print(f"mae  : {mae:.2f}")

mse = mean_squared_error(y_test, predictions)
print(f"mse  : {mse:.2f}")

rmse = root_mean_squared_error(y_test, predictions)
print(f"rmse : {rmse:.2f}")

r2 = r2_score(y_test, predictions)
print(f"r2   : {r2:.4f}\n")

# Feature correlation matrix
print("Correlation matrix:")
print(df.corr(numeric_only=True))

# ---------------------------------------------------------
# Results & Observations:
#
# mae  : 30610.27
# mse  : 963403624.52
# rmse : 31038.74
# r2   : -0.4473
#
# Why the result is not good:
# 1. Tiny dataset: Only 10 sample rows in total.
# 2. Very little information (missing key features):
#    We only used SizeSqFt, Bedrooms, YearBuilt.
#    Real house prices depend on many more factors:
#    - Location / Neighborhood
#    - Garage & Lot size
#    - Bathrooms
#    - School district
#    - Condition & Renovations
#
# ML Lesson:
# Writing the code correctly doesn't mean the model will be accurate.
# A model is only as good as the quality, volume, and features of the data provided!

# ---------------------------------------------------------

# How do ML engineers improve this?

# In this order:

# ✅ More data
# ✅ Better features
# ✅ Remove redundant features
# ✅ Try different algorithms if Linear Regression isn't suitable
# ---------------------------------------------------------