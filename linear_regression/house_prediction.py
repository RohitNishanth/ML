import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)

# 🏡 Initialize Model & Load Dataset
model = LinearRegression()
df = pd.read_csv("house_prices.csv")

# 📊 Select Features & Target
x = df[["SizeSqFt", "Bedrooms", "YearBuilt"]]
y = df["SalePriceUSD"]

# ✂️ Train-Test Split (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# 🧠 Train the Model
model.fit(X_train, y_train)

# 🔮 Predict Price for a New House
new_house = pd.DataFrame({
    "SizeSqFt": [1800],
    "Bedrooms": [3],
    "YearBuilt": [2005],
})
predicted_house = model.predict(new_house)
print(f"✨ Estimated Price for New House: ${predicted_house[0]:,.2f}\n")

# 📈 Predictions on Test Set & Model Evaluation
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
print(f"  • MAE  : ${mae:,.2f}")

mse = mean_squared_error(y_test, predictions)
print(f"  • MSE  : {mse:,.2f}")

rmse = root_mean_squared_error(y_test, predictions)
print(f"  • RMSE : ${rmse:,.2f}")

r2 = r2_score(y_test, predictions)
print(f"  • R²   : {r2:.4f}")

print(df.corr(numeric_only=True))

"""
=====================================================
📌 NOTES & ML TAKEAWAYS
=====================================================
The result shows lower accuracy for a few key reasons:

1️⃣ Tiny dataset (only 10 rows of data)
2️⃣ Limited features:
   We are only using: SizeSqFt, Bedrooms, YearBuilt

   Real house prices depend on many more factors:
   📍 Location
   🚗 Garage
   🛁 Bathrooms
   🎓 School district
   🏚️ Condition
   🌳 Lot size
   🛠️ Renovations

💡 Important ML Lesson:
   "I wrote the code correctly, so the model should be accurate." -> Myth!
   A model is only as good as the quality, volume, and features of the data provided.
   Even perfect code can't compensate for limited data.
"""