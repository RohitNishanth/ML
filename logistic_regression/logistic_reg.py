import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load data
df = pd.read_csv('titanic.csv')

# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Feature encoding
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Separate target and features
drop_cols = [col for col in ['PassengerId', 'SourceURL', 'Survived'] if col in df.columns]
X = df.drop(columns=drop_cols)
y = df['Survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Predict & evaluate
y_pred = model.predict(X_test_scaled)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}\n")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature Weights
weights = pd.DataFrame({
    'Feature': X.columns,
    'Weight': model.coef_[0]
}).sort_values('Weight', ascending=False)

print("\nFeature Weights:")
print(weights.to_string(index=False))

"""
====================================================================
                       REVISION CHEATSHEET
====================================================================
1. Logistic Regression Basics:
   - Used for binary classification (outputs probabilities between 0 and 1).
   - Sigmoid function: p = 1 / (1 + e^-z), where z = w1*x1 + w2*x2 + ... + b
   - Threshold = 0.5 (if p >= 0.5 -> Predict 1, else Predict 0).

2. Data Preprocessing:
   - Missing Numerical (Age) -> Imputed with Median (robust to outliers).
   - Missing Categorical (Embarked) -> Imputed with Mode (most frequent).
   - Binary Mapping (Sex) -> male: 1, female: 0.
   - One-Hot Encoding (Embarked) -> drop_first=True to avoid dummy variable trap.

3. Feature Scaling:
   - StandardScaler: (x - mean) / std_dev.
   - Essential for gradient-based models so feature magnitudes don't skew weights.

4. Model Evaluation Metrics:
   - Accuracy: Overall percentage of correct predictions.
   - Confusion Matrix: [ [TN, FP], [FN, TP] ]
   - Precision: TP / (TP + FP)  -> Accuracy of positive predictions.
   - Recall: TP / (TP + FN)     -> Proportion of actual positives identified.
   - Coefficients: Positive = increases survival probability, Negative = decreases it.
====================================================================
"""