import numpy as np

# ==========================================
# STEP 1: SAMPLE DATA (Student Scores vs Hours Studied)
# ==========================================
# Hours studied (x) and exam score (y)
x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([55, 65, 70, 85, 90], dtype=float)
n = len(x)

print("--- DATASET ---")
print("Hours Studied (X):", x)
print("Exam Scores   (Y):", y)
print(f"Number of observations (n): {n}\n")

# ==========================================
# STEP 2: LEAST SQUARES FORMULA (FROM SCRATCH)
# ==========================================
# Slope m = sum((x - mean_x) * (y - mean_y)) / sum((x - mean_x)^2)
mean_x = np.mean(x)
mean_y = np.mean(y)

numerator = np.sum((x - mean_x) * (y - mean_y))
denominator = np.sum((x - mean_x) ** 2)

m = numerator / denominator
c = mean_y - m * mean_x

print("--- FITTED MODEL (y = m*x + c) ---")
print(f"Slope (m):     {m:.4f}")
print(f"Intercept (c): {c:.4f}")
print(f"Equation: Score = {c:.2f} + {m:.2f} * Hours\n")

# ==========================================
# STEP 3: PREDICTIONS AND RESIDUALS
# ==========================================
y_pred = m * x + c
residuals = y - y_pred

print("--- RESIDUALS ANALYSIS ---")
for i in range(n):
    print(f"X={x[i]:.0f} | Actual Y={y[i]:.1f} | Predicted Y={y_pred[i]:.1f} | Residual={residuals[i]:.2f}")
print()

# ==========================================
# STEP 4: METRICS (SS_tot, SS_res, R^2, F-Statistic)
# ==========================================
ss_tot = np.sum((y - mean_y) ** 2)
ss_res = np.sum(residuals ** 2)

r2 = 1 - (ss_res / ss_tot)

# Degrees of Freedom
# p = number of predictor variables (1 feature: Hours Studied)
p = 1
df_total = n - 1
df_res = n - p - 1 # n - 2 for simple linear regression (estimating slope + intercept)
df_model = p

# Mean Squares & F-Statistic
ms_model = (ss_tot - ss_res) / df_model
ms_res = ss_res / df_res
f_stat = ms_model / ms_res

print("--- MODEL EVALUATION METRICS ---")
print(f"Total Sum of Squares (SS_tot):        {ss_tot:.2f}")
print(f"Sum of Squared Residuals (SS_res):    {ss_res:.2f}")
print(f"Coefficient of Determination (R^2):   {r2:.4f} ({r2*100:.1f}% variation explained)")
print(f"Residual Degrees of Freedom (n - p - 1): {df_res}")
print(f"F-Statistic:                          {f_stat:.4f}\n")
