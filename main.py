import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt

# Load the dataset
df = pd.read_csv('your_dataset.csv')

# Visualizations
# 1. Histogram of Expenses
plt.figure(figsize=(10, 6))
sns.histplot(df['expenses'], kde=True)
plt.xlabel('Expenses')
plt.ylabel('Frequency')
plt.title('Distribution of Medical Expenses')
plt.show()

# 2. Scatter plot of Expenses vs. BMI
plt.figure(figsize=(10, 6))
sns.scatterplot(x='bmi', y='expenses', data=df)
plt.xlabel('BMI')
plt.ylabel('Expenses')
plt.title('BMI vs. Medical Expenses')
plt.show()

# 3. Scatter plot of Expenses vs. Age
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='expenses', data=df)
plt.xlabel('Age')
plt.ylabel('Expenses')
plt.title('Age vs. Medical Expenses')
plt.show()

# 4. Count plot of Smoker vs. Non-Smoker
plt.figure(figsize=(8, 6))
sns.countplot(x='smoker', data=df)
plt.xlabel('Smoker')
plt.ylabel('Count')
plt.title('Count of Smokers and Non-Smokers')
plt.show()

# 5. Scatter plot of Expenses vs. Life Expectancy
plt.figure(figsize=(10, 6))
sns.scatterplot(x='life_expect', y='expenses', data=df)
plt.xlabel('Life Expectancy')
plt.ylabel('Expenses')
plt.title('Life Expectancy vs. Medical Expenses')
plt.show()

# 6. Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# 7. Scatter plot of Expenses vs. Probability of Death
plt.figure(figsize=(10, 6))
sns.scatterplot(x='p(d)', y='expenses', data=df)
plt.xlabel('Probability of Death')
plt.ylabel('Expenses')
plt.title('Probability of Death vs. Medical Expenses')
plt.show()

# 8. Scatter plot of Probability of Death vs. Age
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='p(d)', data=df)
plt.xlabel('Age')
plt.ylabel('Probability of Death')
plt.title('Age vs. Probability of Death')
plt.show()

# Preprocessing
X = df.drop(['expenses'], axis=1)
y = df['expenses']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training and Evaluation
# Random Forest Regression
random_forest_reg = RandomForestRegressor(n_estimators=400, max_depth=5, random_state=13)
random_forest_reg.fit(X_train, y_train)
y_pred_rf = random_forest_reg.predict(X_test)
rf_r2 = r2_score(y_test, y_pred_rf)
rf_rmse = sqrt(mean_squared_error(y_test, y_pred_rf))

# Decision Tree Regression
decision_tree_reg = DecisionTreeRegressor(max_depth=5, random_state=13)
decision_tree_reg.fit(X_train, y_train)
y_pred_dt = decision_tree_reg.predict(X_test)
dt_r2 = r2_score(y_test, y_pred_dt)
dt_rmse = sqrt(mean_squared_error(y_test, y_pred_dt))

# Polynomial Regression
polynomial_features = PolynomialFeatures(degree=3)
X_train_poly = polynomial_features.fit_transform(X_train)
X_test_poly = polynomial_features.fit_transform(X_test)
polynomial_reg = LinearRegression(fit_intercept=False)
polynomial_reg.fit(X_train_poly, y_train)
y_pred_pr = polynomial_reg.predict(X_test_poly)
pr_r2 = r2_score(y_test, y_pred_pr)
pr_rmse = sqrt(mean_squared_error(y_test, y_pred_pr))

# Model Evaluation
print("Random Forest Regression R^2 Score:", rf_r2)
print("Random Forest Regression RMSE:", rf_rmse)
print("Decision Tree Regression R^2 Score:", dt_r2)
print("Decision Tree Regression RMSE:", dt_rmse)
print("Polynomial Regression R^2 Score:", pr_r2)
print("Polynomial Regression RMSE:", pr_rmse)
