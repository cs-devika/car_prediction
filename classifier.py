import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Load dataset
car_data = pd.read_csv("/content/car_prediction_data.csv")

# Drop duplicates
car_data = car_data.drop_duplicates()

# Drop Car_Name
car_data = car_data.drop(['Car_Name'], axis=1)

# Feature engineering: Car_Age
car_data['Car_Age'] = 2026 - car_data['Year']

# Define features and target (4 features)
X = car_data[['Car_Age','Year','Present_Price','Kms_Driven']]
y = car_data['Selling_Price']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Save model
with open("car_model.pkl", "wb") as f:
    pickle.dump(model, f)