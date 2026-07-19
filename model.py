import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Read the dataset
data = pd.read_csv("Human Development Index - Full.csv")

# Display first 5 rows
print(data.head())

print(data.shape)
print(data.columns)
print(data.info())

print(data.isnull().sum())
print(data.describe())

columns = [
    "Life Expectancy at Birth (2021)",
    "Expected Years of Schooling (2021)",
    "Mean Years of Schooling (2021)",
    "Gross National Income Per Capita (2021)",
    "Human Development Index (2021)"
]

new_data = data[columns]

print(new_data.head())
new_data = new_data.dropna()

print(new_data.shape)
print(new_data.head())
# Input Features
X = new_data.drop("Human Development Index (2021)", axis=1)

# Target
y = new_data["Human Development Index (2021)"]

print(X.head())
print(y.head())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("Model Trained Successfully!")
# Predict HDI values
y_pred = model.predict(X_test)

print(y_pred[:5])
from sklearn.metrics import r2_score

score = r2_score(y_test, y_pred)

print("R2 Score:", score)
import joblib

joblib.dump(model, "hdi_model.pkl")

print("Model Saved Successfully!")
