import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
data = pd.read_csv("dataset.csv")

print("Columns in dataset:", data.columns)

# Features and target (MATCH DATASET EXACTLY)
X = data[['Rainfall', 'Temperature', 'Humidity', 'Soil_pH',
          'Nitrogen', 'Phosphorus', 'Potassium']]
y = data['Yield']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ model.pkl saved successfully")
