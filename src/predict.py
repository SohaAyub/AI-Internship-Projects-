import pandas as pd
import joblib

# Load model
model = joblib.load("house_price_model.pkl")

# Example house
new_house = pd.DataFrame([
    [8.0, 25, 6.0, 1.2, 800, 3.0, 37.5, -122.0]
], columns=[
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude"
])

prediction = model.predict(new_house)

print("Predicted Price:", prediction[0])
