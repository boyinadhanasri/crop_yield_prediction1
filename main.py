import tkinter as tk
from tkinter import Label, Entry, Button
import pickle
import numpy as np

# ===============================
# Load trained ML model
# ===============================
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ===============================
# Create main window
# ===============================
root = tk.Tk()
root.title("Crop Yield Prediction System")
root.geometry("400x620")
root.configure(bg="#f5fff5")

# ===============================
# Title
# ===============================
Label(
    root,
    text="Crop Yield Prediction",
    font=("Arial", 18, "bold"),
    fg="green",
    bg="#f5fff5"
).pack(pady=15)

# ===============================
# Helper function for input fields
# ===============================
def create_field(label_text):
    Label(root, text=label_text, bg="#f5fff5").pack()
    entry = Entry(root)
    entry.pack(pady=5)
    return entry

# ===============================
# Input fields (MATCH DATASET ORDER)
# ===============================
rainfall_entry = create_field("Rainfall (mm)")
temperature_entry = create_field("Temperature (°C)")
humidity_entry = create_field("Humidity (%)")
ph_entry = create_field("Soil pH")
n_entry = create_field("Nitrogen")
p_entry = create_field("Phosphorus")
k_entry = create_field("Potassium")

# ===============================
# Prediction function
# ===============================
def predict_yield():
    try:
        input_data = np.array([[
            float(rainfall_entry.get()),
            float(temperature_entry.get()),
            float(humidity_entry.get()),
            float(ph_entry.get()),     # Soil_pH
            float(n_entry.get()),
            float(p_entry.get()),
            float(k_entry.get())
        ]])

        prediction = model.predict(input_data)

        result_label.config(
            text=f"🌾 Predicted Crop Yield: {prediction[0]:.2f} tons/hectare"
        )

    except Exception as e:
        result_label.config(text=f"Error: {e}")

# ===============================
# Predict button
# ===============================
Button(
    root,
    text="Predict Yield",
    command=predict_yield,
    bg="green",
    fg="white",
    font=("Arial", 11, "bold")
).pack(pady=15)

# ===============================
# Output label (STEP 1)
# ===============================
result_label = Label(
    root,
    text="",
    font=("Arial", 12, "bold"),
    fg="blue",
    bg="#f5fff5"
)
result_label.pack(pady=10)

# ===============================
# Run the app
# ===============================
root.mainloop()
