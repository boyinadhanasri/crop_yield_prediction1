from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle
import numpy as np
import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": ""})

@app.post("/", response_class=HTMLResponse)
async def predict(
    request: Request,
    rainfall: float = Form(...),
    temperature: float = Form(...),
    humidity: float = Form(...),
    ph: float = Form(...),
    n: float = Form(...),
    p: float = Form(...),
    k: float = Form(...),
):
    input_data = np.array([[rainfall, temperature, humidity, ph, n, p, k]])
    prediction = model.predict(input_data)[0]
    result = f"🌾 Predicted Crop Yield: {prediction:.2f} tons/hectare"
    return templates.TemplateResponse("index.html", {"request": request, "prediction": result})

# For local testing
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
