# app/main.py

from fastapi import FastAPI
from app.schemas import IrisInput, PredictionOutput
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["https://github.com/Tr-tuhin/iris-insight-api.git"],  # or specify your frontend URL
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# Add this CORS middleware here



# Load the model once at startup
model = joblib.load("model/iris_model.joblib")
##if big in size Model->upload->online(hugging face)->online->load(server)->predict

# Iris class names for readability
class_names = ['setosa', 'versicolor', 'virginica']


# Create FastAPI app
app = FastAPI(
    title="Iris Classifier API",
    description="API for predicting Iris species using FastAPI",
    version="1.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "API is running"}

@app.get("/info")
def model_info():
    """Basic model info"""
    return {
        "model_type": "RandomForestClassifier",
        "classes": class_names
    }

@app.post("/predict", response_model=PredictionOutput)
def predict_species(data: IrisInput):
    """Make prediction from input features"""
    features = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    prediction = model.predict(features)[0]
    predicted_class = class_names[prediction]
    return {"predicted_class": predicted_class}
