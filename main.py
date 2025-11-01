from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("student_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Initialize FastAPI
app = FastAPI(title="E-Learning ML API", version="1.0")

# Input Schema
class StudentData(BaseModel):
    attendance: float
    quiz_avg: float
    assignment_score: float
    video_watch_time: float
    forum_activity: int
    project_score: float

# Prediction Route
@app.post("/predict")
def predict(data: StudentData):
    # Convert input to array
    features = np.array([[
        data.attendance,
        data.quiz_avg,
        data.assignment_score,
        data.video_watch_time,
        data.forum_activity,
        data.project_score
    ]])

    # Scale features
    features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features_scaled)[0]

    # Response
    result = "Completed" if prediction == 1 else "Not Completed"
    return {"prediction": int(prediction), "status": result}

# Root route
@app.get("/")
def home():
    return {"message": "Welcome to the E-Learning ML Prediction API!"}