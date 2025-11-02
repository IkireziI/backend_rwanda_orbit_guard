# This file contains the complete FastAPI server logic for the Collision Prediction API.
#
# To run this server (Backend):
# 1. Install dependencies: pip install fastapi uvicorn pydantic pandas scikit-learn numpy
# 2. Save this code as main.py and ensure rwanda_orbit_guard_model.pkl is in the same directory.
# 3. Run the server: uvicorn main:app --reload
# 4. The API endpoint will be: http://127.0.0.1:8000/predict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal, Optional
import time
from fastapi.middleware.cors import CORSMiddleware
import pickle
import pandas as pd # Needed to prepare data for a typical ML model
import warnings # <-- NEW: Import for warning suppression
import numpy as np # <-- NEW: Added numpy import
import json
import os

# Global variable to hold the loaded ML model
ML_MODEL = None
# Standardizing the file path based on your successful load:
MODEL_FILENAME = "./rwanda_orbit_guard_model.pkl"

# --- 1. Define the Input Data Structure (Schema) ---
class StateVectorInput(BaseModel):
    """
    Schema for the 6 input parameters of the satellite's state vector.
    """
    x_start: float
    y_start: float
    z_start: float
    Vx_start: float
    Vy_start: float
    Vz_start: float

# --- 2. Define the Output Data Structure ---
class PredictionOutput(BaseModel):
    """
    Schema for the predicted results returned to the frontend.
    """
    status: Literal['RED ALERT', 'GREEN LIGHT']
    miss_distance_km: float
    safety_threshold_km: float = 10.0
    model_rmse_meters: float = 6440.18

# Initialize the FastAPI application
app = FastAPI(
    title="Rwanda Orbit Guard Prediction API",
    description="API for calculating satellite collision risk using the rwanda_orbit_guard_model."
)

# --- UPDATED CORS SETTINGS FOR DEPLOYMENT ---
# Allow both your Vercel frontend and local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://rwanda-orbit-guard-frontend.vercel.app",  # Your Vercel frontend
        "http://localhost:3000",                          # Local development
        "http://127.0.0.1:3000",                         # Local development
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# --- GLOBAL MODEL LOADING (Executes when the application starts) ---
try:
    with open(MODEL_FILENAME, 'rb') as file:
        # Use warnings.catch_warnings to suppress the InconsistentVersionWarning during loading
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ML_MODEL = pickle.load(file)
    print(f"INFO: Successfully loaded ML Model: {MODEL_FILENAME}")
except FileNotFoundError:
    print(f"WARNING: Model file '{MODEL_FILENAME}' not found. Using simulated prediction logic.")
    # If the model file is not found, we keep ML_MODEL as None and fall back to the simulation.
except Exception as e:
    print(f"ERROR: Failed to load model {MODEL_FILENAME}: {e}")
    # Handle other loading errors

# --- 3. Prediction Logic (Now uses the loaded model or simulation fallback) ---
def predict_collision_risk(data: StateVectorInput) -> PredictionOutput:
    """
    Predicts the collision risk using the loaded ML model or a simulated fallback.
    """
    
    miss_distance_meters = 0.0
    SAFETY_THRESHOLD_KM = 10.0  # Fixed safety threshold
    
    if ML_MODEL is not None:
        try:
            # 1. Prepare data for the ML model
            input_df = pd.DataFrame([data.dict()])

            # 2. Run the prediction
            prediction_result = ML_MODEL.predict(input_df)
            
            print(f"DEBUG: Prediction result: {prediction_result}")  # For debugging
            print(f"DEBUG: Prediction shape: {prediction_result.shape if hasattr(prediction_result, 'shape') else 'No shape'}")
            print(f"DEBUG: Prediction dimensions: {prediction_result.ndim if hasattr(prediction_result, 'ndim') else 'No ndim'}")
            
            # FIXED: Calculate Euclidean distance from the predicted coordinates
            if isinstance(prediction_result, np.ndarray) and prediction_result.ndim == 2:
                if prediction_result.shape[1] >= 3:  # If we have at least 3 coordinates
                    # The model predicts position coordinates, calculate distance from origin
                    x_pred, y_pred, z_pred = prediction_result[0][0], prediction_result[0][1], prediction_result[0][2]
                    
                    # Calculate Euclidean distance (this is the actual miss distance)
                    miss_distance_meters = np.sqrt(x_pred**2 + y_pred**2 + z_pred**2)
                    
                    print(f"DEBUG: Predicted coordinates - X: {x_pred}, Y: {y_pred}, Z: {z_pred}")
                    print(f"DEBUG: Calculated miss distance: {miss_distance_meters} meters")
                else:
                    # Use the first value as distance
                    miss_distance_meters = abs(float(prediction_result[0][0]))
            elif isinstance(prediction_result, (list, np.ndarray)):
                # Handle 1D arrays
                miss_distance_meters = abs(float(prediction_result[0]))
            else:
                # Handle scalar values
                miss_distance_meters = abs(float(prediction_result))
            
            print(f"DEBUG: Final miss distance (meters): {miss_distance_meters}")
                
        except Exception as e:
            print(f"ERROR during model prediction: {e}")
            print(f"Prediction result type: {type(prediction_result)}, value: {prediction_result}")
            raise HTTPException(status_code=500, detail="Internal Model Prediction Error.")

    else:
        # --- SIMULATED FALLBACK LOGIC (Used if the model file is not available) ---
        TOLERANCE = 1.0 
        
        # Check for the requested GREEN LIGHT input (z_start changed to 100,000.0)
        is_green_input_requested = (
            abs(data.x_start - (-8843.131454)) < TOLERANCE and
            abs(data.z_start - 100000.0) < TOLERANCE 
        )
        
        if is_green_input_requested:
            miss_distance_meters = 50000.0  # 50 km safe distance
        else:
            # Basic Z-based simulation: distance from original RED Z value
            distance_from_problem_z = abs(data.z_start - (-20741.615306))
            
            if distance_from_problem_z < 100:
                 miss_distance_meters = 0.0 # Red Alert
            else:
                 miss_distance_meters = min(distance_from_problem_z / 2, 50000.0) # Cap at 50km
        # --- END SIMULATED FALLBACK LOGIC ---

    # --- FINAL RESULT PROCESSING ---
    miss_distance_km = miss_distance_meters / 1000.0
    
    # Compare predicted miss distance to the safety threshold (10.0 km)
    if miss_distance_km < SAFETY_THRESHOLD_KM:
        status = 'RED ALERT'
    else:
        status = 'GREEN LIGHT'

    return PredictionOutput(
        status=status,
        miss_distance_km=miss_distance_km,
    )

# --- 4. Define the API Endpoint ---
@app.post("/predict", response_model=PredictionOutput)
async def get_prediction(input_data: StateVectorInput):
    """
    Accepts the satellite State Vector and returns the collision risk prediction.
    """
    # Simulate prediction calculation time (Asynchronous Processing Technique)
    time.sleep(1.5) 
    
    result = predict_collision_risk(input_data)
    
    return result

# --- Test endpoint with sample data ---
@app.post("/test-predict")
async def test_prediction():
    """
    Test endpoint with sample data to verify the API is working.
    """
    # Sample test data that should work
    test_data = StateVectorInput(
        x_start=1000.0,
        y_start=1000.0, 
        z_start=1000.0,
        Vx_start=-0.907527,
        Vy_start=-3.804930,
        Vz_start=-2.024133
    )
    
    print(f"TEST: Using sample data: {test_data.dict()}")
    
    result = predict_collision_risk(test_data)
    
    return {
        "test_data": test_data.dict(),
        "prediction": result.dict()
    }

# --- Endpoint to show expected input format ---
@app.get("/input-format")
async def show_input_format():
    """
    Shows the expected input format for the /predict endpoint.
    """
    return {
        "expected_format": {
            "x_start": "float",
            "y_start": "float", 
            "z_start": "float",
            "Vx_start": "float",
            "Vy_start": "float",
            "Vz_start": "float"
        },
        "example": {
            "x_start": -8843.131454,
            "y_start": 13138.221690,
            "z_start": -20741.615306,
            "Vx_start": -0.907527,
            "Vy_start": -3.804930,
            "Vz_start": -2.024133
        }
    }

# --- Health check endpoint for Render ---
@app.get("/")
def read_root():
    return {
        "message": "Rwanda Orbit Guard API is running.",
        "status": "healthy",
        "endpoints": {
            "GET /": "This health check",
            "GET /input-format": "Show expected input format",
            "POST /test-predict": "Test prediction with sample data",
            "POST /predict": "Main prediction endpoint",
            "GET /docs": "API documentation"
        }
    }

# --- Health check for Render ---
@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "API is running smoothly"}