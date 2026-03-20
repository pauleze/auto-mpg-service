"""
Auto MPG Fuel Efficiency Prediction Microservice
EAI 6010 - Module 5 Assignment

Exposes the PyTorch MLP regression model trained in Module 4 as a REST API.
Input:  7 vehicle attributes (JSON)
Output: Predicted fuel efficiency in miles per gallon (MPG)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
import torch.nn as nn
import numpy as np

# ── Model definition (must match Module 4 architecture exactly) ─────────────
class MPGRegressorDropout(nn.Module):
    """
    Two-hidden-layer MLP with dropout, trained on the UCI Auto MPG dataset.
    Architecture: 7 → 64 → 32 → 1
    """
    def __init__(self, n_features: int = 7, dropout_p: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


# ── Scaler parameters (fitted on training data in Module 4) ─────────────────
# StandardScaler statistics from the Auto MPG training split
# Features: cylinders, displacement, horsepower, weight, acceleration, model_year, origin
FEATURE_MEANS = np.array([5.4548, 193.7258, 104.4677, 2966.9355, 15.5419, 75.9839, 1.5726])
FEATURE_STDS  = np.array([1.7010,  104.4940,  38.4038,  847.9042,  2.7571,  3.6915, 0.8013])

# ── Load model ───────────────────────────────────────────────────────────────
model = MPGRegressorDropout(n_features=7)
model.load_state_dict(
    torch.load("auto_mpg_model.pth", map_location=torch.device("cpu"))
)
model.eval()

# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Auto MPG Prediction Service",
    description=(
        "Predicts vehicle fuel efficiency (miles per gallon) from 7 physical attributes "
        "using a PyTorch MLP regression model trained on the UCI Auto MPG dataset. "
        "Developed for EAI 6010 Module 5."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ───────────────────────────────────────────────
class VehicleFeatures(BaseModel):
    cylinders: float = Field(
        ..., ge=3, le=8,
        description="Number of engine cylinders (typically 4, 6, or 8)",
        example=4
    )
    displacement: float = Field(
        ..., ge=68, le=455,
        description="Engine displacement in cubic inches",
        example=140.0
    )
    horsepower: float = Field(
        ..., ge=46, le=230,
        description="Engine horsepower",
        example=90.0
    )
    weight: float = Field(
        ..., ge=1613, le=5140,
        description="Vehicle weight in pounds",
        example=2264
    )
    acceleration: float = Field(
        ..., ge=8.0, le=24.8,
        description="0–60 mph acceleration time in seconds",
        example=15.5
    )
    model_year: float = Field(
        ..., ge=70, le=82,
        description="Model year (last two digits, e.g. 82 = 1982)",
        example=71
    )
    origin: float = Field(
        ..., ge=1, le=3,
        description="Geographic origin: 1 = USA, 2 = Europe, 3 = Japan",
        example=1
    )

    class Config:
        json_schema_extra = {
            "example": {
                "cylinders": 4,
                "displacement": 140.0,
                "horsepower": 90.0,
                "weight": 2264,
                "acceleration": 15.5,
                "model_year": 71,
                "origin": 1
            }
        }


class PredictionResponse(BaseModel):
    predicted_mpg: float = Field(..., description="Predicted fuel efficiency in miles per gallon")
    input_received: dict   = Field(..., description="Echo of the input features for verification")
    model_info: str        = Field(..., description="Brief description of the model used")


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/", summary="Health check")
def root():
    """Returns a simple status message confirming the service is running."""
    return {
        "status": "running",
        "service": "Auto MPG Prediction API",
        "docs": "/docs",
        "predict_endpoint": "/predict"
    }


@app.post("/predict", response_model=PredictionResponse, summary="Predict fuel efficiency")
def predict(vehicle: VehicleFeatures):
    """
    Accepts 7 vehicle attributes and returns a predicted MPG value.

    **Required fields:**
    - `cylinders`: Number of engine cylinders (3–8)
    - `displacement`: Engine displacement in cubic inches (68–455)
    - `horsepower`: Engine horsepower (46–230)
    - `weight`: Vehicle weight in pounds (1613–5140)
    - `acceleration`: 0–60 mph time in seconds (8.0–24.8)
    - `model_year`: Two-digit year (70–82, representing 1970–1982)
    - `origin`: 1 = USA, 2 = Europe, 3 = Japan
    """
    try:
        # Build feature array
        raw = np.array([[
            vehicle.cylinders,
            vehicle.displacement,
            vehicle.horsepower,
            vehicle.weight,
            vehicle.acceleration,
            vehicle.model_year,
            vehicle.origin,
        ]], dtype=np.float32)

        # Standardize using training-set statistics
        scaled = (raw - FEATURE_MEANS) / FEATURE_STDS

        # Run inference
        with torch.no_grad():
            tensor_in = torch.tensor(scaled, dtype=torch.float32)
            prediction = model(tensor_in).item()

        return PredictionResponse(
            predicted_mpg=round(prediction, 2),
            input_received=vehicle.model_dump(),
            model_info=(
                "PyTorch MLP (7→64→32→1, ReLU + Dropout=0.2), "
                "trained on UCI Auto MPG dataset. "
                "Test R²≈0.87, RMSE≈3.1 mpg."
            )
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
