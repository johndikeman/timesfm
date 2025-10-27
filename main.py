import torch
import numpy as np
import timesfm
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn

torch.set_float32_matmul_precision("high")

model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)

model.compile(
    timesfm.ForecastConfig(
        max_context=1024,
        max_horizon=256,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    )
)

app = FastAPI(title="TimesFM inference API")


class PredictionRequest(BaseModel):
    data: List[List[float]]  # 2D array as nested lists
    horizon: int


class PredictionResponse(BaseModel):
    predictions: List[float]


def predict(candle_window: np.ndarray, horizon) -> np.ndarray:
    """
    Your model prediction logic here.

    Parameters:
    -----------
    candle_window : np.ndarray
        Shape: (window_size, 5) where columns are [open, high, low, close, volume]

    Returns:
    --------
    np.ndarray
        Predictions array
    """

    point_forecast, quantile_forecast = model.forecast(
        horizon=horizon, inputs=candle_window
    )

    return point_forecast


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(request: PredictionRequest):
    """
    Endpoint to receive candle data and return predictions.
    """
    try:
        # Convert nested list to numpy array
        data = np.array(request.data)

        horizon = request.horizon

        # Validate input shape
        if data.ndim != 2:
            raise HTTPException(
                status_code=400, detail=f"Expected 2D array, got {data.ndim}D"
            )

        if data.shape[1] != 5:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 5 features (OHLCV), got {data.shape[1]}",
            )

        # Make prediction
        predictions = predict(data, horizon)

        return PredictionResponse(predictions=predictions.tolist())

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Trading Model API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Send candle data for predictions",
            "/health": "GET - Health check",
        },
    }


if __name__ == "__main__":
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000)
