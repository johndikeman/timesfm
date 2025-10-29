import torch
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
import numpy as np
import timesfm
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn

from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.coder import PickleCoder
from redis import asyncio as aioredis

model = {}

import logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    torch.set_float32_matmul_precision("high")

    model["timesfm"] = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )

    model["timesfm"].compile(
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

    logger.info("Initializing Redis cache...")
    try:
        redis = aioredis.from_url("redis://localhost")
        await redis.ping()  # Test connection
        logger.info("Redis connection successful")

        FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")
        logger.info("FastAPICache initialized")
        logger.info(f"Cache backend: {FastAPICache.get_backend()}")
        logger.info(f"Cache prefix: {FastAPICache.get_prefix()}")
    except Exception as e:
        logger.error(f"Failed to initialize cache: {e}")
        raise

    yield


app = FastAPI(title="TimesFM inference API", lifespan=lifespan)


class PredictionRequest(BaseModel):
    data: List[List[float]]  # 2D array as nested lists
    horizon: int

    class Config:
        # This makes the model hashable for caching
        frozen = True

    def __hash__(self):
        # Convert data to tuple of tuples for hashing
        data_tuple = tuple(tuple(row) for row in self.data)
        return hash((data_tuple, self.horizon))


class PredictionResponse(BaseModel):
    predictions: List[List[List[float]]]


def predict(candle_window: np.ndarray, horizon) -> np.ndarray:
    """
    Your model prediction logic here.

    Parameters:
    -----------
    candle_window : np.ndarray

    Returns:
    --------
    np.ndarray
        Predictions array
    """
    model_obj = model.get("timesfm", None)
    if not model_obj:
        raise Exception("model not loaded yet!")

    point_forecast, quantile_forecast = model_obj.forecast(  # type: ignore
        horizon=horizon, inputs=candle_window
    )

    return quantile_forecast


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(request: PredictionRequest) -> PredictionResponse:
    """
    Endpoint to receive candle data and return predictions.
    """
    logger.info(f"Processing prediction request for horizon={request.horizon}")

    # Generate cache key from request data
    cache_key = (
        f"predict:{hash((tuple(tuple(row) for row in request.data), request.horizon))}"
    )
    logger.debug(f"Cache key: {cache_key}")

    # Try to get from cache
    backend = FastAPICache.get_backend()
    cached = await backend.get(cache_key)

    if cached:
        logger.info("Cache HIT - returning cached prediction")
        return PickleCoder.decode(cached)

    logger.info("Cache MISS - generating new prediction")

    try:
        # Convert nested list to numpy array
        data = np.array(request.data)
        horizon = request.horizon

        # Validate input shape
        if data.ndim != 2:
            raise HTTPException(
                status_code=400, detail=f"Expected 2D array, got {data.ndim}D"
            )

        if data.shape[0] != 5:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 5 features (OHLCV), got {data.shape[0]}",
            )

        # Make prediction
        predictions = predict(data, horizon)

        response = PredictionResponse(predictions=predictions.tolist())

        # Store in cache (60 second expiry)
        await backend.set(cache_key, PickleCoder.encode(response.dict()))

        return response

    except Exception as e:
        logger.error(f"Prediction error: {e}")
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

    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(asctime)s %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
        },
        "loggers": {
            "fastapi_cache": {
                "handlers": ["default"],
                "level": "DEBUG",
                "propagate": False,
            },
            "__main__": {  # Add logger for your module
                "handlers": ["default"],
                "level": "DEBUG",
                "propagate": False,
            },
        },
        "root": {"handlers": ["default"], "level": "INFO"},
    }

    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=log_config)
