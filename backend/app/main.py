"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.ml_service import model_service
from app.schemas import HealthResponse, ModelInfoResponse, PredictRequest, PredictResponse

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        model_service.load()
    except FileNotFoundError as exc:
        logger.error("Failed to load model: %s", exc)
    yield


app = FastAPI(
    title=settings.app_name,
    description="AI-powered stellar object classification API",
    version="1.0.0",
    lifespan=lifespan,
)

origins = [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if "*" not in origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(_: Request, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=model_service.is_loaded,
        accuracy=model_service.accuracy,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfoResponse(
        features=model_service.feature_columns,
        classes=model_service.classes,
        accuracy=model_service.accuracy or 0.0,
        model_type=model_service.model_type,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest):
    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        return model_service.predict(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
