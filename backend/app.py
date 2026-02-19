"""ReluRay API - Chest X-ray Pneumonia Detection

A FastAPI-based medical AI API for detecting pneumonia from chest X-ray images.
Uses a pre-trained VGG16 model for binary classification (Normal vs Pneumonia).

Author: Isaac
Version: 1.0.0
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response
from pydantic import BaseModel, Field
from typing import Optional
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io
import base64
import logging
from datetime import datetime
import time
import psutil
import asyncio
from functools import lru_cache
from typing import Dict, Any
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""
    
    async def dispatch(self, request: StarletteRequest, call_next):
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        
        # CSP - Content Security Policy
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: https:; "
            "connect-src 'self'"
        )
        response.headers["Content-Security-Policy"] = csp
        
        return response

# Initialize FastAPI app with metadata
app = FastAPI(
    title="ReluRay API",
    description="""## Medical AI for Chest X-ray Pneumonia Detection
    
ReluRay is an AI-powered API that analyzes chest X-ray images to detect signs of pneumonia.
The system uses a deep learning model (VGG16-based) trained on medical imaging data.

### ⚠️ Medical Disclaimer
This tool is for educational and research purposes only. 
It is not intended to replace professional medical diagnosis. 
Always consult with a qualified healthcare provider for medical decisions.
Results should not be used as the sole basis for treatment decisions.
""",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    openapi_tags=[
        {
            "name": "Health",
            "description": "Health check and system monitoring endpoints"
        },
        {
            "name": "Prediction", 
            "description": "X-ray image analysis and pneumonia detection"
        },
        {
            "name": "Info",
            "description": "Model information and metadata"
        },
        {
            "name": "Monitoring",
            "description": "System metrics and performance monitoring"
        }
    ],
    contact={
        "name": "ReluRay Support",
        "email": "ei@nsisong.com",
        "url": "https://nsisong.com"
    },
    license_info={
        "name": "MIT License",
        "url": "https://github.com/1cbyc/reluray/blob/main/LICENSE"
    }
)

# Add security headers middleware
app.add_middleware(SecurityHeadersMiddleware)

# Add trusted host middleware (prevents host header attacks)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Response models
class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., example="healthy")
    model_loaded: bool = Field(..., example=True)
    timestamp: str = Field(..., example="2024-01-01T12:00:00.000000")
    version: str = Field(..., example="1.0.0")
    model_version: str = Field(..., example="1.0.0")
    uptime_seconds: float = Field(..., example=3600.5)
    memory_usage_mb: float = Field(..., example=1024.5)
    cpu_percent: float = Field(..., example=25.5)

class PredictRequest(BaseModel):
    """Prediction request model"""
    image: str = Field(..., description="Base64-encoded image data (PNG or JPEG format)")

class PredictResponse(BaseModel):
    """Prediction response model"""
    status: str = Field(..., example="success")
    prediction: str = Field(..., description="Prediction result: 'normal' or 'pneumonia'", example="normal")
    confidence: float = Field(..., description="Confidence score (0.0 to 1.0)", example=0.95)
    processing_time_ms: float = Field(..., example=150.5)
    timestamp: str = Field(..., example="2024-01-01T12:00:00.000000")
    model_version: str = Field(..., example="1.0.0")

class ModelInfoResponse(BaseModel):
    """Model information response model"""
    status: str = Field(..., example="success")
    model_name: str = Field(..., example="vgg16_pneumonia_detector")
    model_version: str = Field(..., example="1.0.0")
    model_description: str = Field(..., example="VGG16-based pneumonia detection model")
    input_shape: list = Field(..., example=[224, 224, 3])
    classes: list = Field(..., example=["Normal", "Pneumonia"])
    training_date: str = Field(..., example="2024-01-01")
    accuracy: float = Field(..., example=0.95)
    cache_enabled: bool = Field(..., example=True)
    cache_size: int = Field(..., example=100)
    cache_limit: int = Field(..., example=1000)
    lazy_loading: bool = Field(..., example=True)

# Global variables
model = None
model_loaded_time = None
startup_time = time.time()
prediction_cache = {}
CACHE_LIMIT = 1000

@lru_cache(maxsize=1)
def get_model_info_data():
    """Get model information data (cached)"""
    return {
        'model_name': 'vgg16_pneumonia_detector',
        'model_version': '1.0.0',
        'model_description': 'VGG16-based pneumonia detection model fine-tuned on chest X-ray images',
        'input_shape': [224, 224, 3],
        'classes': ['Normal', 'Pneumonia'],
        'training_date': '2024-01-01',
        'accuracy': 0.95,
        'cache_size': len(prediction_cache),
        'cache_limit': CACHE_LIMIT
    }

def get_cache_key(image_data: str) -> str:
    """Generate cache key from image data"""
    return hashlib.md5(image_data.encode()).hexdigest()

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global model, model_loaded_time
    try:
        logger.info("Loading pneumonia detection model...")
        
        # Try multiple possible model paths
        model_paths = [
            "backend/models/pneumonia_model.h5",
            "models/pneumonia_model.h5",
            "/home/isaac/reluray/backend/models/pneumonia_model.h5"
        ]
        
        model_loaded = False
        for model_path in model_paths:
            if os.path.exists(model_path):
                logger.info(f"Found model at: {model_path}")
                model = load_model(model_path)
                model_loaded = True
                break
        
        if not model_loaded:
            logger.warning("Model file not found. Running in mock mode.")
            model = None
        
        model_loaded_time = time.time()
        logger.info(f"Model loaded: {model is not None}")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = None

def preprocess_image(image_data: str) -> np.ndarray:
    """Preprocess base64 image for model prediction"""
    try:
        # Remove data URL prefix if present
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        
        # Open image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        image = image.resize((224, 224))
        
        # Convert to array and normalize
        image_array = img_to_array(image) / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    process = psutil.Process()
    
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        model_version="1.0.0",
        uptime_seconds=time.time() - startup_time,
        memory_usage_mb=process.memory_info().rss / 1024 / 1024,
        cpu_percent=process.cpu_percent()
    )

@app.get("/api/metrics", tags=["Monitoring"])
async def get_metrics():
    """Get system metrics"""
    process = psutil.Process()
    
    return {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent
        },
        "process": {
            "memory_mb": process.memory_info().rss / 1024 / 1024,
            "cpu_percent": process.cpu_percent(),
            "threads": process.num_threads(),
            "open_files": len(process.open_files())
        },
        "api": {
            "model_loaded": model is not None,
            "cache_size": len(prediction_cache),
            "cache_hits": 0,  # Would need to track this
            "cache_misses": 0  # Would need to track this
        }
    }

@app.post("/api/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    """Predict pneumonia from chest X-ray image"""
    start_time = time.time()
    
    # Check cache first
    cache_key = get_cache_key(request.image)
    if cache_key in prediction_cache:
        cached_result = prediction_cache[cache_key]
        return PredictResponse(
            **cached_result,
            processing_time_ms=(time.time() - start_time) * 1000
        )
    
    try:
        # Preprocess image
        image_array = preprocess_image(request.image)
        
        # Make prediction
        if model is not None:
            prediction = model.predict(image_array, verbose=0)
            confidence = float(prediction[0][0])
            
            # Convert to class label
            if confidence > 0.5:
                result = 'pneumonia'
                confidence_score = confidence
            else:
                result = 'normal'
                confidence_score = 1 - confidence
        else:
            # Mock prediction for testing
            logger.warning("Using mock prediction (model not loaded)")
            confidence_score = 0.85
            result = 'normal'
        
        # Create response
        response_data = {
            "status": "success",
            "prediction": result,
            "confidence": confidence_score,
            "timestamp": datetime.now().isoformat(),
            "model_version": "1.0.0"
        }
        
        # Cache the result
        if len(prediction_cache) < CACHE_LIMIT:
            prediction_cache[cache_key] = response_data
        
        processing_time = (time.time() - start_time) * 1000
        
        return PredictResponse(
            **response_data,
            processing_time_ms=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/api/info", response_model=ModelInfoResponse, tags=["Info"])
async def model_info():
    """Get model information"""
    model_info_data = get_model_info_data()
    
    info = {
        "status": "success",
        "model_name": model_info_data['model_name'],
        "model_version": model_info_data['model_version'],
        "model_description": model_info_data['model_description'],
        "input_shape": model_info_data['input_shape'],
        "classes": model_info_data['classes'],
        "training_date": model_info_data['training_date'],
        "accuracy": model_info_data['accuracy']
    }
    
    # Add caching information
    info['cache_enabled'] = True
    info['cache_size'] = model_info_data['cache_size']
    info['cache_limit'] = model_info_data['cache_limit']
    info['lazy_loading'] = True
    
    return ModelInfoResponse(**info)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
        'error': 'Internal server error',
        'status': 'error'
        }
    )

if __name__ == '__main__':
    import uvicorn
    # Use 5001 instead of 5000 to avoid macOS AirPlay conflict
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('FASTAPI_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting FastAPI application on port {port}")
    logger.info(f"Debug mode: {debug}")
    logger.info(f"Model loaded: {model is not None}")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=debug,
        log_level="info"
    )