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
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self'; connect-src 'self'"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=(), payment=(), usb=(), magnetometer=(), gyroscope=(), accelerometer=()"
        
        # HSTS (only in production)
        if os.environ.get('ENVIRONMENT', 'development').lower() == 'production':
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
        
        return response

# Model caching and optimization
class ModelManager:
    """Manages model loading with caching and lazy loading"""
    
    def __init__(self):
        self._model = None
        self._model_path = None
        self._load_time = 0
        self._cache = {}
        self._cache_size_limit = 100  # Max cached predictions
    
    def find_model_file(self):
        """Find the model file in common locations"""
        versioned_model = f"best_model_{MODEL_VERSION}.keras"
        possible_paths = [
            f'../{versioned_model}',
            f'../../{versioned_model}',
            versioned_model,
            f'./{versioned_model}',
            os.path.join(os.path.dirname(os.path.dirname(__file__)), versioned_model),
            '../best_model.keras',  # From backend/ directory (in repository root)
            '../../best_model.keras',  # Alternative path
            'best_model.keras',      # In current directory
            './best_model.keras',   # Current directory
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'best_model.keras'),  # Absolute from backend/
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Model file found at: {os.path.abspath(path)}")
                return path
        
        return None
    
    @lru_cache(maxsize=32)
    def _get_image_hash(self, image_data: str) -> str:
        """Generate hash for image data to use as cache key"""
        return hashlib.md5(image_data.encode()).hexdigest()
    
    def get_model(self):
        """Lazy load model only when needed"""
        if self._model is None:
            self._model_path = self.find_model_file()
            if self._model_path:
                try:
                    logger.info(f"Loading model from: {self._model_path}")
                    start_load = time.time()
                    self._model = load_model(self._model_path)
                    self._load_time = time.time() - start_load
                    logger.info(f"✅ Model loaded successfully in {self._load_time:.2f}s!")
                    
                    # Log model summary
                    logger.info(f"Model input shape: {self._model.input_shape}")
                    logger.info(f"Model output shape: {self._model.output_shape}")
                except Exception as e:
                    logger.error(f"❌ Error loading model: {e}", exc_info=True)
                    self._model = None
            else:
                logger.error("❌ Model file not found in any expected location")
                logger.error(f"Current working directory: {os.getcwd()}")
                logger.error(f"Files in current directory: {os.listdir('.')}")
        
        return self._model
    
    def get_cached_prediction(self, image_hash: str):
        """Get cached prediction if available"""
        return self._cache.get(image_hash)
    
    def cache_prediction(self, image_hash: str, prediction: Dict[str, Any]):
        """Cache prediction result"""
        # Implement simple LRU by removing oldest entries if cache is full
        if len(self._cache) >= self._cache_size_limit:
            # Remove oldest entry (first item in dict)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[image_hash] = prediction
        logger.debug(f"Cached prediction for hash: {image_hash[:8]}...")
    
    def get_model_info(self):
        """Get model loading information"""
        return {
            'model_loaded': self._model is not None,
            'model_path': self._model_path,
            'load_time_seconds': self._load_time,
            'cache_size': len(self._cache),
            'cache_limit': self._cache_size_limit
        }

# Initialize model manager
model_manager = ModelManager()

# Initialize FastAPI app
app = FastAPI(
    title="ReluRay API",
    description="""# ReluRay Medical AI API
    
## Overview
AI-powered medical image analysis API for chest X-ray pneumonia detection. 
This API provides real-time analysis of chest X-ray images using deep learning models.

## Features
- **Real-time Analysis**: Get instant pneumonia detection results
- **High Accuracy**: VGG16-based model trained on medical datasets
- **Privacy Focused**: Images processed locally, not stored on servers
- **Production Ready**: Built with FastAPI, includes monitoring and caching

## Authentication
Currently no authentication required for public endpoints.

## Rate Limiting
Default rate limit: 10 requests per minute per IP address.

## Base URL
`https://reluray.com/api`

## Support
For API support, contact: ei@nsisong.com

## Important Medical Disclaimer
⚠️ **This tool is for educational and research purposes only.**
It is not intended to replace professional medical diagnosis. 
Always consult with a qualified healthcare provider for medical decisions.
Results should not be used as the sole basis for treatment decisions.
""",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
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

# Add HTTPS redirect middleware in production
if os.environ.get('ENVIRONMENT', 'development').lower() == 'production':
    app.add_middleware(HTTPSRedirectMiddleware)

# Configure CORS
# In production, set CORS_ORIGINS to your frontend domain(s)
# Example: CORS_ORIGINS=https://your-app.vercel.app,https://www.yourdomain.com
cors_origins_env = os.environ.get('CORS_ORIGINS', '')
is_production = os.environ.get('ENVIRONMENT', 'development').lower() == 'production'

if cors_origins_env:
    # Parse comma-separated origins
    cors_origins = [origin.strip() for origin in cors_origins_env.split(',') if origin.strip()]
    allow_origins = cors_origins
    logger.info(f"CORS configured for origins: {cors_origins}")
else:
    # Default behavior: allow all in development, require explicit config in production
    if is_production:
        logger.error("CORS_ORIGINS not set in production! Defaulting to empty list for security.")
        allow_origins = []
    else:
        logger.warning("CORS_ORIGINS not set. Allowing all origins (development mode).")
        allow_origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Configuration
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
MODEL_INPUT_SIZE = (224, 224)
MODEL_VERSION = os.environ.get("MODEL_VERSION", "1.0.0")

# Pydantic models for request/response validation
class PredictRequest(BaseModel):
    """Request model for X-ray image analysis"""
    image: str = Field(
        ...,
        description="""Base64 encoded image data with data URI prefix.
        
        **Format**: `data:image/{format};base64,{base64_encoded_data}`
        
        **Supported formats**: JPEG, PNG, GIF, BMP
        
        **Maximum size**: 10MB
        
        **Example**:
        ```json
        {
            "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        }
        ```
        """,
        example="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )

class HealthResponse(BaseModel):
    """Health check response with system metrics"""
    status: str = Field(..., description="Service status: 'healthy' or 'unhealthy'", example="healthy")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded and ready", example=True)
    timestamp: str = Field(..., description="ISO 8601 timestamp of the check", example="2024-01-01T12:00:00.000000")
    version: str = Field(..., description="API version", example="1.0.0")
    model_version: str = Field(..., description="ML model version", example="1.0.0")
    uptime_seconds: float = Field(..., description="Service uptime in seconds", example=12345.67)
    memory_usage_mb: float = Field(..., description="Memory usage in megabytes", example=256.89)
    cpu_percent: float = Field(..., description="CPU usage percentage", example=12.5)

class PredictResponse(BaseModel):
    """Prediction response for X-ray analysis"""
    prediction: str = Field(
        ...,
        description="""Prediction result.
        
        **Possible values**:
        - `normal`: No signs of pneumonia detected
        - `pneumonia`: Signs of pneumonia detected
        - `error`: Analysis failed
        """,
        example="normal"
    )
    confidence: float = Field(
        ...,
        description="Confidence score between 0 and 1 (higher is more confident)",
        example=0.95,
        ge=0.0,
        le=1.0
    )
    raw_confidence: Optional[float] = Field(
        None,
        description="Raw model output confidence (if available)",
        example=0.8723,
        ge=0.0,
        le=1.0
    )
    timestamp: str = Field(..., description="ISO 8601 timestamp of the analysis", example="2024-01-01T12:00:00.000000")
    processing_time: float = Field(..., description="Processing time in seconds", example=1.23)
    model_version: str = Field(..., description="ML model version used", example="1.0.0")
    status: str = Field(..., description="Request status: 'success' or 'error'", example="success")

class ErrorResponse(BaseModel):
    """Error response for failed requests"""
    error: str = Field(..., description="Error message describing what went wrong", example="Invalid image format")
    status: str = Field(..., description="Always 'error' for error responses", example="error")

class ModelInfoResponse(BaseModel):
    """Model information and metadata"""
    model_name: str = Field(..., description="Name of the ML model", example="VGG16 Pneumonia Detector")
    architecture: str = Field(..., description="Model architecture", example="VGG16 with custom top layers")
    training_data: str = Field(..., description="Dataset used for training", example="Chest X-ray Pneumonia Dataset")
    classes: list = Field(..., description="List of classes the model can predict", example=["normal", "pneumonia"])
    input_size: str = Field(..., description="Input image dimensions", example="224x224")
    framework: str = Field(..., description="ML framework used", example="TensorFlow/Keras")
    model_version: str = Field(..., description="Model version", example="1.0.0")
    model_loaded: bool = Field(..., description="Whether model is currently loaded", example=True)
    model_input_shape: Optional[str] = Field(None, description="Detailed input shape", example="(224, 224, 3)")
    model_output_shape: Optional[str] = Field(None, description="Detailed output shape", example="(1,)")
    status: str = Field(..., description="Response status", example="success")

# Model loading with better path resolution
def find_model_file():
    """Find the model file in common locations"""
    possible_paths = [
        '../best_model.keras',  # From backend/ directory (in repository root)
        '../../best_model.keras',  # Alternative path
        'best_model.keras',      # In current directory
        './best_model.keras',   # Current directory
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'best_model.keras'),  # Absolute from backend/
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"Model file found at: {os.path.abspath(path)}")
            return path
    
    return None

# Load the trained model (using model manager for lazy loading)
model = None  # Will be loaded lazily by model_manager
model_path = None  # Will be set by model_manager
start_time = time.time()

def get_system_metrics():
    """Get system performance metrics for monitoring"""
    try:
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        uptime = time.time() - start_time
        
        return {
            'uptime_seconds': uptime,
            'memory_usage_mb': memory.used / (1024 * 1024),
            'cpu_percent': cpu_percent,
            'memory_total_mb': memory.total / (1024 * 1024),
            'memory_percent': memory.percent
        }
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        return {
            'uptime_seconds': time.time() - start_time,
            'memory_usage_mb': 0,
            'cpu_percent': 0,
            'memory_total_mb': 0,
            'memory_percent': 0
        }

def preprocess_image(image_data: str):
    """Preprocess image for model prediction"""
    try:
        # Validate input
        if not image_data or not isinstance(image_data, str):
            logger.error("Invalid image data: not a string")
            return None
        
        # Extract base64 data
        if image_data.startswith('data:image'):
            # Remove data URL prefix (e.g., "data:image/jpeg;base64,")
            image_data = image_data.split(',')[1]
        
        # Decode base64
        try:
            image_bytes = base64.b64decode(image_data, validate=True)
        except Exception as e:
            logger.error(f"Invalid base64 encoding: {e}")
            return None
        
        # Validate image size
        if len(image_bytes) > MAX_IMAGE_SIZE:
            logger.warning(f"Image too large: {len(image_bytes)} bytes (max: {MAX_IMAGE_SIZE})")
            return None
        
        # Open and validate image
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.verify()  # Verify it's a valid image
        except Exception as e:
            logger.error(f"Invalid image file: {e}")
            return None
        
        # Reopen image (verify() closes it)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        image = image.resize(MODEL_INPUT_SIZE, Image.Resampling.LANCZOS)
        
        # Convert to array and normalize
        img_array = img_to_array(image)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        logger.debug(f"Image preprocessed: shape={img_array.shape}")
        return img_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}", exc_info=True)
        return None

@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health Check Endpoint
    
    Returns the current health status of the API service along with system metrics.
    
    This endpoint is useful for:
    - Monitoring service availability
    - Checking if the ML model is loaded
    - Getting system resource usage
    - Implementing health checks in load balancers or monitoring systems
    
    **Response Codes**:
    - `200`: Service is healthy
    - `500`: Service is unhealthy (internal server error)
    
    **Example Response**:
    ```json
    {
        "status": "healthy",
        "model_loaded": true,
        "timestamp": "2024-01-01T12:00:00.000000",
        "version": "1.0.0",
        "model_version": "1.0.0",
        "uptime_seconds": 12345.67,
        "memory_usage_mb": 256.89,
        "cpu_percent": 12.5
    }
    ```
    """
    metrics = get_system_metrics()
    model_info = model_manager.get_model_info()
    
    return HealthResponse(
        status='healthy',
        model_loaded=model_info['model_loaded'],
        timestamp=datetime.now().isoformat(),
        version='1.0.0',
        model_version=MODEL_VERSION,
        uptime_seconds=round(metrics['uptime_seconds'], 2),
        memory_usage_mb=round(metrics['memory_usage_mb'], 2),
        cpu_percent=round(metrics['cpu_percent'], 2)
    )

@app.get("/api/metrics", tags=["Monitoring"])
async def get_metrics():
    """
    Detailed System Metrics
    
    Returns comprehensive system and application metrics for monitoring purposes.
    
    This endpoint provides more detailed information than the health check endpoint,
    including cache statistics, model information, and detailed system resource usage.
    
    **Use Cases**:
    - Performance monitoring and alerting
    - Capacity planning
    - Debugging performance issues
    - Monitoring cache effectiveness
    
    **Response Structure**:
    - `system`: Hardware and OS-level metrics
    - `application`: ReluRay-specific metrics including model and cache info
    
    **Example Response**:
    ```json
    {
        "status": "success",
        "timestamp": "2024-01-01T12:00:00.000000",
        "system": {
            "uptime_seconds": 12345.67,
            "memory_usage_mb": 256.89,
            "memory_total_mb": 8192.0,
            "memory_percent": 3.14,
            "cpu_percent": 12.5
        },
        "application": {
            "model_loaded": true,
            "model_path": "/home/isaac/reluray/backend/best_model.keras",
            "model_load_time": 2.34,
            "cache_size": 45,
            "cache_limit": 100,
            "version": "1.0.0",
            "model_version": "1.0.0"
        }
    }
    ```
    """
    metrics = get_system_metrics()
    model_info = model_manager.get_model_info()
    
    return {
        'status': 'success',
        'timestamp': datetime.now().isoformat(),
        'system': {
            'uptime_seconds': round(metrics['uptime_seconds'], 2),
            'memory_usage_mb': round(metrics['memory_usage_mb'], 2),
            'memory_total_mb': round(metrics['memory_total_mb'], 2),
            'memory_percent': round(metrics['memory_percent'], 2),
            'cpu_percent': round(metrics['cpu_percent'], 2)
        },
        'application': {
            'model_loaded': model_info['model_loaded'],
            'model_path': model_info['model_path'],
            'model_load_time': model_info['load_time_seconds'],
            'cache_size': model_info['cache_size'],
            'cache_limit': model_info['cache_limit'],
            'version': '1.0.0',
            'model_version': MODEL_VERSION
        }
    }

@app.post("/api/predict", response_model=PredictResponse, tags=["Prediction"], responses={
    200: {"description": "Successful prediction", "model": PredictResponse},
    400: {"description": "Bad request - invalid image or parameters", "model": ErrorResponse},
    413: {"description": "Payload too large - image exceeds 10MB limit"},
    429: {"description": "Too many requests - rate limit exceeded"},
    500: {"description": "Internal server error", "model": ErrorResponse},
    503: {"description": "Service unavailable - model not loaded", "model": ErrorResponse}
})
async def predict(request: PredictRequest):
    """
    Analyze Chest X-ray Image
    
    Analyzes a chest X-ray image for signs of pneumonia using AI.
    
    This is the main endpoint of the ReluRay API. It accepts a base64-encoded
    X-ray image and returns an analysis with confidence scores.
    
    **Medical Disclaimer**: 
    ⚠️ This tool is for educational and research purposes only.
    It is not intended to replace professional medical diagnosis.
    Always consult with a qualified healthcare provider for medical decisions.
    
    **Image Requirements**:
    - Format: JPEG, PNG, GIF, or BMP
    - Maximum size: 10MB
    - Must be a valid chest X-ray image
    - Base64 encoded with data URI prefix
    
    **Caching**: 
    Results are cached based on image hash. Identical images will return
    cached results for faster response times.
    
    **Rate Limiting**:
    Limited to 10 requests per minute per IP address.
    
    **Example Request**:
    ```json
    {
        "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    }
    ```
    
    **Example Success Response**:
    ```json
    {
        "prediction": "normal",
        "confidence": 0.95,
        "raw_confidence": 0.872,
        "timestamp": "2024-01-01T12:00:00.000000",
        "processing_time": 1.234,
        "model_version": "1.0.0",
        "status": "success"
    }
    ```
    
    **Example Error Response**:
    ```json
    {
        "error": "Invalid image format",
        "status": "error"
    }
    ```
    """
    start_time = time.time()
    
    # Get image hash for caching
    image_hash = model_manager._get_image_hash(request.image)
    
    # Check cache first
    cached_result = model_manager.get_cached_prediction(image_hash)
    if cached_result:
        logger.info(f"Cache hit for image hash: {image_hash[:8]}...")
        return PredictResponse(**cached_result)
    
    # Get model (lazy loading)
    model = model_manager.get_model()
    if model is None:
        logger.error("Prediction attempted but model is not loaded")
        raise HTTPException(
            status_code=503,
            detail='Model not loaded. Please check server logs.'
        )
    
    try:
        image_data = request.image
        
        # Preprocess image
        logger.info("Preprocessing image...")
        processed_image = preprocess_image(image_data)
        if processed_image is None:
            logger.warning("Image preprocessing failed")
            raise HTTPException(
                status_code=400,
                detail='Failed to process image. Please ensure the image is valid and under 10MB.'
            )
        
        # Make prediction
        logger.info("Running model prediction...")
        prediction_start = time.time()
        prediction = model.predict(processed_image, verbose=0)
        prediction_time = time.time() - prediction_start
        
        confidence = float(prediction[0][0])
        
        # Determine result
        if confidence > 0.5:
            result = 'Pneumonia'
            final_confidence = confidence
        else:
            result = 'Normal'
            final_confidence = 1 - confidence
        
        total_time = time.time() - start_time
        
        # craeted a better response method
        response_data = {
            'prediction': result,
            'confidence': round(final_confidence, 3),
            'raw_confidence': round(confidence, 3),
            'timestamp': datetime.now().isoformat(),
            'processing_time': round(total_time, 3),
            'model_version': MODEL_VERSION,
            'status': 'success'
        }
        
        # Cache the result
        model_manager.cache_prediction(image_hash, response_data)
        
        logger.info(f"Prediction completed: {result} (confidence: {final_confidence:.3f}, time: {total_time:.2f}s)")
        
        return PredictResponse(**response_data)
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Value error in prediction: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail='Invalid request data')
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail='Failed to analyze image. Please try again.')

@app.get("/api/info", response_model=ModelInfoResponse, tags=["Info"])
async def model_info():
    """
    Get Model Information
    
    Returns detailed information about the ML model used for pneumonia detection.
    
    This endpoint provides metadata about the model architecture, training data,
    capabilities, and current status. Useful for developers integrating with the API
    or for monitoring model deployment.
    
    **Information Included**:
    - Model architecture and framework
    - Training dataset details
    - Input/output specifications
    - Current loading status
    - Cache configuration
    
    **Example Response**:
    ```json
    {
        "model_name": "VGG16 Transfer Learning",
        "architecture": "Convolutional Neural Network",
        "training_data": "Chest X-ray Pneumonia Dataset",
        "classes": ["Normal", "Pneumonia"],
        "input_size": "224x224x3",
        "framework": "TensorFlow/Keras",
        "model_version": "1.0.0",
        "model_loaded": true,
        "model_input_shape": "(224, 224, 3)",
        "model_output_shape": "(1,)",
        "status": "success"
    }
    ```
    
    **Note**: The response includes additional caching-related fields that are
    not part of the formal ModelInfoResponse schema but provide useful
    debugging information.
    """
    model = model_manager.get_model()
    model_info_data = model_manager.get_model_info()
    
    info = {
        'model_name': 'VGG16 Transfer Learning',
        'architecture': 'Convolutional Neural Network',
        'training_data': 'Chest X-ray Pneumonia Dataset',
        'classes': ['Normal', 'Pneumonia'],
        'input_size': '224x224x3',
        'framework': 'TensorFlow/Keras',
        'model_version': MODEL_VERSION,
        'model_loaded': model is not None,
        'status': 'success'
    }
    
    if model is not None:
        info['model_input_shape'] = str(model.input_shape)
        info['model_output_shape'] = str(model.output_shape)
    
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
