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
    description="AI-powered medical image analysis API",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
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
    image: str = Field(..., description="Base64 encoded image data")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str
    version: str
    model_version: str
    uptime_seconds: float
    memory_usage_mb: float
    cpu_percent: float

class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    raw_confidence: Optional[float] = None
    timestamp: str
    processing_time: float
    model_version: str
    status: str

class ErrorResponse(BaseModel):
    error: str
    status: str

class ModelInfoResponse(BaseModel):
    model_name: str
    architecture: str
    training_data: str
    classes: list
    input_size: str
    framework: str
    model_version: str
    model_loaded: bool
    model_input_shape: Optional[str] = None
    model_output_shape: Optional[str] = None
    status: str

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
    """Enhanced health check endpoint with monitoring metrics"""
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
    """Detailed system metrics for monitoring"""
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

@app.post("/api/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    """predict pneumonia from uploaded image with caching"""
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
            result = 'pneumonia'
            final_confidence = confidence
        else:
            result = 'normal'
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
    """Get model information with caching details"""
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
