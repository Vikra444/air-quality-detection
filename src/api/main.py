"""
FastAPI main application for AirGuard.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import traceback
import numpy as np
import time
import logging
from ..config.settings import settings
from ..utils.logger import get_logger, performance_monitor
from ..data.models import (
    AirQualityData, PredictionRequest, PredictionResult,
    HealthAdvisory, HealthRiskLevel, WhatIfRequest,
    UserCreate, UserLogin, User
)
from ..data.api_clients.unified_client import UnifiedAirQualityClient
from ..data.storage import storage
from ..data.cache import cache
from ..data.preprocessing import AirQualityPreprocessor
from ..scheduler.scheduler import scheduler
from ..scheduler.jobs import fetch_air_quality_data
from ..notifications.notification_service import alert_manager
from ..utils.monitoring import metrics_collector, system_monitor, get_prometheus_metrics
from ..security.auth import security_manager, get_current_user, verify_access_token
from ..security.rate_limiting import rate_limiter, rate_limit_dependency, get_client_ip

logger = get_logger("api")

# Global variables
unified_client = None
preprocessor = None
security = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global unified_client, preprocessor
    
    # Startup
    logger.info("Starting AirGuard API server...")
    
    try:
        # Initialize components
        await storage.initialize()
        await cache.initialize()
        
        unified_client = UnifiedAirQualityClient()
        preprocessor = AirQualityPreprocessor()
        
        # Start scheduler
        scheduler.start()
        scheduler.add_interval_job(
            fetch_air_quality_data,
            seconds=settings.data_fetch_interval,
            id="fetch_air_quality"
        )
        
        # Start system monitoring
        app.state.monitoring_task = asyncio.create_task(system_monitor.start_monitoring())
        app.state.startup_time = time.time()
        
        logger.success("AirGuard API server started successfully")
    
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}", error=str(e))
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down AirGuard API server...")
    # Stop monitoring
    system_monitor.stop_monitoring()
    if hasattr(app.state, 'monitoring_task'):
        app.state.monitoring_task.cancel()
    
    scheduler.stop()
    await unified_client.close()
    await cache.close()
    await storage.close()
    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="AirGuard API",
    description="Real-time Air Quality & Health Risk Prediction & Advisory System",
    version="2.0.0",  # Updated version
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency functions
async def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Verify API token using enhanced security system."""
    if not credentials:
        # Allow demo access for unprotected endpoints
        return None
    
    token = credentials.credentials
    
    # First try JWT token
    jwt_payload = verify_access_token(token)  # This should be the standalone function
    if jwt_payload:
        return jwt_payload
    
    # Then try API key
    api_key_data = security_manager.verify_api_key(token)
    if api_key_data:
        return api_key_data
    
    # Fall back to demo token for backward compatibility
    if token == "demo-token":
        return {"sub": "demo", "roles": ["user"]}
    
    raise HTTPException(status_code=401, detail="Invalid token")


# New authentication dependency with role checking
async def get_current_active_user(token_data: dict = Depends(get_current_user)):
    """Get current active user with role validation."""
    if not token_data:
        raise HTTPException(status_code=401, detail="Invalid authentication")
    
    username = token_data.get("sub") or token_data.get("username")
    if not username:
        raise HTTPException(status_code=401, detail="Invalid token data")
    
    # In production, fetch user from database
    user = security_manager.users.get(username)
    if not user or not user.get("is_active"):
        raise HTTPException(status_code=401, detail="Inactive user")
    
    return user


def require_role(required_role: str):
    """Dependency to require a specific role."""
    async def role_checker(current_user: dict = Depends(get_current_active_user)):
        if not security_manager.has_role(current_user, required_role):
            raise HTTPException(
                status_code=403,
                detail=f"Access denied. Required role: {required_role}"
            )
        return current_user
    return role_checker


# Enhanced middleware for logging requests and monitoring performance
@app.middleware("http")
async def performance_monitoring_middleware(request: Request, call_next):
    """Comprehensive middleware for request logging and performance monitoring."""
    # Apply rate limiting
    client_ip = get_client_ip(request)
    if not rate_limiter.is_allowed(client_ip):
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Please try again later."}
        )
    
    # Increment request counter
    metrics_collector.increment_counter("http_requests_total", labels={"method": request.method})
    
    start_time = time.time()
    
    # Log incoming request
    logger.info(
        f"Incoming request: {request.method} {request.url.path}",
        method=request.method,
        path=request.url.path,
        query_params=dict(request.query_params),
        client=request.client.host if request.client else "unknown"
    )
    
    try:
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Record metrics
        performance_monitor.record_api_call(
            request.url.path,
            duration,
            response.status_code
        )
        
        # Record with new metrics collector
        metrics_collector.record_timer(
            "http_request_duration_seconds", 
            duration, 
            labels={"method": request.method, "path": request.url.path, "status": str(response.status_code)}
        )
        
        # Log response
        logger.info(
            f"Request completed: {request.method} {request.url.path} - Status: {response.status_code} - Time: {duration:.3f}s",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration=duration,
            client=request.client.host if request.client else "unknown"
        )
        
        # Add performance headers
        response.headers["X-Response-Time"] = str(duration)
        response.headers["X-Process-Time"] = str(time.time() - start_time)
        
        return response
        
    except Exception as e:
        # Calculate duration even for errors
        duration = time.time() - start_time
        
        # Record error metrics
        performance_monitor.record_api_call(
            request.url.path,
            duration,
            500  # Internal server error
        )
        
        metrics_collector.record_timer(
            "http_request_duration_seconds", 
            duration, 
            labels={"method": request.method, "path": request.url.path, "status": "500"}
        )
        metrics_collector.increment_counter("http_errors_total", labels={"method": request.method, "path": request.url.path})
        
        # Log error
        logger.error(
            f"Request failed: {request.method} {request.url.path} - Error: {str(e)} - Time: {duration:.3f}s",
            method=request.method,
            path=request.url.path,
            error=str(e),
            duration=duration,
            client=request.client.host if request.client else "unknown",
            exc_info=True
        )
        
        # Re-raise the exception
        raise e


# Enhanced error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    logger.warning(
        f"HTTP Exception: {exc.status_code} - {exc.detail}",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat(),
            "path": request.url.path
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    logger.error(
        f"Validation Error: {exc.errors()}",
        errors=exc.errors(),
        path=request.url.path
    )
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "details": exc.errors(),
            "status_code": 422,
            "timestamp": datetime.now().isoformat(),
            "path": request.url.path
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(
        f"Unhandled exception: {exc}",
        error=str(exc),
        path=request.url.path,
        exc_info=True
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat(),
            "path": request.url.path
        }
    )


# Health check endpoint with enhanced metrics
@app.get("/health")
async def health_check(token: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Enhanced health check endpoint with optional authentication."""
    try:
        cache_stats = await cache.get_stats()
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0",
            "components": {
                "database": storage._initialized if storage else False,
                "cache": cache_stats.get("status") == "active",
                "api_client": unified_client is not None,
                "preprocessor": preprocessor is not None
            },
            "cache_stats": cache_stats,
            "authenticated": token is not None
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}", error=str(e))
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


# New metrics endpoint for monitoring
@app.get("/metrics")
async def get_metrics(format: str = "json"):
    """Get application metrics in various formats."""
    if format.lower() == "prometheus":
        # Return Prometheus format
        prometheus_metrics = get_prometheus_metrics()
        return Response(content=prometheus_metrics, media_type="text/plain")
    else:
        # Return JSON format
        metrics = metrics_collector.get_metrics()
        # Add performance monitor metrics
        perf_metrics = performance_monitor.get_metrics()
        metrics["performance_monitor"] = perf_metrics
        return metrics


# Enhanced dependency to get current user from database
async def get_current_db_user(token_data: dict = Depends(get_current_user)):
    """Get current user from database using JWT token."""
    try:
        user_id = int(token_data.get("sub"))
        user = await security_manager.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        return user
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    except Exception as e:
        logger.error(f"Error getting current user: {e}", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )


# Enhanced Air Quality Data Endpoints with proper authentication
@app.get("/api/v1/air-quality/current")
async def get_current_air_quality(
    location_id: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    include_forecast: bool = False,
    current_user: User = Depends(get_current_db_user)  # Require authentication
):
    """Get current air quality data for a location."""
    try:
        if not unified_client:
            raise HTTPException(status_code=503, detail="API client not available")
        
        # Validate coordinates
        if not latitude or not longitude:
            raise HTTPException(status_code=400, detail="latitude and longitude are required")
        
        # Check cache first
        cached_data = await cache.get_air_quality(location_id or "unknown", latitude, longitude)
        if cached_data:
            logger.debug("Returning cached data", location_id=location_id)
            # If forecast requested, add it to the response
            if include_forecast:
                from ..ml.predictors import AirQualityPredictor
                predictor = AirQualityPredictor()
                forecast_request = PredictionRequest(
                    location_id=location_id or f"Location_{latitude}_{longitude}",
                    latitude=latitude,
                    longitude=longitude,
                    prediction_horizon=24
                )
                try:
                    forecast = await predictor.generate_predictions(forecast_request)
                    cached_data_dict = cached_data.dict() if hasattr(cached_data, 'dict') else cached_data
                    cached_data_dict["forecast"] = forecast.dict() if hasattr(forecast, 'dict') else forecast
                    return cached_data_dict
                except Exception as e:
                    logger.warning(f"Could not generate forecast: {e}")
            return cached_data
        
        # Fetch from API with timeout handling
        try:
            raw_data = await unified_client.fetch_air_quality(latitude, longitude)
        except Exception as e:
            logger.warning(f"API fetch failed, using demo data: {e}", error=str(e))
            # Fallback to demo data if API fails
            from ..data.demo_data import generate_demo_air_quality_data
            raw_data = generate_demo_air_quality_data(latitude, longitude, location_id)
        
        if not raw_data:
            raise HTTPException(status_code=503, detail="Failed to fetch air quality data")
        
        # Set location_id in raw_data before preprocessing
        raw_data["location_id"] = location_id or f"Location_{latitude}_{longitude}"
        
        # Preprocess and normalize
        data = preprocessor.normalize_to_model(raw_data)
        
        # Store in database and cache
        await storage.store_air_quality_data(data)
        await cache.set_air_quality(data)
        
        # If forecast requested, add it to the response
        if include_forecast:
            from ..ml.predictors import AirQualityPredictor
            predictor = AirQualityPredictor()
            forecast_request = PredictionRequest(
                location_id=location_id or f"Location_{latitude}_{longitude}",
                latitude=latitude,
                longitude=longitude,
                prediction_horizon=24
            )
            try:
                forecast = await predictor.generate_predictions(forecast_request)
                data_dict = data.dict() if hasattr(data, 'dict') else data
                data_dict["forecast"] = forecast.dict() if hasattr(forecast, 'dict') else forecast
                return data_dict
            except Exception as e:
                logger.warning(f"Could not generate forecast: {e}")
        
        return data
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting current air quality: {e}", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/air-quality/historical")
async def get_historical_air_quality(
    location_id: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 100,
    token: Optional[HTTPAuthorizationCredentials] = Depends(verify_token)
):
    """Get historical air quality data."""
    try:
        if not storage._initialized:
            raise HTTPException(status_code=503, detail="Database not available")
        
        # Default to last 24 hours if no time range specified
        if not end_time:
            end_time = datetime.now()
        if not start_time:
            start_time = end_time - timedelta(hours=24)
        
        historical_data = await storage.get_historical_data(
            location_id, start_time, end_time, limit
        )
        
        return {
            "location_id": location_id,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "count": len(historical_data),
            "data": [item.dict() if hasattr(item, 'dict') else item for item in historical_data]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting historical air quality: {e}", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/air-quality/data")
async def ingest_air_quality_data(
    data: AirQualityData,
    background_tasks: BackgroundTasks,
    token: Optional[HTTPAuthorizationCredentials] = Depends(require_role("admin"))
):
    """Ingest new air quality data (admin only)."""
    try:
        # Store data in background
        background_tasks.add_task(store_data_background, data)
        
        return {
            "status": "success",
            "message": "Data ingested successfully",
            "location_id": data.location_id,
            "timestamp": data.timestamp.isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error ingesting air quality data: {e}", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


async def store_data_background(data: AirQualityData):
    """Background task to store data."""
    try:
        await storage.store_air_quality_data(data)
        await cache.set_air_quality(data)
        logger.info(f"Stored air quality data for {data.location_id}")
    except Exception as e:
        logger.error(f"Error storing data in background: {e}", error=str(e))


# Enhanced Prediction Endpoints
@app.post("/api/v1/predictions/generate", response_model=PredictionResult)
async def generate_predictions(
    request: PredictionRequest,
    token: Optional[HTTPAuthorizationCredentials] = Depends(verify_token)
) -> PredictionResult:
    """Generate air quality predictions."""
    try:
        from ..ml.predictors import AirQualityPredictor
        
        predictor = AirQualityPredictor()
        result = await predictor.generate_predictions(request)
        
        # Store prediction
        await storage.store_prediction(result)
        
        return result
    
    except Exception as e:
        logger.error(f"Error generating predictions: {e}", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/predictions/explain")
async def explain_predictions(
    request: PredictionRequest,
    method: str = "shap",
    token: Optional[HTTPAuthorizationCredentials] = Depends(verify_token)
):
    """Generate explainability analysis for predictions using SHAP/LIME."""
    try:
        from ..ml.predictors import AirQualityPredictor
        from ..ml.explainability import ExplainerFactory
        import numpy as np
        
        # Generate prediction first
        predictor = AirQualityPredictor()
        result = await predictor.generate_predictions(request)
        
        # Get model (use ensemble if available)
        model = predictor.models.get("advanced_ensemble")
        if not model:
            # Fallback to first available model
            model = list(predictor.models.values())[0] if predictor.models else None
        
        if not model or not hasattr(model, 'model'):
            raise HTTPException(status_code=400, detail="No trained model available for explanation")
        
        # Prepare input data
        # Get current air quality data
        raw_data = await unified_client.fetch_air_quality(request.latitude, request.longitude)
        if not raw_data:
            raise HTTPException(status_code=503, detail="Failed to fetch air quality data")
        
        # Extract features (simplified - in production, use proper feature extraction)
        features = np.array([[
            raw_data.get("aqi", 0),
            raw_data.get("pm25", 0),
            raw_data.get("pm10", 0),
            raw_data.get("no2", 0),
            raw_data.get("co", 0),
            raw_data.get("o3", 0),
            raw_data.get("so2", 0),
            raw_data.get("temperature", 20),
            raw_data.get("humidity", 50),
            raw_data.get("wind_speed", 3),
            raw_data.get("wind_direction", 0),
            raw_data.get("pressure", 1013)
        ]])
        
        # Get training data for LIME (simplified - use recent historical data)
        historical_data = await storage.get_historical_data(
            request.location_id,
            datetime.now() - timedelta(days=7),
            datetime.now(),
            limit=100
        )
        
        training_data = None
        if historical_data:
            # Convert to numpy array
            training_samples = []
            for record in historical_data:
                training_samples.append([
                    record.aqi, record.pm25, record.pm10, record.no2, record.co,
                    record.o3, record.so2, record.temperature or 20,
                    record.humidity or 50, record.wind_speed or 3,
                    record.wind_direction or 0, record.pressure or 1013
                ])
            if training_samples:
                training_data = np.array(training_samples)
        
        # Generate explanations
        explanations = ExplainerFactory.explain_prediction(
            model.model if hasattr(model, 'model') else model,
            features[0],
            training_data,
            method=method
        )
        
        return {
            "prediction": result.dict() if hasattr(result, 'dict') else result,
            "explanations": explanations,
            "method": method,
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error explaining predictions: {e}", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Enhanced Simulation Endpoints
@app.post("/api/v1/simulations/what-if")
async def what_if_simulation(
    request: WhatIfRequest,
    token: Optional[HTTPAuthorizationCredentials] = Depends(verify_token)
):
    """Run what-if scenario simulation."""
    try:
        from ..ml.simulation import WhatIfSimulator
        from ..ml.predictors import AirQualityPredictor
        
        latitude = request.latitude
        longitude = request.longitude
        scenario = request.scenario
        location_id = request.location_id
        
        if not latitude or not longitude:
            raise HTTPException(status_code=400, detail="latitude and longitude are required")
        
        # Get current data
        raw_data = await unified_client.fetch_air_quality(latitude, longitude)
        if not raw_data:
            raise HTTPException(status_code=503, detail="Failed to fetch air quality data")
        
        # Initialize simulator
        predictor = AirQualityPredictor()
        simulator = WhatIfSimulator(predictor)
        
        # Run simulation
        result = simulator.simulate(
            raw_data,
            scenario,
            prediction_horizon=24
        )
        
        return {
            "location_id": location_id or f"Location_{latitude}_{longitude}",
            "latitude": latitude,
            "longitude": longitude,
            "scenario": scenario,
            "simulation_result": result,
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running simulation: {e}", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Enhanced Geospatial Endpoints
@app.get("/api/v1/geospatial/morphology")
async def get_geospatial_morphology(
    latitude: float,
    longitude: float,
    token: Optional[HTTPAuthorizationCredentials] = Depends(verify_token)
):
    """Get geospatial morphology classification (urban vs peri-urban)."""
    try:
        from ..data.geospatial import GeospatialMorphologyAnalyzer
        
        analyzer = GeospatialMorphologyAnalyzer()
        result = analyzer.classify_area(latitude, longitude)
        
        return {
            "latitude": latitude,
            "longitude": longitude,
            "morphology": result,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error classifying morphology: {e}", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Enhanced AI Insights Endpoints
@app.get("/api/v1/insights/generate")
async def generate_ai_insights(
    latitude: float,
    longitude: float,
    location_id: Optional[str] = None,
    days: int = 7,
    token: Optional[HTTPAuthorizationCredentials] = Depends(verify_token)
):
    """Generate comprehensive AI insights."""
    try:
        from ..ai_insights import InsightsGenerator
        
        # Get current data
        raw_data = await unified_client.fetch_air_quality(latitude, longitude)
        if not raw_data:
            raise HTTPException(status_code=503, detail="Failed to fetch air quality data")
        
        # Get historical data
        loc_id = location_id or f"Location_{latitude}_{longitude}"
        historical_data = await storage.get_historical_data(
            loc_id,
            datetime.now() - timedelta(days=days),
            datetime.now(),
            limit=1000  # Increased limit for better insights
        )
        
        # Get previous data for comparison
        previous_data = None
        if historical_data and len(historical_data) > 0:
            previous_data = historical_data[0]
        
        # Generate insights
        generator = InsightsGenerator()
        insights = generator.generate_insights(
            raw_data,
            historical_data,
            weather_data=raw_data,
            previous_data=previous_data.dict() if previous_data else None
        )
        
        return {
            "location_id": loc_id,
            "latitude": latitude,
            "longitude": longitude,
            "insights": insights,
            "timestamp": datetime.now().isoformat(),
            "historical_days": days
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating insights: {e}", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Enhanced Health Advisory Endpoints
@app.get("/api/v1/advisory/health")
async def get_health_advisory(
    aqi: float,
    current_user: User = Depends(get_current_db_user)
):
    """Get personalized health advisory based on AQI and user profile."""
    try:
        # Generate personalized advisory
        advisory = generate_personalized_advisory(current_user, aqi)
        
        return {
            "risk_level": advisory["risk_level"],
            "message": advisory["message"],
            "recommendations": advisory["recommendations"],
            "factors": advisory["factors"],
            "aqi": aqi,
            "aqi_category": get_aqi_category(aqi)
        }
    except Exception as e:
        logger.error(f"Error generating health advisory: {e}", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to generate health advisory")


# New endpoint for bulk data processing (admin only)
@app.post("/api/v1/bulk/process")
async def process_bulk_data(
    locations: List[Dict[str, Any]],
    token: Optional[HTTPAuthorizationCredentials] = Depends(require_role("admin"))
):
    """Process air quality data for multiple locations (admin only)."""
    try:
        results = []
        errors = []
        
        for loc_data in locations:
            try:
                latitude = loc_data.get("latitude")
                longitude = loc_data.get("longitude")
                location_id = loc_data.get("location_id")
                
                if not latitude or not longitude:
                    errors.append(f"Invalid data for location {location_id}: missing coordinates")
                    continue
                
                # Fetch data
                raw_data = await unified_client.fetch_air_quality(latitude, longitude)
                if not raw_data:
                    errors.append(f"Failed to fetch data for {location_id}")
                    continue
                
                # Process data
                raw_data["location_id"] = location_id or f"Location_{latitude}_{longitude}"
                data = preprocessor.normalize_to_model(raw_data)
                
                # Store data
                await storage.store_air_quality_data(data)
                await cache.set_air_quality(data)
                
                results.append({
                    "location_id": data.location_id,
                    "status": "success",
                    "aqi": data.aqi
                })
                
            except Exception as e:
                errors.append(f"Error processing {loc_data.get('location_id', 'unknown')}: {str(e)}")
        
        return {
            "processed": len(results),
            "errors": len(errors),
            "results": results,
            "error_details": errors if errors else None
        }
    
    except Exception as e:
        logger.error(f"Error in bulk processing: {e}", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# New endpoint for trend analysis
@app.get("/api/v1/analysis/trends")
async def get_trend_analysis(
    location_id: str,
    days: int = 30,
    token: Optional[HTTPAuthorizationCredentials] = Depends(verify_token)
):
    """Get trend analysis for a location."""
    try:
        # Get historical data
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        historical_data = await storage.get_historical_data(
            location_id, start_time, end_time, limit=1000
        )
        
        if not historical_data:
            raise HTTPException(status_code=404, detail="No historical data found")
        
        # Perform trend analysis
        aqi_values = [record.aqi for record in historical_data]
        timestamps = [record.timestamp for record in historical_data]
        
        # Calculate trends
        if len(aqi_values) > 1:
            # Simple linear trend
            x = list(range(len(aqi_values)))
            slope = np.polyfit(x, aqi_values, 1)[0] if len(aqi_values) > 1 else 0
            
            trend_direction = "increasing" if slope > 0.1 else "decreasing" if slope < -0.1 else "stable"
            trend_magnitude = abs(slope)
        else:
            trend_direction = "insufficient_data"
            trend_magnitude = 0
        
        # Calculate statistics
        stats = {
            "mean_aqi": np.mean(aqi_values),
            "median_aqi": np.median(aqi_values),
            "std_aqi": np.std(aqi_values),
            "min_aqi": np.min(aqi_values),
            "max_aqi": np.max(aqi_values),
            "trend_direction": trend_direction,
            "trend_magnitude": float(trend_magnitude)
        }
        
        return {
            "location_id": location_id,
            "period_days": days,
            "data_points": len(historical_data),
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in trend analysis: {e}", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket Endpoints
active_connections: List[WebSocket] = []

@app.websocket("/ws/air-quality")
async def websocket_air_quality(websocket: WebSocket):
    """WebSocket endpoint for real-time air quality updates."""
    await websocket.accept()
    active_connections.append(websocket)
    logger.info(f"WebSocket connection established. Total connections: {len(active_connections)}")
    
    try:
        while True:
            # Wait for client message (could be location request)
            data = await websocket.receive_json()
            
            if data.get("type") == "subscribe":
                # Client wants to subscribe to updates for a location
                latitude = data.get("latitude")
                longitude = data.get("longitude")
                location_id = data.get("location_id")
                
                if latitude and longitude:
                    # Fetch current data and send
                    try:
                        raw_data = await unified_client.fetch_air_quality(latitude, longitude)
                        if raw_data:
                            await websocket.send_json({
                                "type": "air_quality_update",
                                "data": raw_data,
                                "timestamp": datetime.now().isoformat()
                            })
                    except Exception as e:
                        await websocket.send_json({
                            "type": "error",
                            "message": str(e)
                        })
            
            elif data.get("type") == "ping":
                # Heartbeat
                await websocket.send_json({"type": "pong"})
    
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Remaining connections: {len(active_connections)}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", error=str(e))
        if websocket in active_connections:
            active_connections.remove(websocket)


async def broadcast_air_quality_update(data: Dict[str, Any]):
    """Broadcast air quality update to all connected WebSocket clients."""
    if not active_connections:
        return
    
    message = {
        "type": "air_quality_update",
        "data": data,
        "timestamp": datetime.now().isoformat()
    }
    
    disconnected = []
    for connection in active_connections:
        try:
            await connection.send_json(message)
        except Exception as e:
            logger.warning(f"Failed to send WebSocket message: {e}")
            disconnected.append(connection)
    
    # Remove disconnected connections
    for conn in disconnected:
        if conn in active_connections:
            active_connections.remove(conn)


# System Endpoints (admin only)
@app.get("/api/v1/system/health")
async def get_system_health(token: Optional[HTTPAuthorizationCredentials] = Depends(require_role("admin"))):
    """Get detailed system health metrics (admin only)."""
    try:
        api_health = unified_client.get_health_metrics() if unified_client else {}
        cache_stats = await cache.get_stats()
        
        return {
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "api_clients": api_health,
            "cache": cache_stats,
            "database": {
                "status": "connected" if storage._initialized else "disconnected"
            }
        }
    except Exception as e:
        logger.error(f"Error getting system health: {e}", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# New endpoint for system metrics (admin only)
@app.get("/api/v1/system/metrics")
async def get_system_metrics(token: Optional[HTTPAuthorizationCredentials] = Depends(require_role("admin"))):
    """Get detailed system performance metrics (admin only)."""
    try:
        # Get API client metrics
        api_metrics = unified_client.get_health_metrics() if unified_client else {}
        
        # Get cache metrics
        cache_stats = await cache.get_stats()
        
        # Get database metrics
        db_metrics = {
            "status": "connected" if storage._initialized else "disconnected"
        }
        
        # Get system resource usage (simplified)
        import psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        system_metrics = {
            "cpu_usage": cpu_percent,
            "memory_usage": memory.percent,
            "memory_available": memory.available / (1024 * 1024),  # MB
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "api_metrics": api_metrics,
            "cache_metrics": cache_stats,
            "database_metrics": db_metrics,
            "system_metrics": system_metrics
        }
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# New endpoint for subscribing to notifications
@app.post("/api/v1/notifications/subscribe")
async def subscribe_to_notifications(
    subscription: Dict[str, Any],
    token: Optional[HTTPAuthorizationCredentials] = Depends(verify_token)
):
    """Subscribe to air quality notifications."""
    try:
        # In a real implementation, you would store subscriptions in a database
        # For now, we'll just log the subscription
        logger.info(f"New notification subscription: {subscription}")
        
        return {
            "status": "success",
            "message": "Subscribed to notifications",
            "subscription_id": "sub_" + str(hash(str(subscription)))[:8]
        }
    except Exception as e:
        logger.error(f"Error subscribing to notifications: {e}", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# New endpoint for sending test alerts (admin only)
@app.post("/api/v1/notifications/test-alert")
async def send_test_alert(
    alert_data: Dict[str, Any],
    token: Optional[HTTPAuthorizationCredentials] = Depends(require_role("admin"))
):
    """Send a test alert (admin only)."""
    try:
        recipients = alert_data.get("recipients", [])
        alert_type = alert_data.get("alert_type", "TEST")
        message = alert_data.get("message", "Test alert from AirGuard")
        data = alert_data.get("data", {})
        
        if not recipients:
            raise HTTPException(status_code=400, detail="Recipients are required")
        
        # Send test alert
        await alert_manager.notification_service.send_alert(
            recipients, alert_type, message, data
        )
        
        return {
            "status": "success",
            "message": "Test alert sent",
            "recipients": len(recipients)
        }
    except Exception as e:
        logger.error(f"Error sending test alert: {e}", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# New endpoint for data quality reports
@app.get("/api/v1/data-quality/report")
async def get_data_quality_report(
    location_id: str,
    days: int = 7,
    token: Optional[HTTPAuthorizationCredentials] = Depends(verify_token)
):
    """Get data quality report for a location."""
    try:
        from ..data.quality_assurance import quality_assurance
        
        # Get historical data
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        historical_data = await storage.get_historical_data(
            location_id, start_time, end_time, limit=1000
        )
        
        if not historical_data:
            raise HTTPException(status_code=404, detail="No historical data found")
        
        # Generate quality report
        report = quality_assurance.generate_quality_report(historical_data)
        
        return {
            "location_id": location_id,
            "period_days": days,
            "report": report,
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating data quality report: {e}", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# New endpoint for anomaly detection
@app.get("/api/v1/data-quality/anomalies")
async def detect_anomalies(
    location_id: str,
    days: int = 7,
    token: Optional[HTTPAuthorizationCredentials] = Depends(verify_token)
):
    """Detect anomalies in historical data."""
    try:
        from ..data.quality_assurance import quality_assurance
        
        # Get historical data
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        historical_data = await storage.get_historical_data(
            location_id, start_time, end_time, limit=1000
        )
        
        if not historical_data:
            raise HTTPException(status_code=404, detail="No historical data found")
        
        # Detect anomalies
        anomalies = quality_assurance.detect_anomalies(historical_data)
        
        return {
            "location_id": location_id,
            "period_days": days,
            "anomalies_count": len(anomalies),
            "anomalies": anomalies,
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting anomalies: {e}", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# New monitoring dashboard endpoint (admin only)
@app.get("/api/v1/monitoring/dashboard")
async def get_monitoring_dashboard(token: Optional[HTTPAuthorizationCredentials] = Depends(require_role("admin"))):
    """Get monitoring dashboard data with comprehensive metrics (admin only)."""
    try:
        # Get all metrics
        metrics = metrics_collector.get_metrics()
        perf_metrics = performance_monitor.get_metrics()
        
        # Get system metrics
        system_metrics = system_monitor.metrics_collector.get_metrics()
        
        # Compile dashboard data
        dashboard_data = {
            "timestamp": datetime.now().isoformat(),
            "application_metrics": {
                "total_requests": system_metrics["counters"].get("http_requests_total", 0),
                "total_errors": system_metrics["counters"].get("http_errors_total", 0),
                "active_websocket_connections": len(active_connections),
                "uptime_seconds": time.time() - getattr(app.state, "startup_time", time.time())
            },
            "api_performance": {},
            "model_performance": {},
            "system_resources": {},
            "recent_metrics": {}
        }
        
        # Process API performance metrics
        for endpoint, data in perf_metrics.items():
            if "call_count" in data:
                dashboard_data["api_performance"][endpoint] = {
                    "call_count": data["call_count"],
                    "avg_duration": round(data["avg_duration"], 3),
                    "status_codes": data.get("status_codes", {}),
                    "error_rate": round(
                        (data["status_codes"].get(500, 0) / max(data["call_count"], 1)) * 100, 2
                    ) if data["call_count"] > 0 else 0
                }
        
        # Process model performance metrics
        for model_name, data in perf_metrics.items():
            if "prediction_count" in data:
                dashboard_data["model_performance"][model_name] = {
                    "prediction_count": data["prediction_count"],
                    "avg_duration": round(data["avg_duration"], 3),
                    "avg_accuracy": round(sum(data.get("accuracies", [0])) / max(len(data.get("accuracies", [0])), 1), 3)
                }
        
        # Get system resource metrics
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            dashboard_data["system_resources"] = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_mb": round(memory.available / (1024 * 1024), 2),
                "disk_percent": round((disk.used / disk.total) * 100, 2),
                "disk_free_gb": round(disk.free / (1024 * 1024 * 1024), 2)
            }
        except Exception as e:
            logger.warning(f"Could not collect system metrics: {e}")
        
        # Add recent metrics samples
        dashboard_data["recent_metrics"] = {
            "counters": dict(list(system_metrics["counters"].items())[-10:]) if system_metrics["counters"] else {},
            "gauges": dict(list(system_metrics["gauges"].items())[-10:]) if system_metrics["gauges"] else {},
        }
        
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Error generating monitoring dashboard: {e}", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# New authentication endpoints
@app.post("/api/v1/auth/register", response_model=User)
async def register_user(user_data: UserCreate):
    """Register a new user."""
    try:
        # Validate that passwords match
        if user_data.password != user_data.confirm_password:
            raise HTTPException(
                status_code=400,
                detail="Passwords do not match"
            )
        
        # Create user
        user = await security_manager.create_user(user_data)
        return user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering user: {e}", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/v1/auth/login")
async def login_user(credentials: UserLogin):
    """Login user and return access token."""
    try:
        user = await security_manager.authenticate_user(
            credentials.identifier, 
            credentials.password
        )
        
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid mobile/email or password"
            )
        
        # Create access token
        access_token = security_manager.create_access_token_for_user(user)
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "id": user["id"],
                "full_name": user["full_name"],
                "mobile": user["username"]  # This is actually the mobile
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error logging in user: {e}", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/auth/me")
async def get_current_user_info(current_user: dict = Depends(get_current_active_user)):
    """Get current user information."""
    return {
        "username": current_user["username"],
        "roles": current_user["roles"],
        "created_at": current_user.get("created_at")
    }


@app.get("/api/v1/air-quality/hotspots")
async def get_air_quality_hotspots(
    lat: float,
    lon: float,
    radius: float = 10.0,  # kilometers
    current_user: User = Depends(get_current_db_user)
):
    """Get air quality hotspots around a location."""
    try:
        hotspots = []
        
        # Generate sample hotspots around the user's location
        # In a real implementation, this would fetch actual data from the database
        # or use a more sophisticated algorithm
        
        # Center point
        hotspots.append({
            "lat": lat,
            "lon": lon,
            "aqi": 120  # Sample AQI
        })
        
        # Surrounding points (simplified for demonstration)
        directions = [
            (0.01, 0),    # North
            (-0.01, 0),   # South
            (0, 0.01),    # East
            (0, -0.01),   # West
            (0.007, 0.007),  # Northeast
            (-0.007, 0.007), # Southeast
            (0.007, -0.007), # Northwest
            (-0.007, -0.007) # Southwest
        ]
        
        base_aqi = 120
        for i, (dlat, dlon) in enumerate(directions):
            # Vary AQI values for different points
            aqi_variation = (i * 15) % 100
            hotspot_aqi = max(50, min(300, base_aqi + aqi_variation - 50))
            
            hotspots.append({
                "lat": lat + dlat,
                "lon": lon + dlon,
                "aqi": hotspot_aqi
            })
        
        return hotspots
        
    except Exception as e:
        logger.error(f"Error fetching air quality hotspots: {e}", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to fetch air quality hotspots")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )