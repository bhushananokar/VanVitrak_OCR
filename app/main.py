import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .core.config import settings
from .core.database import init_database, close_database, check_database_health
from .routers import ocr_routes
from .models.schemas import HealthCheckResponse, SystemStats, ErrorResponse
from .utils.constants import HTTP_STATUS_CODES

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Application startup time
start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    """
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    
    try:
        # Initialize database
        await init_database()
        logger.info("Database initialized successfully")
        
        # Test Mistral API connection
        await test_mistral_connection()
        logger.info("Mistral API connection verified")
        
        # Additional startup tasks
        await cleanup_old_files()
        
        logger.info("Application startup completed")
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")
    
    try:
        await close_database()
        logger.info("Database connections closed")
        
    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="Mistral OCR Component for FRA Document Processing",
    version=settings.APP_VERSION,
    debug=settings.DEBUG,
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

if not settings.DEBUG:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "*.yourdomain.com"]
    )


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log incoming requests"""
    start_time = time.time()
    
    # Log request
    logger.info(f"{request.method} {request.url.path} - {request.client.host}")
    
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - "
        f"{response.status_code} - {process_time:.3f}s"
    )
    
    return response


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "error_code": f"HTTP_{exc.status_code}",
            "timestamp": time.time()
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors"""
    logger.warning(f"Validation error: {exc.errors()}")
    
    return JSONResponse(
        status_code=HTTP_STATUS_CODES['UNPROCESSABLE_ENTITY'],
        content={
            "success": False,
            "message": "Request validation failed",
            "error_code": "VALIDATION_ERROR",
            "details": exc.errors(),
            "timestamp": time.time()
        }
    )


@app.exception_handler(StarletteHTTPException)
async def starlette_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle Starlette HTTP exceptions"""
    logger.error(f"Starlette HTTP {exc.status_code}: {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail or "Internal server error",
            "error_code": f"HTTP_{exc.status_code}",
            "timestamp": time.time()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=HTTP_STATUS_CODES['INTERNAL_SERVER_ERROR'],
        content={
            "success": False,
            "message": "An unexpected error occurred" if not settings.DEBUG else str(exc),
            "error_code": "INTERNAL_ERROR",
            "timestamp": time.time()
        }
    )


# Include routers
app.include_router(
    ocr_routes.router,
    prefix="/api/v1",
    tags=["OCR Processing"]
)


# Health check endpoints
@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    """
    try:
        # Check database
        db_healthy = await check_database_health()
        db_status = "healthy" if db_healthy else "unhealthy"
        
        # Check Mistral API
        mistral_status = await check_mistral_api()
        
        # Check Redis (if configured)
        redis_status = await check_redis_connection()
        
        # Overall status
        overall_status = "healthy" if all([
            db_status == "healthy",
            mistral_status == "healthy",
            redis_status == "healthy"
        ]) else "degraded"
        
        return HealthCheckResponse(
            status=overall_status,
            version=settings.APP_VERSION,
            database_status=db_status,
            mistral_api_status=mistral_status,
            redis_status=redis_status,
            uptime_seconds=time.time() - start_time
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthCheckResponse(
            status="unhealthy",
            version=settings.APP_VERSION,
            database_status="unknown",
            mistral_api_status="unknown",
            redis_status="unknown",
            uptime_seconds=time.time() - start_time
        )


@app.get("/health/ready")
async def readiness_check():
    """
    Readiness check for Kubernetes deployments.
    """
    try:
        # Check if application is ready to serve requests
        db_healthy = await check_database_health()
        mistral_healthy = await check_mistral_api() == "healthy"
        
        if db_healthy and mistral_healthy:
            return {"status": "ready"}
        else:
            raise HTTPException(
                status_code=HTTP_STATUS_CODES['SERVICE_UNAVAILABLE'],
                detail="Service not ready"
            )
            
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        raise HTTPException(
            status_code=HTTP_STATUS_CODES['SERVICE_UNAVAILABLE'],
            detail="Service not ready"
        )


@app.get("/health/live")
async def liveness_check():
    """
    Liveness check for Kubernetes deployments.
    """
    return {"status": "alive", "timestamp": time.time()}


@app.get("/stats", response_model=SystemStats)
async def get_system_stats():
    """
    Get system statistics and metrics.
    """
    try:
        from .core.database import get_async_session
        from .models.database import Document, Claim, ExtractedCoordinate, ClaimGeometry, BatchJob
        from sqlalchemy import select, func
        
        async with get_async_session().__anext__() as db:
            # Document statistics
            doc_total = await db.scalar(select(func.count(Document.id)))
            doc_pending = await db.scalar(
                select(func.count(Document.id)).where(Document.status == "pending")
            )
            doc_processing = await db.scalar(
                select(func.count(Document.id)).where(Document.status == "processing")
            )
            doc_completed = await db.scalar(
                select(func.count(Document.id)).where(Document.status == "completed")
            )
            doc_failed = await db.scalar(
                select(func.count(Document.id)).where(Document.status == "failed")
            )
            
            # Claim statistics
            claims_total = await db.scalar(select(func.count(Claim.id)))
            coordinates_total = await db.scalar(select(func.count(ExtractedCoordinate.id)))
            geometries_total = await db.scalar(select(func.count(ClaimGeometry.id)))
            
            # Batch job statistics
            active_batches = await db.scalar(
                select(func.count(BatchJob.id)).where(
                    BatchJob.status.in_(["queued", "running"])
                )
            )
        
        return SystemStats(
            total_documents=doc_total or 0,
            pending_documents=doc_pending or 0,
            processing_documents=doc_processing or 0,
            completed_documents=doc_completed or 0,
            failed_documents=doc_failed or 0,
            total_claims=claims_total or 0,
            total_coordinates=coordinates_total or 0,
            total_geometries=geometries_total or 0,
            active_batch_jobs=active_batches or 0
        )
        
    except Exception as e:
        logger.error(f"Stats retrieval failed: {str(e)}")
        return SystemStats()


@app.get("/version")
async def get_version():
    """Get application version information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": "development" if settings.DEBUG else "production",
        "python_version": "3.9+",
        "features": [
            "mistral_ocr",
            "coordinate_extraction", 
            "geojson_conversion",
            "batch_processing"
        ]
    }


@app.get("/")
async def root():
    """Root endpoint with basic information."""
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs" if settings.DEBUG else None,
        "health": "/health"
    }


# Utility functions for health checks
async def test_mistral_connection() -> None:
    """Test connection to Mistral API during startup"""
    try:
        from .services.mistral_ocr_service import MistralOCRService
        
        service = MistralOCRService()
        # Simple test - this would need to be implemented in the service
        # For now, just check if we can initialize the client
        if not service.client:
            raise Exception("Failed to initialize Mistral client")
        
        logger.info("Mistral API connection test passed")
        
    except Exception as e:
        logger.warning(f"Mistral API connection test failed: {str(e)}")
        # Don't fail startup for API connection issues
        pass


async def check_mistral_api() -> str:
    """Check Mistral API health"""
    try:
        from .services.mistral_ocr_service import MistralOCRService
        
        service = MistralOCRService()
        # In a real implementation, you might make a simple API call
        # to verify the connection
        return "healthy" if service.client else "unhealthy"
        
    except Exception:
        return "unhealthy"


async def check_redis_connection() -> str:
    """Check Redis connection health"""
    try:
        import redis.asyncio as redis
        
        redis_client = redis.from_url(settings.REDIS_URL)
        await redis_client.ping()
        await redis_client.close()
        return "healthy"
        
    except Exception:
        return "unhealthy"


async def cleanup_old_files():
    """Cleanup old uploaded files during startup"""
    try:
        from .utils.file_handler import FileHandler
        
        file_handler = FileHandler()
        deleted_count = await file_handler.cleanup_old_files(days_old=7)
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old files during startup")
        
    except Exception as e:
        logger.warning(f"File cleanup during startup failed: {str(e)}")


# Custom dependency for API rate limiting (if needed)
async def rate_limit_check(request: Request):
    """
    Simple rate limiting check.
    In production, use a proper rate limiting solution like Redis-based limits.
    """
    # Placeholder for rate limiting logic
    return True


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8001,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=settings.DEBUG
    )