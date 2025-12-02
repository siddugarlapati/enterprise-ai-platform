from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import logging
import traceback
from typing import Callable

from app.config.settings import settings

logger = logging.getLogger(__name__)

class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        try:
            return await call_next(request)
        except HTTPException:
            raise
        except Exception as e:
            request_id = getattr(request.state, "request_id", "unknown")
            
            logger.error(
                f"Unhandled exception: {str(e)}",
                extra={
                    "request_id": request_id,
                    "path": request.url.path,
                    "method": request.method,
                    "traceback": traceback.format_exc()
                }
            )
            
            # Return generic error in production
            if settings.ENVIRONMENT == "production":
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": "Internal server error",
                        "request_id": request_id,
                        "message": "An unexpected error occurred. Please try again later."
                    }
                )
            
            # Return detailed error in development
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "request_id": request_id,
                    "message": str(e),
                    "traceback": traceback.format_exc()
                }
            )

async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = getattr(request.state, "request_id", "unknown")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "request_id": request_id
        },
        headers=exc.headers
    )

async def validation_exception_handler(request: Request, exc):
    request_id = getattr(request.state, "request_id", "unknown")
    
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation error",
            "request_id": request_id,
            "details": exc.errors() if hasattr(exc, "errors") else str(exc)
        }
    )
