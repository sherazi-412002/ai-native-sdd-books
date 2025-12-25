from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os
import traceback
import logging
import time

# Load environment variables
load_dotenv()

# Import rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Import API routes
import sys
from pathlib import Path

# Add the project root to the path to resolve imports correctly
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from server.api.v1 import chat, health

# Import metrics service
from server.services.metrics import metrics_service

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Load environment variables
load_dotenv()

def init_logging():
    """
    Initialize logging configuration
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

# Initialize logging
init_logging()
logger = logging.getLogger(__name__)

# Initialize the app
app = FastAPI(
    title="Intelligence Layer - RAG Backend API",
    description="API for the RAG-powered chatbot backend",
    version="1.0.0",
    debug=os.getenv("DEBUG", "False").lower() == "true"
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware for Docusaurus frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add global exception handlers for comprehensive error handling
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler for unhandled exceptions
    """
    logger.error(f"Unhandled exception: {str(exc)}\nTraceback: {traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred",
            "details": str(exc) if app.debug else None
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """
    Exception handler for request validation errors
    """
    logger.warning(f"Validation error: {str(exc)}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "VALIDATION_ERROR",
            "message": "Request validation failed",
            "details": exc.errors() if app.debug else None
        }
    )

# Include API routes
app.include_router(chat.router)  # Rate limiting applied at individual endpoints
health.register_health_routes(app)

@app.get("/")
async def root():
    return {"message": "Intelligence Layer - RAG Backend API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)