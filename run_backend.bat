@echo off
echo Starting FastAPI Backend...
echo.
echo Backend will be available at: http://localhost:8000
echo API Documentation: http://localhost:8000/docs
echo.
cd enterprise-ai-platform
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
