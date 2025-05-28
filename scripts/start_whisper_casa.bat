@echo off
echo Starting Whisper API server...
cd /d J:\programacion\JIT\PYTHON\FASTAPI\whisper-api\
call .\venv\Scripts\activate
cd src
python -m uvicorn whisper_api:app --host 0.0.0.0 --port 8000
pause