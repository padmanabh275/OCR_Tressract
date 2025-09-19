@echo off
echo Starting Document Extraction System with torch_env...

REM Activate torch_env
call conda activate torch_env

REM Check if Ollama is running
echo Checking Ollama service...
curl -s http://localhost:11434/api/tags > nul
if %errorlevel% neq 0 (
    echo Starting Ollama service...
    start "Ollama Service" ollama serve
    timeout /t 5 /nobreak > nul
)

REM Start the document extraction system
echo Starting Document Extraction System...
python app_with_database.py

pause
