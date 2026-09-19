@echo off
title VERIFY.AI Setup
color 0A
:: Hugging Face model cache lives in-repo on D: (C: is nearly full).
:: MUST be set before any Python process starts, or downloads land on C:.
set HF_HOME=D:\DeepGuard\hf_cache
set HF_HUB_CACHE=D:\DeepGuard\hf_cache\hub
set HUGGINGFACE_HUB_CACHE=D:\DeepGuard\hf_cache\hub
set TORCH_HOME=D:\temp\torch_cache
echo.
echo  ====================================
echo   VERIFY.AI - Setup ^& Launch
echo  ====================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found!
    echo  Please install Python from https://python.org
    echo  Make sure to check "Add Python to PATH" during install!
    pause
    exit /b 1
)
echo  [OK] Python found

:: Create virtual environment
if not exist venv (
    echo  Creating virtual environment...
    python -m venv venv
    echo  [OK] Virtual env created
) else (
    echo  [OK] Virtual env exists
)

:: Install packages
echo.
echo  Installing packages (first time takes 5-15 minutes)...
echo  NOTE: torch comes from the CUDA wheel index, NOT PyPI,
echo  so the +cu121 GPU build is installed (never a CPU build).
echo  Please wait...
echo.
venv\Scripts\pip install --quiet --upgrade pip
venv\Scripts\pip install --quiet "torch==2.3.0+cu121" "torchvision==0.18.0+cu121" --index-url https://download.pytorch.org/whl/cu121
venv\Scripts\pip install fastapi "uvicorn[standard]" python-multipart httpx pydantic
venv\Scripts\pip install Pillow "numpy==1.26.4" opencv-python-headless safetensors
venv\Scripts\pip install librosa soundfile transformers timm scipy
echo  [OK] All packages installed

:: Create weights + logs folder
if not exist weights mkdir weights
if not exist logs mkdir logs
echo  [OK] Folders ready

:: Fetch model weights (skips files already present; prints manual
:: steps for weights with no scriptable source, e.g. IAPL + RawNet)
echo.
echo  Downloading model weights (skips existing, may take a while)...
venv\Scripts\python download_weights.py

:: Kill anything on our ports (7 voters + gateway + legacy audio/video)
for %%P in (8000 5001 5004 5005 5009 5010 5011 5013 5002 7001) do (
    for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| find ":%%P "') do (
        taskkill /PID %%a /F >nul 2>&1
    )
)

echo.
echo  Starting the 7 model services...
echo  (RawNet :5002 is NOT started — no usable weights exist. See README.)
echo.

start "NPR :5001"        /min cmd /c "cd /d %~dp0\models\npr          && %~dp0venv\Scripts\uvicorn main:app --host 0.0.0.0 --port 5001 > %~dp0logs\npr.log 2>&1"
start "UFD :5004"        /min cmd /c "cd /d %~dp0\models\ufd          && %~dp0venv\Scripts\uvicorn main:app --host 0.0.0.0 --port 5004 > %~dp0logs\ufd.log 2>&1"
start "IAPL :5005"       /min cmd /c "cd /d %~dp0\models\iapl         && %~dp0venv\Scripts\uvicorn main:app --host 0.0.0.0 --port 5005 > %~dp0logs\iapl.log 2>&1"
start "SDXL :5009"       /min cmd /c "cd /d %~dp0\models\sdxl_detector && %~dp0venv\Scripts\uvicorn main:app --host 0.0.0.0 --port 5009 > %~dp0logs\sdxl.log 2>&1"
start "UMM :5010"        /min cmd /c "cd /d %~dp0\models\umm_maybe    && %~dp0venv\Scripts\uvicorn main:app --host 0.0.0.0 --port 5010 > %~dp0logs\umm.log 2>&1"
start "CapCheck :5011"   /min cmd /c "cd /d %~dp0\models\capcheck     && %~dp0venv\Scripts\uvicorn main:app --host 0.0.0.0 --port 5011 > %~dp0logs\capcheck.log 2>&1"
start "Nonescape :5013"  /min cmd /c "cd /d %~dp0\models\nonescape    && %~dp0venv\Scripts\uvicorn main:app --host 0.0.0.0 --port 5013 > %~dp0logs\nonescape.log 2>&1"
start "CrossViT :7001"   /min cmd /c "cd /d %~dp0\models\crossvit     && %~dp0venv\Scripts\uvicorn main:app --host 0.0.0.0 --port 7001 > %~dp0logs\crossvit.log 2>&1"

echo.
echo  Waiting for model services to report healthy (up to 10 min on
echo  first launch while large checkpoints load)...
venv\Scripts\python scripts\wait_for_services.py 5001 5004 5005 5009 5010 5011 5013
if errorlevel 1 (
    echo.
    echo  [WARN] Not all models came up — check logs\*.log for tracebacks.
    echo  The gateway will still start; failed models are excluded per-request.
)

echo.
echo  Starting gateway (only after models are up)...
start "Gateway :8000"    /min cmd /c "cd /d %~dp0 && venv\Scripts\uvicorn gateway.main:app --host 0.0.0.0 --port 8000 > logs\gateway.log 2>&1"
timeout /t 4 /nobreak >nul

echo  [OK] Gateway        -^>  http://localhost:8000
echo  [OK] NPR model      -^>  http://localhost:5001
echo  [OK] UFD model      -^>  http://localhost:5004
echo  [OK] IAPL model     -^>  http://localhost:5005
echo  [OK] SDXL model     -^>  http://localhost:5009
echo  [OK] UMM model      -^>  http://localhost:5010
echo  [OK] CapCheck model -^>  http://localhost:5011
echo  [OK] Nonescape model-^>  http://localhost:5013
echo  [OK] CrossViT       -^>  http://localhost:7001
echo  [--] RawNet2 skipped (no usable weights — see README ^& download_weights.py)
echo.
timeout /t 4 /nobreak >nul

echo  ====================================
echo   All done! Opening API docs...
echo  ====================================
echo.
echo  API:   http://localhost:8000
echo  Docs:  http://localhost:8000/docs
echo.
echo  To stop everything, close this window
echo  and run stop.bat
echo.
start "" "http://localhost:8000/docs"
pause
