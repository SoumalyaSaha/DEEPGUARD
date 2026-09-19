@echo off
:: Hugging Face model cache lives in-repo on D: (C: is nearly full).
:: MUST be set before any Python process starts, or downloads land on C:.
set HF_HOME=D:\DeepGuard\hf_cache
set HF_HUB_CACHE=D:\DeepGuard\hf_cache\hub
set HUGGINGFACE_HUB_CACHE=D:\DeepGuard\hf_cache\hub
set TORCH_HOME=D:\temp\torch_cache
echo Starting VERIFY.AI (7 voters + gateway)...
start "NPR Model" cmd /k "cd /d D:\DeepGuard && venv\Scripts\activate && cd models\npr && uvicorn main:app --port 5001"
start "UFD Model" cmd /k "cd /d D:\DeepGuard && venv\Scripts\activate && cd models\ufd && uvicorn main:app --port 5004"
start "IAPL Model" cmd /k "cd /d D:\DeepGuard && venv\Scripts\activate && cd models\iapl && uvicorn main:app --port 5005"
start "SDXL Model" cmd /k "cd /d D:\DeepGuard && venv\Scripts\activate && cd models\sdxl_detector && uvicorn main:app --port 5009"
start "UMM Model" cmd /k "cd /d D:\DeepGuard && venv\Scripts\activate && cd models\umm_maybe && uvicorn main:app --port 5010"
start "CapCheck Model" cmd /k "cd /d D:\DeepGuard && venv\Scripts\activate && cd models\capcheck && uvicorn main:app --port 5011"
start "Nonescape Model" cmd /k "cd /d D:\DeepGuard && venv\Scripts\activate && cd models\nonescape && uvicorn main:app --port 5013"
echo Waiting for models to report healthy (up to 10 min on first launch)...
cd /d D:\DeepGuard && venv\Scripts\python scripts\wait_for_services.py 5001 5004 5005 5009 5010 5011 5013
start "Gateway" cmd /k "cd /d D:\DeepGuard && venv\Scripts\activate && uvicorn gateway.main:app --host 0.0.0.0 --port 8000"
echo All servers starting! Gateway launches last so early requests never hit cold models.
