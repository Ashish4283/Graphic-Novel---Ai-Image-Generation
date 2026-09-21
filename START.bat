@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

rem ---- Configure here if your paths differ ----
set "COMFY_DIR=C:\ComfyUI"
set "COMFY_PORT=8188"
set "STUDIO_PORT=8500"
set "COMFY_ARGS=--lowvram"

echo.
echo  ==============================================================
echo    Graphic Novel Studio - Start
echo  ==============================================================
echo.

python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo  ERROR: Python not found on PATH.
    echo  Open a new terminal, or add Python to PATH, then retry.
    pause
    exit /b 1
)

if not exist "%COMFY_DIR%\main.py" (
    echo  ERROR: ComfyUI not found at %COMFY_DIR%
    echo  Edit COMFY_DIR at the top of this file.
    pause
    exit /b 1
)

rem ---- 1. ComfyUI, the renderer ----
call :is_up %COMFY_PORT%
if "!UP!"=="1" (
    echo  [1/2] ComfyUI already running on port %COMFY_PORT%
    goto studio
)

echo  [1/2] Starting ComfyUI on port %COMFY_PORT% ...
start "ComfyUI" /min cmd /c "cd /d "%COMFY_DIR%" && python main.py %COMFY_ARGS% --port %COMFY_PORT%"
echo        waiting for startup, up to 2 minutes...
set /a TRIES=0

:wait_comfy
call :sleep 3
set /a TRIES+=1
call :is_up %COMFY_PORT%
if "!UP!"=="1" goto comfy_ready
if !TRIES! GEQ 40 (
    echo  ERROR: ComfyUI did not start. Check the ComfyUI window.
    pause
    exit /b 1
)
goto wait_comfy

:comfy_ready
echo        ComfyUI ready.

rem ---- 2. Studio, the production manager ----
:studio
call :is_up %STUDIO_PORT%
if "!UP!"=="1" (
    echo  [2/2] Studio already running on port %STUDIO_PORT%
    goto done
)

echo  [2/2] Starting Studio on port %STUDIO_PORT% ...
start "Graphic Novel Studio" /min cmd /c "cd /d "%~dp005_app" && python server.py --port %STUDIO_PORT% --comfy http://127.0.0.1:%COMFY_PORT%"
set /a TRIES=0

:wait_studio
call :sleep 2
set /a TRIES+=1
call :is_up %STUDIO_PORT%
if "!UP!"=="1" goto studio_ready
if !TRIES! GEQ 20 (
    echo  ERROR: Studio did not start. Check the Studio window.
    echo  Missing packages?  pip install -r 05_app\requirements.txt
    pause
    exit /b 1
)
goto wait_studio

:studio_ready
echo        Studio ready.

:done
echo.
echo  ==============================================================
echo    Studio  : http://127.0.0.1:%STUDIO_PORT%
echo    ComfyUI : http://127.0.0.1:%COMFY_PORT%
echo.
echo    Both run minimised. Close them with STOP.bat
echo  ==============================================================
echo.
start "" "http://127.0.0.1:%STUDIO_PORT%"
exit /b 0

rem ---- helper: is anything LISTENING on a port? ----
:is_up
set "UP=0"
for /f "tokens=*" %%L in ('netstat -ano ^| findstr /r /c:"LISTENING" ^| findstr /c:":%~1 "') do set "UP=1"
exit /b 0

rem ---- helper: sleep N seconds. "timeout" fails when stdin is redirected
rem      (scheduled task, CI, piped run), so use ping instead. ----
:sleep
ping -n %~1 127.0.0.1 >nul 2>&1
exit /b 0
