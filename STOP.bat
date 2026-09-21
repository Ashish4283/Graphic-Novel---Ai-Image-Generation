@echo off
setlocal enabledelayedexpansion

set "COMFY_PORT=8188"
set "STUDIO_PORT=8500"

echo.
echo  ==============================================================
echo    Graphic Novel Studio - Stop
echo  ==============================================================
echo.

rem Kill only the process actually listening on each port. Killing every
rem python.exe would take out unrelated work.
call :kill_port %STUDIO_PORT% Studio
call :kill_port %COMFY_PORT% ComfyUI

echo.
echo  Checking...
ping -n 3 127.0.0.1 >nul 2>&1
call :report %STUDIO_PORT% Studio
call :report %COMFY_PORT% ComfyUI
echo.
echo  GPU memory is released when ComfyUI exits.
echo.
exit /b 0

:kill_port
set "FOUND=0"
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /r /c:"LISTENING" ^| findstr /c:":%~1 "') do (
    if not "%%P"=="0" (
        echo  Stopping %~2 on port %~1, PID %%P ...
        taskkill /PID %%P /T /F >nul 2>&1
        set "FOUND=1"
    )
)
if "!FOUND!"=="0" echo  %~2 was not running on port %~1
exit /b 0

:report
set "STILL=0"
for /f "tokens=*" %%L in ('netstat -ano ^| findstr /r /c:"LISTENING" ^| findstr /c:":%~1 "') do set "STILL=1"
if "!STILL!"=="1" (
    echo    [!] %~2 still listening on port %~1
) else (
    echo    [ok] %~2 stopped
)
exit /b 0
