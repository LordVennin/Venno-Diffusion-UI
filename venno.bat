@echo off
rem Venno Diffusion UI (Windows). First start: creates venv and installs everything.
rem Options go straight to the server, e.g.  venno.bat --listen --lowvram   (venno.bat --help lists them)
rem Settings you always want: set VENNO_ARGS=--models-dir D:\models
cd /d "%~dp0"
if not defined PYTHON set PYTHON=python
if defined VIRTUAL_ENV goto run
if defined VENNO_NO_VENV goto run
if not exist venv\Scripts\python.exe (
    echo [Venno] creating venv...
    %PYTHON% -m venv venv || goto fail
)
set PYTHON=venv\Scripts\python.exe
:run
%PYTHON% -m venno.install %VENNO_ARGS% %* || goto fail
%PYTHON% -m venno %VENNO_ARGS% %*
goto end
:fail
echo [Venno] could not start.
pause
:end
