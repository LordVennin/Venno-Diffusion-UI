@echo off
rem Venno MOSS: test MOSS-Image LoRAs (Windows). First start: creates venv-moss and installs everything.
rem It is separate from Venno's venv because MOSS-Image needs newer libraries; both can run at the same time.
rem Options go straight to the server, e.g.  moss.bat --memory low --lora-dir D:\venno-trainer\output
cd /d "%~dp0"
if not defined PYTHON set PYTHON=python
if not exist venv-moss\Scripts\python.exe (
    echo [Venno MOSS] creating venv-moss...
    %PYTHON% -m venv venv-moss || goto fail
)
venv-moss\Scripts\python.exe -m venno_moss.install %MOSS_ARGS% %* || goto fail
venv-moss\Scripts\python.exe -m venno_moss %MOSS_ARGS% %*
goto end
:fail
echo [Venno MOSS] could not start.
pause
:end
