@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Local AI Player-Action Simulator

rem ---- find Python (3.10+; 3.11 recommended) ---------------------------------
set "PYCMD="
where py >nul 2>nul && set "PYCMD=py -3"
if not defined PYCMD ( where python >nul 2>nul && set "PYCMD=python" )
if not defined PYCMD (
    echo [ERROR] Python was not found. Install Python 3.11 from https://www.python.org/downloads/
    echo         and tick "Add python.exe to PATH" during setup.
    goto :fail
)
%PYCMD% -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"
if errorlevel 1 (
    echo [ERROR] Python 3.10 or newer is required ^(3.11 recommended^).
    goto :fail
)

rem ---- virtual environment ---------------------------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    %PYCMD% -m venv .venv
    if errorlevel 1 goto :fail
)
call ".venv\Scripts\activate.bat"

rem ---- dependencies (first run, or when requirements change) -----------------
if exist "requirements.lock.txt" ( set "REQ=requirements.lock.txt" ) else ( set "REQ=requirements.txt" )
set "STAMP=.venv\.deps_%REQ%"
for %%F in (%REQ%) do set "REQTIME=%%~tF"
set "OLD="
if exist "%STAMP%" set /p OLD=<"%STAMP%"
if not "%OLD%"=="%REQTIME%" (
    echo Installing dependencies ^(first run takes a few minutes^)...
    python -m pip install --upgrade pip
    pip install -r %REQ%
    if errorlevel 1 goto :fail
    echo Installing optional extras ^(failures are fine^)...
    for /f "usebackq eol=# tokens=*" %%P in ("requirements-optional.txt") do pip install %%P
    >"%STAMP%" echo %REQTIME%
)

rem ---- optional: start a local MongoDB service if one is installed ----------
sc query MongoDB >nul 2>nul && (sc query MongoDB | find "RUNNING" >nul || net start MongoDB >nul 2>nul)

echo Starting Local AI Player-Action Simulator...
python main.py
if errorlevel 1 goto :fail
goto :end

:fail
echo.
echo The application stopped with an error. See the messages above.
pause
:end
endlocal
