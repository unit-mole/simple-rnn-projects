@echo off
setlocal
cd /d "%~dp0"

set "SHORT_VENV=C:\venvs\textgen"
set "LOCAL_VENV=%CD%\.venv"

if exist "%SHORT_VENV%\Scripts\python.exe" (
    set "VENV_DIR=%SHORT_VENV%"
) else if exist "%LOCAL_VENV%\Scripts\python.exe" (
    set "VENV_DIR=%LOCAL_VENV%"
) else (
    echo [ERROR] No usable virtual environment was found.
    echo.
    echo Recommended setup for this OneDrive project:
    echo   if not exist "C:\venvs" mkdir "C:\venvs"
    echo   python -m venv "C:\venvs\textgen"
    echo   call "C:\venvs\textgen\Scripts\activate.bat"
    echo   python -m pip install --upgrade pip setuptools wheel
    echo   python -m pip install -r requirements.txt -r requirements-ci.txt
    echo.
    exit /b 1
)

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 exit /b 1

python -m streamlit run app\streamlit_app.py
set "EXIT_CODE=%ERRORLEVEL%"

endlocal & exit /b %EXIT_CODE%
