@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found.
    echo Create it with: py -3.12 -m venv .venv
    exit /b 1
)

call ".venv\Scripts\activate.bat"
python -m streamlit run app\streamlit_app.py
endlocal
