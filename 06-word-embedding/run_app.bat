@echo off
setlocal
cd /d "%~dp0"

if exist "C:\venvs\wordembed\Scripts\python.exe" (
    call "C:\venvs\wordembed\Scripts\activate.bat"
) else (
    echo Virtual environment not found at C:\venvs\wordembed
    echo Create it with: python -m venv C:\venvs\wordembed
    exit /b 1
)

python -m streamlit run app\streamlit_app.py
endlocal
