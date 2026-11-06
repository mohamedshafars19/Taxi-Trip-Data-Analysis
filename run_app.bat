@echo off
REM NYC Taxi Data Analysis - Streamlit Interactive Web Application Launcher
echo ========================================================
echo Launching NYC Taxi Interactive Web App (Streamlit)...
echo ========================================================
streamlit run app.py
if errorlevel 1 (
    echo.
    echo Trying fallback Streamlit launcher in Python...
    python -m streamlit run app.py
    if errorlevel 1 (
        echo.
        echo Trying AppData Python executable...
        "%LOCALAPPDATA%\Python\bin\python.exe" -m streamlit run app.py
        if errorlevel 1 (
            "%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe" -m streamlit run app.py
        )
    )
)
pause
