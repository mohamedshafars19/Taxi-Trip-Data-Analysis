@echo off
REM NYC Taxi Data Analysis Execution Script
echo ========================================================
echo Running NYC Yellow Taxi Data Science Pipeline...
echo ========================================================
python taxi_analysis.py %*
if errorlevel 1 (
    echo.
    echo Trying fallback Python executable in AppData...
    "%LOCALAPPDATA%\Python\bin\python.exe" taxi_analysis.py %*
    if errorlevel 1 (
        "%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe" taxi_analysis.py %*
    )
)
pause
