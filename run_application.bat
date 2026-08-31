@echo off
title TalentMatrix AI - Enterprise Resume Screening & ATS System
echo ======================================================================
echo   TalentMatrix AI(TM) - Enterprise Candidate Intelligence System
echo ======================================================================
echo.
echo [1/3] Verifying Python environment...
python --version
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    pause
    exit /b 1
)

echo.
echo [2/3] Checking dependencies & database...
python launch.py --init-only

echo.
echo [3/3] Launching TalentMatrix AI Enterprise Dashboard...
start "" http://localhost:8501
streamlit run app/streamlit_app.py

pause
