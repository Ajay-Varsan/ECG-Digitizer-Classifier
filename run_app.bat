@echo off
echo =========================================================================
echo Starting CardioDigitize AI: ECG Digitizer ^& 15-Disorder Classifier Studio
echo =========================================================================
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m streamlit run app_combined.py --server.port 8501
) else if exist "ecg (2)\ecg\.venv\Scripts\python.exe" (
    "ecg (2)\ecg\.venv\Scripts\python.exe" -m streamlit run app_combined.py --server.port 8501
) else (
    streamlit run app_combined.py --server.port 8501
)
pause
