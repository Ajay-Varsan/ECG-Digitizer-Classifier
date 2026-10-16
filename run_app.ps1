# Run CardioDigitize AI Streamlit Studio
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir
Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host "Starting CardioDigitize AI: ECG Digitizer & 15-Disorder Classifier Studio" -ForegroundColor Green
Write-Host "=========================================================================" -ForegroundColor Cyan

if (Test-Path "$ScriptDir\.venv\Scripts\python.exe") {
    & "$ScriptDir\.venv\Scripts\python.exe" -m streamlit run app_combined.py --server.port 8501
} elseif (Test-Path "$ScriptDir\ecg (2)\ecg\.venv\Scripts\python.exe") {
    & "$ScriptDir\ecg (2)\ecg\.venv\Scripts\python.exe" -m streamlit run app_combined.py --server.port 8501
} else {
    streamlit run app_combined.py --server.port 8501
}
