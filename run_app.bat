@echo off
REM Convenience launcher for Bala Support.
REM Uses "python -m streamlit" because this venv resolves packages
REM through a .pth link and has no console-script shims.
"%~dp0.venv\Scripts\python.exe" -m streamlit run "%~dp0app.py"
