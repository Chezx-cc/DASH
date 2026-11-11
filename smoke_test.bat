@echo off
setlocal
python "%~dp0scripts\run_smoke.py"
if errorlevel 1 exit /b %errorlevel%

