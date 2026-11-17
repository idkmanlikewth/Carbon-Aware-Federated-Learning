@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe py -3 -m venv .venv
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe -m pip install -r requirements-research.txt
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe demo_server.py
