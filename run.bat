@echo off
REM Etymon Windows Launcher
REM This script starts Etymon with proper error handling

echo Starting Etymon - World Generator...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or later and try again
    pause
    exit /b 1
)

REM Check if requirements are installed
python -c "import pygame, numpy, scipy, noise, PIL" >nul 2>&1
if errorlevel 1 (
    echo Error: Required packages not installed
    echo Running setup to install dependencies...
    python setup.py
    if errorlevel 1 (
        echo Setup failed. Please check error messages above.
        pause
        exit /b 1
    )
)

REM Start the application
python main.py

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo Application exited with an error
    pause
)