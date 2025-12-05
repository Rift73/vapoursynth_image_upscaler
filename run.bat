@echo off
:: VapourSynth Image Upscaler Launcher
:: This batch file launches the GUI without a console window

:: Find pythonw.exe and run the application
start "" pythonw "%~dp0run.pyw"
