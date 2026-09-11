@echo off
rem Libra management console (GUI). Runs libra-console.ps1 next to this file without a console window.
start "" powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -STA -File "%~dp0libra-console.ps1"
