@echo off
rem Libra management console (GUI). Looks for libra-console.ps1 next to this file, then in %USERPROFILE%\libra.
set "S=%~dp0libra-console.ps1"
if not exist "%S%" set "S=%USERPROFILE%\libra\libra-console.ps1"
if not exist "%S%" (
  echo libra-console.ps1 not found: %S%
  echo run tools/windows/install.sh from WSL first.
  pause
  exit /b 1
)
start "" powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -STA -File "%S%"
