@echo off
chcp 65001 >nul
title 標籤條碼檢查
cd /d "%~dp0"
set "PY=python"
where py >nul 2>&1 && set "PY=py -3"
start "label-check-server" /min cmd /c "%PY% web_app.py --no-browser"
ping -n 4 127.0.0.1 >nul
start http://localhost:8720
exit
