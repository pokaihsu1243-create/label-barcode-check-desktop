@echo off
chcp 65001 >nul
title 標籤條碼檢查
set "PY=python"
where py >nul 2>&1 && set "PY=py -3"
set "APP=%~dp0check_labels.py"
echo ================================
echo   標籤條碼檢查（條碼 vs 下方文字）
echo ================================
echo.
echo 填了應有條碼總數才會做總數核對；不填的話，就算逐張都一致也不會顯示「可以放行」。
set "EXPECT="
set /p "EXPECT=這份稿應有幾個條碼？（不知道就直接按 Enter）: "
echo.
if "%EXPECT%"=="" (
  %PY% "%APP%" %*
) else (
  %PY% "%APP%" %* --expect %EXPECT%
)
echo.
echo 檢查結束，報告已在瀏覽器開啟。
pause
