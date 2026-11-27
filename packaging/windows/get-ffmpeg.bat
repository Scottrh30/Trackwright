@echo off
setlocal
rem ------------------------------------------------------------------
rem  Trackwright - get FFmpeg
rem  Downloads ffmpeg.exe and ffprobe.exe into this folder, next to
rem  trackwright.exe. Trackwright uses them to open video files and to
rem  make MP4s from renders. Image sequences work without FFmpeg.
rem ------------------------------------------------------------------
cd /d "%~dp0"
set "TW_DEST=%~dp0"
title Trackwright - get FFmpeg

if exist "%~dp0ffmpeg.exe" if exist "%~dp0ffprobe.exe" (
    echo FFmpeg is already in this folder. Nothing to do.
    goto done
)

echo Downloading FFmpeg (about 100 MB) from gyan.dev ...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference = 'Stop';" ^
  "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12;" ^
  "$zip = Join-Path $env:TEMP 'trackwright-ffmpeg.zip';" ^
  "$dir = Join-Path $env:TEMP 'trackwright-ffmpeg';" ^
  "$ProgressPreference = 'SilentlyContinue';" ^
  "Invoke-WebRequest -Uri 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip' -OutFile $zip;" ^
  "if (Test-Path $dir) { Remove-Item $dir -Recurse -Force };" ^
  "Expand-Archive -Path $zip -DestinationPath $dir -Force;" ^
  "Get-ChildItem -Path $dir -Recurse -Include ffmpeg.exe, ffprobe.exe | Copy-Item -Destination $env:TW_DEST -Force;" ^
  "Remove-Item $zip, $dir -Recurse -Force"
if exist "%~dp0ffmpeg.exe" if exist "%~dp0ffprobe.exe" (
    echo.
    echo Done. FFmpeg is now next to trackwright.exe.
    goto done
)

echo.
echo The direct download did not work. Trying Windows Package Manager (winget) ...
where winget >nul 2>nul
if errorlevel 1 goto manual
winget install -e --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements
if errorlevel 1 goto manual
echo.
echo Done. FFmpeg was installed for your user account.
echo Close and reopen Trackwright so it can find it.
goto done

:manual
echo.
echo Could not install FFmpeg automatically.
echo Download "ffmpeg-release-essentials.zip" from https://www.gyan.dev/ffmpeg/builds/
echo and copy ffmpeg.exe and ffprobe.exe (from its bin folder) next to trackwright.exe.

:done
echo.
pause
endlocal
