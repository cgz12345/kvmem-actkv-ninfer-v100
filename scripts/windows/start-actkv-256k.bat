@echo off
setlocal
rem Reference ActKV text + MTP profile; Qwen GSQ-RCO IQ3_S on V100 16 GB.
rem mean reproduces the reference README report; max is a separate experimental arm.
if "%~1"=="" (
  echo Usage: %~nx0 "C:\models\Qwen.ninfer" [on^|off^|max]
  exit /b 2
)
set "NINFER_ACTKV_SWITCH=--kvmem-action-lrfu"
set "NINFER_ACTKV_AGG=mean"
if not "%~2"=="" if /I not "%~2"=="on" if /I not "%~2"=="off" if /I not "%~2"=="max" (
  echo Mode must be on, off, or max.
  exit /b 2
)
if /I "%~2"=="off" set "NINFER_ACTKV_SWITCH="
if /I "%~2"=="max" set "NINFER_ACTKV_AGG=max"
set "NINFER_ACTKV_MODEL=%~f1"
if not exist "%NINFER_ACTKV_MODEL%" (
  echo Model not found: "%NINFER_ACTKV_MODEL%"
  exit /b 2
)
set "NINFER_ACTKV_EXE=%~dp0..\..\build-sm70\apps\ninfer-serve.exe"
if not exist "%NINFER_ACTKV_EXE%" (
  echo Server not found: "%NINFER_ACTKV_EXE%"
  exit /b 2
)
"%NINFER_ACTKV_EXE%" "%NINFER_ACTKV_MODEL%" ^
  --host 127.0.0.1 --port 18200 --model-id ninfer-27b ^
  --max-context 262144 --kv-capacity auto --kv-headroom-mib 512 ^
  --max-concurrency 1 --default-max-tokens 65536 --prefill-chunk 256 ^
  --kv-dtype int8 --spec mtp --draft-tokens 4 ^
  --pending-timeout-ms 600000 ^
  --kvmem-selected 36864 --kvmem-reserve 16384 ^
  --kvmem-sink 0 --kvmem-recent 1024 ^
  --kvmem-host-mib 12288 --kvmem-retained 4 ^
  %NINFER_ACTKV_SWITCH% --kvmem-lrfu-lambda 0.50 ^
  --kvmem-lrfu-top-frac 0.90 --kvmem-lrfu-fresh-frac 0 ^
  --kvmem-score-agg %NINFER_ACTKV_AGG% --cors --log-level info
exit /b %ERRORLEVEL%
