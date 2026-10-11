@echo off
setlocal
rem Native Qwen/Swift dense 27B GGUF. Optional second argument: mtp; third argument: matching mmproj GGUF.
if "%~1"=="" (
  echo Usage: %~nx0 "C:\models\Swift.gguf" [mtp] ["C:\models\mmproj.gguf"]
  exit /b 2
)
set "NINFER_GGUF_MODEL=%~f1"
if not exist "%NINFER_GGUF_MODEL%" (
  echo Model not found: "%NINFER_GGUF_MODEL%"
  exit /b 2
)
set "NINFER_GGUF_SPEC="
if /I "%~2"=="mtp" (
  set "NINFER_GGUF_SPEC=--spec mtp --draft-tokens 3 --lm-head-draft"
) else if not "%~2"=="" (
  echo Second argument must be mtp or omitted.
  exit /b 2
)
set "NINFER_GGUF_VISION="
set "NINFER_GGUF_WINDOW=--kvmem-selected 24576 --kvmem-reserve 8192 --kv-headroom-mib 512"
if not "%~3"=="" (
  if not exist "%~f3" (
    echo Projector not found: "%~f3"
    exit /b 2
  )
  set NINFER_GGUF_VISION=--vision --mmproj "%~f3"
  rem BF16 vision + MTP + indexed head fit a 20K window on V100 16GB.
  set "NINFER_GGUF_WINDOW=--kvmem-selected 16384 --kvmem-reserve 4096 --kv-headroom-mib 256"
)
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "PATH=%~dp0..\..\..\..\..\release\bin;%CUDA_PATH%\bin;%PATH%"
"%~dp0..\..\build-sm70\apps\ninfer-serve.exe" ^
  "%NINFER_GGUF_MODEL%" ^
  --host 127.0.0.1 --port 8088 --model-id qwen3.8-27b ^
  --max-context 262144 --kv-capacity auto ^
  --max-concurrency 1 --default-max-tokens 65536 ^
  --prefill-chunk 2048 --kv-dtype int8 ^
  %NINFER_GGUF_SPEC% %NINFER_GGUF_VISION% --preserve-thinking ^
  --pending-timeout-ms 600000 ^
  %NINFER_GGUF_WINDOW% ^
  --kvmem-sink 5120 --kvmem-recent 64 ^
  --kvmem-host-mib 12288 --kvmem-retained 4 ^
  --cors --log-level info
exit /b %ERRORLEVEL%
