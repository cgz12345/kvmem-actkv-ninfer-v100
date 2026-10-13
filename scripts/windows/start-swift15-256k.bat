@echo off
rem ============================================================================
rem  NInfer V100 + KVMem - Swift-1.5-Qwen3.8-27B (GSQ-RCO IQ3_S) + MTP + Vision
rem  Large Agent headers use 30K selected + 2K reserve in the same 32K KV window.
rem
rem  ActKV for this artifact: LRFU with mean aggregation plus the small-window
rem  fresh-slot mitigation (verified equal to baseline; see
rem  docs/maintainer/actkv-reproduction.md and KVMMEM-PORT-NOTES.md section 35).
rem
rem  --default-thinking-budget bounds interactive thinking: clients that do not
rem  ask for a budget (the bundled WebUI sends none) can otherwise spend the
rem  whole output budget on reasoning without emitting an answer.
rem ============================================================================
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set PATH=%CUDA_PATH%\bin;%PATH%
"%~dp0..\..\build-sm70\apps\ninfer-serve.exe" ^
  "%~dp0..\..\models\swift15_iq3s_mtp.ninfer" ^
  --host 127.0.0.1 --port 8088 ^
  --model-id qwen3.8-27b ^
  --max-context 262144 ^
  --kv-capacity auto --kv-headroom-mib 512 ^
  --max-concurrency 1 ^
  --default-max-tokens 65536 ^
  --default-thinking-budget 16384 ^
  --prefill-chunk 2048 ^
  --kv-dtype int8 ^
  --spec mtp --draft-tokens 3 ^
  --lm-head-draft ^
  --preserve-thinking ^
  --pending-timeout-ms 600000 ^
  --kvmem-selected 30720 --kvmem-reserve 2048 ^
  --kvmem-sink 5120 --kvmem-recent 64 ^
  --kvmem-host-mib 12288 --kvmem-retained 4 ^
  --vision ^
  --kvmem-action-lrfu --kvmem-score-agg mean --kvmem-lrfu-fresh-frac 0.5 ^
  --cors ^
  --log-level info
