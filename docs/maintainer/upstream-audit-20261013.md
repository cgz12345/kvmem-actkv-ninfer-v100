# Upstream audit and V100 discovery entry — 2026-10-13

## Current deployment
Swift IQ3_S production on localhost:8088 enables ActKV LRFU, GQA mean and fresh fraction 0.5. Health and automatic get_weather tool call passed. The separate Swift production A/B completed: off 12/12 (511.2s, 98.5 tok/s), mean + fresh 0.5 12/12 (510.8s, 98.4 tok/s). This is distinct from the Qwen 33-needle workload.

## Incoming code
- JimmyMax/ninfer-v100: origin/master 0d24f83, 12 commits ahead of the previously integrated upstream baseline. db947f9 chained FP16 attention PV accumulation adapted from attention/ to our softmax_attention/ directory and current kernel API. Product and numerical-test builds passed. The complete expanded attention suite passed, including six added 32K/128K fragmented INT8 cases without relaxing tolerances. NVFP4/FP8 decode staging and residual fusion from 6856d32 also passed the independent NVFP4 A16 Linear, NVFP4 LinearAdd and FP8 LinearAdd suites. The optional draft attention window is not included: it changes draft attention and needs a separate acceptance/quality qualification. The serving binary was rebuilt and deployed by the concurrent Agent-interface work; its SHA256 matches the final source build.
- melis2023/ninfer-kvmem-actkv: c14704d, four README-only updates since pinned 1c75f79. No new algorithm to import.
- kvmem/kvmem-llama.cpp: master 7db86eb, stable 0.18.0. Strict constrained tool output, SSD tiering and shared LRFU merit separate integration. Current upstream patch already reserves one scalar RoPE axis for text-only KVMem: do not submit our same 1KB fix as an unresolved upstream bug.
- Neroued/ninfer: HEAD 81c8ce0; registered Linux/5090 scope. Our Windows/V100 adaptation belongs in a community-port discovery entry, rather than a claim of a supported upstream GPU.

## Local operator A/B

Tesla V100, Windows/MSVC 14.44, CUDA 12.8. Graph cached-only INT8 attention, batch 1, cold cache, 5 warmups and 30 repetitions per shape. An idle Swift service remained resident because stopping it was rejected by automatic approval review; there was no inference request during this benchmark. This measures the operator, not end-to-end generation.

Command: `ninfer_causal_softmax_attention_bench --entry cached --geometry all --kv-dtype int8 --tokens 1,4 --context 8192,32768,131072 --execution graph --cache cold --warmup 5 --repeat 30 --csv-out ...`

Compare `NINFER_SM70_ATTN_V3=0` (old per-group accumulation) and `=1` (chained PV, new default). `NINFER_SM70_ATTN_V2=0` still selects the older pre-v2 kernel.

| Geometry | Keys | Width | Old median µs | New median µs | Time reduction |
|---|---:|---:|---:|---:|---:|
| d256-h24-kv4 | 8192 | 1 | 94.208 | 79.872 | 15.2% |
| d256-h24-kv4 | 8192 | 4 | 122.880 | 102.400 | 16.7% |
| d256-h24-kv4 | 32768 | 1 | 238.592 | 191.488 | 19.7% |
| d256-h24-kv4 | 32768 | 4 | 262.144 | 218.112 | 16.8% |
| d256-h24-kv4 | 131072 | 1 | 748.544 | 605.184 | 19.2% |
| d256-h24-kv4 | 131072 | 4 | 772.096 | 693.248 | 10.2% |
| d256-h16-kv2 | 8192 | 1 | 65.536 | 61.440 | 6.2% |
| d256-h16-kv2 | 8192 | 4 | 116.736 | 113.664 | 2.6% |
| d256-h16-kv2 | 32768 | 1 | 132.096 | 119.808 | 9.3% |
| d256-h16-kv2 | 32768 | 4 | 171.008 | 159.744 | 6.6% |
| d256-h16-kv2 | 131072 | 1 | 417.792 | 389.120 | 6.9% |
| d256-h16-kv2 | 131072 | 4 | 536.576 | 507.904 | 5.3% |

Local logs: `runtime-logs/attention-v3-{long-oracle,oracle-1,bench-0,bench-1}.log` and benchmark CSVs. No private requests or raw logs are published.

## Deployment and sharing

ActKV remains enabled in the existing Swift production service. Final `build-sm70/apps` and `release/bin` server/CLI/perplexity executables are synchronized. Production PID 29688 is healthy, uses ActKV mean + fresh 0.5 and a 16384 thinking budget; the 22 Agent probes passed. The earlier stop rejection no longer blocks deployment: another authorized session had already rebuilt and restarted the server with these shared-tree changes. No claim of whole-model speedup is made.

The user cancelled external replies on 2026-10-13. No discussion, issue comment or PR has been posted. README improvements were independently published by the concurrent session. This optimization and its evidence are synchronized separately.

## Qualification commands

- Product build: `development/ninfer-build/rebuild-targets.bat` (passed, correct MSVC header tracking).
- `ninfer_softmax_attention_test`: PASS for the expanded suite, independent FP64 attention oracle and unchanged INT8 relative-L2 limit 0.00315. Six new fragmented 32K/128K cases have relative-L2 0.00178–0.00187.
- `ninfer_linear_nvfp4_a16_test`: exit 0, `OK NVFP4_A16 Linear`.
- `ninfer_linear_add_nvfp4_test`: exit 0, `OK NVFP4 linear_add`.
- `ninfer_linear_add_fp8_test`: exit 0, `OK FP8 linear_add`.
- `python tests/serve_agent_compat.py`: 22/22; production `/health`: 200.

The NVFP4/FP8 changes do not establish a speed improvement for Swift IQ3_S linear layers. The new INT8 attention path benefits both documented attention geometries in the measured operator sample. No exhaustive best-possible performance or universal 256K quality claim is made.
