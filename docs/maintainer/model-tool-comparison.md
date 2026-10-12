# Swift / Qwen tool-call comparison — 2026-10-12

## Question and tested scope

The user reported visible malformed XML such as `<tool_call> Read <function=Bash> ...` and asked whether replacing Swift with Qwen or IQ4 would help. This comparison tests the available deployed quantized artifacts; it is not an intelligence benchmark or an isolated fine-tuning experiment.

- Swift: `models/swift15_iq3s_mtp.ninfer`, Swift-1.5 GSQ-RCO IQ3_S.
- Qwen: `models/qwen38-27b-iq3s-ninfer-v3.ninfer`, WaveCut Qwen GSQ-RCO IQ3_S. This is also processed/quantized, not untouched official full-precision Qwen.
- Latest engine: `build-sm70/apps/ninfer-serve.exe`, source master `7481c23`; V100-SXM2-16GB, INT8 KV, 32K physical KVMem window, MTP3, indexed proposal head, vision disabled in both latest-engine comparisons.
- Identical public requests, temperature 0.6/top_p 0.95; OpenAI seed42. Omitted sampler fields use each artifact's defaults. Artifact templates, mixed quantization and original fine-tuning remain potential differences.

## Cases and observable criteria

Sixteen OpenAI task cases each run with thinking off and low thinking (32 requests): report reading, sed ranges, Chinese/space paths, arithmetic, declared integer/strings, multiline content, literal tool XML in a string, editing, tool-result follow-up, a roughly4K background and contaminated XML history. Eight cases each run through Anthropic adaptive low/high effort (16 requests). No constrained decoding or forced tool_choice was enabled.

The harness checks structured API tool calls, required argument types, selected first tool and the relevant checked arguments. It does not execute generated shell commands or arbitrary file writes. Follow-up uses controlled fixture tool results. These are format/core-argument tests; they do not demonstrate successful unrestricted Agent execution, all read-range semantics, or long-session reliability.

| Configuration | OpenAI valid calls/schema | Anthropic valid calls/schema | Checked first-tool/core arguments |
|---|---:|---:|---:|
| Swift, latest engine |32/32|16/16|48/48|
| Qwen GSQ-RCO, latest engine |32/32|16/16|47/48|
| Swift, previous production binary |32/32|not run|32/32|

Qwen's single first-tool mismatch was arithmetic with thinking disabled: it chose Bash to calculate before the requested record_result call. That can be a legitimate intermediate plan; this harness did not continue that case. It is not malformed XML or proof of lower intelligence. No tested case leaked an actual tool region into visible text. The literal XML writing case produced a structured Write call with the expected string.

Mean request latency including TTFT: Swift latest OpenAI2.91s / Anthropic2.71s; Qwen2.92s /2.63s. Each result is a single small workload sample. There is no meaningful speed advantage established here.

## Real failure remains outside the reproduced conditions

Existing production log `runtime-logs/native-gguf-final-production.stderr.log` records req#59: Anthropic,48097 input tokens,10891 output tokens,201.1s total, stop token, followed by `tool markup returned as text | malformed structure`. The screenshot's stray Read label is not handled by the new missing-closer recovery. The exact original request/history has not been captured, so the controlled short comparison neither reproduces nor fixes that case. No conclusion that Swift is worse than Qwen, or that IQ4 fixes it, follows from these results.

## IQ4 availability and capacity

Verified via the pinned Hugging Face model API and author card/layout:
`bartowski/ukisai_Swift-1.5-Qwen3.8-27b-GGUF`, revision `ce202c7d9a5d71969c60bf62a4f1c3e41747520c`.

| Candidate |Exact file bytes|GiB|Assessment on V10016GB|
|---|---:|---:|---|
|IQ4_XS|15475951936|14.41|Full current MTP/vision/32K configuration cannot fit. A text-only, no-MTP/indexed-head, small-window configuration may fit; not measured.|
|IQ4_NL|17442400576|16.25|No useful full-resident runtime budget on this GPU.|
|Q4_K_S|16363514176|15.24|Also very tight once runtime/KV/optional components are included; not measured.|
|Q4_K_M|17442400576|16.25|Same capacity problem as IQ4_NL.|

Current Swift GGUF is12120016896bytes (11.29GiB); IQ4_XS is3.36GB larger. Files include metadata/optional weights, so file size is not an exact device-residency number. A small text window such as4–8K is a tentative configuration to investigate, not a supported or validated recommendation.

The author explicitly stores MTP matrices at Q4_0 except Q8_0. The current native GGUF directory/projection registry does not support Q4_0, and constructs embedded MTP objects even when the backend is off. This exact IQ4 file therefore cannot currently be loaded unchanged just by disabling MTP; loading and/or Q4_0 operator support must first be extended and qualified. Its scalar metadata and RoPE/template compatibility also require checking.

Only API metadata, README and allocation files were fetched. Attempts to fetch the first16MB header failed with TLS errors; no complete15GB IQ4 model was downloaded and no IQ4 quality/performance claim is made. Evidence is in `runtime-logs/agent-model-candidate-list.json` and `candidate-iq4-*`.

## Practical next steps

1. Keep the current Swift configuration provisionally: this comparison does not justify replacing it with Qwen for tool reliability.
2. Capture and replay the original long Anthropic request with identical history/tools/sampling. Compare old/latest parser, low/high thinking and both models on that request; distinguish malformed syntax, output truncation, history contamination and KVMem effects. Do not indiscriminately execute repaired text as a command.
3. If IQ4 remains desired, first add qualified Q4_0 support or a deliberate text-only loading route, then measure actual residency with small KV capacity and no vision/speculation. Benchmark quality only after it genuinely loads.

All test services were stopped and the original Swift release service restored on8088 (recordedPID42180; verify again before acting). Source/runtime binaries were not replaced or committed for this comparison. Raw result files: `runtime-logs/tool-model-ab-{swift-old,swift-new,qwen}.json` and `tool-anthropic-ab-{swift-new,qwen}.json`; rerunnable harnesses are under `development/ninfer-build/_scratch/model-tool-*.py`.


## Follow-up: raw-generation and header preservation (2026-10-12)

An opt-in `ninfer_tool_raw_stress` diagnostic decodes accepted generated token IDs independently of output parsing. It runs through the public Engine with real MTP/KVMem and records original text, finish reason, actual input/output counts and prefix reuse. Invoke it as `MODEL MANIFEST.json RESULTS.json [mtp|plain] [sink_tokens] [verify]`. Fixtures and manifests are under the project `_scratch/raw-tool-stress`; output files are in runtime-logs. These are synthetic stress histories, not the captured original ZCode request.

Swift produced valid raw calls in10 cases at up to47855 prompt tokens, including contaminated old XML history, unlimited high effort, a1024-token thinking budget and no thinking. With76 tool definitions, the header probe was13886tokens and actual protected preamble13875tokens (217pages). Both Swift and Qwen produced valid raw calls in five further cases each, up to61061tokens. Thus this test did not establish that replacing Swift fixes the user's actual failure.

However, a concrete memory-policy gap was confirmed: with the previous5120-token fixed sink, the first post-query selected set retained only124/217 header pages, omitting93pages. Before the query arrives, fallback recent-page selection can also omit non-sink instruction pages. Successful output in these samples does not make that instruction loss correct, nor prove it caused the screenshot.

The implementation now marks the exact leading instruction/tool boundary at preparation and makes its pages mandatory during all KVMem pressure selections, independently of cache-control policy. The same unmodified5120 startup setting retained all217/217 pages after the fix; eight checked selections also passed native payload/state verification. Five post-fix raw cases passed, including four61061-token requests and history reuse60998/61061. The cold long request completed in135.37seconds and warm cases in7.9–8.6seconds; these are samples, not a general speed claim. Frontend/RuntimeMechanisms/ServeOptions/OpenAI/Anthropic regressions passed.

The production Swift service has been updated to this build without changing the model,65536 output default, MTP3 or32K physical window. Local failed-request capture is enabled so the next real malformed request can be replayed instead of diagnosed from screenshots alone. Oversized protected instruction requests were observed to returnHTTP400 while health remainedOK; an intentionally malformed local display-test request was captured with byte-exact original body. No IQ4 model was installed and no claim of universally reliable Agent operation is made.


## Real Agent admission regression and corrected allocation

The first real request after automatic protection was rejected: 76 tools plus leading instructions required28889 protected tokens, exceeding the24576 selected budget. This is a genuine usability regression of the old startup allocation with the new mandatory protection, not malformed model output. Bounded opt-in diagnostics now also retain context-length admission failures and the error reports the exact protected and selected counts.

The Swift launcher and deployed service now use selected30720/reserve2048, preserving the same32768-token physical window and2048-token prefill chunk. The original failing Anthropic SSE request was replayed unchanged, including its32000 output budget: HTTP200,33208 input tokens,316 output tokens, end_turn,83.22seconds. It returned text/thinking and did not require a tool execution, so this proves successful admission/completion for this request, not universal tool reliability. The replay and conversation body remain local and are excluded from the source publication. The larger selected allocation leaves less room for history and more frequent pressure selection; no general throughput improvement is claimed.
