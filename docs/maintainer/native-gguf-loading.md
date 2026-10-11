# Native GGUF loading

NInfer can load supported Qwen/Swift dense 27B GGUF files directly. It uses the existing CUDA engine and KVMem, without writing a converted `.ninfer` model or re-quantizing block weights.

## Supported scope

- Little-endian GGUF v2/v3; `qwen35` architecture, 64 text layers (optional embedded 65th MTP layer is supported), hidden size 5120, vocabulary 248320, and the registered 27B geometry.
- Projection formats: Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS. A model may mix these formats.
- Text generation, structured tool calls and the existing KVMem route. Embedded MTP is supported with `--spec mtp --draft-tokens 3`; a file without the embedded layer fails clearly when MTP is selected. Optimized draft heads are selected with `--lm-head-draft`. Image/video input uses `--vision --mmproj MATCHING_PROJECTOR.gguf`. DFlash is not supported on this native route.
- This is not support for arbitrary GGUF architectures, split GGUF shards, FP16/BF16 projection models or arbitrary vision projectors. The supported external projector is the matching Qwen dense 27B `qwen3vl_merger` with 27 layers and BF16/F16/F32 tensors.

## Loading and frontend resources

The GGUF contains its token inventory, BPE merge order and registered chat template. Native loading reads them directly; no `.assets` directory is needed. The target validates the Qwen tokenizer domain and uses its registered image/video processor defaults. Unknown architectures, templates and mismatched geometry fail explicitly.

```powershell
.\ninfer-serve.exe C:\models\Swift.gguf --host 127.0.0.1 --port 8088 --max-context 262144 --kv-dtype int8 --kvmem-selected 24576 --kvmem-reserve 8192 --kvmem-host-mib 12288 --spec mtp --draft-tokens 3 --lm-head-draft
```

Omit the speculative flags when the file has no embedded MTP layer. Add `--vision --mmproj C:\models\mmproj-Swift-bf16.gguf` for a matching projector. `--mmproj` is available through serving, CLI and `EngineOptions::mmproj_path`; `.ninfer` artifacts already carry their vision bundle and reject an external projector.

The indexed proposal head gathers 131072 rows from the original output head, preserving its GGUF block format. Its embedded token shortlist follows the existing frequency ranking and registered special tokens; regenerate/check it with `tools/convert/qwen3_6_27b/generate_native_gguf_shortlist.py [--check]`. It does not write or re-quantize a model file.

Vision projection matrices retain mapped BF16 storage. F16/F32 values are converted to BF16 through bounded reads. Temporal patch weights are stacked as `[out, channel, time, y, x]`; LayerNorm scales are preserved without the text RMSNorm offset. Supply a projector from the same checkpoint; geometry validation alone does not prove checkpoint identity.

## Implementation and validation

The artifact adapter exposes bounded virtual reads of mapped GGUF rows. Attention query/gate interleaving, GDN head ordering and convolution layouts are restored during materialization. Small control weights and normalization vectors are converted to the formats expected by the registered target. Original block matrix bytes remain unchanged.

2026-10-12: Windows/MSVC build passes. The real Swift-1.5 IQ3_S mixed-quant GGUF loaded on a V100 16 GB in 11.4 seconds. A CPU comparison against the existing Swift artifact passed for 858 objects / 11,768,132,657 bytes: block/BF16/resource bytes exact, FP32 tolerance 1e-6. Chinese arithmetic returned 391 for 17*23; tool generation produced `record_result({"value":391})` with `finish_reason: tool_calls`. A 44,077-token prompt correctly retrieved the middle needle `ZX-7749-QK` through the 32,768-token GPU window. The next turn again returned the needle and reused 44,130 of 44,153 prompt tokens. Logs: project runtime-logs/native-gguf-smoke.stderr.log, native-gguf-tool.json, native-gguf-long.json, native-gguf-reuse.json and native-gguf-compare.log. Existing artifact reader/materialization tests and GGUF directory tests pass. Production was restored to the existing Swift MTP+vision release after testing. These checks do not establish performance across all advertised formats.

The directory test rejects malformed headers, unsupported block formats, invalid dimensions/alignment, overlapping payloads and every truncated prefix of its fixture. `ninfer_native_gguf_qwen_check MODEL.gguf REFERENCE.ninfer` performs the real-model CPU comparison; it supports the reference converter's fused row layouts and compares tokenizer/shortlist semantics. An optional third mmproj argument also checks vision tensor mapping against the raw projector with an independent temporal stack and BF16 rounding oracle.

GGUF format reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

## Embedded MTP validation (2026-10-12)

The native adapter maps `blk.64.nextn.*`, the MTP attention/MLP tensors and norm scales without a model-sized repack. Q/gate rows are un-interleaved; separate GGUF parents retain their individual block formats. The MTP execution leaf accepts separate projection parents and uses contiguous Gate/Up buffers for multi-token CUDA Graphs. Runtime norms use `1 + weight`, so all GGUF MTP scales are stored minus one, matching the existing converted artifact exactly.

The complete text+MTP CPU comparison passed: 873 objects / 12,116,549,681 bytes; block/BF16/resource bytes exact, FP32 tolerance 1e-6. Plain decode with the embedded MTP disabled still starts normally. MTP3 CUDA Graph initialization and generation pass on V100 16 GB.

One uncached greedy comparison: arithmetic 43.2 → 85.4 tokens/s, code 43.2 → 92.6 tokens/s, 3576-input / 256-output explanation 42.1 → 66.5 tokens/s. Arithmetic and code text/reasoning matched ordinary decode exactly; the explanation differed. These are small workload samples, not a universal speed or output-identity guarantee.

A 37,477-token MTP prompt retrieved `ZX-7749-QK` beyond the 32,768-token GPU window. The next turn again returned it and reused 37,558 of 37,577 prompt tokens. See runtime-logs/native-gguf-mtp-compare.log, native-gguf-mtp-plain.json, native-gguf-mtp-mtp.json, native-gguf-mtp-long.json, native-gguf-mtp-reuse.json and native-gguf-mtp-smoke2.stderr.log.

The adapter also rejects mismatched RoPE dimensions/sections/base/scaling, norm epsilon, context geometry and inconsistent MTP metadata. Same matrix dimensions alone do not establish target compatibility.

Windows convenience launcher (after building and installing dependency DLLs):

```bat
scripts\windows\start-native-gguf.bat "C:\models\Swift.gguf" mtp
```

Omit `mtp` for ordinary decode; supply a matching mmproj as the third argument for vision. MTP mode enables the indexed proposal head. The launcher checks the model and optional projector, uses INT8 KV with a 32K text window or a 20K window for BF16 vision on V100 16GB, and retains a 65536-token default output budget. Stop any other resident large-model service before launching it on a V100 16 GB.

A same-configuration `.ninfer` MTP3 comparison also matched arithmetic/code messages exactly. The 3576-token free-form explanation differed among ordinary decode, native MTP and converted-artifact MTP; generation identity is not established for all prompts. Weight-layout parity alone does not guarantee bitwise-identical greedy text across execution layouts.

## Complete native route (2026-10-12)

Embedded frontend metadata, indexed MTP proposal head and matching external vision projector are now implemented. The main model remains in its original quantization. The projector must come from the same checkpoint; the tested pair is Swift-1.5 GSQ/RCO IQ3_S main weights and its compatible Swift-1.5 BF16 `qwen3vl_merger` projector.

On V100 16GB, BF16 vision + MTP3 + the indexed Q4_K head require a smaller physical KV window: `--kvmem-selected 16384 --kvmem-reserve 4096 --kv-headroom-mib 256`. Startup uses 12.5 GiB weights and 1.86 GiB runtime, with approximately 478 MiB free. The 262144 logical context setting is retained. The 32K text configuration with the same vision bundle and 512 MiB headroom is rejected by the memory planner; it must not be advertised as fitting this GPU. Existing `.ninfer` quantized vision remains available with its previous 32K configuration.

```bat
scripts\windows\start-native-gguf.bat "C:\models\Swift.gguf" mtp "C:\models\mmproj-Swift-bf16.gguf"
```

The BF16 Linear registry now includes the seven vision projection shapes. The production route passed naive FP64 dot-product checks under the existing A16 criterion at raw-patch extents 4/8/64/132/512 and merged-token extents 1/3/128/129/512. Input weights and output buffer guards were preserved. Real-model CPU checks passed for all token bytes/domain/special flags, BPE behavior, the exact 131072-token shortlist, gathered head rows, text/MTP payloads and 333 vision tensors; the vision oracle independently stacks temporal patch tensors and rounds F32 to BF16, including unaligned reads.

The opt-in HTTP check is `python tests/native_gguf_http.py --url http://127.0.0.1:8089 --output RESULTS.json`. It checks arithmetic, structured tools, image high/low, Responses images, a four-frame AVI, SSE completion/usage, an over-window needle and `previous_response_id` reuse. It expects a caller-managed native GGUF server with MTP, indexed head and vision enabled; it does not start or stop a service.

Final HTTP validation passed on the complete native MTP3 + indexed-head + BF16-vision configuration: 17×23=391, structured `record_result({"value":391})`, red image through high/low Chat input and Responses, real four-frame AVI, and SSE completion with usage. Image high/low prompt sizes were 1050/282 tokens. The 40077-token middle-needle request answered `ZX-7749-QK` in 153.6 seconds; the follow-up answered it again, reusing 40153/40175 input tokens (99.95%) and completing in 1.7 seconds. Decode samples were 80.8/72.5 tokens/s for these two requests, not a general throughput guarantee. Health remained OK. Evidence: project `runtime-logs/native-gguf-complete-final-http.json`, `native-gguf-complete-fixed.stderr.log`, `native-gguf-final-oracle.log`, `native-gguf-vision-linear-oracle.log`. The original Swift `.ninfer` production service was restored after validation; new binaries remain in `build-sm70/apps`, with no commit, push or release performed.

The full 256K retrieval-quality and 65536-token generation boundaries from the project handoff remain unqualified. Native loading does not change those claims. Unrelated upstream features and arbitrary GGUF architectures are outside this implemented route.
