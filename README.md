# KVMem NInfer · Windows V100 · Qwen3.8-27B (Swift-1.5)

**English** | [简体中文](README.zh-CN.md)

KVMem NInfer runs Qwen3.8-27B (recommended artifact: the **Swift-1.5 GSQ-RCO IQ3_S** community
finetune, MTP included) on a single **Tesla V100 16 GB** under Windows, with a host-archived KV
working set, OpenAI/Anthropic-compatible HTTP serving, image/video input, and a native GGUF
loader. It is the Windows/V100 KVMem line of the [NInfer](#upstream-artifacts-and-credits)
engine, maintained from real 16 GB measurements, not paper specs.

## What this project adds

- **KVMem long context** — history KV lives in host memory as paged records; the GPU keeps a
  bounded working window (`--kvmem-selected` B + `--kvmem-reserve` R). 262144 is the logical
  context ceiling, 32768 the GPU window on the 16 GB production profile — KVMem selects history
  pages, it is **not** full attention over the archive. Cross-request history reuse restores
  thousands of pages with >99% cache hit in the verified multi-round tests.
- **Instruction and tool-header protection** — the exact prepared-token span of the
  system/developer instructions and the complete tool preamble is pinned in every pressure
  selection, independent of client cache-control hints
  ([serving.md](docs/serving.md#kvmem-instruction-and-tool-preservation)).
- **ActKV retrieval, opt-in** — `--kvmem-action-lrfu`, `--kvmem-lrfu-lambda`,
  `--kvmem-lrfu-top-frac`, `--kvmem-lrfu-fresh-frac`, `--kvmem-score-agg mean|max`, ported from
  the llama.cpp KVMem backend. Default stays the historical one-shot cosine ranking; measured
  neutral at the production 32K window and guarded under small windows by the fresh-slot
  reservation ([details](docs/serving.md#actkv-action-oriented-retrieval-opt-in)).
- **Native GGUF loading** — Qwen/Swift dense 27B `.gguf` runs directly on the CUDA engine,
  keeping the original K/IQ quantization blocks: text, embedded MTP, indexed draft heads and
  matching `--mmproj` vision, without converting the model
  ([scope and launch](docs/maintainer/native-gguf-loading.md)).
- **Long-context correctness fixes** — per-chunk statistics origin (fixes late-query zero Q),
  CUDA-graph allocation capture, host-history restore envelope, GGUF scratch growth during
  capture.
- **Tool-call robustness** — tolerant parser for malformed/truncated tool markup with strict
  rejection of ambiguous cases, plus a bounded local failure-capture diagnostic
  (`--failed-tool-request-dir`).
- **Monitoring** — `GET /stats` (JSON), `GET /metrics` (Prometheus), `GET /slots` (execution-lane
  snapshot).

## Quick start

```powershell
git clone https://github.com/cgz12345/kvmem-ninfer-v100.git
cd kvmem-ninfer-v100
```

1. Set up Windows, Visual Studio C++, CUDA 12.8, CMake, Ninja and the vision dependencies
   ([install guide](docs/windows/SWIFT-KVMEM.md)).
2. Download the Swift IQ3_S artifact to `models/swift15_iq3s_mtp.ninfer` and respect its license.
3. Build, then run `scripts/windows/start-swift15-256k.bat`.
4. API at `http://127.0.0.1:8088/v1`; model name `qwen3.8-27b`; auth disabled, any key works.

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8088/v1", api_key="local")
r = client.chat.completions.create(
    model="qwen3.8-27b",
    messages=[{"role": "user", "content": "Hello, introduce yourself"}],
    max_tokens=512,
)
print(r.choices[0].message.content)
```

## Verified scope

- Swift IQ3_S, MTP K=3, int8 KV, 32K window / 2048 prefill, vision on: 4030-in / 256-out twice at
  **96.6 tok/s decode**; MTP acceptance 86%; canary and multi-round history replay pass.
- 256K-class retrieval spot checks: a 260,936-token haystack answers the planted mid-document
  code correctly after the statistics-origin fix; a 100,091-token request completes in 412 s
  with intact SSE/usage. Three passing samples are **not** a 256K quality certification.
- Default output budget 65,536; the engine emits until the model stops (observed 22,421 tokens);
  a full 65,536-token single generation is not yet verified.
- DFlash2 is not usable on this card with the current artifacts (VRAM + head format); MTP is the
  supported speculative backend. Drive-by caveats and all raw evidence live in
  [KVMMEM-PORT-NOTES.md](../KVMMEM-PORT-NOTES.md) and `runtime-logs/`.

## When something goes wrong

Include GPU model, driver, build revision, launch arguments and sanitized logs in issues. The
repo's `development` notes record the exact commands and evidence for every verified claim.

## Build

Windows MSVC recipe: `development/ninfer-build/build-vision.bat` (VC 14.44, CUDA 12.8, Ninja) —
see [V100 build notes](docs/V100-BUILD.md) for flags and upstream Linux builds.

## Upstream, artifacts and credits

This repository is the Windows/V100 **KVMem** maintenance line of **NInfer**
([Neroued/ninfer](https://github.com/Neroued/ninfer) → geoffwatts → liujun-7788 → JimmyMax →
this line). Supported `.ninfer` artifact versions: **v2 and v3**, including `nvfp4`,
`nvfp4full` (per-tensor format probing) and community conversions resolved through the
converter recipe without touching artifact bytes. Hardware target: **V100 (sm_70)**.
Upstream's own V100 benchmark numbers (`nvfp4` artifacts, 219 tok/s decode) and historical
material are preserved in [docs/v100.md](docs/v100.md) and
[docs/V100-BUILD.md](docs/V100-BUILD.md); attribution and licenses in [NOTICE](NOTICE.md).

## License and contributions

The engine follows the root [LICENSE](LICENSE). Swift weights carry their own Swift Open License
and base-model terms from their download source. Issues and pull requests are welcome; this fork
is not affiliated with or endorsed by the upstream authors.
