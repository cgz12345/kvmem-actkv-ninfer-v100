# NInfer for V100 — KVMem, ActKV & native GGUF

**English** | [简体中文](README.zh-CN.md)

KVMem NInfer runs Qwen3.8-27B (recommended artifact: the **Swift-1.5 GSQ-RCO IQ3_S** community
finetune, MTP included) on a single **Tesla V100 16 GB** under Windows, with a host-archived KV
working set, OpenAI/Anthropic-compatible HTTP serving, image/video input, and a native GGUF
loader. It is the Windows/V100 KVMem line of the [NInfer](#upstream-artifacts-and-credits)
engine, maintained from real 16 GB measurements, not paper specs.

**[Windows setup](docs/windows/SWIFT-KVMEM.md) · [Native GGUF support](docs/maintainer/native-gguf-loading.md) · [ActKV results](docs/maintainer/actkv-reproduction.md)**

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
  `--kvmem-lrfu-top-frac`, `--kvmem-lrfu-fresh-frac`, `--kvmem-score-agg mean`, ported from
  the llama.cpp KVMem backend. Default stays the historical one-shot cosine ranking; measured
  neutral at the production 32K window and guarded under small windows by the fresh-slot
  reservation ([details](docs/serving.md#actkv-action-oriented-retrieval-opt-in)).
  The [reference reproduction](docs/maintainer/actkv-reproduction.md) provides a mean-profile
  launcher and the original 33-needle/80-call workload. The Qwen mean profile passed 33/33
  immediate and final recall, 80 tool calls and 14/14 cross-tool recall; baseline also passed.
  Mean and max are evaluated separately; no advantage over baseline is established.
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
git clone https://github.com/cgz12345/kvmem-actkv-ninfer-v100.git
cd kvmem-actkv-ninfer-v100
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

## Serving profiles

Three launchers cover the tested configurations; each keeps the model, KV geometry and
speculation flags explicit:

| Launcher | Artifact | KV window (selected + reserve) | Notes |
|---|---|---|---|
| `scripts/windows/start-swift15-256k.bat` | Swift-1.5 GSQ-RCO IQ3_S | 30720 + 2048 | Local production profile: sink 5120, recent 64, vision, MTP K=3, int8 KV, **ActKV mean + fresh-frac 0.5** |
| `scripts/windows/start-actkv-256k.bat MODEL on\|off\|max` | Qwen3.8-27B GSQ-RCO IQ3_S | 36864 + 16384 | ActKV reference profile: sink 0, recent 1024, MTP K=4, int8 KV; `on` = LRFU mean (validated 33/33), `off` = baseline, `max` = experimental arm |
| `scripts/windows/start-native-gguf.bat MODEL.gguf [mtp] [MMPROJ.gguf]` | Qwen/Swift dense 27B `.gguf` | per configuration | Native GGUF route: original K/IQ blocks, embedded MTP and draft head, optional mmproj vision |

All profiles serve the same HTTP surface (port 8088 for the Swift profile, 18200 for the reference
instance) with model id `qwen3.8-27b` (`ninfer-27b` for the reference launcher).

The Swift launcher defaults to 262,144 logical context tokens, a 32,768-token GPU KV
window, 65,536 output tokens and a 16,384-token thinking budget. Native effort tiers are
`low`, `medium`, `xhigh`; `minimal` maps to `low`, `high`/`max` to `xhigh`, and `none`
disables thinking. Request fields override sampling defaults. `json_object` is a prompt-level
hint, not grammar-constrained JSON; named forced tool selection and `json_schema` are rejected.

## Measured results

The numbers below come from this card (Tesla V100 16 GB, Windows, CUDA 12.8) and the binaries in
this repository; raw logs stay local in `runtime-logs/`.

### ActKV reference reproduction

The [reference profile](docs/maintainer/actkv-reproduction.md) runs the community author's own
harness ([tools/bench/actkv/niah33.py](tools/bench/actkv/niah33.py)): 15 write rounds, 33 needles,
a final recall sweep and cross-tool checks over a 258,202-token session on the
Qwen3.8-27B GSQ-RCO IQ3_S artifact. Acceptance is exit 0 = 33/33 same-turn, 33/33 strict final,
14/14 cross-tool, at least 80 tool calls:

| Configuration | Immediate recall | Strict final | Tool calls | Cross-tool |
|---|---:|---:|---:|---:|
| Baseline, LRFU off | 33/33 | 33/33 | 80 | 14/14 |
| **ActKV LRFU, mean aggregation** | **33/33** | **33/33** | **80** | **14/14** |
| ActKV LRFU, max aggregation | 28/33 | 30/33 | 67 | 6/14 |

The mean profile reproduces the community result exactly on this fixture; the max aggregation
regresses. The engine keeps ActKV off by default and the shipped launchers enable only tested
profiles ([start-actkv-256k.bat](scripts/windows/start-actkv-256k.bat) for the reference
artifact, `start-swift15-256k.bat` for Swift below). No speed or accuracy advantage over the
baseline is established.

### Swift-1.5 production profile (ActKV mean + fresh-frac 0.5 enabled)

- 4030-in / 256-out, twice, no cache: **96.6 tok/s decode**, ~570 tok/s prefill, MTP acceptance 86%.
- Multi-round history reuse restores >99% cached tokens; 36K and 260K session canaries pass.
- Launched by `scripts/windows/start-swift15-256k.bat`: KVMem 30720+2048, sink 5120, recent 64,
  vision, MTP K=3, int8 KV, plus the ActKV mean profile with `--kvmem-lrfu-fresh-frac 0.5`.

### Animated pelican test

The local production profile generated an animated "pelican riding a bicycle" SVG (SMIL
animation: wheels spinning, legs pedaling, gentle bobbing). Prompt: *"Draw a beautiful, lively
animated SVG of a pelican riding a bicycle … Return only the SVG code, no explanation."*

| Artifact | Thinking | Output tokens | Wall time | Decode |
|---|---|---:|---:|---:|
| [Swift-1.5 IQ3_S](docs/assets/pelican-swift-bicycle.svg), production profile | `reasoning_effort: high` (16,421 reasoning) | 21,843 | 293.8 s | 74.7 tok/s |

![Animated pelican on a bicycle](docs/assets/pelican-swift-bicycle.gif)

Context for the numbers: with `reasoning_effort: none` the first, non-animated version completed
in 18.8 s / 1,756 tokens at 96.0 tok/s; with the default thinking mode the same request spent an
entire 8,192-token budget on reasoning and produced no SVG. Budget for thinking whenever the
output must be machine-readable. The GIF capture renders 14 frames of the 1.4 s animation loop
(10 fps); the SVG file animates natively on the GitHub file view.

### Long-context spot checks

A 260,936-token haystack retrieved a planted mid-document code; a 100,091-token request
completed in 412 seconds with intact SSE and usage. See the limitations below and the
[Windows qualification notes](docs/windows/SWIFT-KVMEM.md) for the exact boundaries.

## Reproducing the measurements

```powershell
# ActKV reference: 33 needles, 80 tool calls, deterministic scoring (Qwen GSQ-RCO artifact)
.\scripts\windows\start-actkv-256k.bat "D:\models\Qwen3.8-27B-GSQ-RCO-IQ3_S.ninfer" on
python tools/bench/actkv/niah33.py --endpoint http://127.0.0.1:18200/v1/chat/completions `
    --output out/actkv/mean.json

# Deterministic ActKV A/B against any live server (seeded workload, identical prompts per arm)
python tests/actkv_ab.py plan                  # print the workload, no server needed
python tests/actkv_ab.py run on actkv-on 8089  # run against a server started with chosen flags
```

Both harnesses are self-contained Python; the A/B workload regenerates deterministically from a
fixed seed, so two arms can be compared with byte-identical prompts.

## API surface

- OpenAI-compatible `/v1/chat/completions`, `/v1/responses`, `/v1/models` and Anthropic
  `/v1/messages`; SSE streaming with `usage`; tool calls with streamed argument fragments.
- Sampling defaults come from the loaded model; per-request overrides cover temperature,
  top-p/top-k, min-p, presence/frequency penalties, seed, `max_tokens` (service default 65,536)
  and `reasoning_effort: none|minimal|low|medium|high|xhigh`.
- Image/video input on the Vision and native-GGUF routes; `--mmproj` loads a matching GGUF
  projector.
- Operations: `GET /health`, `GET /props`, `GET /stats`, `GET /metrics`, `GET /slots`.
- Reliability: system instructions and the complete tool preamble are pinned in every KVMem
  pressure selection; malformed or truncated tool markup is repaired where unambiguous and
  rejected otherwise, with the original request optionally captured locally
  (`--failed-tool-request-dir`).

## Repository layout

```text
targets/        model execution (Qwen3.6/3.8 27B, 35B-A3B), layouts, request planning
ops/            CUDA operators (attention, linear, MoE, GDN/ReplaySSM, quantization)
runtime/        engine core, scheduling, resource and context management
serve/          HTTP server, protocol adapters, tool handling
artifact/       container readers, including the native GGUF path
kvmem/          host KV store, retrieval scoring, residency and IO contracts
docs/           user guides + maintainer references (index in docs/README.md)
scripts/windows launchers for the tested serving profiles
tools/bench     benchmarks, including the ActKV reference harness
tests/          C++ suites and the deterministic ActKV A/B harness
```

## Limitations and roadmap

- 262,144 is a **logical** ceiling with a bounded GPU window (32K production, 53K reference
  profile). KVMem selects history pages; it is not full attention over the archive, and passing
  long-context samples are not a 256K quality certification.
- ActKV is an opt-in KVMem adaptation: the mean profile matches but does not beat the baseline,
  and the max aggregation regresses (table above). The engine default stays off; only the tested
  launcher profiles enable it.
- Swift ships without DFlash2 weights and uses MTP. A separate Qwen artifact passed a limited
  6K-window DFlash2 test; large-window, vision and mixed MTP+DFlash operation are not qualified
  ([details](docs/windows/SWIFT-KVMEM.md#dflash2-gguf-验证2026-10-11)).
- An Engine benchmark generated 65,536 tokens with EOS stopping disabled. This proves the
  tested execution budget, not arbitrary 65K API output or long-conversation quality; API
  generation can finish earlier at EOS. Complex OCR/video and 24-hour stability are not qualified.
- FFmpeg GPL shared-build binaries are not distributed until their complete corresponding source
  is assembled.

## When something goes wrong

Include GPU model, driver, build revision, launch arguments and sanitized logs in issues. The
[Windows qualification notes](docs/windows/SWIFT-KVMEM.md), [ActKV results](docs/maintainer/actkv-reproduction.md), and [optimization audit](docs/maintainer/upstream-audit-20261013.md) record reproducible commands and validation boundaries. Raw logs remain local.

## Build

Follow the repository-relative commands in [Windows setup](docs/windows/SWIFT-KVMEM.md)
(MSVC 14.44, CUDA 12.8, Ninja). [V100 build notes](docs/V100-BUILD.md) include the upstream Linux path.

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
