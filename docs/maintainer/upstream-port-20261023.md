# Upstream ActKV/KVMem port — 2026-10-23

Source: `melis2023/ninfer-kvmem-actkv` at `49bf662` (61 commits after the `1c75f79` already
carried by this tree). The port is limited to changes that act on the Windows/V100 sm70 engine
and does not absorb the upstream sm70/sm75 source drop or the DFlash2/NVFP4 work.

## Ported

| Upstream | Change in this tree |
|---|---|
| `b6c8337` merge recall page runs | Spill/restore now pass an empty record-group set because every record is a slice of the one pinned staging allocation. Consecutive physical pages merge into one strided copy per plane. |
| `cbf62d4` reselect at 80% | `prepare_window` starts the swap at 80% of the window when the selected window is below that watermark. Configurations whose selected window is nearly the whole window wait for half of their reserve instead, which avoids reselecting on every chunk after compaction. |
| `3e8fbb7` restore only changed pages | Selection is sorted, compared slot-by-slot with the previous window, and a delta restore copies only changed slots. The path is used when at most half the slots changed and the byte-exact verification mode is off. The new visible frontier is applied with `destructive_truncate` before `commit_frontier`, including the zero-change case. `KVMEM_RESTORE_DIFF=1` prints `prev/sel/changed/inc`. |
| `82118ad` score token-only prompts on their tail | `Frontend::prepare_tokens` sets `memory_query` to the final 1024 tokens (short prompts use the whole prompt) instead of leaving the retrieval query empty. |
| `7431320` retire GQA max | Scoring always uses the folded mean. `--kvmem-score-agg max` is refused with an explanatory error. The fresh-slot mitigation (`--kvmem-lrfu-fresh-frac`) remains. |

Docs, the reference launcher, and the README now describe `mean` as the only aggregation.
The historical max-arm measurement remains in the reproduction record.

## Verification

Build: Windows/MSVC 14.44, CUDA 12.8, sm70. `rebuild-targets.bat` rebuilt `ninfer-serve`,
`ninfer`, and `ninfer-perplexity`; header dependency tracking reported 534 entries and no
unexpected zero dependencies.

Tests (all exit 0): `kvmem_store_test`, `mean_key_retrieval_test`, `memory_contract_test`,
`kvmem_runtime_test`, `ninfer_token_sums_test`, `ninfer_qwen3_6_runtime_mechanisms_test`,
`ninfer_serve_options_test`, `ninfer_qwen3_6_frontend_test` (the new case checks the
1024-token tail query). A server started with `--kvmem-score-agg max` fails before loading
weights with the retirement message.

Reference profile (Qwen3.8 IQ3_S, selected 36864 + reserve 16384, prefill 256, MTP4, ActKV
mean, 33-needle fixture). The 15-round run was stopped after round 4 to preserve the test
window; it is not a new 33/33 claim. Same-turn recall was 3/3, 3/3, 3/3, 2/2. Compared with
the 2026-10-13 baseline on the same fixture:

| Round | Write baseline | Write now | Prefill baseline | Prefill now | Probe baseline | Probe now |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 31.71 s | 31.7 s | 525.1 tok/s | 525.7 tok/s | 1.68 s | 1.6 s |
| 2 | 69.04 s | 68.9 s | 481.2 tok/s | 482.3 tok/s | 2.52 s | 2.1 s |
| 3 | 113.96 s | 112.8 s | 439.9 tok/s | 444.4 tok/s | 4.70 s | 3.0 s |
| 4 | 205.18 s | 199.7 s | 326.8 tok/s | 336.0 tok/s | 2.93 s | 1.3 s |

A 5-round/13-needle variant then completed end to end: same-turn 13/13, final strict
13/13, final union 13/13, cross-tool 5/5, 31 tool calls, exit 0
(`runtime-logs/port-5round/`). This checks that needles from rounds 1-3 are still recalled
after later reselections. The full 33-needle/15-round final sweep was not rerun.

Single-request long prompt (Qwen3.8 IQ3_S, same ActKV/window profile): 98,508 prompt tokens,
answer `N07-015838` correct, 225.87 s total, prefill 436.7 tok/s, decode 104.4 tok/s, MTP
8/8 accepted, finish `stop`, health 200 before and after (`runtime-logs/port-100k/`).

Swift production profile (selected 30720 + reserve 2048, sink 5120, recent 64, MTP3, ActKV
mean + fresh 0.5, 36K prompt): five cache-recovery cases correct (`ORCHID-7319`); cold
72.36 s and 0.95/1.23/2.59/2.06 s on the four reuses, cache > 99.9%
(`runtime-logs/port-swift/`). The zero-change delta path was observed (inc=1).

The changed-slot delta path was exercised with a stable-selection variant (recent 8192,
ActKV off): `prev=482 sel=480 changed=239 inc=1`, same five correct cases
(`runtime-logs/port-delta/`).

## Boundaries

- The full 33-needle/15-round final sweep was not rerun in this round; the 2026-10-13
  full-mean result remains the reference for that full run. This round proves same-turn
  recall, the 5-round/13-needle final sweep, and per-round timing.
- The 100K request is a single fixture, not a controlled A/B against the earlier 100K
  record (different fixture and output length).
- Delta restore falls back to a full restore when more than half the slots change; the
  256K reference still takes the full path.
- `release/bin` was not replaced and no production service was started. The source changes are
  committed as `e44a779` and pushed to `mine/master`; the release binaries remain unchanged.
- DFlash2/NVFP4 and the constrained-tool-call patch were not ported.
