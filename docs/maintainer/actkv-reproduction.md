# ActKV reference reproduction

## Scope and source

Reference: [melis2023/ninfer-kvmem-actkv](https://github.com/melis2023/ninfer-kvmem-actkv), code revision `1c75f798eb59ca5629f0b8db259c2b3c0df6a84d`. Upstream `c14704d` only changes the README; the benchmark and policy code are unchanged.

The community report uses LRFU with **mean** aggregation. The previously tested LRFU+max combination is a separate configuration. The port carries LRFU, nonnegative keep scores, once-per-round updates and deferred selection commits; the optional fresh-slot extension remains zero in reference runs. This is the community KVMem adaptation, not a full implementation of the ActKV paper's action-attention recovery and confidence-driven budget control.

## Reproduce

Build the Windows sm70 server following [V100 build](../V100-BUILD.md). Use a separate test instance and the Qwen3.8 dense 27B GSQ-RCO IQ3_S MTP artifact. The qualified profile is text-only; it does not qualify native BF16 vision or the Swift production preset.

```powershell
.\scripts\windows\start-actkv-256k.bat "D:\models\Qwen.ninfer" on
```

In another terminal:

```powershell
python tools/bench/actkv/niah33.py --endpoint http://127.0.0.1:18200/v1/chat/completions --output out/actkv/mean.json
```

Restart the same launcher with `off` for the baseline and use a different result path. `max` enables LRFU+max as a separate experimental arm. All three modes retain the same model and window: selected36864/reserve16384, 53248 physical tokens, recent1024, prefill256, INT8 KV, MTP4, max-context262144, Host archive12GiB. The reference has no extra fixed sink setting; automatic instruction protection remains enabled.

The workload and recall scoring match the original script: 15 writes with435 filler lines, 33 needles, final recall, and14 cross-tool checks. Only endpoint/output paths and final acceptance are adapted. Exit0 requires33/33 immediate,33/33 strict final,14/14 cross-tool and at least80 tool calls; a successful retry union alone is insufficient. Tool execution uses local controlled stubs, including send_alert; no messages are sent externally.

## Qualification

The complete reference workload passed on2026-10-13. Full mean used the exact profile above with fresh-frac0; no fresh-slot extension or forced tool choice was needed.

| Configuration | Immediate recall | Strict final recall | Tool calls | Cross-tool recall |
|---|---:|---:|---:|---:|
| Baseline, LRFU off / mean |33/33|33/33|80|14/14|
| LRFU on / mean / fresh0 |33/33|33/33|80|14/14|
| LRFU on / max / fresh0 |28/33|30/33|67|6/14|

The mean run completed normally with no retry needed. Peak observed prompt was **258,202 tokens** under the262144 logical ceiling, using the53248-token physical window. The post-fill write-prefill medians were193.3tok/s baseline and193.3tok/s mean. These are single-run stage samples; the reference harness reports the final HTTP hop for some multi-hop probe timings, so they must not be summed as whole-trace wall time.

This qualifies the community mean profile on this fixture and resolves the incomplete13-round result. Both baseline and mean are perfect here: it does **not** establish an ActKV speed or accuracy advantage, and the max arm regresses. Keep the global option off unless selecting a tested profile. Smaller-window Swift results and fresh-frac0.5 mitigation remain separate experiments. Local raw logs and model weights are not part of source publication.

The platform is V10016GB versus the reference RTX5060Ti16GB. The engine revision and model-byte identity with the author's private artifact are not identical verified conditions. Current MTP context lookup supports15 copied drafts; the reference explicitly requests31 with adaptive MTP. CPU vision residency is also outside this text-only reproduction. Absolute GPU speeds and all hardware configurations are not qualified by this test.

## Workspace correction

Text-only compressed prefill uses an additional scalar RoPE position tensor. The startup workspace plan now reserves that axis as well as the existing three-axis vision route. At prefill256, the missing allocation was1024 bytes. The original65813-token failing request and subsequent reuse completed after the fix. This correction is independent of the ActKV ranking policy.
