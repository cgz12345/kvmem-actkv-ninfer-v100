# V100 版 NInfer — KVMem、ActKV 与原生 GGUF

[English](README.md) | **简体中文**

KVMem NInfer 让你在**单卡 Tesla V100 16GB** 的 Windows 机器上跑 Qwen3.8-27B（推荐制品：
**Swift-1.5 GSQ-RCO IQ3_S** 社区微调，自带 MTP）：KV 历史归档到主机内存、有限 GPU 工作窗口、
OpenAI/Anthropic 兼容 HTTP 服务、图片/视频输入，以及原生 GGUF 加载。本仓库是 NInfer 的
**Windows/V100 KVMem 维护线**，所有结论来自 16GB 实机测量，不是纸面参数。

## 想在 V100 上用 NInfer / KVMem / ActKV？

这里是 **Windows + NVIDIA Tesla V100（sm_70）社区移植版**。如果上游构建面向更新的显卡，可以从这里开始。

| 功能 | 本项目提供什么 |
|---|---|
| 硬件 | 单卡 Tesla V100 16GB、Windows、CUDA 12.8 |
| 长上下文 | KVMem 将历史 KV 归档到主机内存；逻辑上限 262,144 tokens，GPU 保留有限窗口 |
| 检索 | 可选 ActKV / LRFU、mean/max 聚合，附实测对照结果 |
| 量化 | 原生加载支持范围内的 Qwen/Swift dense 27B GGUF K/IQ 格式，无需转换整个模型 |
| Agent 客户端 | OpenAI / Anthropic API、MTP、工具头保护与视觉输入 |

**[Windows 安装](docs/windows/SWIFT-KVMEM.md) · [原生 GGUF 范围](docs/maintainer/native-gguf-loading.md) · [ActKV 实测与启动](docs/maintainer/actkv-reproduction.md)**

本项目是独立社区分支，上游署名见 [NOTICE](NOTICE.md)。GGUF 支持限定为文档列出的 Qwen/Swift dense 27B，不是任意架构。

## 本项目做了什么

- **KVMem 长上下文** —— 历史 KV 以分页记录存放在主机内存，GPU 只保留有界工作窗口
  （`--kvmem-selected` B + `--kvmem-reserve` R）。262144 是逻辑上下文上限、32768 是 16GB
  生产配置的 GPU 窗口——KVMem 是**选择历史页面**，不是对全部历史做完整注意力。
  跨请求历史复用实测可恢复数千页、复用率 >99%。
- **指令与工具头保护** —— system/developer 指令和完整工具定义前缀按精确 token 边界在每次
  压力选择中固定，不依赖客户端 cache-control 提示
  （[serving.md](docs/serving.md#kvmem-instruction-and-tool-preservation)）。
- **ActKV 检索（可选）** —— `--kvmem-action-lrfu`、`--kvmem-lrfu-lambda`、
  `--kvmem-lrfu-top-frac`、`--kvmem-lrfu-fresh-frac`、`--kvmem-score-agg mean|max`，
  自 llama.cpp KVMem 后端移植。默认保持历史的一次性余弦排序；生产 32K 窗口实测持平，
  小窗口由 fresh 保底槽位兜底
  （[说明](docs/serving.md#actkv-action-oriented-retrieval-opt-in)）。
  [参考复现](docs/maintainer/actkv-reproduction.md)提供 mean 启动配置与原版 33 针/80 次工具测试，
  Qwen mean 配置已通过同轮/最终33/33、80次工具调用、跨工具14/14；关闭的基线也满分。
  mean 和 max 分开评估，目前没有优于基线的证据。
- **原生 GGUF 加载** —— Qwen/Swift dense 27B `.gguf` 直接跑 CUDA 引擎，保留原始 K/IQ
  量化块：文本、内嵌 MTP、优化草稿头、搭配 `--mmproj` 的视觉，无需转换整个模型
  （[范围与启动](docs/maintainer/native-gguf-loading.md)）。
- **长上下文正确性修复** —— 逐 chunk 统计起点（修复晚查询 Q 全零）、CUDA Graph 捕获中
  分配、历史恢复信封、GGUF scratch 捕获期扩容。
- **工具调用可靠性** —— 对残缺/截断的工具标记做容忍解析（歧义仍拒绝），并提供有界本地
  失败抓取诊断（`--failed-tool-request-dir`）。
- **监控** —— `GET /stats`（JSON）、`GET /metrics`（Prometheus）、`GET /slots`（执行 lane 快照）。

## 快速开始

```powershell
git clone https://github.com/cgz12345/kvmem-actkv-ninfer-v100.git
cd kvmem-actkv-ninfer-v100
```

1. 按安装指南配置 Windows、Visual Studio C++、CUDA 12.8、CMake、Ninja 和视觉依赖
   （[安装指南](docs/windows/SWIFT-KVMEM.md)）。
2. 下载 Swift IQ3_S 模型到 `models/swift15_iq3s_mtp.ninfer`，遵守模型许可证。
3. 构建后执行 `scripts/windows/start-swift15-256k.bat`。
4. API 地址 `http://127.0.0.1:8088/v1`；模型名 `qwen3.8-27b`；未启用鉴权，key 可填 `local`。

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8088/v1", api_key="local")
r = client.chat.completions.create(
    model="qwen3.8-27b",
    messages=[{"role": "user", "content": "你好，介绍一下你自己"}],
    max_tokens=512,
)
print(r.choices[0].message.content)
```

## 服务配置（三个已验证启动器）

| 启动器 | 制品 | KV 窗口（selected + reserve） | 说明 |
|---|---|---|---|
| `scripts/windows/start-swift15-256k.bat` | Swift-1.5 GSQ-RCO IQ3_S | 30720 + 2048 | 本地生产配置：sink 5120、recent 64、视觉、MTP K=3、int8 KV，**ActKV mean + fresh-frac 0.5** |
| `scripts/windows/start-actkv-256k.bat MODEL on\|off\|max` | Qwen3.8-27B GSQ-RCO IQ3_S | 36864 + 16384 | ActKV 参考配置：sink 0、recent 1024、MTP K=4、int8 KV；`on` = LRFU mean（已验证 33/33）、`off` = 基线、`max` = 实验臂 |
| `scripts/windows/start-native-gguf.bat MODEL.gguf [mtp] [MMPROJ.gguf]` | Qwen/Swift dense 27B `.gguf` | 按配置 | 原生 GGUF 路线：原始 K/IQ 块、内嵌 MTP 与草稿头、可选 mmproj 视觉 |

三个配置都提供同一套 HTTP 接口（Swift 生产用 8088，参考实例用 18200），模型名分别为
`qwen3.8-27b` 与 `ninfer-27b`。

## 实测结果

以下数字来自本机（Tesla V100 16GB、Windows、CUDA 12.8）与本仓库二进制；原始日志保存在本地
`runtime-logs/`。

### ActKV 参考复现

[参考配置](docs/maintainer/actkv-reproduction.md)运行社区作者自己的基准
（[tools/bench/actkv/niah33.py](tools/bench/actkv/niah33.py)）：15 轮写入、33 根针、最终召回
与跨工具检查，会话峰值 258,202 tokens，模型为 Qwen3.8-27B GSQ-RCO IQ3_S。验收标准为
exit 0 = 同轮 33/33、严格最终 33/33、跨工具 14/14、工具调用 ≥80：

| 配置 | 同轮召回 | 严格最终 | 工具调用 | 跨工具 |
|---|---:|---:|---:|---:|
| 基线（LRFU 关） | 33/33 | 33/33 | 80 | 14/14 |
| **ActKV LRFU + mean 聚合** | **33/33** | **33/33** | **80** | **14/14** |
| ActKV LRFU + max 聚合 | 28/33 | 30/33 | 67 | 6/14 |

mean 配置在这套夹具上精确复现社区成绩；max 聚合会退化。引擎默认关闭 ActKV，随库启动脚本
只启用经过测试的配置（参考制品用 [start-actkv-256k.bat](scripts/windows/start-actkv-256k.bat)，
Swift 见下）。**没有证据表明 ActKV 比基线更快或更准。**

### Swift-1.5 生产配置（已启用 ActKV mean + fresh-frac 0.5）

- 4030 输入 / 256 输出、无缓存两次：**96.6 tok/s** decode、约 570 tok/s prefill、MTP 接受率 86%。
- 跨轮历史复用 >99% 缓存命中；36K 与 260K 会话金丝雀通过。
- 由 `scripts/windows/start-swift15-256k.bat` 启动：KVMem 30720+2048、sink 5120、recent 64、
  视觉、MTP K=3、int8 KV，并启用 ActKV mean 配置 `--kvmem-lrfu-fresh-frac 0.5`。

### 动画鹈鹕测试

本地生产配置生成了一个"会动的鹈鹕骑自行车"SVG（SMIL 动画：车轮持续旋转、双腿蹬踏、
车身轻微起伏）。提示词：*"Draw a beautiful, lively animated SVG of a pelican riding a
bicycle … Return only the SVG code, no explanation."*

| 制品 | 思考 | 输出 tokens | 耗时 | Decode |
|---|---|---:|---:|---:|
| [Swift-1.5 IQ3_S](docs/assets/pelican-swift-bicycle.svg)（生产配置） | `reasoning_effort: high`（其中 16,421 推理） | 21,843 | 293.8 秒 | 74.7 tok/s |

![会动的鹈鹕](docs/assets/pelican-swift-bicycle.gif)

数字的背景：同提示词 `reasoning_effort: none` 时第一版（静态）18.8 秒 / 1756 tokens / 96.0 tok/s
完成；默认思考模式下同一请求会把 8192 token 预算全部花在推理上、没有产出 SVG——输出需要机器
可读时记得给思考留预算。GIF 以 10fps 捕捉 1.4 秒动画循环的 14 帧；SVG 文件在 GitHub 文件页可
原生播放动画。

### 验证范围与边界

- 256K 级检索抽查：统计起点修复后，260,936-token 干草堆能正确回答中段埋针；100,091-token
  请求 412 秒完成、SSE/usage 完整。通过的样本**不构成** 256K 质量认证。
- 默认输出预算 65,536；引擎生成到模型自行停止（实测 22,421 tokens）；完整 65,536
  单次生成尚未验证。
- DFlash2 在本卡与现有制品上不可用（显存 + 头格式），MTP 是受支持的投机后端。完整边界与
  原始证据见 [KVMMEM-PORT-NOTES.md](../KVMMEM-PORT-NOTES.md) 与 `runtime-logs/`。

## 复现方法

```powershell
# ActKV 参考测试：33 针、80 次工具调用、确定性评分（需要 Qwen GSQ-RCO 制品）
.\scripts\windows\start-actkv-256k.bat "D:\models\Qwen3.8-27B-GSQ-RCO-IQ3_S.ninfer" on
python tools/bench/actkv/niah33.py --endpoint http://127.0.0.1:18200/v1/chat/completions `
    --output out/actkv/mean.json

# 确定性 ActKV A/B：对任意运行中的服务生效（固定种子，两臂提示词逐字节相同）
python tests/actkv_ab.py plan                  # 打印工作负载，无需服务
python tests/actkv_ab.py run on actkv-on 8089  # 对按指定旗标启动的服务运行
```

两个测试脚本都是自包含 Python；A/B 工作负载由固定种子确定性生成，两臂可直接对比。

## API 能力

- OpenAI 兼容 `/v1/chat/completions`、`/v1/responses`、`/v1/models`，以及 Anthropic
  `/v1/messages`；SSE 流式输出带 `usage`；工具调用参数流式解析。
- 采样默认值来自模型本身；可按请求覆盖 temperature、top-p/top-k、min-p、惩罚项、seed、
  `max_tokens`（服务默认 65,536）与 `reasoning_effort: none|minimal|low|medium|high|xhigh`。
- Vision 与原生 GGUF 路线支持图片/视频输入；`--mmproj` 加载匹配的 GGUF 投影器。
- 运维接口：`GET /health`、`GET /props`、`GET /stats`、`GET /metrics`、`GET /slots`。
- 可靠性：system 指令与完整工具前缀在每次 KVMem 压力选择中固定；残缺/截断的工具标记在无歧义
  时修复、否则拒绝，并可本地抓取原始请求（`--failed-tool-request-dir`）。

## 仓库结构

```text
targets/       模型执行（Qwen3.6/3.8 27B、35B-A3B）、布局与请求规划
ops/           CUDA 算子（注意力、线性、MoE、GDN/ReplaySSM、量化）
runtime/       引擎核心、调度、资源与上下文管理
serve/         HTTP 服务、协议适配、工具处理
artifact/      容器读取，含原生 GGUF 路径
kvmem/         主机 KV 存储、检索评分、驻留与 IO 契约
docs/          用户指南 + 维护参考（索引见 docs/README.md）
scripts/windows 三个已验证服务配置的启动器
tools/bench    基准工具，含 ActKV 参考夹具
tests/         C++ 测试套件与确定性 ActKV A/B 夹具
```

## 限制与路线

- 262,144 是**逻辑**上限，GPU 只保留有界窗口（生产 32K、参考配置 53K）。KVMem 选择历史
  页面，不是对全部历史做完整注意力；通过的长上下文样本不构成 256K 质量认证。
- ActKV 是可选的 KVMem 适配：mean 配置与基线持平而非超越，max 聚合会退化（见上表）。引擎
  默认关闭，只有经过测试的启动器配置才启用。
- DFlash2 在本卡与现有制品上不可用（显存预算 + 选择头格式）；MTP 是受支持的投机后端。
- 完整 65,536-token 单次生成尚未端到端验证；默认输出预算为 65,536，模型自行停止即结束。
- FFmpeg GPL 共享构建二进制在补齐完整对应源码之前不对外分发。

## 出问题时提供什么

请在 issue 中附显卡型号、驱动、构建版本、启动参数和脱敏日志。仓库的 `development` 笔记
记录了每个已验证结论对应的命令与证据。

## 构建

Windows MSVC 配方：`development/ninfer-build/build-vision.bat`（VC 14.44、CUDA 12.8、Ninja）；
构建参数与上游 Linux 构建见 [V100 构建说明](docs/V100-BUILD.md)
（[中文版](docs/V100-BUILD.zh-CN.md)）。

## 上游、制品与署名

本仓库是 **NInfer** 的 **Windows/V100 KVMem** 维护线
（[Neroued/ninfer](https://github.com/Neroued/ninfer) → geoffwatts → liujun-7788 → JimmyMax →
本线）。支持的 `.ninfer` 制品版本：**v2 与 v3**，包括 `nvfp4`、`nvfp4full`（按张量探测格式）
以及通过转换 recipe 解析身份的社区制品，不修改制品字节。硬件目标：**V100（sm_70）**。
上游自己的 V100 基准（`nvfp4` 制品、219 tok/s decode）与历史资料保留在
[docs/v100.md](docs/v100.md) 与 [docs/V100-BUILD.md](docs/V100-BUILD.md)；署名与第三方许可
见 [NOTICE](NOTICE.md)。

## 许可与贡献

引擎遵守根目录 [LICENSE](LICENSE)。Swift 权重另适用其下载源的 Swift Open License 和底座
许可证。欢迎通过 Issues / Pull Requests 反馈和贡献；本分支与上游作者无隶属或背书关系。
