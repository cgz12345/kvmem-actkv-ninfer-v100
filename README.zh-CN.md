# KVMem NInfer · Windows V100 · Qwen3.8-27B（Swift-1.5）

[English](README.md) | **简体中文**

KVMem NInfer 让你在**单卡 Tesla V100 16GB** 的 Windows 机器上跑 Qwen3.8-27B（推荐制品：
**Swift-1.5 GSQ-RCO IQ3_S** 社区微调，自带 MTP）：KV 历史归档到主机内存、有限 GPU 工作窗口、
OpenAI/Anthropic 兼容 HTTP 服务、图片/视频输入，以及原生 GGUF 加载。本仓库是 NInfer 的
**Windows/V100 KVMem 维护线**，所有结论来自 16GB 实机测量，不是纸面参数。

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
git clone https://github.com/cgz12345/kvmem-ninfer-v100.git
cd kvmem-ninfer-v100
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

## 验证范围

- Swift IQ3_S、MTP K=3、int8 KV、32K 窗口 / 2048 prefill、开启视觉：4030 输入 / 256 输出
  两次，**96.6 tok/s** decode；MTP 接受率 86%；金丝雀与多轮历史回放通过。
- 256K 级检索抽查：统计起点修复后，260,936-token 干草堆能正确回答中段埋针；100,091-token
  请求 412 秒完成、SSE/usage 完整。三个通过样本**不构成** 256K 质量认证。
- 默认输出预算 65,536；引擎生成到模型自行停止（实测 22,421 tokens）；完整 65,536
  单次生成尚未验证。
- DFlash2 在本卡与现有制品上不可用（显存 + 头格式），MTP 是受支持的投机后端。完整边界与
  原始证据见 [KVMMEM-PORT-NOTES.md](../KVMMEM-PORT-NOTES.md) 与 `runtime-logs/`。

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
