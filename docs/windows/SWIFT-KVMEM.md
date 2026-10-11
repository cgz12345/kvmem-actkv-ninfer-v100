# Windows V100 / Swift-1.5 / KVMem

2026-10-11，Tesla V100-SXM2 16GB，Windows，CUDA 12.8 / MSVC。

本分支提供 Windows V100 移植、KVMem 主机归档、多轮历史、MTP、图片输入和监控接口。
推荐模型为 Swift-1.5-Qwen3.8-27B-GSQ-RCO IQ3_S。API 名称沿用 `qwen3.8-27b`。

## 模型与启动

从 https://huggingface.co/fyb1214/Swift-1.5-Qwen3.8-27B-GSQ-RCO-NInfer 下载
`swift15_iq3s_mtp.ninfer`，放到仓库 `models/`。
固定 revision：`7da3e38b9ee7a6c8888cf2112d3c5a48a952a2a8`。
SHA256：`0e1771e4df425a7e05b0ffce8662cb500dc2546f3759767fe6ac2699d64c0816`。
模型具有 text/vision/mtp 组件，无 DFlash2。自报身份通过转换 recipe 解析，不要改写模型头。
权重适用 Swift Open License v1.0；底座 Apache-2.0。请阅读下载源 LICENSE/NOTICE。
源码仓库不分发模型、DLL 或其他二进制。

构建需要 CMake、Ninja、Visual Studio C++、CUDA 12.8；视觉需要 FFmpeg 开发文件和 curl。
设置 NINFER_FFMPEG_ROOT（include 和 lib/av*.lib）、NINFER_CURL_ROOT（include 和 libcurl.lib）。
在 x64 VS 开发终端执行：

```bat
chcp 65001
cmake -S . -B build-sm70 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70 -DNINFER_ENABLE_VISION=ON -DNINFER_BUILD_MEDIA_ACQUIRE=ON -DNINFER_FFMPEG_ROOT="%NINFER_FFMPEG_ROOT%" -DNINFER_CURL_ROOT="%NINFER_CURL_ROOT%" -DCMAKE_POLICY_VERSION_MINIMUM=3.5
python scripts/windows/fix-showincludes-prefix.py --repair build-sm70
cmake --build build-sm70 --target ninfer-serve ninfer ninfer-perplexity -j
python scripts/windows/fix-showincludes-prefix.py --check build-sm70
scripts\windows\start-swift15-256k.bat
```

依赖 DLL 放到 build-sm70/apps 或 PATH。嵌入网页另需配置 NINFER_WEBUI_DIR，参见 docs/serving.md。
启动脚本使用仓库相对路径；CUDA 安装路径不同时修改脚本。

API：`http://127.0.0.1:8088/v1`；模型 `qwen3.8-27b`；未启用鉴权，客户端 key 可填 `local`。
逻辑上下文 262144；GPU KV 窗口 32768；默认输出 65536（受输入长度及模型 EOS 限制）。
MTP K=3、INT8 KV、vision 开启。KVMem 页面选择不是全量 256K 注意力。

## 实测与边界

最新 Swift：4030 输入 + 256 输出、temperature=0、thinking off、两次无缓存请求。
输出 96.6 tok/s；预填充 577.2 / 579.7 tok/s；总耗时 9.75 / 9.67 秒。
同期 Qwen 制品为 92.7 tok/s，本样本 Swift 输出约快 4.2%，不代表所有任务。
Swift 图片 high/low 及 Responses low 均正确识别红色；low 282 vs high 1050 输入 tokens。
默认输出预算和 active/idle 监控验证通过。

同一运行时：原始 Qwen 制品 260936-token 中段埋针正确召回；两个 52047-token
不同位置样本通过。Swift lookup 图 6000-token 请求和小窗口 9000-token 换页回归通过。
修复统计逻辑起点、非对齐页统计、CUDA Graph 捕获内 scratch 分配及传输同步/历史恢复问题。
相关独立 CUDA 数学与图生命周期测试通过。详细实现与协议见现有 docs/。

未声称所有 256K 检索可靠、完整生成 65536 已验证、复杂 OCR/视频/音频全面验收。
DFlash2 已完成本机初步适配和实机验证，见下节；Swift 当前制品没有 DFlash2 权重，正式配置仍使用 MTP。

## 分发

本次只发布源码。现有本地 FFmpeg GPL full shared DLL 的完整对应源码及依赖构建资料
尚未齐备，故不公开上传该运行包。发布二进制前补齐对应源码或重新构建合规依赖。
保留根目录 LICENSE、NOTICE 与上游署名；不要将模型许可证误写成引擎许可证。

## 历史重渲染后的前缀复用

KVMem 会尝试较早的完整 Base 检查点。最近响应的思考/文本重渲染改变尾部时，
仍可恢复逐 token 相同的历史前缀，再重放变化部分；MTP 边界和部分页统计会按检查点重建。
36K 冷启动 74.5 秒，改变问题及回放响应的后续请求 2.6–3.3 秒，复用率超过99.8%，答案正确。
这是固定样本的结果；若客户端改写所有保存检查点之前的头部或历史，仍需重新预填充。
测试脚本：scripts/windows/test-history-prefix-recovery.py（生产窗口，建议 --kvmem-verify）。

## DFlash2 GGUF 验证（2026-10-11）

V100 上 GGUF_Q4_K 完整/精简词表现在支持 `linear_topk`，使用调用者提供的工作区。
每批最多128列，向量及矩阵路径均以独立 FP64 解量化参考验证。
此前不稳定分数来自测试缓冲区的默认流初始化与nonblocking执行流竞争；
在执行前同步初始化后宽列路径通过，已移除八列限制。
完整词表忽略填充行；精简词表扫描实际131072行并映射到全局token ID。
测试覆盖1、7、8、9、16、33、129列、并列排序、边界保护和CUDA图重复回放。
`ninfer_linear_topk_test --gguf-only` 和包含原有格式的完整测试均通过。

DFlash目标验证图的注意力包络现按实际GPU窗口裁剪；此前6144窗口的最高图层
构造6151位置的包络，造成启动失败。准备图与运行时选择使用同一裁剪规则。

实测使用 `qwen38-27b-iq3s-ninfer-v3.ninfer`，不是Swift。该制品含DFlash2权重。
文本模式、int8 KV、窗口6144（selected4096/reserve2048）、prefill2048、headroom128、
`--lm-head-draft`、单并发。DFlash2 K7权重13.4GiB，运行后约348MiB显存余量。
启动、短算术回答和5265-token输入请求成功；同一输入、冷缓存、输出256tokens：

| 模式 | 预填充秒 | 解码tok/s | 接受比例 |
|---|---:|---:|---:|
| DFlash2 K7 | 9.373 | 51.05 | 180/525 |
| MTP K3 | 9.136 | 67.89 | 155/296 |

这是单一固定样本，不能推断所有任务的速度。DFlash2未替代Swift生产模式，
也未验证大窗口、视觉和长时间稳定性；MTP与DFlash同时运行的混合模式尚未实现。

上游实时对表：ninfer-all最新8fe9fc2，包括共享缓存诊断越界修复f928d88、
多GPU视觉修复284c6f2和Gemma4支持ce0eac8。当前缓存代码没有该越界诊断分支；
后两项不属于当前单V100/Swift执行路径。新增Windows/MSVC修复f26387a中的
near/far宏问题在本项目已处理；MoE CPU专家和Gemma执行器本项目未采用。
其余参考仓库远端未发现新提交。

小GPU窗口1536下长预填充的bad allocation已定位并修复：Swift混合权重包含IQ1_M，
工作区估算遗漏其反量化转换slab，现覆盖该路径。相同窗口/prefill512/vision/MTP3，
24063-token输入成功完成，预填充83.43秒，输出32tokens，health200。
历史回放脚本增加`--archive-repeats`，可用较短归档复现小窗口路径。

最终容量计划还显式计入MLP中间激活，gate/up与down的临时空间按互斥生命周期取峰值。
1536窗口、vision/MTP3、`--kvmem-verify`下五轮7866–7948-token历史验证通过，
冷请求16.94秒，后续0.88–1.19秒，复用99%以上，native KV与StateImage精确检查通过。

连续生成预算已验证：`ninfer_bench -n 65535 -r 1 --warmup 0 --prefill-chunk 512
--kv-dtype int8 --spec mtp --draft-tokens 3 --lm-head-draft`实际生成65536tokens，
以OutputLimit结束，退出码0，连续运行449.39秒。使用Swift、原生65542上下文，
基准工具关闭默认EOS停止；这验证引擎预算及连续执行，不代表API会强制模型写满，
也不等于KVMem下任意65K自然对话或24小时稳定性认证。
该重复续写样本145.89tok/s，不用于替代普通对话速度测量。
Windows基准目标现统一链接静态CUDA runtime，避免重复符号；正常引擎关闭不再打印故障诊断。

完成容量修复后，DFlash2的最终试跑配置将prefill从2048调为1024，保留6144窗口，
避免按真实最坏路径预留工作区时超出16GB容量。启动后约598MiB空闲，算术回答正确且health200。
上表是初次2048分块的历史A/B结果；当前Swift生产继续使用2048分块和32768窗口。
