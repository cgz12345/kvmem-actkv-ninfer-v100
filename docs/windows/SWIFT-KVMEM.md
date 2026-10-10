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
DFlash 本机适配按用户要求跳过。

## 分发

本次只发布源码。现有本地 FFmpeg GPL full shared DLL 的完整对应源码及依赖构建资料
尚未齐备，故不公开上传该运行包。发布二进制前补齐对应源码或重新构建合规依赖。
保留根目录 LICENSE、NOTICE 与上游署名；不要将模型许可证误写成引擎许可证。
