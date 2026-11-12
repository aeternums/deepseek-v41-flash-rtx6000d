# 调优：为什么是这几个参数

所有数字来自 4×RTX 6000D、TP4、同一套合成负载（约 355 token 输入、1024 token 输出、
30 秒补充请求后排空）。换硬件或换负载形态都要重测。

## 真正起作用的两项

| 参数 | 改前 | 改后 | 16 并发聚合输出 |
| --- | --- | --- | --- |
| `--max-num-seqs` | 8 | 16 | — |
| `--max-cudagraph-capture-size` | 8 | 64 | 382 → 890–950 tok/s |

原因不复杂：默认槽位少，decode 阶段每步只有 8 条请求可以拼批，4 张卡吃不满；
把槽位和 CUDA Graph 捕获尺寸一起放大，单步能塞进更多序列，且更多形状命中已捕获的 graph，
省掉每步的重新捕获。两项要一起改——只放大槽位、捕获尺寸不动，会退化成频繁的 graph miss。

## 次要项

- `--max-num-batched-tokens 4096`：prefill 的批预算。调大能提高长输入吞吐，但会挤占 decode 的显存和调度时隙。
- `--gpu-memory-utilization 0.95`：留给 KV cache 的比例。再往上抬，长上下文下容易 OOM。
- `--kv-cache-dtype fp8`：KV 用量减半，长上下文收益明显。
- `--engram-table-memory ram`：Engram 表放内存而不是显存，把显存留给权重和 KV。
- DSpark 深度 5：试过 3 / 5 / 7。7 的单路略快但 16 并发更差，3 在长输入上更差，5 是折中。

## 试过但没用的

- `--max-cudagraph-capture-size 8192`：启动阶段直接退出，捕获开销撑爆了启动预算。
- 32 槽位 / capture 128：在 1M 上下文配置下起不来，显存不够。
- SGLang 0.5.22：同条件下 16 并发只有 225–326 tok/s，比 vLLM 低一截，且多个 DSpark 组合启动失败。
- 关掉投机解码：短输入单路从 228 掉到约 150 tok/s。

## 怎么复测

用镜像自带的 bench（会把硬件拓扑、时钟、降频原因一起记录）：

```bash
docker exec -it <container> lil-bench --no-upload
```

或者用 vLLM 自带的 `vllm bench serve` 固定输入输出长度。注意两点：

1. 分别报首字延迟和端到端聚合速度，不要把单路 decode 速度当成整体吞吐。
2. 冷输入和前缀缓存命中要分开跑，否则数字没有可比性。
