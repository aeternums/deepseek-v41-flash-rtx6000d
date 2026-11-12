# 在 4×RTX 6000D 上跑 DeepSeek-V4.1-Flash

一套可以直接复刻的部署配方：镜像、权重、启动参数、验证方法和实测数字。

镜像构建在上游公开仓库完成。本仓库只做一件事——把版本钉死，给出能一次跑起来的组合，
并记录我们在这台机器上实际测到的结果和踩过的坑。

## 这套方案由什么组成

| 层 | 内容 | 说明 |
| --- | --- | --- |
| 权重 | `local-inference-lab/DeepSeek-V4.1-Flash-1017-lossless-CSF`（463 GB） | 基座 `deepseek-ai/DeepSeek-V4.1-Flash-1017` 的 FP4-CSF 无损压缩版 |
| 镜像 | `ghcr.io/local-inference-lab/vllm:karmic-kraken-20261016.2` | 上游 `local-inference-lab/blackwell-llm-docker` 构建，commit `3d1ccd84` |
| 底座 | `nvcr.io/nvidia/pytorch@sha256:33ef5fc1…` | NVIDIA PyTorch 26.08，CUDA 13.4.1 |
| 引擎 | vLLM `0.1.dev21836+g6d66eed35.cu134` | FlashInfer + B12X 注意力/MoE + DeepGEMM，按 SM120 编译 |
| 硬件 | 8× RTX 6000D 85 GB（SM120，PCIe，无 NVLink） | 本配方占用 4 张卡，TP4 |

为什么是这套：RTX 6000D 是 SM120，PyPI 上的通用轮子没有为它编译 FlashInfer、DeepGEMM 和 B12X 后端，
直接 `pip install vllm` 跑不出同样速度。上游把这套编译结果做成一个镜像加多个模型 profile，
换模型不用换镜像。

## 三步拉起

```bash
# 1. 权重（约 463 GB，公开可下载）
hf download local-inference-lab/DeepSeek-V4.1-Flash-1017-lossless-CSF \
  --local-dir /models/DeepSeek-V4.1-Flash-1017-lossless-CSF

# 2. 镜像
docker pull ghcr.io/local-inference-lab/vllm@sha256:faf0b1f08d688b1aaa7d5161dc3b724c701d17d68278e7d32ebdd3a6b5127ce2

# 3. 起服务（先看 deploy/.env.example）
cd deploy && cp .env.example .env
docker compose up -d
docker compose logs -f | grep -m1 'Application startup complete'
```

验证：

```bash
./scripts/smoke.sh
curl -s localhost:8000/v1/models | jq -r '.data[].id'
```

首次启动要加载 463 GB 权重并捕获 CUDA Graph，4 卡实测约 5 分钟（权重加载 75 s、
引擎初始化 86 s）；`/cache` 卷保存 JIT 和 graph 缓存，重启会快很多。

## 实测结果

4×RTX 6000D / TP4 / DSpark5 / FP8 KV / Engram 表放内存。合成负载：约 355 token 输入、
固定 1024 token 输出、30 秒补充请求后排空，统计聚合端到端输出速度。

| 场景 | 结果 |
| --- | --- |
| 短输入，1 并发 | 228 tok/s |
| 短输入，16 并发（8 槽位 / capture 8） | 382 tok/s |
| 短输入，16 并发（16 槽位 / capture 64） | 890–950 tok/s |
| 130K 输入，单路 | 首字 10.3 s，端到端 77 tok/s |
| 512K 输入，单路 | 首字 52.5 s，端到端 18 tok/s |

16 并发那一行是本方案的主要收益：只把 `--max-num-seqs` 从 8 提到 16、
`--max-cudagraph-capture-size` 从 8 提到 64，吞吐约 2.4 倍。原因见 [docs/tuning.md](docs/tuning.md)。

这些是合成负载下的框架吞吐，不含业务正确性评价，也不能外推成多少人同时用没问题。

## 边界

- 无 NVLink，TP4 的通信走 PCIe；卡越多 all-reduce 越吃亏。本机 8 卡里 4 卡跑这个大模型，另外 4 卡跑小模型。
- 单机单实例。扩到多机需要另一套并行策略，本仓库不覆盖。
- 长上下文要按实际并发重新测。声明 1M 上下文不等于 1M 上下文下多人并发可用。
- 权重遵循 LIL-1.0 许可（在 Apache-2.0 上附加条件），商用前自行读一遍 `LICENSE`。

## 上游

- 镜像与构建：<https://github.com/local-inference-lab/blackwell-llm-docker>
- 硬件 wiki：<https://github.com/local-inference-lab/rtx6kpro>
- 模型 profile 说明：上游 `runtime/README.md`（含 `--print-config`、生成的 compose、基准协议）
