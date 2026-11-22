# DeepSeek-V4.1-Flash on 4× RTX 6000D

[简体中文](#zh) · [English](#en)

<a id="zh"></a>

## 简体中文

**2026-11-22 采用的配置**为 TP4、原版权重、B12X、FP8 KV、DSpark 最多 7 步自适应验证、1,048,576 token 上下文、16 个调度槽位。目标是混合服务短对话、代码和逐轮累积的长会话。结果来自与生产一致的配置，使用合成负载。

### 1. 主机与固定版本

| 项目 | 实测环境 |
|---|---|
| GPU | 4× NVIDIA RTX 6000D，SM120，每卡 85,651 MiB，PCIe 5.0 ×16，无 NVLink |
| 主机 | AMD EPYC 9335，约 755 GiB 可见 RAM；四张测试卡共用 NUMA 0，CPU 0–31，容器内存及 swap 上限均为 360 GiB |
| 驱动 | NVIDIA 615.78.08；Linux x86-64、Docker Engine、Compose v2 或更高、NVIDIA Container Toolkit |
| 权重 | [DeepSeek-V4.1-Flash-1017 固定 revision](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170)，48 个 shard，约 510 GB |
| BRRR | [Fraser Price 上游](https://github.com/fraserprice/DeepSeek-4.1-Flash-1017-BRRR)，commit `2aff97254f4b9005e181b91386c120505e091c18` |
| 推理环境 | vLLM `0.1.dev21836+g6d66eed35`、PyTorch `2.14.0a0+4fdf77b940.nv26.08`、B12X 和 CUDA 依赖见 [固定镜像](brrr/build.sh) |
| 本地修改 | [Engram buffered I/O 补丁](brrr/engram-buffered.patch)，将小读也走页缓存，适合本机 RAM 和 NVMe 条件 |

建议模型和 Docker 存储预留至少 650 GB 可用 NVMe 空间。记录自己的驱动、显存、PCIe 拓扑和 RAM。实测主机还有其他 GPU，复刻时只需要四张目标卡。内存容量不足可能导致页缓存回收，尤其会改变 Engram 读盘和尾延迟。生产镜像的本地 image ID 为 `sha256:9267440df38315dfd62b6026e32e68273084e0668b6ea9aae8057f80d81d71b7`。复建时以源提交、补丁与底镜像 digest 为准，构建元数据可导致本地 image ID 不同。

### 2. 下载、构建与启动

以下从本仓库 `brrr/` 目录执行。主机需有 Git、curl、jq、Python 3.12 和 Hugging Face CLI。GPU 必须空闲，先用 `nvidia-smi` 和 `nvidia-smi topo -m` 找到同一 NUMA 节点的四张卡，按实际编号修改环境变量。

```bash
export MODEL_DIR=/models/DeepSeek-V4.1-Flash-1017
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 --local-dir "$MODEL_DIR"
find "$MODEL_DIR" -maxdepth 1 -name 'model-*-of-00048.safetensors' | wc -l  # 应为 48

./build.sh
export GPU0=0 GPU1=1 GPU2=2 GPU3=3 CPUSET=0-31 NUMA_NODE=0
docker compose config -q
docker compose create deepseek
docker update --cpuset-mems "$NUMA_NODE" "$(docker compose ps -aq deepseek)"
docker compose start deepseek
docker compose logs -f deepseek
```

等日志出现 `Application startup complete` 后运行 `./smoke.sh`。预期模型列表显示 1,048,576 上下文、算术返回 `323`。引擎日志检查 CUDA Graph 捕获、DSpark speculator 和约 **1,457,122 token** 的 KV 池。首启需要预编译和捕获，保留 Compose 的 `jit` 卷。端点绑定到本机 `127.0.0.1:18136`。停止用 `docker compose stop deepseek`。

[Compose](brrr/compose.yaml) 把关键参数固定为 4,096 批 token、2,048 单请求长预填充阈值、Graph 128、GPU 内存利用率 0.975、每 GPU 1.5 GiB 显式 KV 预算；其余引擎默认值由 BRRR 上游 launcher 固定。1M 是每个请求的输入与输出总窗口，16 是调度槽位，约 146 万 token 的 KV 池由所有并发请求共享。默认 8K 是**自适应验证启动成本表的标定长度**。

### 3. 复刻测试

专用测试机上串行运行下列基准。测试前记录 GPU 温度和时钟。计时从客户端发请求开始，首 token 指首个可见内容。GPU 计时与客户端计时应分别报告。

```bash
# 加热本次使用的 Engram shard 页缓存；制作报告时记录冷或热状态。
ionice -c2 -n6 cat "$MODEL_DIR"/model-000{47,48}-of-00048.safetensors >/dev/null
mkdir -p runs
python3 bench/short_chat.py --self-test
python3 bench/short_chat.py http://127.0.0.1:18136 1 8 16 > runs/short.jsonl

python3.12 -m venv .venv
.venv/bin/pip install -r bench/requirements.txt
GPU8_BENCH_NAMESPACE=short-cost-ab .venv/bin/python bench/code.py \
  --port 18136 -s 2048 -c 1 4 8 16 -r 5 --temperature 1 --top-p 1 > runs/code.log

python3 bench/warm_chat.py tokenize http://127.0.0.1:18136
python3 bench/warm_chat.py bench http://127.0.0.1:18136 > runs/warm.jsonl
python3 bench/mixed_million.py
```

- **短对话**固定合成请求，温度 1、每路 512 输出 token，每个并发三轮。聚合吞吐是总输出 token 除以总批耗时，含首 token 时间。
- **代码**约 2,148 输入 token，关闭思考，温度 1，最多生成 2,048 token，每个并发五轮。先发送未缓存的预填充探针，再测相同输入的解码窗口。重复测同一个正在运行的实例时换 namespace，避免冷预填充探针命中旧缓存。脚本等待可见 GPU 降到 70°C，可在只暴露测试卡的环境运行。
- **长对话**10 万、20 万输入，预热前缀后按请求查验至少 95% 命中及检索标记，每个负载三轮，统计该组端到端聚合吞吐的中位数。20 万独立前缀用 C4，因为 C8 超过测试 KV 池。
- **近 1M 混合**随机化长提示开头和检索标记，首 token 到达后立刻发 8 个短对话。脚本断言检索正确、上下文符合预算、所有短请求在长生成结束前完成，结果写到 `mixed-million-result.json`。

### 4. 技术栈与配置

| 层 | 生产实现 |
|---|---|
| 推理框架 | Local Inference Lab 的 vLLM 构建 `0.1.dev21836+g6d66eed35`，通过 BRRR 上游 Dockerfile 应用 `vllm.patch` 和 SM120 自定义内核 |
| 模型与并行 | DeepSeek-V4.1-Flash-1017 原版 checkpoint、TP4，同 NUMA 节点四卡，B12X PCIe all-reduce |
| 算子 | BRRR 融合解码结合 B12X attention、MoE 和 linear 后端；CUDA Graph 模式 `FULL_AND_PIECEWISE`，最大捕获尺寸 128 |
| 投机解码 | DSpark 最多 7 步，自适应验证、概率草稿采样和 block rejection。默认成本系数 1.0，启动标定上下文 8K |
| 内存与调度 | FP8 KV，每 GPU 显式预留 1.5 GiB；chunked prefill，每次调度 4,096 批 token，单个长请求每块最多 2,048 token，16 槽，异步调度 |
| Engram | 表从原版权重读取，启用 projection TP；本地补丁将小块读取交给 Linux 页缓存。热缓存基准先读取第 47、48 个 shard |
| 服务接口 | vLLM OpenAI 兼容 HTTP API，DeepSeek V4.1 tokenizer、reasoning 和 tool-call parser |

[Compose](brrr/compose.yaml)保存部署覆盖参数，[上游 launcher](https://github.com/fraserprice/DeepSeek-4.1-Flash-1017-BRRR/blob/2aff97254f4b9005e181b91386c120505e091c18/entrypoint.sh)保存其余引擎默认值。

### 5. 生产配置实测成绩

以下取同一生产配置已记录的各并发档最好结果，可来自不同批次。按脚本的并发档统计展示。

| 负载 | 单流 | 4 流 | 8 流 | 16 流 |
|---|---:|---:|---:|---:|
| 短代码预填充，约 2.15K 输入 | **5,868** | **9,022** | **11,044** | **11,172** |
| 短代码解码（约 2.15K 输入） | **548** | **1,134** | **1,627** | **2,226** |
| 短对话端到端（512 输出） | **246** | — | **842** | — |

单位 token/s。代码单流取解码中位数；多流取共同解码窗口的聚合吞吐。短对话使用端到端计时，8 流首 token p95 约 **353 ms**。代码 [单流原始记录](brrr/results/base-code.log)与 [并发原始记录](brrr/results/production-c1-c16-code.log)来自分别运行的基准。

预填充数据取约 2.15K token 的短代码提示，在前缀缓存未命中的情况下发送请求，每路最多生成 1 token。每批用总输入 token 数除以该批最慢首 token 时间，多轮以总 token 数除以总批时间。单流为单路输入吞吐，多流为合计。这是包含排队、请求处理和首 token 的客户端有效速率。预填充与解码的最好成绩可来自不同批次。 [单流原始记录](brrr/results/base-code.log) · [4／8／16 流预填充记录](brrr/results/production-prefill-code.log)

- **预热长会话**。10 万独立前缀 8 流合计 **750 token/s**，20 万共享前缀 8 流合计 **679 token/s**。 [原始汇总](brrr/results/warm-chat-summary.jsonl)
- **近 1M 混合负载**。1,035,067 输入 token，4,096 输出 token，检索通过。冷长请求首 token **190.1 秒**，生成 **144.8 token/s**。长请求开始生成后发出的 8 个短请求均在其结束前完成，短请求首 token p50 / p95 **210 / 506 ms**，合计端到端 **581 token/s**。 [逐请求结果](brrr/results/mixed-million.json)

混合测试覆盖了长请求生成阶段。冷长请求预填充期间的短请求尾延迟尚未验证。GPU 时钟、NUMA、页缓存和请求内容会影响复测数字。

### 6. 参考来源

- [DeepSeek 原版模型、许可和固定权重](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170)
- [Fraser Price 的 BRRR 固定源码、Dockerfile 和基准](https://github.com/fraserprice/DeepSeek-4.1-Flash-1017-BRRR/tree/2aff97254f4b9005e181b91386c120505e091c18)
- [Local Inference Lab 的 vLLM 镜像构建](https://github.com/local-inference-lab/blackwell-llm-docker)与 [vLLM fork](https://github.com/local-inference-lab/vllm)
- [Local Inference Lab B12X](https://github.com/local-inference-lab/b12x)与 [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
- [vLLM 的 DSpark 自适应验证说明](https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-08-14-dspark-adaptive-verification.md)

---

<a id="en"></a>

## English

The production configuration selected on November 22, 2026 uses TP4, official model weights, B12X, FP8 KV cache, and DSpark adaptive verification with up to seven draft tokens per step. It provides a **1,048,576-token context window and 16 scheduler slots**. The benchmarks use synthetic chat, coding, and long-context workloads on the production configuration.

### 1. Host and pinned versions

| Component | Measured setup |
|---|---|
| GPU | 4× NVIDIA RTX 6000D (SM120), 85,651 MiB per GPU, PCIe 5.0 ×16, no NVLink |
| Host | AMD EPYC 9335, about 755 GiB of visible RAM. All four GPUs on NUMA node 0, CPUs 0–31. Docker memory and memory-plus-swap limits both set to 360 GiB |
| Driver and runtime | NVIDIA 615.78.08, x86-64 Linux, Docker Engine, Docker Compose v2 or later, NVIDIA Container Toolkit |
| Model | [Pinned DeepSeek-V4.1-Flash-1017 revision](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170), 48 shards and about 510 GB |
| BRRR | [Fraser Price's upstream repository](https://github.com/fraserprice/DeepSeek-4.1-Flash-1017-BRRR), commit `2aff97254f4b9005e181b91386c120505e091c18` |
| Inference runtime | vLLM `0.1.dev21836+g6d66eed35`, PyTorch `2.14.0a0+4fdf77b940.nv26.08`, B12X and CUDA dependencies from the [pinned base image](brrr/build.sh) |
| Local patch | [Engram buffered I/O](brrr/engram-buffered.patch), which uses the Linux page cache for small reads |

Allow at least 650 GB of free NVMe space for weights and Docker storage. Record the driver, GPU memory, PCIe topology, and RAM on the test host. The original host had additional GPUs, but this recipe uses four. RAM pressure can evict Engram pages and change disk I/O and tail latency. The measured local image ID was `sha256:9267440df38315dfd62b6026e32e68273084e0668b6ea9aae8057f80d81d71b7`. Use the pinned commit, patch, and base image digests to recreate it. Build metadata can change the local image ID.

### 2. Download, build, and start

Run these commands from the repository's `brrr/` directory. The host needs Git, curl, jq, Python 3.12, and the Hugging Face CLI. Use `nvidia-smi` and `nvidia-smi topo -m` to select four idle GPUs on the same NUMA node. Adjust GPU IDs, CPU affinity, and NUMA settings to match the host.

```bash
export MODEL_DIR=/models/DeepSeek-V4.1-Flash-1017
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 --local-dir "$MODEL_DIR"
find "$MODEL_DIR" -maxdepth 1 -name 'model-*-of-00048.safetensors' | wc -l  # Expect 48

./build.sh
export GPU0=0 GPU1=1 GPU2=2 GPU3=3 CPUSET=0-31 NUMA_NODE=0
docker compose config -q
docker compose create deepseek
docker update --cpuset-mems "$NUMA_NODE" "$(docker compose ps -aq deepseek)"
docker compose start deepseek
docker compose logs -f deepseek
```

Wait for `Application startup complete`, then run `./smoke.sh`. The model list should report a 1,048,576-token context window, and the arithmetic request should return `323`. Look for CUDA Graph capture, the DSpark speculator, and a KV cache capacity of about **1,457,122 tokens** in the logs. Keep the Compose `jit` volume for subsequent starts. The API binds to `127.0.0.1:18136`. Use `docker compose stop deepseek` to stop the service.

[Compose](brrr/compose.yaml) sets a batch budget of 4,096 tokens, a long-prefill threshold of 2,048 tokens per request, maximum graph capture size 128, GPU memory utilization of 0.975, and an explicit KV cache budget of 1.5 GiB per GPU. BRRR pins the other defaults in its launcher. The 1M limit includes input and output for one request. The 16 scheduler slots share the approximately 1.46M-token KV pool. The default 8K context is used to calibrate the adaptive verification cost table at startup.

### 3. Benchmark reproduction

Run the benchmarks sequentially on a dedicated test host. Record GPU temperature and clocks. Client timing starts when the request is sent, and time to first token (TTFT) measures the arrival of first visible content. Report client and GPU timings separately.

```bash
# Warm the Engram shard page cache and report whether the benchmark used a warm cache.
ionice -c2 -n6 cat "$MODEL_DIR"/model-000{47,48}-of-00048.safetensors >/dev/null
mkdir -p runs
python3 bench/short_chat.py --self-test
python3 bench/short_chat.py http://127.0.0.1:18136 1 8 16 > runs/short.jsonl

python3.12 -m venv .venv
.venv/bin/pip install -r bench/requirements.txt
GPU8_BENCH_NAMESPACE=short-cost-ab .venv/bin/python bench/code.py \
  --port 18136 -s 2048 -c 1 4 8 16 -r 5 --temperature 1 --top-p 1 > runs/code.log

python3 bench/warm_chat.py tokenize http://127.0.0.1:18136
python3 bench/warm_chat.py bench http://127.0.0.1:18136 > runs/warm.jsonl
python3 bench/mixed_million.py
```

- **Short chat.** Fixed synthetic prompts, temperature 1, 512 output tokens per request, and three rounds at each concurrency. Aggregate throughput divides total output tokens by total batch wall time, including TTFT.
- **Coding.** Approximately 2,148 input tokens, thinking disabled, temperature 1, up to 2,048 output tokens. The command runs five rounds per concurrency. It first sends an uncached prefill probe, then measures decode on the same input. Change the namespace for another run against the same instance to prevent unexpected prefix hits. The script waits for the visible GPUs to cool to 70°C.
- **Warm long chat.** Approximately 100K and 200K input tokens. Warm the prefixes, then check that each request has at least a 95% cache hit and returns the retrieval marker. Run three rounds per scenario and report median end-to-end aggregate throughput. At 200K, use four independent prefixes to stay within the tested KV pool.
- **Near-1M mixed workload.** Randomize the long prompt's beginning and retrieval marker. Send eight short chats when the long request emits its first token. The script checks retrieval and context budget, and asserts that all short requests complete before the long request ends. It writes `mixed-million-result.json`.

### 4. Inference stack and scheduling

| Layer | Production implementation |
|---|---|
| Engine | Local Inference Lab's vLLM build `0.1.dev21836+g6d66eed35`. The upstream BRRR Dockerfile applies `vllm.patch` and builds SM120 kernels |
| Model and parallelism | Official DeepSeek-V4.1-Flash-1017 checkpoint, TP4 across four GPUs on the same NUMA node, and B12X PCIe all-reduce |
| Operators and graphs | BRRR fused decode with B12X attention, MoE, and linear backends. CUDA Graph mode `FULL_AND_PIECEWISE`, maximum capture size 128 |
| Speculative decode | DSpark with up to seven draft tokens, adaptive verification, probabilistic draft sampling, and block rejection. Default verification cost scale 1.0, startup calibration context 8K |
| Cache and scheduling | FP8 KV cache, 1.5 GiB per GPU, chunked prefill, 4,096 tokens per batch, up to 2,048 prefill tokens per long request per chunk, and 16 slots with asynchronous scheduling |
| Engram | Tables from the official weights, tensor-parallel projection, buffered reads through the Linux page cache. Warm benchmarks read shards 47 and 48 first |
| API | vLLM OpenAI-compatible HTTP API with DeepSeek V4.1 tokenizer, reasoning parser, and tool-call parser |

[Compose](brrr/compose.yaml) contains the deployment overrides. The [pinned upstream launcher](https://github.com/fraserprice/DeepSeek-4.1-Flash-1017-BRRR/blob/2aff97254f4b9005e181b91386c120505e091c18/entrypoint.sh) supplies the remaining defaults.

### 5. Production configuration results

The table reports the best archived results at each concurrency on the same production configuration. Values may come from separate batches.

| Workload | 1 stream | 4 streams | 8 streams | 16 streams |
|---|---:|---:|---:|---:|
| Short coding prefill, about 2.15K input tokens | **5,868** | **9,022** | **11,044** | **11,172** |
| Short coding decode, about 2.15K input tokens | **548** | **1,134** | **1,627** | **2,226** |
| Short chat, end to end, 512 output tokens | **246** | — | **842** | — |

Prefill uses input tokens/s. The probe allows at most one output token per request and requires a prefix-cache miss. Each batch is timed to its slowest TTFT. Across rounds, total input tokens are divided by total batch time. This client-observed rate includes queueing, request processing, and time to first token. [Single-stream log](brrr/results/base-code.log) · [Concurrent prefill log](brrr/results/production-prefill-code.log)

Decode and end-to-end chat use output tokens/s. Single-stream coding decode is the median per-round rate. Concurrent coding decode is aggregate throughput over the interval when every stream is decoding. Short-chat TTFT p95 was about **353 ms** at eight streams. [Single-stream coding log](brrr/results/base-code.log) · [Concurrent coding log](brrr/results/production-c1-c16-code.log)

- **Warm long chat.** About **750 tokens/s** in aggregate with eight independent 100K prefixes, and **679 tokens/s** with eight requests sharing a 200K prefix. [Results](brrr/results/warm-chat-summary.jsonl)
- **Near-1M mixed load.** The long request retrieved the marker with **1,035,067 input tokens**, generated **4,096 output tokens**, and reached its first token in **190.1 s**. Its decode rate was **144.8 tokens/s**. All eight short requests sent after its first token completed before long decode ended. Their TTFT p50 / p95 was **210 / 506 ms**, and their aggregate end-to-end throughput was **581 tokens/s**. [Per-request results](brrr/results/mixed-million.json)

The mixed test measures short requests during long decode. Short-request tail latency during cold long prefill remains unmeasured. GPU clocks, NUMA placement, page-cache residency, and prompt content can change the results.

### 6. References

- [Pinned official DeepSeek checkpoint and license](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170)
- [Pinned Fraser Price BRRR source, Dockerfile, and benchmarks](https://github.com/fraserprice/DeepSeek-4.1-Flash-1017-BRRR/tree/2aff97254f4b9005e181b91386c120505e091c18)
- [Local Inference Lab container build](https://github.com/local-inference-lab/blackwell-llm-docker) and [vLLM fork](https://github.com/local-inference-lab/vllm)
- [Local Inference Lab B12X](https://github.com/local-inference-lab/b12x) and [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
- [vLLM DSpark adaptive verification](https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-08-14-dspark-adaptive-verification.md)
