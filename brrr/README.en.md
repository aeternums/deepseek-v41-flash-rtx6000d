# Reproducing DeepSeek-V4.1-Flash with native BRRR on 4× RTX 6000D

[简体中文](README.md) · English

The production configuration selected on November 22, 2026 uses TP4, official model weights, B12X, FP8 KV cache, and DSpark adaptive verification with up to seven draft tokens per step. It provides a **1,048,576-token context window and 16 scheduler slots**. The benchmarks use synthetic chat, coding, and long-context workloads on the production configuration.

## 1. Host and pinned versions

| Component | Measured setup |
|---|---|
| GPU | 4× NVIDIA RTX 6000D (SM120), 85,651 MiB per GPU, PCIe 5.0 ×16, no NVLink |
| Host | AMD EPYC 9335, about 755 GiB of visible RAM. All four GPUs on NUMA node 0, CPUs 0–31. Docker memory and memory-plus-swap limits both set to 360 GiB |
| Driver and runtime | NVIDIA 615.78.08, x86-64 Linux, Docker Engine, Docker Compose v2 or later, NVIDIA Container Toolkit |
| Model | [Pinned DeepSeek-V4.1-Flash-1017 revision](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170), 48 shards and about 510 GB |
| BRRR | [Fraser Price's upstream repository](https://github.com/fraserprice/DeepSeek-4.1-Flash-1017-BRRR), commit `2aff97254f4b9005e181b91386c120505e091c18` |
| Inference runtime | vLLM `0.1.dev21836+g6d66eed35`, PyTorch `2.14.0a0+4fdf77b940.nv26.08`, B12X and CUDA dependencies from the [pinned base image](build.sh) |
| Local patch | [Engram buffered I/O](engram-buffered.patch), which uses the Linux page cache for small reads |

Allow at least 650 GB of free NVMe space for weights and Docker storage. Record the driver, GPU memory, PCIe topology, and RAM on the test host. The original host had additional GPUs, but this recipe uses four. RAM pressure can evict Engram pages and change disk I/O and tail latency. The measured local image ID was `sha256:9267440df38315dfd62b6026e32e68273084e0668b6ea9aae8057f80d81d71b7`. Use the pinned commit, patch, and base image digests to recreate it. Build metadata can change the local image ID.

## 2. Download, build, and start

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

[Compose](compose.yaml) sets a batch budget of 4,096 tokens, a long-prefill threshold of 2,048 tokens per request, maximum graph capture size 128, GPU memory utilization of 0.975, and an explicit KV cache budget of 1.5 GiB per GPU. BRRR pins the other defaults in its launcher. The 1M limit includes input and output for one request. The 16 scheduler slots share the approximately 1.46M-token KV pool. The default 8K context is used to calibrate the adaptive verification cost table at startup.

## 3. Benchmark reproduction

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

## 4. Inference stack and scheduling

| Layer | Production implementation |
|---|---|
| Engine | Local Inference Lab's vLLM build `0.1.dev21836+g6d66eed35`. The upstream BRRR Dockerfile applies `vllm.patch` and builds SM120 kernels |
| Model and parallelism | Official DeepSeek-V4.1-Flash-1017 checkpoint, TP4 across four GPUs on the same NUMA node, and B12X PCIe all-reduce |
| Operators and graphs | BRRR fused decode with B12X attention, MoE, and linear backends. CUDA Graph mode `FULL_AND_PIECEWISE`, maximum capture size 128 |
| Speculative decode | DSpark with up to seven draft tokens, adaptive verification, probabilistic draft sampling, and block rejection. Default verification cost scale 1.0, startup calibration context 8K |
| Cache and scheduling | FP8 KV cache, 1.5 GiB per GPU, chunked prefill, 4,096 tokens per batch, up to 2,048 prefill tokens per long request per chunk, and 16 slots with asynchronous scheduling |
| Engram | Tables from the official weights, tensor-parallel projection, buffered reads through the Linux page cache. Warm benchmarks read shards 47 and 48 first |
| API | vLLM OpenAI-compatible HTTP API with DeepSeek V4.1 tokenizer, reasoning parser, and tool-call parser |

[Compose](compose.yaml) contains the deployment overrides. The [pinned upstream launcher](https://github.com/fraserprice/DeepSeek-4.1-Flash-1017-BRRR/blob/2aff97254f4b9005e181b91386c120505e091c18/entrypoint.sh) supplies the remaining defaults.

## 5. Production configuration results

The table reports the best archived results at each concurrency on the same production configuration. Values may come from separate batches.

| Workload | 1 stream | 4 streams | 8 streams | 16 streams |
|---|---:|---:|---:|---:|
| Short coding prefill, about 2.15K input tokens | **5,868** | **9,022** | **11,044** | **11,172** |
| Short coding decode, about 2.15K input tokens | **548** | **1,134** | **1,627** | **2,226** |
| Short chat, end to end, 512 output tokens | **246** | — | **842** | — |

Prefill uses input tokens/s. The probe allows at most one output token per request and requires a prefix-cache miss. Each batch is timed to its slowest TTFT. Across rounds, total input tokens are divided by total batch time. This client-observed rate includes queueing, request processing, and time to first token. [Single-stream log](results/base-code.log) · [Concurrent prefill log](results/production-prefill-code.log)

Decode and end-to-end chat use output tokens/s. Single-stream coding decode is the median per-round rate. Concurrent coding decode is aggregate throughput over the interval when every stream is decoding. Short-chat TTFT p95 was about **353 ms** at eight streams. [Single-stream coding log](results/base-code.log) · [Concurrent coding log](results/production-c1-c16-code.log)

- **Warm long chat.** About **750 tokens/s** in aggregate with eight independent 100K prefixes, and **679 tokens/s** with eight requests sharing a 200K prefix. [Results](results/warm-chat-summary.jsonl)
- **Near-1M mixed load.** The long request retrieved the marker with **1,035,067 input tokens**, generated **4,096 output tokens**, and reached its first token in **190.1 s**. Its decode rate was **144.8 tokens/s**. All eight short requests sent after its first token completed before long decode ended. Their TTFT p50 / p95 was **210 / 506 ms**, and their aggregate end-to-end throughput was **581 tokens/s**. [Per-request results](results/mixed-million.json)

The mixed test measures short requests during long decode. Short-request tail latency during cold long prefill remains unmeasured. GPU clocks, NUMA placement, page-cache residency, and prompt content can change the results.

## 6. References

- [Pinned official DeepSeek checkpoint and license](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170)
- [Pinned Fraser Price BRRR source, Dockerfile, and benchmarks](https://github.com/fraserprice/DeepSeek-4.1-Flash-1017-BRRR/tree/2aff97254f4b9005e181b91386c120505e091c18)
- [Local Inference Lab container build](https://github.com/local-inference-lab/blackwell-llm-docker) and [vLLM fork](https://github.com/local-inference-lab/vllm)
- [Local Inference Lab B12X](https://github.com/local-inference-lab/b12x) and [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
- [vLLM DSpark adaptive verification](https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-08-14-dspark-adaptive-verification.md)
