# DeepSeek: 2226 tok/s at 16-way

[简体中文](SOCIAL.md) · English

DeepSeek-V4.1-Flash runs on four RTX 6000D GPUs with a 1,048,576-token context window and 16 scheduler slots. The context limit covers input and output for one request. The active requests share the KV cache. The benchmarks cover short chat, coding, and long conversations to measure both throughput and the wait for a first token under mixed load.

## Inference stack

The deployment uses Local Inference Lab's vLLM image, Fraser Price's BRRR patch and SM120 kernels, and the official DeepSeek checkpoint. It runs TP4 with B12X for attention, MoE, linear layers, and PCIe collectives. CUDA Graphs replay common batch shapes.

DSpark proposes up to seven draft tokens per step and adapts how many it verifies. Long input is processed in chunks of up to 2,048 tokens per request with a 4,096-token batch budget. The deployment uses FP8 KV cache and the Linux page cache for small Engram reads.

## Benchmarks

| Workload | 1 stream | 4 streams | 8 streams | 16 streams |
|---|---:|---:|---:|---:|
| Coding prefill, input tokens/s | 5,868 | 9,022 | 11,044 | 11,172 |
| Coding decode, output tokens/s | 548 | — | 1,627 | **2,226** |
| Short chat, end-to-end output tokens/s | 246 | — | 842 | — |

The coding prompt is about 2.15K tokens. Prefill uses uncached prefixes and at most one output token per request, timed to the slowest first token in each batch. Single-stream decode uses a median. Concurrent decode measures aggregate throughput while every stream is decoding. The figures come from independent synthetic runs on the same configuration. Eight-stream short chat reached p95 first-token latency of **353 ms**.

Eight independent, prewarmed 100K prefixes reached **750 tokens/s** in aggregate. In a separate cold-input test, a **1,035,067-token** request reached its first token in about **190 seconds**. All eight short requests inserted after that first token finished before long decode ended. Their first-token p95 was **506 ms**. Short-request latency during cold long prefill was not measured.

The repository includes pinned weights and image digests, the Engram patch, Compose configuration, scripts, raw results, and upstream references. [Reproduction guide](README.en.md)
