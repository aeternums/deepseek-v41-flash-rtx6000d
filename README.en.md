# DeepSeek-V4.1-Flash on 4× RTX 6000D

[简体中文](README.md) · English

The production deployment uses native BRRR with vLLM, B12X, and DSpark adaptive verification with up to seven draft tokens per step. It supports a 1M-token context window and 16 scheduler slots.

On the short coding workload, median single-stream decode throughput was **548 output tokens/s**. The best aggregate rates were **1,627** at eight streams and **2,226** at sixteen. These were separate runs on the same production configuration, measured during the window when all streams were decoding.

For roughly 2.15K input tokens per request, effective prefill throughput was **5,868 input tokens/s** on one stream and **9,022 / 11,044 / 11,172** in aggregate at four, eight, and sixteen streams. The prefill probe used uncached prefixes and client-observed time to first token.

[Deployment and reproduction guide](brrr/README.en.md) · [Raw benchmark results](brrr/results/) · [Shareable technical summary](brrr/SOCIAL.en.md)
