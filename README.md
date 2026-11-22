# DeepSeek-V4.1-Flash on 4× RTX 6000D

现行生产方案使用原生 BRRR、vLLM、B12X 和 DSpark 7 步自适应验证，保留 1M 上下文和 16 个调度槽位。

短代码实测单流中位 **548 token/s**，8 流最高合计 **1,627 token/s**，16 流最高合计 **2,226 token/s**。各并发档来自同配置的独立基准，使用共同解码窗口。

[技术栈、部署与复现指南](brrr/README.md) · [原始基准结果](brrr/results/) · [社交媒体短稿](brrr/SOCIAL.md)
