# 四张 RTX 6000D 跑 DeepSeek V4.1 Flash，长短请求怎么兼顾？

我把原生 BRRR 的配置和复现脚本放上 GitHub 了。四张 RTX 6000D、TP4，保留 1M 上下文与 16 个调度槽位，DSpark 最多 7 步自适应验证。

- 普通短对话端到端单流 **246 tok/s**，8 流合计 **842 tok/s**。
- 上游短代码基准单流中位 **548 tok/s**，8 流解码窗口合计 **1,600 tok/s**。
- 冷 1M 请求首 token 约 **190 秒**。开始生成后插进 8 个短请求，全部在长请求结束前完成，短请求首 token p95 **506 ms**。

验证成本调到 0.5，20 万独立前缀 4 流快了 **6.5%**，短对话却略慢。实际流量有长有短，我最终保留默认成本标定。700+ 单流还没达到，冷长请求预填充时的短请求尾延迟也待测。

固定权重 revision、镜像底座 digest、Engram 页缓存补丁、Compose、测试脚本和原始结果都在 [GitHub 复现指南](https://github.com/aeternums/deepseek-v41-flash-rtx6000d/tree/main/brrr)。
