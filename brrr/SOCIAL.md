# 四张 RTX 6000D 跑 DeepSeek V4.1 Flash

我把现行生产配置和复现脚本放上 GitHub 了。原版权重跑在 **vLLM + BRRR + B12X** 上，4 卡 TP4、FP8 KV、DSpark 最多 7 步自适应验证，保留 **1M 上下文和 16 个调度槽位**。

- 短代码解码单流中位 **548 tok/s**，8 流最高合计 **1,627 tok/s**，16 流最高合计 **2,226 tok/s**。
- 普通短对话端到端单流 **246 tok/s**，8 流合计 **842 tok/s**。
- 冷 1M 请求首 token 约 **190 秒**。开始生成后插入 8 个短请求，全部在长请求结束前完成，短请求首 token p95 **506 ms**。

各吞吐数字来自同一生产配置的独立基准。固定权重、镜像、Engram 补丁、启动参数、测试脚本和参考来源都在 [GitHub 复现指南](https://github.com/aeternums/deepseek-v41-flash-rtx6000d/tree/main/brrr)。
