# 4× RTX 6000D 上的 DeepSeek-V4.1-Flash 原生 BRRR 复现指南

**2026-11-22 采用的配置**为 TP4、原版权重、B12X、FP8 KV、DSpark 最多 7 步自适应验证、1,048,576 token 上下文、16 个调度槽位。目标是混合服务短对话、代码和逐轮累积的长会话。仓库首页的 CSF 方案是另一套历史配方。

## 1. 主机与固定版本

| 项目 | 实测环境 |
|---|---|
| GPU | 4× NVIDIA RTX 6000D，SM120，每卡 85,651 MiB，PCIe 5.0 ×16，无 NVLink |
| 主机 | AMD EPYC 9335，约 755 GiB 可见 RAM；四张测试卡共用 NUMA 0，CPU 0–31，容器内存及 swap 上限均为 360 GiB |
| 驱动 | NVIDIA 615.78.08；Linux x86-64、Docker Engine、Compose v2 或更高、NVIDIA Container Toolkit |
| 权重 | [DeepSeek-V4.1-Flash-1017 固定 revision](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170)，48 个 shard，约 510 GB |
| BRRR | [Fraser Price 上游](https://github.com/fraserprice/DeepSeek-4.1-Flash-1017-BRRR)，commit `2aff97254f4b9005e181b91386c120505e091c18` |
| 推理环境 | vLLM `0.1.dev21836+g6d66eed35`、PyTorch `2.14.0a0+4fdf77b940.nv26.08`、B12X 和 CUDA 依赖见 [固定镜像](build.sh) |
| 本地修改 | [Engram buffered I/O 补丁](engram-buffered.patch)，将小读也走页缓存，适合本机 RAM 和 NVMe 条件 |

建议模型和 Docker 存储预留至少 650 GB 可用 NVMe 空间。记录自己的驱动、显存、PCIe 拓扑和 RAM。实测主机还有其他 GPU，复刻时只需要四张目标卡。内存容量不足可能导致页缓存回收，尤其会改变 Engram 读盘和尾延迟。生产镜像的本地 image ID 为 `sha256:9267440df38315dfd62b6026e32e68273084e0668b6ea9aae8057f80d81d71b7`。复建时以源提交、补丁与底镜像 digest 为准，构建元数据可导致本地 image ID 不同。

## 2. 下载、构建与启动

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

[Compose](compose.yaml) 把关键参数固定为 4,096 批 token、2,048 单请求长预填充阈值、Graph 128、GPU 内存利用率 0.975、每 GPU 1.5 GiB 显式 KV 预算；其余引擎默认值由 BRRR 上游 launcher 固定。1M 是每个请求的输入与输出总窗口，16 是调度槽位，约 146 万 token 的 KV 池由所有并发请求共享。默认 8K 是**自适应验证启动成本表的标定长度**。

## 3. 复刻测试

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
  --port 18136 -s 2048 -c 1 8 -r 5 --temperature 1 --top-p 1 > runs/code.log

python3 bench/warm_chat.py tokenize http://127.0.0.1:18136
python3 bench/warm_chat.py bench http://127.0.0.1:18136 > runs/warm.jsonl
python3 bench/mixed_million.py
```

- **短对话**固定合成请求，温度 1、每路 512 输出 token，每个并发三轮。聚合吞吐是总输出 token 除以总批耗时，含首 token 时间。
- **代码**约 2,148 输入 token，关闭思考，温度 1，最多生成 2,048 token，每个并发五轮。先发送未缓存的预填充探针，再测相同输入的解码窗口。重复测同一个正在运行的实例时换 namespace，避免冷预填充探针命中旧缓存。脚本等待可见 GPU 降到 70°C，可在只暴露测试卡的环境运行。
- **长对话**10 万、20 万输入，预热前缀后按请求查验至少 95% 命中及检索标记，每个负载三轮，统计该组端到端聚合吞吐的中位数。20 万独立前缀用 C4，因为 C8 超过测试 KV 池。
- **近 1M 混合**随机化长提示开头和检索标记，首 token 到达后立刻发 8 个短对话。脚本断言检索正确、上下文符合预算、所有短请求在长生成结束前完成，结果写到 `mixed-million-result.json`。

成本系数 A/B 在启动命令前分别设置 `export COST_SCALE=0.5`、`export COST_SCALE=2.0`，默认 1.0。64K 标定使用 `export PROFILE_CONTEXT=65536` 并保持成本系数 1.0。每次变更运行 `docker compose create --force-recreate deepseek`，重新设置 `--cpuset-mems`、启动、等待健康，再按相同顺序测。试验结束用 `unset COST_SCALE PROFILE_CONTEXT` 恢复默认。

完全关闭 DSpark 的对照需要同时去掉上游 launcher 和 Compose 的投机参数。先停掉占用测试端口的实例，再生成这组配置：

```bash
docker compose stop deepseek
unset COST_SCALE PROFILE_CONTEXT
docker run --rm --entrypoint cat deepseek-v41-brrr:2aff972-buffered /usr/local/bin/brrr-serve |
  sed '/^[[:space:]]*--speculative-config /d' > brrr-serve-off
chmod +x brrr-serve-off
docker compose config --format json |
  jq --arg launcher "$PWD/brrr-serve-off" '
    .services.deepseek.command |= (index("--speculative-config") as $i | del(.[$i:$i+2])) |
    .services.deepseek.volumes += [{"type":"bind","source":$launcher,"target":"/usr/local/bin/brrr-serve","read_only":true}]
  ' > compose-off.json
docker compose -f compose-off.json create --force-recreate deepseek
docker update --cpuset-mems "$NUMA_NODE" "$(docker compose -f compose-off.json ps -aq deepseek)"
docker compose -f compose-off.json start deepseek
```

待健康后重复页缓存预热和基准。用 `docker compose -f compose-off.json stop deepseek` 停止对照。重新执行基线启动流程时为 `create` 加上 `--force-recreate`。

## 4. 实测结果

C8、C4 是并发合计。对话是端到端，短代码是所有流同处解码的公共窗口。代码单流列给每轮速率的中位数。

| 方案 | 短对话 C1 / C8 | 短代码 C1 / C8 | 100K 共享 C8 | 100K 独立 C8 | 200K 共享 C8 | 200K 独立 C4 |
|---|---:|---:|---:|---:|---:|---:|
| BRRR 7 步自适应，成本 1.0 | 246 / 842 | 548 / 1600 | 741 | 750 | 679 | 453 |
| 成本 0.5 | 240 / 829 | 553 / 1600 | 717 | 735 | 680 | 483 |
| 成本 2.0 | 233 / 753 | 524 / 1590 | 678 | 659 | 617 | 444 |
| 关闭 DSpark | 164 / 697 | 176 / 764 | 622 | 639 | 574 | 384 |
| 64K 成本标定 | 235 / 803 | 542 / 1595 | 746 | 758 | 669 | 475 |

单位均为 token/s。成本 0.5 和 64K 标定能提升部分 200K 独立会话吞吐，但短对话下降。基线短对话 C8 首 token p95 约 353 ms。生产因此采用成本 1.0、8K 标定。

**长短混合实测。** 1,035,067 输入 token、4,096 输出 token。冷长请求首 token **190.1 s**，解码 **144.8 token/s**。长请求开始生成后发出的 8 个短请求全部在其结束前完成，短请求首 token p50 / p95 为 **210 / 506 ms**，合计端到端 **581 token/s**。

**可解释的长上下文开销。** 单流 GPU trace 中，2K、100K、200K 的验证图中位耗时分别为 7.29、7.88、8.44 ms。图内候选筛选核按每次图调用归一后约为 0.06、0.77、1.32 ms。相关数据见 [profile-summary.jsonl](results/profile-summary.jsonl)。

**社区对照。** 上游 BRRR 在四张 RTX PRO 6000 Max-Q 上保留固定 DSpark，约 2.15K 代码负载测得 703 单流中位、1847 八流合计。我们的实测条件和验证策略不同，单流未达到 700。Moet 在本机曾测到冷 1M 首 token 约 73 秒，但此次混合选型保留 BRRR。 [上游公开测试](https://github.com/fraserprice/DeepSeek-4.1-Flash-1017-BRRR#results)

[原始合成基准与汇总](results/)包含逐请求时间与 token 数。配置和脚本以复刻相同实验条件为目标，时钟、NUMA、页缓存、并发流量与输出接受长度都会影响测速。当前混合用例测的是长请求**生成期间**插入短请求，真实流量重放和冷 1M **预填充期间**的短请求 p95 尚无结果。
