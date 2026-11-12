# 镜像：拉取还是自建

## 首选：直接拉公开镜像

不需要自己编译。上游把 SM120 的编译结果发布到 GHCR：

```bash
docker pull ghcr.io/local-inference-lab/vllm@sha256:faf0b1f08d688b1aaa7d5161dc3b724c701d17d68278e7d32ebdd3a6b5127ce2
docker tag ghcr.io/local-inference-lab/vllm@sha256:faf0b1f0… ds41:v1
```

用 digest 而不是 tag：tag 会移动，digest 不会。上面这个 digest 对应 tag
`karmic-kraken-20261016.2`，和我们生产机上跑的镜像 rootfs 逐层一致（68 层 diff_id 相同，
创建时间相同，revision 都是 `3d1ccd84`）。

确认自己拉对了：

```bash
docker image inspect ds41:v1 --format '{{index .Config.Labels "org.opencontainers.image.revision"}}'
# 3d1ccd84c34ff2de6ad79004464d11474df80cb1
docker image inspect ds41:v1 --format '{{len .RootFS.Layers}}'
# 68
```

## 自建：从上游源码构建

只有需要改动引擎（打补丁、换内核版本）时才需要。构建分两步，都要 sm120 构建机、
约 60 GB 空闲磁盘和一个 buildx builder（默认名 `lil-wheel-cu134-sm120`）。

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker
cd blackwell-llm-docker
git checkout 3d1ccd84c34ff2de6ad79004464d11474df80cb1

# 1) foundation bundle：把钉死的 NGC 底座重打成带哈希的轮子包
bash tools/jovian_wheel_runtime/build_foundation_bundle.sh dist/foundation

# 2) 应用层镜像：在 foundation 上装 vLLM / FlashInfer / B12X / DeepGEMM
bash tools/jovian_wheel_runtime/build_runtime_image.sh \
  dist/foundation/bundle local/lil-cu134:source-locked
```

约束：

- 构建脚本要求 git 工作区干净，镜像 label 会记录当前 commit。
- 第 1 步会用 `nvcr.io/nvidia/pytorch` 的固定 digest 拉底座并校验；
  底座是 `sha256:33ef5fc1…`（NVIDIA PyTorch 26.08 / CUDA 13.4.1）。
- 第 2 步要求 bundle 里有完整的应用轮子。上游每个 release 都带
  `container-release.json`，列出全部组件（flashinfer、vLLM、B12X、DeepGEMM、
  LMCache、InstantTensor）的仓库、commit、归档名和 sha256，照它逐个取即可。
- 设置 `RUNTIME_GPU=<物理卡号>` 会在加载镜像后跑一次 GPU 校验（导入 vLLM 和
  LMCache 原生扩展、在 SM120 上做一次 BF16 矩阵乘）。

细节以上游 `docs/jovian-cu134-wheel-runtime.md` 为准，那是构建契约的维护源。

## 为什么我们没自建

我们用的就是上游发布的那一版，rootfs 一致。本地重新构建只会得到同样的文件系统和一个
不同的 config digest，却让「我们跑的到底是哪版」变得难追溯。除非要改引擎代码，自建没有收益。
