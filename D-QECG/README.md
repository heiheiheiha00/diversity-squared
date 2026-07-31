# D-QECG

D-QECG 是基于 LLaVA-1.5 的两阶段视觉 token 筛选项目。项目目录统一为
`D-QECG`，Python 包名为 `dqecg`；内部保留 `d_squared_*` 配置键以兼容已有
模型实现、检查点和 lmms-eval wrapper。

## 服务器目录

```text
/home/majie/code_junle/D-QECG
/home/majie/majie_data/base_model/llava-v1.5-7b
/home/majie/majie_data/base_model/llava-v1.5-13b
/home/majie/.cache/huggingface/datasets/lmms-lab___mme
/home/majie/.conda/envs/D-QECG
/home/majie/majie_data/D-QECG
```

原有 `/home/majie/.conda/envs/llava` 环境不会被读取或修改。

## 第一次上服务器

把整个 `D-QECG` 目录放入 `/home/majie/code_junle`，然后执行：

```bash
cd /home/majie/code_junle/D-QECG
bash server/create_dqecg_env.sh
```

脚本会：

1. 在 `/home/majie/.conda/envs/D-QECG` 创建独立环境；
2. 安装固定版本的 PyTorch、Transformers、lmms-eval 和 FlashAttention；
3. 验证新环境；
4. 使用本地 LLaVA-1.5-7B 和缓存的 MME 跑 2 条样本。

MME 数据默认离线读取；首次运行如果缺少 LLaVA 引用的
`openai/clip-vit-large-patch14-336`，会通过 Hugging Face 镜像下载到
`/home/majie/.cache/huggingface`，后续可离线复用。

只安装环境、不运行 MME：

```bash
RUN_MME=false bash server/create_dqecg_env.sh
```

## 运行 MME

```bash
conda run --no-capture-output \
  -p /home/majie/.conda/envs/D-QECG \
  bash /home/majie/code_junle/D-QECG/server/run_mme.sh
```

默认运行 LLaVA eager baseline。运行 D-QECG：

```bash
conda run --no-capture-output \
  -p /home/majie/.conda/envs/D-QECG \
  env MODE=dqecg \
      D2_USE_CACHE=true \
      D2_STATIC_KV_CACHE=true \
      D2_ATTN_IMPLEMENTATION=flash_attention_2 \
  bash /home/majie/code_junle/D-QECG/server/run_mme.sh
```

MME 的 `LIMIT` 必须是偶数。详细环境和 benchmark 用法见
[server/README.md](server/README.md)。
