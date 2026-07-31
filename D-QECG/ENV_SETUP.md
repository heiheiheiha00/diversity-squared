# D-QECG 环境说明

本项目只使用独立环境：

```text
/home/majie/.conda/envs/D-QECG
```

不会检查、激活、安装或修改服务器原有的 `llava` 环境。

## 一键创建

```bash
cd /home/majie/code_junle/D-QECG
bash server/create_dqecg_env.sh
```

固定运行时：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| PyTorch / torchvision | 2.5.1 / 0.20.1，CUDA 12.4 |
| Transformers | 4.44.2 |
| lmms-eval | 0.7.1 |
| FlashAttention | 2.8.3.post1 |

脚本可以重复执行；重复执行时只刷新
`/home/majie/.conda/envs/D-QECG`，不接触其他 Conda 环境。

LLaVA 从项目内的 `src/LLaVA` 直接加载。MME 推理不安装 Gradio、
FastAPI、bitsandbytes 等 Web UI、量化或训练专用依赖。
lmms-eval 源码保存在
`/home/majie/.conda/envs/D-QECG/src/lmms-eval-0.7.1`，不会被项目目录同步覆盖。

不运行 MME smoke test：

```bash
RUN_MME=false bash server/create_dqecg_env.sh
```

使用 13B 模型：

```bash
MODEL_PATH=/home/majie/majie_data/base_model/llava-v1.5-13b \
bash server/create_dqecg_env.sh
```

进入环境：

```bash
conda activate /home/majie/.conda/envs/D-QECG
export DQECG_ROOT=/home/majie/code_junle/D-QECG
export PYTHONPATH="$DQECG_ROOT/src:$DQECG_ROOT/src/LLaVA:${PYTHONPATH:-}"
```
