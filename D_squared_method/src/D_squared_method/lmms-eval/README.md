# 使用官方 LMMs-Eval 评测 D-squared

这条评测链路和 FastV 一样：使用官方 `lmms-eval` 跑任务，让 `lmms-eval` 加载本项目里的 LLaVA 和本项目里的 `transformers`。当前推荐使用“手动修改 LLaVA wrapper 参数”的稳定方式，不把 D-squared token 数量放进 `--model_args`。

## 1. 安装 D-squared 本地代码

在 AutoDL 或 Linux 环境中：

```bash
cd /root/autodl-tmp/D_squared_method
pip install -e src/LLaVA
pip install -e src/transformers
```

运行评测前设置：

```bash
export D_SQUARED_ROOT=/root/autodl-tmp/D_squared_method
export PYTHONPATH="$D_SQUARED_ROOT/src:$D_SQUARED_ROOT/src/LLaVA:$D_SQUARED_ROOT/src/transformers/src:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
```

## 2. 下载并安装官方 lmms-eval

建议和 FastV 一样直接使用官方仓库：

```bash
cd /root/autodl-tmp
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
cd lmms-eval
pip install -e .
```

如果需要 Hugging Face token，先登录：

```bash
huggingface-cli login
```

老版本 `huggingface_hub` 可能没有 `get_token()`，可以用下面这个检查：

```bash
python -c "from huggingface_hub import HfFolder; print('HF token found' if HfFolder.get_token() else 'NO HF TOKEN')"
```

## 3. 修改 lmms-eval 的 LLaVA wrapper

先定位实际使用的 wrapper 文件：

```bash
grep -R "class Llava" -n /root/autodl-tmp/lmms-eval/lmms_eval
grep -R "register_model(\"llava\")" -n /root/autodl-tmp/lmms-eval/lmms_eval
```

当前 AutoDL 上常见路径是：

```text
/root/autodl-tmp/lmms-eval/lmms_eval/models/simple/llava.py
```

### 3.1 手动设置 D-squared 参数

在 wrapper 中找到 `load_pretrained_model(...)` 调用后面的这一段：

```python
self._config = self._model.config
```

在它前面插入：

```python
self._model.config.use_d_squared = True
self._model.config.d_squared_inplace = True
self._model.config.d_squared_sys_length = 36
self._model.config.d_squared_image_token_length = 576
self._model.config.d_squared_visual_keep_count = 144
self._model.config.d_squared_llm_keep_count = 144
self._model.config.d_squared_agg_layer = 3
self._model.model.reset_d_squared()
```

这里当前实验配置是：

```text
第一阶段 visual selector 保留 144 个 token
第二阶段 LLM selector 补充保留 144 个 token
最终保留集合是两者并集，正常情况下共 288 个 visual token
剪枝层位置 d_squared_agg_layer = 3，和 FastV 对齐
```

如果跑原版 LLaVA baseline，则改成：

```python
self._model.config.use_d_squared = False
self._model.model.reset_d_squared()
```

### 3.2 修改 generate 参数

在同一个 wrapper 文件里，找到 `self.model.generate(...)` 调用中的：

```python
use_cache=self.use_cache,
```

D-squared token-drop 评测时改成：

```python
use_cache=False,
output_attentions=True,
```

D-squared 的第二阶段 selector 需要上一层 attention，所以必须 `output_attentions=True`。真实 token-drop 会改变序列长度，所以必须 `use_cache=False`。

### 3.3 检查 wrapper 语法

改完后先检查语法：

```bash
python -m py_compile /root/autodl-tmp/lmms-eval/lmms_eval/models/simple/llava.py
```

没有输出就表示语法通过。

## 4. 全量 MME 命令

注意：当前稳定版不在 `--model_args` 里传 D-squared 参数，token 数量在 wrapper 里手动改。

```bash
cd /root/autodl-tmp/lmms-eval

export D_SQUARED_ROOT=/root/autodl-tmp/D_squared_method
export PYTHONPATH="$D_SQUARED_ROOT/src:$D_SQUARED_ROOT/src/LLaVA:$D_SQUARED_ROOT/src/transformers/src:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

accelerate launch \
  --num_processes=1 \
  --num_machines=1 \
  --mixed_precision=no \
  --dynamo_backend=no \
  -m lmms_eval \
  --model llava \
  --model_args pretrained=/root/autodl-tmp/models/llava-v1.5-7b \
  --tasks mme \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix llava_v1_5_7b_mme_d_squared_k3_v144_l144_full \
  --output_path ./logs/
```

全量 MME 不要加 `--limit`。如果只想快速冒烟测试，可以临时加：

```bash
--limit 20
```

## 5. 常见问题

如果报：

```text
Unexpected kwargs: {'use_d_squared': ...}
```

说明你把 D-squared 参数放进了 `--model_args`。当前稳定版不要这样做，应该在 wrapper 里手动写配置。

如果报：

```text
piece id is out of range
```

通常是 LLaVA 的输入里有 `IMAGE_TOKEN_INDEX=-200`，但 lmms-eval 解码时把 prompt 也一起 decode 了。本项目已经在本地 LLaVA 的 `llava_llama.py` 中兼容了 lmms-eval 路径；确认 AutoDL 上同步的是最新的：

```text
/root/autodl-tmp/D_squared_method/src/LLaVA/llava/model/language_model/llava_llama.py
```

如果报：

```text
cannot import name '_filelock' from 'datasets.utils'
```

这是 lmms-eval 和 `datasets` 版本的兼容问题。优先不要大规模升级环境，可以先在 `lmms_eval/evaluator.py` 的 `_enable_reentrant_filelocks()` 里加兼容 `try/except`，让没有 `_filelock` 的版本直接跳过这个增强逻辑。
