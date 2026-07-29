## 单机最简命令：
1. 默认 debug + Profiling
```sh
NGPU=8 LOG_RANK=0 bash examples/deepseek_v4/debug_deepseek_v4_single_node.sh \
  --profiling.enable_profiling
  ```
2. 首次导出 Hugging Face Checkpoint
```sh
NGPU=8 LOG_RANK=0 bash examples/deepseek_v4/debug_deepseek_v4_single_node.sh \
  --dump_folder ./export_ckpt \
  --training.steps 1 \
  --debug.seed 42 \
  --debug.deterministic \
  --checkpoint.enable \
  --checkpoint.no_load_only \
  --checkpoint.last_save_in_hf
 ```
3. 加载 Debug Hugging Face Checkpoint 训练
```sh
NGPU=8 LOG_RANK=0 bash examples/deepseek_v4/debug_deepseek_v4_single_node.sh \
  --training.steps 1 \
  --debug.seed 42 \
  --debug.deterministic \
  --checkpoint.enable \
  --checkpoint.initial_load_path ./export_ckpt/checkpoint/step-1 \
  --checkpoint.initial_load_in_hf
```
4. DeepSeek V4 Pro debug
```sh
CONFIG=debug_deepseek_v4_pro_single_node \
  NGPU=8 LOG_RANK=0 \
  bash examples/deepseek_v4/debug_deepseek_v4_single_node.sh
```
5. DeepSeek V4 Pro MXFP8 debug
```sh
CONFIG=debug_deepseek_v4_pro_single_node_mxfp8 \
  NGPU=8 LOG_RANK=0 \
  bash examples/deepseek_v4/debug_deepseek_v4_single_node.sh
```
