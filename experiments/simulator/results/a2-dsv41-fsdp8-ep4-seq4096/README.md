# a2-dsv41-fsdp8-ep4-seq4096 platform run results

sim-platform run of the DeepSeek-V4.1 debugmodel a2 geometry, FSDP8 +
expert-parallel 4, seq_len 4096, captured on the 030-qwen3.5 simulator
stack (branch `030-qwen3.5-261106`).

## Files

- `a2_dsv41.toml` — platform run configuration
- `config.json` — GLM5Next unified_mm raw-model config
  (`glm-adapt/titan-qwen35/.../raw_model/unified_mm`) used alongside this run
- `reproduce_config.txt` — full reproduce parameters
- `run.log` — outer run log
- `runs.tar.zst` — per-window artifacts (`000_F` / `001_B` / `002_OPTIMIZER`
  kernel CSVs and chrome traces, `task.log`, `statistics/`), compressed
  with zstd; the raw `runs/` tree is 257 MB and one trace exceeds the
  GitHub 100 MB file limit, hence the archive

## Extract

```bash
tar --zstd -xf runs.tar.zst
```
