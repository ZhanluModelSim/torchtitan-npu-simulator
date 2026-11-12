# a2-qwen35-fsdp8-ep4-seq4096 platform run results

sim-platform run of the Qwen3.5 debugmodel a2 geometry, FSDP8 +
expert-parallel 4, seq_len 4096, captured on the 030-qwen3.5 simulator
stack (branch `030-qwen3.5-261106`).

## Files

- `a2_qwen35.toml` — platform run configuration
- `qwen35_config.json` — model configuration used by the run
- `reproduce_config.txt` — full reproduce parameters
- `run.log` — outer run log
- `runs/` — per-window artifacts (`l1/` kernel CSVs and chrome traces for
  `000_F` / `001_B` / `002_OPTIMIZER`, `statistics/` aggregates) and
  `task.log`

All files are small enough to be stored raw; no archive is needed.
