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

## Operator modeling status (2026-11-12 rerun)

Coverage 1896/57/0 (template nodes): flex_attention forward/backward,
aten convolution + backward and constant_pad_nd are all modeled
(opm gdn-sla2: 02e5bbf convolution, 23a3860 flex_attention, 8ec632b
constant_pad_nd).  Remaining misses are c10d collectives (communication
formulas out of scope) and a handful of small aten ops.  End-to-end
54755.8 us.
