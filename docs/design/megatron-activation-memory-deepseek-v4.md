# Megatron Activation Estimate vs DeepSeek V4

## Scope

This note compares Megatron-LM `compute_activation_memory()` with the
simulator's autograd saved-tensor capture for this case:

```text
B=1, S=2048, H=7168, TP=1
61 DeepSeek V4 layers + 1 MTP layer
384 experts, top_k=6, moe_inter_dim=3072, hc_mult=4
activation checkpoint = none
```

The simulator result below is the logical aggregate of saved activation
lifetimes. It is not the offloaded-device peak.

## What Megatron Computes

`compute_activation_memory()` uses a dense-Transformer closed form:

```text
L * S * B * (18 * H + 4 * F) / TP
+ embedding / final norm / output-logits terms
```

It represents a standard self-attention plus dense MLP layer. The activation
formula does not use MoE expert count/top-k, MLA/DSA projection widths, MHC,
or concrete operator outputs. It also documents that different query
projection and hidden sizes are not handled.

Source: <https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/theoretical_memory_usage.py>

## Major DeepSeek V4 Terms Not Represented

The following values come from `activation_offload_tensors.csv` in the target
run. Each row is an observed saved lifetime; totals cover all 62 captured
layers.

| Structure | Observed saved shape | Per layer | Total | Why the formula misses it |
| --- | --- | ---: | ---: | --- |
| Sparse-attention Q/KV cat | `3 x [1,2048,128,512]` bf16 | 768 MiB | 46.50 GiB | Formula has no `n_heads=128`, `head_dim=512`, or number of retained Q/KV views. |
| Wide attention projection | `[2048,65536]` bf16 | 256 MiB | 15.50 GiB | `65536 = 128 * 512`, while the formula only uses `H=7168`. |
| MHC HcPre branch | `2 x [1,2048,28672]` bf16 | 224 MiB | 13.56 GiB | `28672 = 4 * H`; standard Transformer has no MHC branch. |
| Extra H-size values | multiple `[1,2048,7168]` bf16 | about 197 MiB | 12.03 GiB | V4 retains values around MHC, norms, and quantized projections. |
| Routed-MoE GMM + SwiGLU | `[12288,6144]` plus `2 x [12288,3072]` bf16 | 288 MiB | 17.44 GiB | `12288 = S * top_k = 2048 * 6`; dense `4*S*F` has no routed-token expansion. |
| MHC reshape/clone | `[1,2048,16,1024]` bf16 | 64 MiB | 3.88 GiB | MHC-specific 4D intermediate. |
| Quant matmul H-size saves | `2 x [2048,7168]` bf16 | 56 MiB | 3.39 GiB | Actual quant-matmul backward saves are not modeled. |
| Shared experts | multiple `[2048,3072]` bf16 | 48 MiB | 2.91 GiB | Shared expert work is additional to routed experts. |
| Logits/loss path | `2 x [2048,129280]` fp32 | n/a | 1.97 GiB | Megatron has one coarse output term, not the observed FP32 saved lifetimes. |

## Interpretation

Megatron's approximately 20 GiB result is a useful dense-Transformer lower
bound, not a DeepSeek V4 activation estimate. The simulator observes 3,599
mapped saved activations totaling 121.03 GiB logical bytes. With activation
offload enabled the modeled device peak is 26.19 GiB; without offload it is
143.23 GiB. Therefore 121.03 GiB must not be read as simultaneous device
residency after offload.

The gap is structural: the largest missing terms are sparse-attention Q/KV
representations, MHC's `4 * H` branches, and `top_k=6` routed-MoE expansion,
not small norm or bookkeeping tensors.
