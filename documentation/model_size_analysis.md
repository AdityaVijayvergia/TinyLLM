# Model Size and Memory Analysis

## 1. Impact of Model Depth

Tested with fixed `max_seq_len = 64` and `batch_size = 50`.
The GPU has 6.00 GiB of VRAM (RTX 1660 Ti).

| Depth | Params (M) | Model VRAM (MB) | Peak Training VRAM (MB) |
|-------|------------|-----------------|-------------------------|
| 2     | 88.1       | 272.3           | 2392.2                  |
| 4     | 109.1      | 368.9           | 2899.7                  |
| 8     | 151.0      | 528.9           | 3894.6                  |
| 12    | 192.9      | 688.9           | 4890.3                  |
| 16    | 234.9      | 848.9           | OOM                     |
| 20    | 276.8      | 1008.9          | OOM                     |
| 24    | 318.8      | 1168.9          | OOM                     |

**Observations:**
- Model parameter memory scales linearly with depth.
- Training VRAM requirements scale very aggressively as depth increases, causing OOM at >=16 depth on a 6GB card at these batch sizes.

## 2. Impact of Max Sequence Length

Tested with fixed `batch_size = 16` and `depth = 4`.

| Seq Len | Params (M) | Model VRAM (MB) | Peak Training VRAM (MB) |
|---------|------------|-----------------|-------------------------|
| 64      | 109.1      | 352.3           | 1292.6                  |
| 128     | 109.1      | 368.9           | 2053.6                  |
| 256     | 109.1      | 369.5           | 3659.0                  |
| 512     | 109.1      | 371.0           | OOM                     |
| 1024    | 109.1      | 373.2           | OOM                     |

**Observations:**
- As sequence length doubles, the training VRAM scales nearly linearly/super-linearly due to activation memory, easily exceeding 6GB by length 512 even for a tiny depth=4 model.

## 3. Impact of Batch Size

Tested with fixed `max_seq_len = 256` and `depth = 4`.

| Batch Size | Params (M) | Model VRAM (MB) | Peak Training VRAM (MB) |
|------------|------------|-----------------|-------------------------|
| 1          | 109.1      | 369.5           | 1295.0                  |
| 4          | 109.1      | 369.5           | 1300.1                  |
| 16         | 109.1      | 369.5           | 3659.0                  |
| 32         | 109.1      | 369.5           | OOM                     |
| 64         | 109.1      | 369.5           | OOM                     |
| 128        | 109.1      | 369.5           | OOM                     |

**Observations:**
- The base overhead (model + optimizer + minimal activations) takes ~1.3GB. 
- Beyond small batch sizes, activation memory dominates and scales roughly linearly with batch size. 

## 4. Theoretical Memory Scaling Formula

The total memory during training comes from two major sources: Fixed overhead (parameters + optimizer) and Dynamic overhead (activations saved for the backward pass).

1. **Depth ($L$) -> $\mathcal{O}(L)$** (Strictly Linear)
   - Every additional layer adds a proportional amount of parameters, optimizer states, and most importantly, another full set of activations to save.
2. **Batch Size ($B$) -> $\mathcal{O}(B)$** (Strictly Linear)
   - Tensors flowing through the network generally have the shape `[Batch Size, ...]`. If you hold sequence length constant and double batch size, exactly twice as much math needs to be saved.
3. **Sequence Length ($T$) -> $\mathcal{O}(T) + \mathcal{O}(T^2)$** (Quadratic)
   - **Quadratic limit:** The Attention map itself is a `[B, Heads, T, T]` tensor. For small sequences (like 512), the linear $\mathcal{O}(T)$ parts actually use more memory than the small $512 \times 512$ attention matrix. However, as $T$ grows beyond 4096 or 8192, the $\mathcal{O}(T^2)$ squared term explodes and becomes the #1 cause of Out of Memory errors.

## 5. GPU Selection and Hardware Utilization (MFU)

When renting a GPU to train the full 321M parameter model (target: depth=20, seq_len=1024), we calculate the required FLOPs and estimate training time and hardware limits.

**Total Training Computation**
- **Parameters:** ~321M
- **Total Training Tokens:** ~6.4 Billion (Chinchilla rule: 20 * Parameters)
- **Total FLOPs:** ~12.3 Quintillion FLOPs

**Renting an A100 80GB (Recommended)**
- **Cost:** ~$0.918/hr
- **Speed:** ~312 theoretical TFLOPS (using bfloat16 Tensor Cores)
- **MFU (Model FLOPs Utilization):** Expect ~40-45%. MFU measures how efficiently memory feeds the compute cores.
- **Estimated Training Time:** ~24.5 hours (1 Day)
- **Total Estimated Cost:** ~$22.50

## 6. Optimizing Batch Sizes on the A100

There are two critical batch size parameters to configure in `scripts/base_train_modified.py` to get perfect utilization on the A100 80GB:

1. `device_batch_size` (Hardware Limit)
   - **Goal:** Set this as high as possible to maximize MFU and token throughput before the GPU runs Out Of Memory (OOM).
   - **Estimation:** An 80GB card can typically hold a `device_batch_size` of **~24 to 32** for a 20-layer, 1024-sequence model.
   - **How to Tune:** Run a quick binary search. Start at 32; if it OOMs immediately, drop to 24; if it runs perfectly, push to 36. Find the highest stable ceiling.

2. `total_batch_size` (Learning Quality)
   - **Goal:** Adhere to the Chinchilla scaling laws, averaging gradients over ~500,000 tokens per optimizer step for optimal learning stability and quality.
   - **Configuration:** Uncomment `total_batch_size = 524288` in the script.
   - **How It Works (Gradient Accumulation):** The script automatically divides `total_batch_size` by the total tokens in your `device_batch_size` to calculate `grad_accum_steps`. For example, if your device runs 32,768 tokens per pass, the script will loop 16 fast, sequential micro-loops on the hardware before executing a single optimizer step, simulating the massive batch size perfectly.