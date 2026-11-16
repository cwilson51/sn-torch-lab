## GPU_IDLE_HIGH
The GPU spends a large share of the step doing nothing. Look at the idle breakdown to see why, then apply the fixes for HOST_BOUND or LAUNCH_BOUND.
- Increase the work per step (a larger batch size, if memory allows) so fixed per-step overheads are amortized.
- Overlap host work with GPU work: `pin_memory=True` and `.to(device, non_blocking=True)`.
- Avoid synchronizing every step: `loss.item()`, `.cpu()`, `print(tensor)` and `torch.cuda.synchronize()` stall the CPU until the GPU catches up. Log every N steps instead.

## HOST_BOUND
The GPU waits for the CPU, usually for data loading or Python overhead.
- DataLoader: raise `num_workers` (start with the number of CPU cores allocated, e.g. `$SLURM_CPUS_PER_TASK`), set `persistent_workers=True` and `prefetch_factor=2..4`, and use `pin_memory=True`.
- Move expensive augmentation to the GPU (e.g. torchvision v2 transforms on GPU tensors, or NVIDIA DALI), or pre-process the dataset once.
- On a cluster, copy the dataset to node-local storage (`$SLURM_TMPDIR`) at the start of the job instead of reading many small files from a shared filesystem.
- Request more CPU cores per GPU in the Slurm job if the workers are CPU-starved.

## LAUNCH_BOUND
Kernels are so short that launching them costs as much as running them.
- `torch.compile(model)` fuses small elementwise kernels into fewer, larger ones.
- CUDA graphs (`torch.compile(mode="reduce-overhead")`, or `torch.cuda.graphs`) remove per-kernel launch cost for static shapes.
- Increase the batch size so each kernel does more work.
- Use fused optimizers (`torch.optim.AdamW(..., fused=True)` or `foreach=True`).

## TOP_KERNEL_DOMINANT
One kernel takes most of the GPU time. That is often expected (a large GEMM or convolution); the question is whether it runs as fast as it can.
- If it is a fp32 GEMM/conv, see NO_AMP_SUSPECTED.
- For convolutions: `torch.backends.cudnn.benchmark = True` (fixed input sizes), and `channels_last` memory format with AMP.
- If it is not a GEMM/conv (e.g. an elementwise or reduction kernel), `torch.compile` may fuse it with its neighbours.

## MEMCPY_HIGH
A large share of GPU time is spent copying data between host and device.
- Keep tensors on the GPU between steps; avoid `.cpu()` / `.numpy()` round trips inside the loop.
- Use `pin_memory=True` in the DataLoader and `non_blocking=True` on `.to(device)` so copies overlap with compute.
- Send compact dtypes over PCIe (e.g. uint8 images) and convert/normalize on the GPU.

## NO_AMP_SUSPECTED
Matrix multiplies and convolutions run in fp32 (or TF32) on a GPU with tensor cores for fp16/bf16. TF32 is faster than fp32, but fp16/bf16 is typically faster still and halves activation memory.
- Use automatic mixed precision: `with torch.autocast("cuda", dtype=torch.bfloat16):` around the forward pass and loss (bf16 needs no GradScaler; supported on compute capability ≥ 8.0).
- On GPUs without bf16 (compute capability 7.x), use `dtype=torch.float16` with `torch.amp.GradScaler("cuda")`.
- If staying in fp32, at least enable TF32: `torch.set_float32_matmul_precision("high")`.
- Check accuracy after switching; most CNN and transformer training is unaffected.
