## GPU_IDLE_HIGH
The GPU spends a large share of the step doing nothing. Look at the idle breakdown to see why, then apply the fixes for HOST_BOUND or LAUNCH_BOUND.
- Increase the work per step (a larger batch size, if memory allows) so fixed per-step overheads are amortized.
- Overlap host work with GPU work: `pin_memory=True` and `.to(device, non_blocking=True)`.
- Avoid synchronizing every step: `loss.item()`, `.cpu()`, `print(tensor)` and `torch.cuda.synchronize()` stall the CPU until the GPU catches up. Log every N steps instead.

## HOST_BOUND
The GPU waits for the CPU, usually for data loading or Python overhead.
- DataLoader: raise `num_workers` (start with the CPU cores available per GPU: `cpu_cores_per_gpu` in the run config if it is given, otherwise e.g. `$SLURM_CPUS_PER_TASK`), set `persistent_workers=True` and `prefetch_factor=2..4`, and use `pin_memory=True`.
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
Note: PyTorch runs cuDNN convolutions in TF32 by default (`torch.backends.cudnn.allow_tf32 = True`), even when matmuls stay fp32 (`torch.set_float32_matmul_precision("highest")`, the default). So a model written in plain fp32 usually shows TF32 convolution kernels; that is expected, not a misconfiguration.
- Use automatic mixed precision: `with torch.autocast("cuda", dtype=torch.bfloat16):` around the forward pass and loss (bf16 needs no GradScaler; supported on compute capability ≥ 8.0).
- On GPUs without bf16 (compute capability 7.x), use `dtype=torch.float16` with `torch.amp.GradScaler("cuda")`.
- If staying in fp32, at least enable TF32: `torch.set_float32_matmul_precision("high")`.
- Check accuracy after switching; most CNN and transformer training is unaffected.

## COMM_HEAVY
Communication (NCCL collectives) takes a large share of GPU time. Part of it may be ranks waiting for a slower rank inside a collective: check RANK_STRAGGLER and RANK_IDLE_IMBALANCE first, since fixing those also shrinks communication time.
- Communicate less often: with gradient accumulation, skip gradient synchronization on all but the last micro-batch (the framework's no-sync option), so collectives run once per optimizer step.
- Give each GPU more work per step (a larger per-GPU batch, if memory allows), so communication is a smaller share of the step.
- Communicate less data: reduce gradients in bf16/fp16 where the framework supports it (e.g. a gradient compression or reduce-dtype setting).
- Keep a run on as few nodes as fit: GPUs in one node talk over NVLink/PCIe, which is much faster than the network between nodes. Request whole nodes (`--gpus-per-node`) rather than GPUs spread across nodes.

## POOR_OVERLAP
Communication mostly runs while no compute runs, so it adds directly to each step.
- Let communication start during the backward pass: gradient bucketing does this by default in PyTorch DDP. Tune the bucket size (e.g. DDP's `bucket_cap_mb`): smaller buckets start communicating sooner, larger ones have less overhead.
- Avoid synchronizing the GPU inside the training step (`loss.item()`, `.cpu()`, `torch.cuda.synchronize()`), which forces communication to finish before compute continues.
- If the model has parameters unused in some steps, avoid settings that make the framework search for them every step (e.g. DDP `find_unused_parameters=True`), since that delays communication.

## RANK_STRAGGLER
One rank does more GPU work per step than the others, and every other rank waits for it at each collective.
- Check that every rank gets the same amount of work: equal per-rank batch sizes, a `DistributedSampler` (or equivalent) so the data is split evenly, and no extra work on rank 0 (logging, evaluation, checkpointing) inside the training step.
- With variable-size inputs (sequences, images), bucket or pad inputs so per-rank batches cost about the same.
- If the slow rank is always the same GPU or node across runs, suspect hardware: a throttled or shared GPU, or a slower node. Compare `nvidia-smi` clocks and temperatures on that node, and report it to cluster support if it persists.

## RANK_IDLE_IMBALANCE
Some ranks' GPUs sit idle much more than others', usually because their CPU side (data loading, Python work) is slower.
- Give every rank the same number of DataLoader workers and enough CPU cores (`--cpus-per-task` per GPU in the Slurm job).
- Copy the dataset to node-local storage (`$SLURM_TMPDIR`) on every node, so no rank reads from a slower shared filesystem.
- Move rank-specific CPU work (logging, metric computation, checkpointing on rank 0) out of the step, or do it every N steps.

## SCALING_INEFFICIENT
Adding GPUs bought less extra throughput than expected: each GPU does less useful work per second than in the baseline run. The lost time is usually communication or waiting, so check COMM_HEAVY, POOR_OVERLAP, RANK_STRAGGLER and RANK_IDLE_IMBALANCE on the scaled run first, and compare its communication tables with the baseline's.
- Give each GPU more work per step (a larger per-GPU batch, if memory allows), so the fixed communication cost per step is a smaller share.
- Keep the job on as few nodes as fit: communication between nodes is much slower than within a node. Request whole nodes (`--gpus-per-node`) rather than GPUs spread across nodes.
- Make communication overlap the backward pass (see POOR_OVERLAP), and communicate less often with gradient accumulation and the framework's no-sync option.
- If per-GPU work also changed between the runs (different per-GPU batch size, precision or model), the efficiency compares unlike runs: check the config diff before acting on it.
