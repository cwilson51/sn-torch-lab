"""Profile ResNet50 training on synthetic ImageNet-sized data and export a trace.

Author: Collin Wilson at 2026-11-13, https://github.com/cwilson51

Run with ``python -m torch_trace_lab.train BATCH_SIZE [options]``.
"""

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function, schedule
from torchvision import transforms
from torchvision.datasets import FakeData
from torchvision.models import resnet50
from tqdm import tqdm


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m torch_trace_lab.train",
        description="Profile ResNet50 training on synthetic ImageNet-sized data.",
    )
    parser.add_argument("batch_size", type=int, help="Batch size used for training.")
    parser.add_argument("--nWorkers", type=int, default=1,
                        help="Number of DataLoader worker processes.")
    parser.add_argument("--nEpochs", type=int, default=1, help="Number of epochs to train.")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory to write the trace JSON to.")
    parser.add_argument("--compile", action="store_true",
                        help="Compile the model and optimizer.")
    parser.add_argument("--amp", action="store_true",
                        help="Use automatic mixed precision (AMP) for training.")

    # Profiling options
    parser.add_argument("--skip_first", type=int, default=1,
                        help="Number of steps to skip before profiling.")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Number of profiler warmup steps.")
    parser.add_argument("--active", type=int, default=1,
                        help="Number of steps to record.")
    return parser.parse_args(argv)


MODEL_NAME = "ResNet-50 (synthetic FakeData)"


def torch_lab_metadata(args, device_type: str) -> dict:
    """The run config that ``torch-lab analyze-traces`` reads from the trace's ``torch_lab`` key."""
    if args.amp:
        precision = "fp16-amp" if device_type == "cuda" else "bf16-amp"
    elif args.compile and device_type == "cuda":
        precision = "tf32"  # --compile sets float32 matmul precision to "high"
    else:
        precision = "fp32"
    return {
        "model": MODEL_NAME,
        "batch_size_per_gpu": args.batch_size,
        "dataloader_workers": args.nWorkers,
        "torch_compile": bool(args.compile),
        "precision": precision,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "profiler_schedule": {
            "skip_first": args.skip_first,
            "wait": 0,
            "warmup": args.warmup,
            "active": args.active,
        },
    }


def embed_metadata(trace_path: Path, metadata: dict) -> None:
    """Add ``metadata`` as a top-level ``torch_lab`` object in an exported trace.

    ``prof.add_metadata_json`` drops metadata added outside the active window of a
    profiler schedule, so the JSON is rewritten after export instead.
    """
    with open(trace_path) as f:
        trace = json.load(f)
    trace["torch_lab"] = metadata
    with open(trace_path, "w") as f:
        json.dump(trace, f)


def main(argv=None):
    args = parse_args(argv)

    job_id = os.environ.get("SLURM_JOB_ID") or time.strftime("%Y%m%d-%H%M%S")
    iterations = args.skip_first + args.warmup + args.active + 5  # five extra as a buffer

    if torch.cuda.is_available():
        device = torch.device("cuda")
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    else:
        device = torch.device("cpu")
        activities = [ProfilerActivity.CPU]
        warnings.warn("CUDA is not available; profiling on CPU only.", stacklevel=1)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_ds = FakeData(
        size=10_000,
        image_size=(3, 224, 224),
        num_classes=1000,
        transform=transform,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.nWorkers,
    )

    model = resnet50()
    if args.compile:
        model = torch.compile(model)
        torch.set_float32_matmul_precision("high")
    model.to(device)

    my_schedule = schedule(
        skip_first=args.skip_first,
        wait=0,
        warmup=args.warmup,
        active=args.active,
        repeat=1,
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # fp16 autocast + GradScaler on CUDA; bf16 autocast (no scaler needed) on CPU.
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None

    @torch.compile(fullgraph=False)
    def compiled_step():
        optimizer.step()

    def optimizer_step():
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        elif args.compile:
            compiled_step()
        else:
            optimizer.step()

    with profile(
        activities=activities,
        profile_memory=False,
        record_shapes=True,
        schedule=my_schedule,
        with_stack=True,
    ) as prof:
        for epoch in range(args.nEpochs):
            print(f"EPOCH {epoch + 1}:")

            model.train(True)
            running_loss = 0.0

            with record_function("training"):
                for i, (inputs, labels) in enumerate(tqdm(train_dataloader)):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    if args.amp:
                        with torch.autocast(device_type=device.type, dtype=amp_dtype):
                            outputs = model(inputs)
                            loss = loss_fn(outputs, labels)
                    else:
                        outputs = model(inputs)
                        loss = loss_fn(outputs, labels)

                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    optimizer_step()
                    optimizer.zero_grad()

                    running_loss += loss.item()
                    if i == iterations - 1:
                        avg_loss = running_loss / iterations
                        print(f"  batch {i + 1} loss: {avg_loss}")
                        break

                    prof.step()

    sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    print(f"Top operators by {sort_key}:")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=10))
    print(prof.key_averages(group_by_input_shape=True).table(sort_by=sort_key, row_limit=10))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    compiled = "_compiled" if args.compile else ""
    amp = "_amp" if args.amp else ""
    trace_path = output_dir / (
        f"{job_id}_train_batch_{args.batch_size}_nworkers_{args.nWorkers}{compiled}{amp}_trace.json"
    )
    prof.export_chrome_trace(str(trace_path))
    embed_metadata(trace_path, torch_lab_metadata(args, device.type))
    print(f"Trace written to {trace_path}")


if __name__ == "__main__":
    main()
