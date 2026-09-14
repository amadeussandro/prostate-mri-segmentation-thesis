"""Lightweight training-throughput profiler for Experiment 02 (T4 timeout audit).

Measures WHERE per-step wall-clock goes, WITHOUT running a full 100-epoch job and
WITHOUT touching any real checkpoint. It reuses the exact production pipeline
(src/train.py + src/dataset.py + src/model.py) -- it never redefines model,
preprocessing, split, loss, or optimizer -- so its timings reflect the real run.

What it reports (a few warmup + N measured batches):
  - dataset construction time (one-time: NIfTI load + preprocessing, cached)
  - pure data-loading time per batch (no model) -- the DataLoader's own cost
  - per-stage train-step time: data fetch / host->device / forward / backward /
    optimizer step
  - validation time per batch
  - checkpoint save time (to a TEMP dir -- never the real results dir)
  - GPU memory + utilization (when CUDA is present)
  - optional num_workers sweep (pure data loading)
  - optional AMP (mixed precision) vs FP32 step timing (CUDA only; MEASUREMENT
    ONLY -- it does not change how any real run trains)
  - optional cudnn.benchmark on/off step timing (CUDA only; MEASUREMENT ONLY)

Then it extrapolates an estimated per-epoch time from the measured step time and
the real steps-per-epoch, so you can compare before/after a config change.

Usage (Colab, on the real config):
    python scripts/benchmark_training.py \
        --config configs/experiments/exp02_a_zresample_on.yaml \
        --train-batches 30 --val-batches 20 \
        --num-workers-sweep 0,2,4 --measure-amp --measure-cudnn-benchmark

Nothing here trains a real model to convergence, mutates a real checkpoint, or
changes the scientific configuration.
"""

import argparse
import os
import sys
import tempfile
import time
from contextlib import contextmanager

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.model import build_model  # noqa: E402
from src.train import (  # noqa: E402
    _build_dataset,
    _build_train_sampler,
    _ensure_global_stats,
    _resolve_split_ids,
    compute_loss,
)
from src.utils import get_device, load_config, set_seed  # noqa: E402


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


@contextmanager
def _timer(device):
    _sync(device)
    t0 = time.perf_counter()
    box = {}
    try:
        yield box
    finally:
        _sync(device)
        box["dt"] = time.perf_counter() - t0


def _fmt(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.3f} s"


def _make_loader(dataset, batch_size, sampler, num_workers, device):
    kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )
    if sampler is not None:
        kwargs["sampler"] = sampler
    else:
        kwargs["shuffle"] = False
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, **kwargs)


def _pure_data_loading(dataset, batch_size, sampler, num_workers, device, n_batches):
    loader = _make_loader(dataset, batch_size, sampler, num_workers, device)
    it = iter(loader)
    # warmup one batch (worker spin-up not counted)
    try:
        next(it)
    except StopIteration:
        return None
    n, t0 = 0, time.perf_counter()
    for _ in range(n_batches):
        try:
            next(it)
        except StopIteration:
            break
        n += 1
    dt = time.perf_counter() - t0
    del loader, it
    return dt / n if n else None


def main():
    ap = argparse.ArgumentParser(description="Exp02 training-throughput profiler (no full training).")
    ap.add_argument("--config", default="configs/experiments/exp02_a_zresample_on.yaml")
    ap.add_argument("--train-batches", type=int, default=30)
    ap.add_argument("--val-batches", type=int, default=20)
    ap.add_argument("--num-workers", type=int, default=0, help="workers for the main per-stage timing loop")
    ap.add_argument("--num-workers-sweep", default="", help="comma list, e.g. 0,2,4 (pure data-loading only)")
    ap.add_argument("--measure-amp", action="store_true", help="CUDA only: time AMP vs FP32 (measurement only)")
    ap.add_argument("--measure-cudnn-benchmark", action="store_true", help="CUDA only: time cudnn.benchmark on/off")
    ap.add_argument("--max-train-patients", type=int, default=0,
                    help="Benchmark-only: cap #train patients loaded (0=all). For RAM-limited machines; "
                         "does NOT affect any real run -- per-step timings are unaffected by cohort size.")
    ap.add_argument("--max-val-patients", type=int, default=0, help="Benchmark-only: cap #val patients (0=all).")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    config = load_config(args.config)
    seed = config.get("training", {}).get("seed", 42)
    set_seed(seed)
    device = args.device or get_device()

    print("=" * 70)
    print("EXP02 TRAINING THROUGHPUT BENCHMARK (no full training run)")
    print("=" * 70)
    print(f"  config : {args.config}")
    print(f"  device : {device}")
    if device == "cuda":
        print(f"  gpu    : {torch.cuda.get_device_name(0)}")
    print(f"  torch  : {torch.__version__}")
    print(f"  cudnn.benchmark={torch.backends.cudnn.benchmark} "
          f"deterministic={torch.backends.cudnn.deterministic}")

    training_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    batch_size = training_cfg.get("batch_size", 8)
    num_classes = model_cfg.get("out_channels", 3)
    loss_type = training_cfg.get("loss_function", "cross_entropy")

    # --- split + (arm B) train-only global stats, exactly as run_training does ---
    train_ids, val_ids, _ = _resolve_split_ids(config)
    _ensure_global_stats(config, train_ids)  # uses the FULL train split (leakage-safe), before any cap
    if args.max_train_patients > 0:
        train_ids = list(train_ids)[: args.max_train_patients]
        print(f"  [benchmark] capping train patients to {len(train_ids)} (RAM-limited profiling only)")
    if args.max_val_patients > 0:
        val_ids = list(val_ids)[: args.max_val_patients]
        print(f"  [benchmark] capping val patients to {len(val_ids)} (RAM-limited profiling only)")

    # --- dataset construction (one-time: NIfTI load + preprocessing, then cached) ---
    with _timer(device) as t:
        train_dataset = _build_dataset(config, train_ids)
    build_train_dt = t["dt"]
    with _timer(device) as t:
        val_dataset = _build_dataset(config, val_ids)
    build_val_dt = t["dt"]

    steps_per_epoch = max(1, len(train_dataset) // batch_size)
    print("\n-- dataset --")
    print(f"  train slices : {len(train_dataset)}  ({len(train_ids)} patients)")
    print(f"  val slices   : {len(val_dataset)}  ({len(val_ids)} patients)")
    print(f"  build train  : {_fmt(build_train_dt)}  (one-time, cached in RAM)")
    print(f"  build val    : {_fmt(build_val_dt)}  (one-time, cached in RAM)")
    print(f"  steps/epoch  : {steps_per_epoch} (batch_size={batch_size})")

    sampler = _build_train_sampler(train_dataset, seed, use_weighted_sampling=True)

    # --- pure data loading (no model) at the main num_workers ---
    pure_dt = _pure_data_loading(
        train_dataset, batch_size, sampler, args.num_workers, device, args.train_batches
    )
    print("\n-- pure data loading (no model) --")
    print(f"  num_workers={args.num_workers}: {_fmt(pure_dt)}/batch"
          if pure_dt else "  (not enough batches)")

    # --- per-stage train step timing ---
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg.get("learning_rate", 1e-3),
        weight_decay=training_cfg.get("weight_decay", 1e-4),
    )
    model.train()
    loader = _make_loader(train_dataset, batch_size, sampler, args.num_workers, device)
    it = iter(loader)

    agg = {"data": 0.0, "h2d": 0.0, "fwd": 0.0, "bwd": 0.0, "opt": 0.0}
    measured = 0
    warmup = 2
    total_iters = args.train_batches + warmup
    for i in range(total_iters):
        t_data = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        data_dt = time.perf_counter() - t_data

        with _timer(device) as th:
            images = batch["image"].float().to(device, non_blocking=(device == "cuda"))
            masks = batch["mask"].long().to(device, non_blocking=(device == "cuda"))
        with _timer(device) as tf:
            optimizer.zero_grad(set_to_none=True)
            preds = model(images)
            loss = compute_loss(preds, masks, loss_type=loss_type)
        with _timer(device) as tb:
            loss.backward()
        with _timer(device) as to:
            optimizer.step()

        if i >= warmup:
            agg["data"] += data_dt
            agg["h2d"] += th["dt"]
            agg["fwd"] += tf["dt"]
            agg["bwd"] += tb["dt"]
            agg["opt"] += to["dt"]
            measured += 1

    per = {k: v / measured for k, v in agg.items()} if measured else agg
    step_total = sum(per.values())
    print("\n-- per-stage TRAIN step (mean over {} batches, num_workers={}) --".format(measured, args.num_workers))
    for k in ("data", "h2d", "fwd", "bwd", "opt"):
        pct = (per[k] / step_total * 100) if step_total else 0
        print(f"  {k:5s}: {_fmt(per[k]):>10}   ({pct:4.1f}%)")
    print(f"  TOTAL: {_fmt(step_total):>10}/step")

    # --- validation step timing (forward only, no_grad) ---
    val_loader = _make_loader(val_dataset, batch_size, None, args.num_workers, device)
    model.eval()
    val_agg, val_n = 0.0, 0
    with torch.no_grad():
        vit = iter(val_loader)
        for i in range(args.val_batches + warmup):
            try:
                batch = next(vit)
            except StopIteration:
                break
            with _timer(device) as tv:
                images = batch["image"].float().to(device, non_blocking=(device == "cuda"))
                _ = model(images)
            if i >= warmup:
                val_agg += tv["dt"]
                val_n += 1
    val_per = (val_agg / val_n) if val_n else 0.0
    val_steps = max(1, len(val_dataset) // batch_size)
    print("\n-- validation step (forward only) --")
    print(f"  {_fmt(val_per)}/batch   x {val_steps} val batches")

    # --- checkpoint save timing (to TEMP, never the real results dir) ---
    with tempfile.TemporaryDirectory() as d:
        ckpt_path = os.path.join(d, "bench_ckpt.pt")
        with _timer(device) as tc:
            torch.save(
                {
                    "epoch": 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_metric": 0.0,
                    "config": config,
                },
                ckpt_path,
            )
        save_dt = tc["dt"]
        save_mb = os.path.getsize(ckpt_path) / 1e6
    print("\n-- checkpoint save --")
    print(f"  {_fmt(save_dt)}  ({save_mb:.1f} MB, to temp dir)")

    # --- epoch-time extrapolation ---
    est_train = step_total * steps_per_epoch
    est_val = val_per * val_steps
    est_epoch = est_train + est_val + save_dt
    print("\n-- estimated per-epoch (extrapolated) --")
    print(f"  train : {_fmt(est_train)}  ({steps_per_epoch} steps x {_fmt(step_total)})")
    print(f"  val   : {_fmt(est_val)}")
    print(f"  save  : {_fmt(save_dt)} (only when val improves)")
    print(f"  EPOCH ~= {est_epoch / 60:.2f} min   |   100 epochs ~= {est_epoch * 100 / 3600:.2f} h")

    if device == "cuda":
        print("\n-- GPU --")
        print(f"  mem allocated : {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  mem reserved  : {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        try:
            print(f"  utilization   : {torch.cuda.utilization()}%")
        except Exception:
            print("  utilization   : (torch.cuda.utilization unavailable)")

    # --- optional: num_workers sweep (pure data loading) ---
    if args.num_workers_sweep:
        print("\n-- num_workers sweep (pure data loading, no model) --")
        for w in [int(x) for x in args.num_workers_sweep.split(",") if x.strip() != ""]:
            dt = _pure_data_loading(train_dataset, batch_size, sampler, w, device, args.train_batches)
            print(f"  num_workers={w:2d}: {_fmt(dt)}/batch" if dt else f"  num_workers={w}: n/a")

    # --- optional: AMP vs FP32 (CUDA only, measurement only) ---
    if args.measure_amp:
        if device != "cuda":
            print("\n-- AMP measurement skipped (needs CUDA) --")
        else:
            print("\n-- AMP (mixed precision) vs FP32 -- MEASUREMENT ONLY, real runs unchanged --")
            def _time_steps(use_amp):
                m = build_model(config).to(device)
                opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
                scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
                m.train()
                lit = iter(_make_loader(train_dataset, batch_size, sampler, args.num_workers, device))
                tot, cnt = 0.0, 0
                for i in range(args.train_batches + warmup):
                    try:
                        b = next(lit)
                    except StopIteration:
                        lit = iter(_make_loader(train_dataset, batch_size, sampler, args.num_workers, device)); b = next(lit)
                    with _timer(device) as ts:
                        img = b["image"].float().to(device, non_blocking=True)
                        msk = b["mask"].long().to(device, non_blocking=True)
                        opt.zero_grad(set_to_none=True)
                        with torch.autocast(device_type="cuda", enabled=use_amp):
                            out = m(img)
                            l = compute_loss(out, msk, loss_type=loss_type)
                        scaler.scale(l).backward()
                        scaler.step(opt)
                        scaler.update()
                    if i >= warmup:
                        tot += ts["dt"]; cnt += 1
                return tot / cnt if cnt else 0.0
            fp32 = _time_steps(False)
            amp = _time_steps(True)
            print(f"  FP32 step : {_fmt(fp32)}")
            print(f"  AMP  step : {_fmt(amp)}   (speedup x{fp32 / amp:.2f})" if amp else "  AMP n/a")
            print("  NOTE: AMP changes numerics vs the FP32 Exp01 baseline -- OPTIONAL, not enabled by default.")

    # --- optional: cudnn.benchmark on/off (CUDA only, measurement only) ---
    if args.measure_cudnn_benchmark:
        if device != "cuda":
            print("\n-- cudnn.benchmark measurement skipped (needs CUDA) --")
        else:
            print("\n-- cudnn.benchmark on/off -- MEASUREMENT ONLY, real runs unchanged --")
            def _time_plain(flag):
                prev_b, prev_d = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
                torch.backends.cudnn.benchmark = flag
                torch.backends.cudnn.deterministic = not flag
                try:
                    m = build_model(config).to(device)
                    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
                    m.train()
                    lit = iter(_make_loader(train_dataset, batch_size, sampler, args.num_workers, device))
                    tot, cnt = 0.0, 0
                    for i in range(args.train_batches + warmup):
                        try:
                            b = next(lit)
                        except StopIteration:
                            lit = iter(_make_loader(train_dataset, batch_size, sampler, args.num_workers, device)); b = next(lit)
                        with _timer(device) as ts:
                            img = b["image"].float().to(device, non_blocking=True)
                            msk = b["mask"].long().to(device, non_blocking=True)
                            opt.zero_grad(set_to_none=True)
                            out = m(img); l = compute_loss(out, msk, loss_type=loss_type)
                            l.backward(); opt.step()
                        if i >= warmup:
                            tot += ts["dt"]; cnt += 1
                    return tot / cnt if cnt else 0.0
                finally:
                    torch.backends.cudnn.benchmark = prev_b
                    torch.backends.cudnn.deterministic = prev_d
            off = _time_plain(False)
            on = _time_plain(True)
            print(f"  benchmark=False (current, deterministic): {_fmt(off)}")
            print(f"  benchmark=True  (autotune, nondet)      : {_fmt(on)}   (speedup x{off / on:.2f})" if on else "  n/a")
            print("  NOTE: benchmark=True picks nondeterministic conv algos -- OPTIONAL, not enabled by default.")

    print("\nDone. No full training was run; no real checkpoint was created or modified.")


if __name__ == "__main__":
    main()
