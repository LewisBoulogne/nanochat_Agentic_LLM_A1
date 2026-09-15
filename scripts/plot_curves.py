"""Task 2: plot bits-per-byte curves from the jsonl training log."""

import os
import sys
import json
import math
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LN2 = math.log(2)


def load_events(path):
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"warning: skipping unparseable line", file=sys.stderr)
    return events


def pick(events, kind):
    return [e for e in events if e.get("event") == kind]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="path to the base_train jsonl log")
    ap.add_argument("--bytes-per-token", type=float, default=4.74, help="tokenizer compression on ClimbMix train, from tok_eval")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--smooth", type=int, default=1, help="moving-average window over training steps (1 = none)")
    args = ap.parse_args()

    events = load_events(args.jsonl)
    meta = (pick(events, "run_meta") or [{}])[0]
    train = pick(events, "train")
    val = pick(events, "val")

    if not val:
        sys.exit("no 'val' events found (check the jsonl path)")

    total_batch = meta.get("total_batch_size")
    num_iters = meta.get("num_iterations")
    print(f"run: {os.path.basename(args.jsonl)}")
    print(f"  train events: {len(train)}   val events: {len(val)}")
    if total_batch:
        print(f"  total_batch_size: {total_batch:,}   num_iterations: {num_iters:,}")
        print(f"  total tokens: {meta.get('total_tokens'):,}")

    # validation curve
    val_steps = [e["step"] for e in val]
    val_bpb = [e["val_bpb"] for e in val]

    # training curve
    have_real_train_bpb = any("train_bpb" in e and e["train_bpb"] is not None for e in val)
    if have_real_train_bpb:
        tr_steps = [e["step"] for e in val if e.get("train_bpb") is not None]
        tr_bpb = [e["train_bpb"] for e in val if e.get("train_bpb") is not None]
        train_label = "train (bpb, measured)"
        print("  using measured train_bpb from the eval block")
    else:
        tr_steps = [e["step"] for e in train]
        tr_bpb = [e["train_loss"] / (LN2 * args.bytes_per_token) for e in train]
        train_label = f"train (converted, {args.bytes_per_token} bytes/token)"
        print(f"  converting train_loss to bpb using {args.bytes_per_token} bytes/token")

    if args.smooth > 1 and len(tr_bpb) > args.smooth:
        w = args.smooth
        sm = [sum(tr_bpb[max(0, i - w + 1): i + 1]) / len(tr_bpb[max(0, i - w + 1): i + 1]) for i in range(len(tr_bpb))]
        tr_bpb = sm

    # numbers summary
    print("\n=== summary ===")
    print(f"  final val bpb: {val_bpb[-1]:.4f}")
    print(f"  minimum val bpb: {min(val_bpb):.4f} (step {val_steps[val_bpb.index(min(val_bpb))]})")
    print(f"  first val bpb: {val_bpb[0]:.4f} (step {val_steps[0]})")
    print(f"  final train bpb: {tr_bpb[-1]:.4f}")
    print(f"  train-val gap: {tr_bpb[-1] - val_bpb[-1]:+.4f}")
    print(f"  uniform baseline: {math.log2(meta.get('vocab_size', 32768)) / args.bytes_per_token:.4f} (log2(V) / bytes_per_token)")
    
    # report bpb at fractions of training
    for frac in (0.1, 0.25, 0.5, 0.75, 1.0):
        target = frac * (num_iters or val_steps[-1])
        nearest = min(val, key=lambda e: abs(e["step"] - target))
        print(f"  val bpb at {frac:>4.0%} of training (step {nearest['step']:>4}): {nearest['val_bpb']:.4f}")

    if train:
        dts = [e["dt"] for e in train if "dt" in e]
        tps = [e["tok_per_sec"] for e in train if "tok_per_sec" in e]
        if dts:
            print(f"  median step time: {sorted(dts)[len(dts)//2]*1000:.0f} ms")
        if tps:
            print(f"  median tok/sec: {sorted(tps)[len(tps)//2]:,}")

    # plot
    os.makedirs(args.outdir, exist_ok=True)

    for logx in (False, True):
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        ax.plot(tr_steps, tr_bpb, lw=1.0, alpha=0.75, label=train_label, color="#4C72B0")
        ax.plot(val_steps, val_bpb, lw=1.6, marker="o", ms=3.0, label="validation (bpb)", color="#C44E52")

        ax.set_xlabel("optimization step")
        ax.set_ylabel("bits per byte")
        if logx:
            ax.set_xscale("log")
            ax.set_xlim(left=max(1, min(tr_steps[1:] or [1])))
        ax.grid(alpha=0.3, lw=0.5)
        ax.legend(frameon=False, fontsize=8)

        # secondary axis in tokens
        if total_batch:
            sec = ax.secondary_xaxis("top", functions=(lambda s: s * total_batch / 1e6, lambda t: t * 1e6 / total_batch))
            sec.set_xlabel("training tokens (millions)", fontsize=8)
            sec.tick_params(labelsize=8)

        fig.tight_layout()
        name = "bpb_curve_logx" if logx else "bpb_curve"
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(args.outdir, f"{name}.{ext}"), dpi=200)
        plt.close(fig)
        print(f"  wrote {args.outdir}/{name}.pdf")

    # learning rate schedule
    if train and "lrm" in train[0]:
        fig, ax = plt.subplots(figsize=(5.5, 2.0))
        ax.plot([e["step"] for e in train], [e["lrm"] for e in train], lw=1.2, color="#55A868")
        ax.set_xlabel("optimization step")
        ax.set_ylabel("LR multiplier")
        ax.grid(alpha=0.3, lw=0.5)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(args.outdir, f"lr_schedule.{ext}"), dpi=200)
        plt.close(fig)
        print(f"  wrote {args.outdir}/lr_schedule.pdf")


if __name__ == "__main__":
    main()