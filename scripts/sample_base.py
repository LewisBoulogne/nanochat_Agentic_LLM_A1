"""Sample completions from a trained checkpoint, reproducibly."""

import os
import json
import inspect
import argparse
import torch
from nanochat.engine import Engine
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model, load_model_from_dir


DEFAULT_PROMPTS = [
    "The capital of France is",
    "Water boils at a temperature of",
    "The first person to walk on the moon was",
    "In 1969, scientists discovered that",
    "The main difference between a virus and a bacterium is",
]


def load(source, model_tag, step, device, phase="eval"):
    """load_model(source, *args, **kwargs) forwards to load_model_from_dir,
    so inspect the *target* signature, not the wrapper."""
    sig = inspect.signature(load_model_from_dir)
    print(f"load_model_from_dir signature: {sig}")

    args_pos = [device, phase]
    kwargs = {}
    for name, val in (("model_tag", model_tag), ("step", step)):
        if name in sig.parameters and val is not None:
            kwargs[name] = val

    print(f"calling load_model({source!r}, {args_pos}, {kwargs})")
    out = load_model(source, *args_pos, **kwargs)

    if isinstance(out, tuple):
        model = out[0]
        tokenizer = out[1] if len(out) > 1 else None
        meta = out[2] if len(out) > 2 else {}
    else:
        model, tokenizer, meta = out, None, {}
    if tokenizer is None:
        from nanochat.tokenizer import get_tokenizer
        tokenizer = get_tokenizer()
    return model, tokenizer, meta


def generate(engine, tokenizer, prompt, temperature, max_tokens, seed, top_k=None):
    tokens = tokenizer(prompt, prepend="<|bos|>")
    kwargs = dict(num_samples=1, max_tokens=max_tokens, temperature=temperature, seed=seed)
    if top_k is not None:
        kwargs["top_k"] = top_k
    out = engine.generate_batch(tokens, **kwargs)
    sample = out[0] if isinstance(out, tuple) else out
    text = tokenizer.decode(sample[0])
    # strip the BOS marker for readability in the report
    return text.replace("<|bos|>", "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="base", choices=["base", "sft", "rl"])
    ap.add_argument("--model-tag", default="d4")
    ap.add_argument("--step", type=int, default=None, help="checkpoint step (default: latest)")
    ap.add_argument("--temperatures", type=float, nargs="+", default=[0.0])
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--prompts-file", default=None,help="text file, one prompt per line (default: built-in list)")
    ap.add_argument("--out", default="logs/samples_base.json")
    args = ap.parse_args()

    prompts = DEFAULT_PROMPTS
    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = [ln.strip() for ln in f if ln.strip()]

    device_type = autodetect_device_type()
    _, _, _, _, device = compute_init(device_type)

    model, tokenizer, meta = load(args.source, args.model_tag, args.step, device)
    model.eval()
    engine = Engine(model, tokenizer)

    step = meta.get("step", args.step)
    print(f"\nloaded: source={args.source} tag={args.model_tag} step={step}")
    print(f"seed={args.seed} max_tokens={args.max_tokens} top_k={args.top_k}\n")

    records = []
    for temp in args.temperatures:
        print("=" * 78)
        print(f"temperature = {temp}")
        print("=" * 78)
        for i, prompt in enumerate(prompts):
            text = generate(engine, tokenizer, prompt, temp, args.max_tokens, args.seed + i, args.top_k)
            completion = text[len(prompt):].strip() if text.startswith(prompt) else text
            records.append({
                "prompt": prompt,
                "temperature": temp,
                "seed": args.seed + i,
                "max_tokens": args.max_tokens,
                "top_k": args.top_k,
                "source": args.source,
                "model_tag": args.model_tag,
                "step": step,
                "full_text": text,
                "completion": completion,
            })
            print(f"\n[{i}] {prompt!r}")
            print(f"    -> {completion}")
        print()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {len(records)} samples to {args.out}")

    # markdown block
    md = args.out.rsplit(".", 1)[0] + ".md"
    with open(md, "w") as f:
        f.write(f"Samples from `{args.source}` / `{args.model_tag}` at step {step}. "
                f"Seed {args.seed}+i, max_tokens {args.max_tokens}.\n\n")
        for temp in args.temperatures:
            f.write(f"\n**Temperature {temp}**\n\n")
            f.write("| prompt | completion |\n|---|---|\n")
            for r in records:
                if r["temperature"] != temp:
                    continue
                p = r["prompt"].replace("|", "\\|")
                c = r["completion"].replace("|", "\\|").replace("\n", " ")
                f.write(f"| {p} | {c} |\n")
    print(f"wrote markdown table to {md}")


if __name__ == "__main__":
    main()