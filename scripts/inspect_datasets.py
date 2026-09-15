"""Task 3 sanity check: inspect the datasets used in mid-training and SFT."""

import os
import json
import inspect
import argparse

from tasks.mmlu import MMLU
from tasks.gsm8k import GSM8K
from tasks.smoltalk import SmolTalk
from nanochat.tokenizer import get_tokenizer


def describe_dataset(name, ds, n_examples, raw_chars):
    print("=" * 78)
    print(f"{name}")
    print("=" * 78)
    try:
        n = len(ds)
        print(f"  rows: {n:,}")
    except TypeError:
        n = None
        print("  rows: (no __len__)")

    print(f"  type: {type(ds).__name__}")
    methods = [m for m in dir(ds) if not m.startswith("_")]
    print(f"  public attrs/methods: {', '.join(methods[:15])}")

    for i in range(n_examples):
        try:
            row = ds[i]
        except Exception as e:
            print(f"  [could not index row {i}: {e}]")
            break
        print(f"\n  --- row {i} ---")
        print(f"  python type: {type(row).__name__}")
        if isinstance(row, dict):
            print(f"  keys: {list(row.keys())}")
        text = json.dumps(row, indent=2, ensure_ascii=False, default=str)
        if len(text) > raw_chars:
            text = text[:raw_chars] + f"\n  ... [{len(text) - raw_chars} more chars]"
        for line in text.splitlines():
            print("  " + line)
    print()
    return n


def find_render_fn(tokenizer):
    """Locate the method that turns a conversation into (ids, mask)."""
    candidates = [m for m in dir(tokenizer) if not m.startswith("_") and any(k in m.lower() for k in ("render", "conversation", "chat", "apply"))]
    print(f"candidate rendering methods on the tokenizer: {candidates}")
    for name in candidates:
        fn = getattr(tokenizer, name)
        if callable(fn):
            try:
                print(f"  {name}{inspect.signature(fn)}")
            except (TypeError, ValueError):
                print(f"  {name}(?)")
    for preferred in ("render_conversation", "render_for_training", "apply_chat_template"):
        if preferred in candidates:
            return getattr(tokenizer, preferred), preferred
    if candidates:
        return getattr(tokenizer, candidates[0]), candidates[0]
    return None, None


def render_and_show(tokenizer, row, label, max_tokens):
    """Print a token-by-token table with the loss mask, for the report."""
    fn, fname = find_render_fn(tokenizer)
    if fn is None:
        print("  could not find a rendering method; available methods:")
        print("   ", [m for m in dir(tokenizer) if not m.startswith("_")])
        return None

    try:
        out = fn(row)
    except Exception as e:
        print(f"  {fname}() raised {type(e).__name__}: {e}")
        print("  try passing row['messages'] or row['conversation'] instead, "
              "and check how chat_sft.py calls it")
        return None

    if isinstance(out, tuple) and len(out) >= 2:
        ids, mask = out[0], out[1]
    elif isinstance(out, dict) and "ids" in out:
        ids, mask = out["ids"], out.get("mask")
    else:
        print(f"  {fname}() returned {type(out).__name__}, not (ids, mask); inspect manually")
        return None

    print(f"\n  --- rendered with {fname}() : {label} ---")
    print(f"  total tokens: {len(ids):,}")
    if mask is not None:
        n_sup = sum(1 for m in mask if m)
        print(f"  supervised tokens (mask=1): {n_sup:,}  ({100*n_sup/len(ids):.1f}%)")
        print(f"  masked tokens     (mask=0): {len(ids)-n_sup:,}  ({100*(len(ids)-n_sup)/len(ids):.1f}%)")

    print(f"\n  {'idx':>5} {'id':>7}  {'mask':>4}  token")
    for i, tid in enumerate(ids[:max_tokens]):
        piece = tokenizer.decode([tid]).replace(" ", "\u2423").replace("\n", "\\n")
        m = mask[i] if mask is not None else "?"
        print(f"  {i:>5} {tid:>7}  {m:>4}  {piece!r}")
    if len(ids) > max_tokens:
        print(f"  ... [{len(ids)-max_tokens} more tokens]")
    return ids, mask


def mask_statistics(tokenizer, ds, name, n):
    """What fraction of tokens actually contribute to the loss?"""
    fn, _ = find_render_fn(tokenizer)
    if fn is None:
        return
    tot_tokens, tot_sup, n_ok = 0, 0, 0
    for i in range(min(n, len(ds))):
        try:
            out = fn(ds[i])
            ids, mask = (out[0], out[1]) if isinstance(out, tuple) else (out["ids"], out["mask"])
        except Exception:
            continue
        tot_tokens += len(ids)
        tot_sup += sum(1 for m in mask if m)
        n_ok += 1
    if n_ok:
        print(f"  {name}: over {n_ok} rows, {tot_sup:,}/{tot_tokens:,} tokens supervised "
              f"({100*tot_sup/tot_tokens:.1f}%), mean length {tot_tokens/n_ok:.0f} tokens")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-examples", type=int, default=2)
    ap.add_argument("--raw-chars", type=int, default=1200)
    ap.add_argument("--max-tokens", type=int, default=60, help="tokens to print in the mask table")
    ap.add_argument("--mask-sample", type=int, default=200, help="rows to average mask statistics over")
    ap.add_argument("--mmlu-epochs", type=int, default=3)
    ap.add_argument("--gsm8k-epochs", type=int, default=4)
    args = ap.parse_args()

    tokenizer = get_tokenizer()
    print(f"tokenizer vocab size: {tokenizer.get_vocab_size():,}\n")

    datasets = {
        "MMLU (all / auxiliary_train)": MMLU(subset="all", split="auxiliary_train"),
        "GSM8K (main / train)": GSM8K(subset="main", split="train"),
        "SmolTalk (train)": SmolTalk(split="train"),
    }

    sizes = {}
    for name, ds in datasets.items():
        sizes[name] = describe_dataset(name, ds, args.n_examples, args.raw_chars)

    # mixture arithmetic
    print("=" * 78)
    print("mixture sizes")
    print("=" * 78)
    mmlu = sizes.get("MMLU (all / auxiliary_train)") or 0
    gsm = sizes.get("GSM8K (main / train)") or 0
    smol = sizes.get("SmolTalk (train)") or 0
    mid = mmlu * args.mmlu_epochs + gsm * args.gsm8k_epochs
    print(f"  Stage 1 (midtrain): MMLU {mmlu:,} x{args.mmlu_epochs} "
          f"+ GSM8K {gsm:,} x{args.gsm8k_epochs} = {mid:,} rows")
    print(f"  Stage 2 (sft):      SmolTalk {smol:,} x1 = {smol:,} rows")
    print(f"  nanochat default:   {mid + smol:,} rows (all three together)")
    print()

    # token-level mask example
    print("=" * 78)
    print("loss mask example (for the report)")
    print("=" * 78)
    for label, ds in (("GSM8K", datasets["GSM8K (main / train)"]), ("SmolTalk", datasets["SmolTalk (train)"])):
        try:
            render_and_show(tokenizer, ds[0], label, args.max_tokens)
        except Exception as e:
            print(f"  [{label}] failed: {type(e).__name__}: {e}")

    # mask statistics
    print("\n" + "=" * 78)
    print(f"mask statistics (first {args.mask_sample} rows of each)")
    print("=" * 78)
    for name, ds in datasets.items():
        mask_statistics(tokenizer, ds, name, args.mask_sample)


if __name__ == "__main__":
    main()