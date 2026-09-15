"""Task 1: a minimal BPE trainer, used to produce a merge table with counts and a merge tree for a concrete example word."""

import os
import glob
import json
import argparse
from collections import Counter

try:
    import regex as re  # supports \p{L} etc.
    HAVE_REGEX = True
except ImportError:
    import re
    HAVE_REGEX = False


SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
FALLBACK_PATTERN = r"\s?\w+|\s?[^\s\w]+|\s+"


def vis(s):
    """Make whitespace visible in printed output."""
    return s.replace(" ", "\u2423").replace("\n", "\\n").replace("\t", "\\t")


# Data
def load_text(max_chars):
    """Read up to max_chars from the first ClimbMix training shard."""
    base = os.environ["NANOCHAT_BASE_DIR"]
    files = glob.glob(os.path.join(base, "base_data_climbmix", "shard_00000.parquet"))
    if not files:
        raise SystemExit("could not find shard_00000.parquet under NANOCHAT_BASE_DIR")
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(files[0])
    chunks, total = [], 0
    for rg in range(pf.metadata.num_row_groups):
        for doc in pf.read_row_group(rg).column("text").to_pylist():
            chunks.append(doc)
            total += len(doc)
            if total >= max_chars:
                return "\n\n".join(chunks)[:max_chars]
    return "\n\n".join(chunks)[:max_chars]


def pretokenize(text):
    """Split text into word-like chunks, then count how often each chunk occurs.

    BPE is trained on this frequency table rather than on the raw stream, which
    is what stops merges from ever crossing a word boundary.
    """
    pattern = SPLIT_PATTERN if HAVE_REGEX else FALLBACK_PATTERN
    if not HAVE_REGEX:
        print("warning: 'regex' package not installed, using a cruder split pattern")
    pieces = re.findall(pattern, text)
    return Counter(pieces)


# Training
def get_pair_counts(words):
    """words: {tuple_of_symbols: frequency}. Returns Counter over adjacent pairs.

    A pair's count is the number of times it occurs across the corpus, i.e. the
    sum of word frequencies, not the number of distinct words containing it.
    """
    pairs = Counter()
    for symbols, freq in words.items():
        for a, b in zip(symbols, symbols[1:]):
            pairs[(a, b)] += freq
    return pairs


def merge_pair(words, pair):
    """Replace every adjacent occurrence of `pair` with the concatenated symbol."""
    a, b = pair
    merged = a + b
    out = {}
    for symbols, freq in words.items():
        if len(symbols) < 2:
            out[symbols] = out.get(symbols, 0) + freq
            continue
        new, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new.append(merged)
                i += 2
            else:
                new.append(symbols[i])
                i += 1
        key = tuple(new)
        out[key] = out.get(key, 0) + freq
    return out


def train_bpe(word_freqs, num_merges):
    """Return (merges, final_words).

    merges is a list of dicts: rank, pair, count at time of merging, result.
    """
    words = {tuple(w): f for w, f in word_freqs.items()}
    merges = []
    for rank in range(num_merges):
        pairs = get_pair_counts(words)
        if not pairs:
            break
        # ties broken by the pair itself, so the run is deterministic
        (a, b), count = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))
        if count < 2:
            break
        words = merge_pair(words, (a, b))
        merges.append({"rank": rank, "pair": (a, b), "count": count, "result": a + b})
    return merges, words


# Build a tree with one node per occurrence
class Node:
    """One symbol occurrence in a merge tree.

    Keyed by a unique id rather than by its string, so that two occurrences of
    the same substring inside one word stay separate boxes in the figure.
    """
    _next = 0

    def __init__(self, label, rank=None, count=None, children=None):
        self.id = Node._next
        Node._next += 1
        self.label = label
        self.rank = rank
        self.count = count
        self.children = children


def apply_merges_tree(word, merges):
    """Tokenize `word` by applying merges in rank order.

    Encoding replays training order: at each step take the *lowest-rank*
    applicable pair, not the most frequent one in this word. Doing it the other
    way produces a different tokenization from the trainer.

    Returns (roots, history) where roots are the final tokens as Node trees.
    """
    ranks = {m["pair"]: (m["rank"], m["count"]) for m in merges}
    nodes = [Node(c) for c in word]
    history = []
    while len(nodes) >= 2:
        cands = [
            (ranks[(nodes[i].label, nodes[i + 1].label)][0], i)
            for i in range(len(nodes) - 1)
            if (nodes[i].label, nodes[i + 1].label) in ranks
        ]
        if not cands:
            break
        rank, i = min(cands)
        left, right = nodes[i], nodes[i + 1]
        _, count = ranks[(left.label, right.label)]
        parent = Node(left.label + right.label, rank, count, (left, right))
        history.append({"rank": rank, "left": left.label,
                        "right": right.label, "result": parent.label})
        nodes = nodes[:i] + [parent] + nodes[i + 2:]
    return nodes, history


def print_tree(node, prefix="", is_last=True):
    tag = "" if node.rank is None else f"  [#{node.rank}, {node.count:,}]"
    print(prefix + ("└── " if is_last else "├── ") + repr(vis(node.label)) + tag)
    if node.children:
        p = prefix + ("    " if is_last else "│   ")
        print_tree(node.children[0], p, False)
        print_tree(node.children[1], p, True)


def write_dot(roots, path):
    """Graphviz DOT of the merge tree, annotated with merge rank and count."""
    lines = [
        "digraph merges {",
        '  node [shape=box, fontname="monospace"];',
        "  rankdir=BT;",
        "  ordering=out;",
    ]

    def emit(n):
        label = vis(n.label)
        if n.rank is not None:
            label += f"\\nmerge #{n.rank}, count {n.count:,}"
        lines.append(f'  n{n.id} [label="{label}"];')
        if n.children:
            for c in n.children:
                emit(c)
                lines.append(f"  n{c.id} -> n{n.id};")

    for r in roots:
        emit(r)
    lines.append("}")
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chars", type=int, default=3_000_000)
    ap.add_argument("--merges", type=int, default=40)
    ap.add_argument("--word", type=str, default=None, help="word to trace; if omitted, a deep example is chosen automatically")
    ap.add_argument("--dot", type=str, default="logs/merge_tree.dot")
    args = ap.parse_args()

    print(f"reading {args.max_chars:,} characters from shard_00000 ...")
    text = load_text(args.max_chars)
    word_freqs = pretokenize(text)
    alphabet = set(c for w in word_freqs for c in w)
    print(f"pre-tokenized into {sum(word_freqs.values()):,} chunks "
          f"({len(word_freqs):,} distinct)")
    print(f"initial alphabet: {len(alphabet):,} distinct characters\n")

    merges, final_words = train_bpe(word_freqs, args.merges)

    print("| rank | pair | count | result |")
    print("|---|---|---|---|")
    for m in merges:
        a, b = m["pair"]
        print(f"| {m['rank']} | `{vis(a)}` + `{vis(b)}` | {m['count']:,} | `{vis(m['result'])}` |")

    os.makedirs("logs", exist_ok=True)
    with open("logs/toy_bpe_merges.json", "w") as f:
        json.dump([{**m, "pair": list(m["pair"])} for m in merges], f, indent=2)
    print("\nmerges written to logs/toy_bpe_merges.json")

    # pick word to trace
    if args.word:
        target = args.word
    else:
        best, best_depth = None, -1
        for w, _ in word_freqs.most_common(400):
            _, hist = apply_merges_tree(w, merges)
            if len(hist) > best_depth:
                best, best_depth = w, len(hist)
        target = best
        print(f"\nauto-selected example: {vis(target)!r} ({best_depth} merges fire)")

    roots, history = apply_merges_tree(target, merges)
    symbols = [n.label for n in roots]

    print(f"\n=== merge trace for {vis(target)!r} ===")
    print(f"start:  {[vis(c) for c in target]}")
    for h in history:
        print(f"  merge #{h['rank']:>2}: {vis(h['left'])!r} + {vis(h['right'])!r}"
              f" -> {vis(h['result'])!r}")
    print(f"final:  {[vis(s) for s in symbols]}   ({len(symbols)} tokens)")

    print("\n=== tree ===")
    for n in roots:
        print_tree(n)

    write_dot(roots, args.dot)
    print(f"\ngraphviz source written to {args.dot}")
    print(f"render with: mkdir -p figures && dot -Tpdf {args.dot} -o figures/merge_tree.pdf")


if __name__ == "__main__":
    main()