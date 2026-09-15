""" Task 1: compare the 8,192 and 32,768 vocab BPE tokenizers across text types."""

import os
import sys
import glob
import argparse

from nanochat.tokenizer import RustBPETokenizer

def load_tokenizer(dirname):
    base = os.environ.get("NANOCHAT_BASE_DIR")
    if base is None:
        sys.exit("NANOCHAT_BASE_DIR is not set")
    path = os.path.join(base, dirname)
    if not os.path.isdir(path):
        sys.exit(f"missing tokenizer directory: {path}")
    return RustBPETokenizer.from_directory(path)


def encode(tok, text):
    """Return a list of token ids, with no BOS/special tokens prepended."""
    for attempt in (
        lambda: tok.encode(text),
        lambda: tok(text),
    ):
        try:
            ids = attempt()
        except (AttributeError, TypeError):
            continue
        return list(ids)
    sys.exit("could not find a working encode method on the tokenizer")


def decode_one(tok, tid):
    """Decode a single token id to its string form, for display."""
    try:
        s = tok.decode([tid])
    except Exception:
        return f"<{tid}>"
    # make whitespace visible in the printed output
    return s.replace(" ", "\u2423").replace("\n", "\\n").replace("\t", "\\t")


def climbmix_sample(n_chars=4000):
    base = os.environ["NANOCHAT_BASE_DIR"]
    pattern = os.path.join(base, "base_data_climbmix", "shard_00000.parquet")
    files = glob.glob(pattern)
    if not files:
        return None
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None
    pf = pq.ParquetFile(files[0])
    docs = pf.read_row_group(0).column("text").to_pylist()
    out = []
    total = 0
    for d in docs:
        out.append(d)
        total += len(d)
        if total >= n_chars:
            break
    return "\n\n".join(out)[:n_chars]


ENGLISH = (
    "The Rijksmuseum in Amsterdam holds more than a million objects, of which "
    "roughly eight thousand are on display at any one time. Its collection of "
    "seventeenth century Dutch painting is the most comprehensive anywhere, and "
    "the building itself, completed in 1885 to a design by Pierre Cuypers, was "
    "restored over a ten year period before reopening in 2013."
)

DUTCH = (
    "De Universiteit Leiden werd in 1575 gesticht en is daarmee de oudste "
    "universiteit van Nederland. Volgens de overlevering kregen de inwoners van "
    "de stad de keuze tussen een universiteit en vrijstelling van belasting, en "
    "zij kozen voor de universiteit. Het academiegebouw staat aan het Rapenburg."
)

CODE = '''def merge_pair(words, pair):
    """Replace every occurrence of `pair` with the joined symbol."""
    a, b = pair
    out = {}
    for symbols, freq in words.items():
        i, new = 0, []
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new.append(a + b)
                i += 2
            else:
                new.append(symbols[i])
                i += 1
        out[tuple(new)] = freq
    return out
'''

NUMBERS = (
    "3.14159265358979 1234567 2026 0.001 1,000,000 42 "
    "The population was 17549457 in 2024, up from 16829289 in 2014. "
    "IBAN NL91ABNA0417164300, order #99812733, 12/09/2026."
)

GREEK = (
    "Η Βιβλιοθήκη της Αλεξάνδρειας ήταν μία από τις μεγαλύτερες βιβλιοθήκες "
    "του αρχαίου κόσμου. Ιδρύθηκε στις αρχές του τρίτου αιώνα π.Χ. και "
    "αποτελούσε τμήμα του Μουσείου."
)

CJK = (
    "東京は日本の首都であり、世界最大の都市圏を形成している。"
    "人口はおよそ千四百万人で、都市圏全体では三千七百万人を超える。"
)

MISSPELLED = (
    "Protect your business and employee's during flu seaon. In 2017, bythe end "
    "of November only about 41% of the recomended US population had been "
    "vacinated aginst the flu accordng to the CDC."
)

EMOJI = "Training went fine 🎉🔥 but the loss curve looked odd 📉😕 — retrying 🤞"

def build_samples():
    samples = []
    cm = climbmix_sample()
    if cm:
        samples.append(("ClimbMix doc (English)", cm))
    else:
        print("note: could not read ClimbMix shard, skipping that row\n")
    samples += [
        ("English prose", ENGLISH),
        ("Dutch prose", DUTCH),
        ("Python source", CODE),
        ("Numbers / IDs", NUMBERS),
        ("Greek", GREEK),
        ("Japanese", CJK),
        ("Misspelled English", MISSPELLED),
        ("Emoji", EMOJI),
    ]
    return samples

def measure(tok, text):
    ids = encode(tok, text)
    n_chars = len(text)
    n_bytes = len(text.encode("utf-8"))
    n_tokens = len(ids)
    return {
        "chars": n_chars,
        "bytes": n_bytes,
        "tokens": n_tokens,
        "tok_per_char": n_tokens / n_chars,
        "bytes_per_tok": n_bytes / n_tokens,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", default="tokenizer_v8192")
    ap.add_argument("--large", default="tokenizer_v32768")
    args = ap.parse_args()

    tok_s = load_tokenizer(args.small)
    tok_l = load_tokenizer(args.large)
    v_s, v_l = tok_s.get_vocab_size(), tok_l.get_vocab_size()
    print(f"small vocab: {v_s:,}   ({args.small})")
    print(f"large vocab: {v_l:,}   ({args.large})\n")

    samples = build_samples()

    header = (
        f"| Text type | chars | tokens @{v_s//1024}k | tokens @{v_l//1024}k | "
        f"tok/char @{v_s//1024}k | tok/char @{v_l//1024}k | bytes/tok @{v_l//1024}k | reduction |"
    )
    print(header)
    print("|" + "---|" * 8)

    for name, text in samples:
        a = measure(tok_s, text)
        b = measure(tok_l, text)
        reduction = 100 * (a["tokens"] - b["tokens"]) / a["tokens"]
        print(
            f"| {name} | {a['chars']:,} | {a['tokens']:,} | {b['tokens']:,} | "
            f"{a['tok_per_char']:.4f} | {b['tok_per_char']:.4f} | "
            f"{b['bytes_per_tok']:.2f} | {reduction:+.1f}% |"
        )

    # Token-level detail for the failure-case discussion
    print("\n\n=== token-level breakdowns (\u2423 = space) ===")

    detail = [
        ("Numbers", "3.14159265 1234567 2026 42"),
        ("Code indentation", "    if i < len(symbols) - 1:\n        new.append(a + b)"),
        ("Misspellings", "flu seaon bythe recomended vacinated"),
        ("Correct spellings", "flu season by the recommended vaccinated"),
        ("Greek", "Η Βιβλιοθήκη"),
        ("Emoji", "🎉🔥📉"),
    ]
    for name, text in detail:
        print(f"\n--- {name} ---")
        print(f"    text: {text!r}")
        for label, tok in ((f"{v_s//1024}k", tok_s), (f"{v_l//1024}k", tok_l)):
            ids = encode(tok, text)
            pieces = [decode_one(tok, t) for t in ids]
            print(f"    @{label:>3s} ({len(ids):3d} tokens): " + " | ".join(pieces))

    # Embedding cost implication
    print("\n\n=== parameter cost of the vocabulary ===")
    for depth in (2, 4):
        d = max(128, ((depth * 64 + 127) // 128) * 128)
        for v, label in ((v_s, f"{v_s//1024}k"), (v_l, f"{v_l//1024}k")):
            lm_head = v * d
            emb = v * d
            print(
                f"    depth {depth} (d={d}), vocab {label}: "
                f"embedding {emb:,} + lm_head {lm_head:,} = {emb + lm_head:,} params"
            )


if __name__ == "__main__":
    main()