"""Regenerate the registered native GGUF proposal shortlist (stdlib only)."""
import argparse
import hashlib
import struct
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    ranking = root / "tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64"
    raw = ranking.read_bytes()
    counts = struct.unpack("<248320q", raw[:248320 * 8])
    # Registered Qwen/Swift token domain; tool/think/FIM tokens are ordinary added tokens.
    forced = set(range(248044, 248058)) | set(range(248070, 248077))
    order = sorted(range(248077), key=lambda token: -counts[token])
    picked = [token for token in order if token not in forced][:131072 - len(forced)]
    selected = sorted(picked + sorted(forced), key=lambda token: -counts[token])
    header = "// Generated from tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64\n"
    header += "// SHA256 " + hashlib.sha256(raw).hexdigest() + "\n"
    header += "// Stable descending frequency, registered special tokens forced into the shortlist.\n"
    header += "inline constexpr std::uint32_t kGgufQwenShortlist[] = {\n"
    header += "".join("    " + ",".join(map(str, selected[i:i + 16])) + ",\n"
                      for i in range(0, len(selected), 16)) + "};\n"
    output = root / "src/artifact/gguf_qwen_shortlist.inc"
    if args.check:
        if output.read_text() != header:
            raise SystemExit("native GGUF shortlist is stale")
        print("native GGUF shortlist matches registered ranking")
    else:
        output.write_text(header)


if __name__ == "__main__":
    main()
