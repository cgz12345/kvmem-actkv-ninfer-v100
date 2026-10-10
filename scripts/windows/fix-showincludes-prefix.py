#!/usr/bin/env python3
"""Keep Ninja's MSVC header-dependency tracking working with a localized MSVC.

Ninja records `#include` dependencies for MSVC by parsing cl.exe's /showIncludes
output. It matches the lines against the `msvc_deps_prefix` variable that CMake
wrote into <build>/CMakeFiles/rules.ninja, byte for byte.

On a localized (here: Chinese) MSVC the prefix is localized too, and cl.exe
emits those message bytes in the encoding of the *console code page* of the
process that runs the build. Two consequences, both observed on this machine:

  * configuring and building under different console code pages leaves a prefix
    in rules.ninja that can never match the compiler output, and
  * CMake's own probe can re-encode the prefix (double encoding), producing a
    prefix that matches nothing either.

When the prefix does not match, ninja silently records zero dependencies for the
translation unit and header-only edits stop triggering rebuilds.

This script repairs that: it probes cl.exe with /showIncludes in the current
environment, extracts the exact prefix bytes, and rewrites the
`msvc_deps_prefix` line in rules.ninja when it differs. `--check` reports
whether the recorded dependency log still contains units with zero dependencies.

Both modes are safe to run repeatedly; they never touch anything but the
`msvc_deps_prefix` line of the generated rules.ninja (and a temp dir).

Usage:
    python fix-showincludes-prefix.py --repair <build-dir>
    python fix-showincludes-prefix.py --check  <build-dir>

The caller is expected to have the MSVC environment ready (vcvars) and to keep
the console code page unchanged between --repair and the following ninja run.
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import os
import re
import subprocess
import sys
import tempfile
import shutil
from collections import Counter

RULES_REL = os.path.join("CMakeFiles", "rules.ninja")
PREFIX_KEY = b"msvc_deps_prefix = "
PATH_START_RE = re.compile(rb"[A-Za-z]:[\\/]")
COMPILER_CACHE_GLOB = os.path.join("CMakeFiles", "*", "CMakeC*Compiler.cmake")
CACHE_PREFIX_RE = re.compile(rb'set\((CMAKE_C(?:XX)?_CL_SHOWINCLUDES_PREFIX) "([^"]*)"\)')


def console_code_page() -> str:
    try:
        cp = ctypes.windll.kernel32.GetConsoleOutputCP()
        return f"console code page {cp}"
    except Exception:  # pragma: no cover - non-Windows or no console
        return "unknown console code page"


def probe_prefix() -> bytes:
    """Return the exact /showIncludes prefix bytes cl.exe emits here."""
    tmp = tempfile.mkdtemp(prefix="ninfer-deps-probe-")
    src = os.path.join(tmp, "probe.c")
    with open(src, "w", encoding="ascii") as fh:
        fh.write('#include <stdio.h>\n#include <string.h>\nint probe(void){return 0;}\n')
    obj = os.path.join(tmp, "probe.obj")
    proc = subprocess.run(
        ["cl", "/nologo", "/showIncludes", "/c", src, "/Fo" + obj],
        cwd=tmp,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = proc.stdout
    shutil.rmtree(tmp)
    if proc.returncode:
        raise SystemExit("ERROR: compiler probe failed:\n" + output.decode("utf-8", "replace")[:2000])
    candidates = Counter()
    for line in output.split(b"\n"):
        line = line.rstrip(b"\r")
        match = PATH_START_RE.search(line)
        if match and match.start() > 0:
            candidates[line[: match.start()]] += 1
    if not candidates:
        raise SystemExit(
            "ERROR: could not detect the /showIncludes prefix; cl.exe output was:\n"
            + output.decode("utf-8", "replace")[:2000]
        )
    # MSVC indents the path by the include depth, so the candidate for a deeply
    # nested header is longer than the message prefix. Take the shortest one: it
    # is the prefix of every other candidate, and a shorter prefix can only make
    # ninja's remaining-whitespace trimming do more work, never less.
    prefix = min(candidates, key=len)
    if len(prefix) < 8 or not all(c.startswith(prefix) for c in candidates):
        raise SystemExit(
            f"ERROR: inconsistent /showIncludes prefix candidates: {dict(candidates)}"
        )
    return prefix


def repair_compiler_cache(build_dir: str, prefix: bytes) -> list[str]:
    """Correct the cached /showIncludes prefix CMake reuses on every configure.

    CMake caches the detected prefix in `CMakeFiles/<version>/CMakeCCompiler.cmake` and
    `CMakeCXXCompiler.cmake` and writes it into rules.ninja on every configure. A value cached
    from an earlier mis-decoded detection therefore comes back on every regenerate, so the cache
    files have to be corrected as well; otherwise the repair of rules.ninja is undone by the next
    build's automatic CMake re-run.
    """
    try:
        text = prefix.decode("utf-8")
    except UnicodeDecodeError:
        print(
            "NOTE: cl.exe output is not valid UTF-8, so the CMake cache cannot carry it; "
            "run configure and build under the same console code page instead."
        )
        return []
    changed: list[str] = []
    for path in glob.glob(os.path.join(build_dir, COMPILER_CACHE_GLOB)):
        with open(path, "rb") as fh:
            data = fh.read()
        replacement = text.encode("utf-8")
        new_data, hits = CACHE_PREFIX_RE.subn(
            lambda match: b"set(" + match.group(1) + b' "' + replacement + b'")', data
        )
        if hits and new_data != data:
            with open(path, "wb") as fh:
                fh.write(new_data)
            changed.append(path)
    return changed


def repair(build_dir: str) -> int:
    rules = os.path.join(build_dir, RULES_REL)
    if not os.path.isfile(rules):
        print(f"ERROR: {rules} not found; run the cmake configure step first.")
        return 1
    prefix = probe_prefix()
    cache_changed = repair_compiler_cache(build_dir, prefix)
    with open(rules, "rb") as fh:
        data = fh.read()
    lines = data.split(b"\n")
    hits = [i for i, line in enumerate(lines) if line.startswith(PREFIX_KEY)]
    if len(hits) != 1:
        print(f"WARNING: expected exactly one msvc_deps_prefix line, found {len(hits)}.")
        return 1
    idx = hits[0]
    old_line = lines[idx].rstrip(b"\r")
    old_prefix = old_line[len(PREFIX_KEY):]
    for path in cache_changed:
        print(f"Repaired cached compiler /showIncludes prefix in {path}")
    if old_prefix == prefix:
        print(f"showIncludes prefix OK ({prefix.hex()}, {console_code_page()}).")
        return 0
    # Keep the file's existing line ending style.
    eol = b"\r\n" if lines[idx].endswith(b"\r") else b"\n"
    lines[idx] = PREFIX_KEY + prefix + (b"\r" if eol == b"\r\n" else b"")
    with open(rules, "wb") as fh:
        fh.write(b"\n".join(lines))
    print(
        "Repaired msvc_deps_prefix in rules.ninja:\n"
        f"  was: {old_prefix.hex()}\n"
        f"  now: {prefix.hex()}  ({prefix.decode('utf-8', 'replace')!r}, {console_code_page()})"
    )
    return 0


def check(build_dir: str) -> int:
    proc = subprocess.run(
        ["ninja", "-C", build_dir, "-t", "deps"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    text = proc.stdout.decode("utf-8", "replace")
    if proc.returncode:
        print(text)
        return proc.returncode
    entries = 0
    zero = 0
    zero_samples: list[str] = []
    for line in text.splitlines():
        if ": #deps " not in line:
            continue
        entries += 1
        deps = line.split(": #deps ", 1)[1].split(",", 1)[0].strip()
        if deps == "0":
            # This generator includes only standard library headers. Ninja's
            # MSVC parser intentionally filters system include paths.
            target = line.split(": #deps ", 1)[0].replace("\\", "/")
            if target in {"src/CMakeFiles/ninfer_webui_embed.dir/serve/webui/embed.cpp.obj",
                          "tests/CMakeFiles/ninfer_softmax_attention_test.dir/ops/softmax_attention/main.cpp.obj"}:
                continue
            zero += 1
            if len(zero_samples) < 3:
                zero_samples.append(line.split(": #deps ", 1)[0])
    if entries == 0:
        print("ERROR: no dependency entries found; run a build before --check.")
        return 1
    if zero:
        print(
            f"WARNING: {zero} of {entries} objects have zero recorded header dependencies,\n"
            "so header edits will not trigger their rebuild. Samples:\n  "
            + "\n  ".join(zero_samples)
            + "\nRemove only the affected build object files, then rebuild to repopulate. "
              "Simply rerunning ninja may leave these objects unchanged."
        )
        return 1
    # Check real runtime dependencies rather than counting unrelated old entries.
    for variant in ("qwen3_6_27b", "qwen3_6_35b_a3b"):
        target = f"src/CMakeFiles/ninfer_engine.dir/targets/{variant}/impl/variant.cpp.obj"
        result = subprocess.run(["ninja", "-C", build_dir, "-t", "deps", target],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        recorded = result.stdout.decode("utf-8", "replace").replace("\\", "/")
        required = ("kvmem_window_impl.h", "program_impl.h")
        if result.returncode or "(VALID)" not in recorded or any(h not in recorded for h in required):
            print(f"ERROR: runtime header dependencies missing or stale: {target}\n{recorded[:500]}")
            return 1
    print(f"Header dependency tracking OK: {entries} entries, no unexpected zero deps; both runtime variants verified.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--repair", metavar="BUILD_DIR", help="probe cl and fix rules.ninja")
    group.add_argument("--check", metavar="BUILD_DIR", help="report zero-dependency objects")
    args = ap.parse_args()
    if args.repair:
        return repair(args.repair)
    return check(args.check)


if __name__ == "__main__":
    sys.exit(main())
