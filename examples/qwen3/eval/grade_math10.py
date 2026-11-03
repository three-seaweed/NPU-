#!/usr/bin/env python3
"""Grade the board's MATH-500 first-10 outputs.

Usage: grade_math10.py <board_eval_dir_snapshot>  (defaults to examples/qwen3/eval/board_run)
Reads out_<i>.log + timing.txt + math500_first10.jsonl.
"""
import json, re, sys, os

def last_boxed(text):
    """Return contents of the LAST \\boxed{...} with brace matching; None if absent."""
    key = "\\boxed{"
    idx = text.rfind(key)
    if idx < 0:
        return None
    i = idx + len(key)
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(c)
        i += 1
    return "".join(out) if depth == 0 else None

def norm(s):
    if s is None:
        return ""
    s = s.strip()
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace(";", "").replace("\\ ", "")
    s = s.replace(" ", "").replace("~", "")
    s = s.replace("^\\circ", "").replace("^{\\circ}", "")
    s = s.replace("\\text{", "").replace("}", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.lower()
    return s

def reply_text(log):
    """Reply = text between the config line and the first [xnpu line."""
    lines = log.split("\n")
    start = None
    for i, l in enumerate(lines):
        if l.startswith("config:"):
            start = i + 1
            break
    if start is None:
        return ""
    end = len(lines)
    for i in range(start, len(lines)):
        if lines[i].startswith("[xnpu") or lines[i].startswith("==="):
            end = i
            break
    return "\n".join(lines[start:end]).strip()

def parse_prof(log):
    m = re.search(r"kick wall=([0-9.]+) ms.*?evict=([0-9.]+) ms", log)
    k = re.search(r"kicks=(\d+)", log)
    return (float(m.group(1)), float(m.group(2)), int(k.group(1)) if k else 0) if m else None

d = sys.argv[1] if len(sys.argv) > 1 else "examples/qwen3/eval/board_run"
ref = {json.loads(l)["problem"][:50]: json.loads(l) for l in open("examples/qwen3/eval/math500_first10.jsonl")}
refs = [json.loads(l) for l in open("examples/qwen3/eval/math500_first10.jsonl")]
timing = {}
tf = os.path.join(d, "timing.txt")
if os.path.exists(tf):
    for l in open(tf):
        p = l.split()
        if len(p) == 2 and p[0].isdigit():
            timing[int(p[0])] = int(p[1])

n_ok = 0
rows = []
for i in range(10):
    p = os.path.join(d, f"out_{i}.log")
    if not os.path.exists(p):
        rows.append((i, "MISSING", "", "", 0, 0))
        continue
    log = open(p, encoding="utf-8", errors="replace").read()
    rep = reply_text(log)
    got = last_boxed(rep)
    exp = refs[i]["answer"]
    ok = norm(got) == norm(exp) and norm(got) != ""
    n_ok += ok
    wall = timing.get(i, 0)
    prof = parse_prof(log)
    rows.append((i, "OK" if ok else "WRONG", (got or "")[:40], exp[:40], wall, prof))

print(f"{'#':>2} {'res':6} {'boxed(got)':42} {'reference':30} {'wall_s':>7}  prefill_ms kicks")
for r in rows:
    prof_s = f"{r[5][0]:9.1f} {r[5][2]:5d}" if r[5] else "    -         -"
    print(f"{r[0]:>2} {r[1]:6} {r[2]:42} {r[3]:30} {r[4]:7}  {prof_s}")
print(f"\nSCORE: {n_ok}/10")
tot = sum(r[4] for r in rows if r[4])
print(f"total wall: {tot}s ({tot/60:.1f} min)")
