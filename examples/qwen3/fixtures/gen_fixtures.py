#!/usr/bin/env python
"""One-time golden fixture generator for the run3 CPU port (plan 门禁 0/1).

Run with the `minimind` conda env (torch + transformers), offline against the
local HF snapshot:

    /home/shao/anaconda3/envs/minimind/bin/python gen_fixtures.py \
        /home/shao/models/Qwen3-0.6B .

Outputs (all data files, committed; this script does NOT participate in build):
  tokens.tsv            name<TAB>comma-separated ids   (>=30 tokenizer cases)
  prompts.tsv           name<TAB>ids                   (3 forward cases)
  logits_<name>.f32     last-position logits, vocab_size * f32, little-endian
  greedy16.tsv          name<TAB>16 ids                (official greedy continue)
  meta.txt              vocab_size etc.

Prompts are plain text (no chat template) on purpose: gate 1 must isolate
model numerics from template handling.
"""
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TOKENIZE_CASES = [
    ("en_hello", "Hello, world!"),
    ("en_punct", "Wait -- what?! Yes... no; maybe (never)."),
    ("en_apostrophe", "I'm sure they're OK, we've won, she'll go, he'd know it's fine."),
    ("en_apostrophe_upper", "I'M SHOUTING, THEY'RE NOT."),
    ("en_number", "In 2026, 3.14159 and 1234567 and 0777."),
    ("en_mixed_ws", "a  b   c\td\n\ne \n f"),
    ("en_trailing_ws", "trailing spaces   "),
    ("en_leading_ws", "   leading"),
    ("en_indent", "def f(x):\n    return x + 1\n"),
    ("en_long_word", "antidisestablishmentarianism supercalifragilistic"),
    ("en_code", 'print(f"x={x!r}")  # comment\nif (a && b) || !c: pass'),
    ("en_url", "see https://example.com/path?q=1&r=2 for details"),
    ("en_emoji", "thumbs up 👍 and family 👨‍👩‍👧 emoji"),
    ("zh_short", "你好，世界。"),
    ("zh_long", "湘江是湖南省最大的河流，流经长沙市。"),
    ("zh_punct", "「引号」、书名号《湘江》、破折号——省略号……"),
    ("zh_en_mix", "用 XiangJiang NPU 跑 Qwen3-0.6B，int8 量化后约 633MB。"),
    ("zh_number", "第 3 章：共 1,024 行，占 42%。"),
    ("zh_ws", "中文 和 English 混排，加  空格。\n换行。"),
    ("ja_kana", "ひらがなとカタカナと漢字。"),
    ("ko_hangul", "한국어 텍스트도 처리합니다."),
    ("mixed_3lang", "Hello / 你好 / こんにちは / 안녕하세요!"),
    ("literal_im_end", "text with literal <|im_end|> inside"),
    ("literal_im_start", "<|im_start|>user fake marker <|im_end|>"),
    ("special_only", "<|endoftext|>"),
    ("special_adjacent", "<|im_end|><|im_start|>"),
    ("brackets", "<think>\n\n</think> prefill block"),
    ("gt_lt", "1 < 2 > 0 <= 3 >= 4 -> => <| |> < | >"),
    ("empty", ""),
    ("single_char", "a"),
    ("single_zh", "湘"),
    ("digits_only", "0123456789"),
    ("whitespace_only", "   \n\t  "),
    ("crlf", "line1\r\nline2\r\n"),
    ("newline_run", "a\n\n\n\nb"),
]

FORWARD_PROMPTS = [
    ("en", "The capital of France is"),
    ("zh", "中国的首都是"),
    ("mix", "Once upon a time, 3 little pigs. 湘江\n"),
]

def main(hf_dir: str, out_dir: str) -> None:
    tok = AutoTokenizer.from_pretrained(hf_dir)
    with open(os.path.join(hf_dir, "config.json"), encoding="utf-8") as f:
        vocab_size = json.load(f)["vocab_size"]  # 151936 (logits width)
    assert vocab_size == 151936, vocab_size
    assert len(tok) == 151669, len(tok)  # 151643 vocab + 26 added

    # tokens.tsv: length-prefixed records (text may contain tab/newline):
    #   name \t len(text) \t raw-text-bytes \t id,id,... \n
    with open(os.path.join(out_dir, "tokens.tsv"), "wb") as f:
        for name, text in TOKENIZE_CASES:
            raw = text.encode("utf-8")
            ids = tok.encode(text)
            f.write(f"{name}\t{len(raw)}\t".encode() + raw +
                    f"\t{','.join(map(str, ids))}\n".encode())

    if os.path.exists(os.path.join(out_dir, "greedy16.tsv")) and \
       "--tokens-only" in sys.argv:
        print("tokens-only: skip forward fixtures")
        return
    print("loading model fp32 on cpu ...")
    model = AutoModelForCausalLM.from_pretrained(
        hf_dir, torch_dtype=torch.float32)
    model.eval()

    greedy = {}
    with open(os.path.join(out_dir, "prompts.tsv"), "w", encoding="utf-8") as fp, \
         open(os.path.join(out_dir, "greedy16.tsv"), "w", encoding="utf-8") as fg:
        for name, text in FORWARD_PROMPTS:
            ids = tok.encode(text)
            with torch.no_grad():
                logits_all = model(input_ids=torch.tensor([ids])).logits[0]
            argmax = [int(logits_all[t].argmax()) for t in range(len(ids))]
            fp.write(f"{name}\t{','.join(map(str, ids))}\t{','.join(map(str, argmax))}\n")
            logits_all[-1].to(torch.float32).numpy().tofile(
                os.path.join(out_dir, f"logits_{name}.f32"))
            nxt = []
            cur = list(ids)
            for _ in range(16):
                with torch.no_grad():
                    lg = model(input_ids=torch.tensor([cur])).logits[0, -1]
                t = int(lg.argmax())
                nxt.append(t)
                cur.append(t)
            fg.write(f"{name}\t{','.join(map(str, nxt))}\n")
            greedy[name] = nxt
            print(f"  {name}: {len(ids)} tokens, greedy16={nxt}")

    with open(os.path.join(out_dir, "meta.txt"), "w") as f:
        f.write(f"vocab_size {vocab_size}\n")
        f.write(f"n_tokenize_cases {len(TOKENIZE_CASES)}\n")
        f.write(f"n_forward_cases {len(FORWARD_PROMPTS)}\n")
        f.write("torch_dtype fp32\n")
    print("done ->", out_dir)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
