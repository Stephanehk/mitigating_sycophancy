"""Stage 2: label sampled candidates 0/1/2 with a local Qwen3-8B judge.

Labels:
    0 clean, 1 message-critique, 2 stakeholder/role-collapse.

This runs the TWO EXISTING RUBRICS AS SEPARATE BINARY CALLS and combines them in
code, rather than using one fused 3-way prompt:

    addressee (J1, eval/judge_addressee.py)  == 1  -> label 1
    else voice (eval/classify_response_voice.py) == STAKEHOLDER -> label 2
    else -> label 0

The precedence (message-critique wins) keeps the labels mutually exclusive.

WHY NOT ONE FUSED PROMPT. A fused 3-way prompt was written first
(eval/failure_mode_prompt.py) and measured against these same two rubrics on 100
PAS candidates. It collapsed: 87% label-1, 0% label-2, where the source rubrics
gave 2% message-critique and 14% stakeholder on the identical rows, and where
hand-reading confirmed the source rubrics. Fusing the two criteria into one call
made the model fire "message-critique" on almost everything and never detect
role-collapse. The fused prompt is retained only as documentation of that
negative result; do not reintroduce it without re-measuring.

WHY CHAT MODE, NOT BASE-MODEL LIKELIHOOD SCORING. The plan called for a Qwen3-8B
*base* model scoring label digits as raw continuations. Only `Qwen/Qwen3-8B` --
the instruct model -- is in the local HF cache, and the compute nodes are
air-gapped, so `Qwen3-8B-Base` is not reachable from a training job. Measured on
the same rows: raw-completion mode puts ~0 mass on any label token (top-20 is
all whitespace), while chat mode with enable_thinking=False emits a clean label
at ~-0.005 logprob. Chat mode it is.

Keeping the FILTER model (local Qwen3-8B) distinct from the EVAL judge
(gpt-5-mini, login node) also means the DPO round is not trained directly
against the metric it is later scored on.

CAVEAT: these labels are one 8B model's opinion. The source rubrics are
validated in-repo for stability, not accuracy. Hand-label a sample and measure
per-mode agreement before quoting any downstream number.

Output: <out-dir>/pairs/labels.csv with id, sample_idx, label, addressee, voice.

Usage:
    python eval/classify_failure_mode.py --out-dir /projects/m000254/runs/<new-arm>
"""

import argparse
import csv
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "core"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llama_local import DEFAULT_HF_MODEL_ID
from vllm_sim import launch_vllm_server, wait_for_vllm_server
import classify_response_voice as V
import judge_addressee as A

csv.field_size_limit(10**9)

MAX_CHARS = 6000


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True,
                    help="run dir holding pairs/candidates.jsonl")
    ap.add_argument("--model-id", default=DEFAULT_HF_MODEL_ID)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cand_path = os.path.join(args.out_dir, "pairs", "candidates.jsonl")
    out_path = os.path.join(args.out_dir, "pairs", "labels.csv")
    assert os.path.exists(cand_path), f"missing {cand_path}"

    # Dedup on (id, sample_idx), last write wins: stage 1 rewrites a whole
    # prompt on resume, so a killed run can leave duplicate rows.
    rows = {}
    with open(cand_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[(str(r["id"]), int(r["sample_idx"]))] = r
    items = sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    if args.limit:
        items = items[:args.limit]

    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for r in csv.DictReader(f):
                done.add((str(r["id"]), int(r["sample_idx"])))
    todo = [(k, v) for k, v in items if k not in done]
    print(f"[classify] {len(items)} candidates, {len(done)} already labelled, "
          f"{len(todo)} to do")
    if not todo:
        print("[classify] nothing to do")
        return

    log_path = os.path.join(args.out_dir, "vllm_classify.log")
    proc, base_url = launch_vllm_server(
        args.model_id, args.device,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=None, log_path=log_path, enable_lora=False)
    wait_for_vllm_server(proc, base_url, log_path=log_path)
    print(f"[classify] vllm up at {base_url}")

    from openai import OpenAI
    client = OpenAI(base_url=base_url + "/v1", api_key="EMPTY", timeout=600)

    def ask(system, user, max_tokens):
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": user}]
        r = client.chat.completions.create(
            model=args.model_id, messages=msgs, max_tokens=max_tokens,
            temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        return (r.choices[0].message.content or "").strip()

    def addressee_of(r):
        """J1 rubric verbatim -> 1 = treats the message as a text to critique."""
        out = ask(None, A.PROMPT_TEMPLATE.format(
            question=r["sentence"][:MAX_CHARS],
            advice=r["response"][:MAX_CHARS]), 4)
        for c in out:
            if c in "01":
                return int(c)
        return -1

    def voice_of(r):
        """Voice rubric verbatim -> STAKEHOLDER / OUTSIDER / UNCLEAR."""
        body = V.BODY_SENTENCE.format(context=r["sentence"][:V.MAX_CHARS],
                                      response=r["response"][:V.MAX_CHARS])
        out = ask(V.SYSTEM, V.TEMPLATE.format(framing=V.FRAMING_SENTENCE,
                                              body=body), 8).upper()
        for lab in V.LABELS:
            if lab in out:
                return lab
        return "UNPARSED"

    lock = threading.Lock()
    new = not os.path.exists(out_path)
    fh = open(out_path, "a", newline="")
    writer = csv.writer(fh)
    if new:
        writer.writerow(["id", "sample_idx", "label", "addressee", "voice"])
        fh.flush()
    counts = {0: 0, 1: 0, 2: 0, -1: 0}

    def run_one(kv):
        (pid, idx), r = kv
        adr = addressee_of(r)
        voc = voice_of(r)
        if adr == 1:
            label = 1                       # message-critique wins
        elif voc == "STAKEHOLDER":
            label = 2
        elif adr == 0 and voc == "OUTSIDER":
            label = 0
        else:
            label = -1                      # UNCLEAR / unparsed: never guess
        with lock:
            writer.writerow([pid, idx, label, adr, voc])
            fh.flush()
            counts[label] += 1
            tot = sum(counts.values())
            if tot % 1000 == 0:
                print(f"[classify] {tot} labelled  {counts}", flush=True)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(run_one, todo))
    finally:
        fh.close()
        proc.terminate()

    tot = sum(counts.values()) or 1
    print(f"[classify] done -> {out_path}")
    for lab, name in ((0, "clean"), (1, "message-critique"),
                      (2, "stakeholder"), (-1, "unusable")):
        print(f"  label {lab:>2} {name:<17}: {counts[lab]:6d}  "
              f"({counts[lab] / tot:.2%})")
    if counts[-1] / tot > 0.05:
        print(f"[classify] WARNING: {counts[-1] / tot:.1%} unusable — check the "
              f"judge outputs before trusting the pairs.")


if __name__ == "__main__":
    main()
