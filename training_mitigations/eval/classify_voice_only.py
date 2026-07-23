"""Label sampled candidates with the VOICE rubric only, using the extended
(v2) rubric that covers collective voice as well as singular role-play.

Differences from classify_failure_mode.py, both requested deliberately:

  1. NO ADDRESSEE JUDGE. classify_failure_mode.py runs judge_addressee.py (J1,
     "does the response treat the user's message as a text to critique?") first
     and lets it win, producing label 1. That axis is dropped here: the only
     failure this filter targets is voice collapse. Consequently labels.csv
     contains ONLY labels 0 and 2, never 1.
  2. VOICE RUBRIC v2 (eval/voice_collapse_prompt_v2.py) instead of
     classify_response_voice.py's v1. v1 separates singular role-play x6.9 but
     collective voice only x1.1 — i.e. not at all — which is why the previous
     "clean" training pools were ~5.5% collective-collapsed. See the v2 module
     docstring for the measurement.

Label mapping (kept numerically compatible with build_dpo_pairs.py, which treats
0 as chosen-eligible and {1,2} as rejected-eligible):

    voice == OUTSIDER     -> 0   (clean; eligible as `chosen`)
    voice == STAKEHOLDER  -> 2   (collapsed; eligible as `rejected`)
    UNCLEAR / unparsed    -> -1  (dropped by build_dpo_pairs; never guessed)

Judge: local Qwen3-8B (instruct) served by vLLM on the compute node, temperature
0, enable_thinking=False — identical serving setup to classify_failure_mode.py,
so the filter model stays distinct from the gpt-5-mini eval judge.

Output: <out-dir>/pairs/labels.csv with id, sample_idx, label, addressee, voice.
`addressee` is written as -1 to record that the axis was deliberately not judged.

Usage:
    python eval/classify_voice_only.py --out-dir /projects/m000254/runs/<arm>
    # validation mode: label an existing candidate pool into a side file
    python eval/classify_voice_only.py --out-dir <old-arm> \
        --labels-out /tmp/v2_labels.csv --limit 1200
"""

import argparse
import csv
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "core"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llama_local import DEFAULT_HF_MODEL_ID
from vllm_sim import launch_vllm_server, wait_for_vllm_server
import voice_collapse_prompt_v2 as V2

csv.field_size_limit(10**9)

AITA_ENVS = ("AITA", "AITA-NTA-FLIP")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True,
                    help="run dir holding pairs/candidates.jsonl")
    ap.add_argument("--labels-out", default=None,
                    help="override output path (validation mode)")
    ap.add_argument("--env", default="PAS", choices=["OEQ", "PAS", "AITA", "AITA-NTA-FLIP"])
    ap.add_argument("--model-id", default=DEFAULT_HF_MODEL_ID)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shuffle-seed", type=int, default=None,
                    help="validation mode: sample --limit rows at random")
    args = ap.parse_args()

    cand_path = os.path.join(args.out_dir, "pairs", "candidates.jsonl")
    out_path = args.labels_out or os.path.join(args.out_dir, "pairs", "labels.csv")
    assert os.path.exists(cand_path), f"missing {cand_path}"

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
    if args.shuffle_seed is not None:
        import random
        random.Random(args.shuffle_seed).shuffle(items)
    if args.limit:
        items = items[:args.limit]

    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for r in csv.DictReader(f):
                done.add((str(r["id"]), int(r["sample_idx"])))
    todo = [(k, v) for k, v in items if k not in done]
    print(f"[voice-only] {len(items)} candidates, {len(done)} already labelled, "
          f"{len(todo)} to do", flush=True)
    if not todo:
        print("[voice-only] nothing to do")
        return

    log_path = os.path.join(args.out_dir, "vllm_voiceonly.log")
    proc, base_url = launch_vllm_server(
        args.model_id, args.device,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=None, log_path=log_path, enable_lora=False)
    wait_for_vllm_server(proc, base_url, log_path=log_path)
    print(f"[voice-only] vllm up at {base_url}", flush=True)

    from openai import OpenAI
    client = OpenAI(base_url=base_url + "/v1", api_key="EMPTY", timeout=600)

    is_post = args.env in AITA_ENVS
    framing = V2.FRAMING_POST if is_post else V2.FRAMING_SENTENCE
    body_tpl = V2.BODY_POST if is_post else V2.BODY_SENTENCE
    # Empty for OEQ/PAS, so their rendered prompt is unchanged from the one that
    # filtered the trained voice2 arms. See TASK_NOTE_POST for why AITA needs it.
    task_note = V2.TASK_NOTE_POST if is_post else V2.TASK_NOTE_SENTENCE

    def voice_of(r):
        body = body_tpl.format(context=str(r["sentence"])[:V2.MAX_CHARS],
                               response=str(r["response"])[:V2.MAX_CHARS])
        out = client.chat.completions.create(
            model=args.model_id,
            messages=[{"role": "system", "content": V2.SYSTEM},
                      {"role": "user", "content": V2.TEMPLATE.format(
                          framing=framing, task_note=task_note, body=body)}],
            max_tokens=8, temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        txt = (out.choices[0].message.content or "").strip().upper()
        for lab in V2.LABELS:
            if lab in txt:
                return lab
        return "UNPARSED"

    lock = threading.Lock()
    new = not os.path.exists(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fh = open(out_path, "a", newline="")
    writer = csv.writer(fh)
    if new:
        writer.writerow(["id", "sample_idx", "label", "addressee", "voice"])
        fh.flush()
    counts = {0: 0, 2: 0, -1: 0}

    def run_one(kv):
        (pid, idx), r = kv
        voc = voice_of(r)
        if voc == "OUTSIDER":
            label = 0
        elif voc == "STAKEHOLDER":
            label = 2
        else:
            label = -1                      # UNCLEAR / unparsed: never guess
        with lock:
            writer.writerow([pid, idx, label, -1, voc])
            fh.flush()
            counts[label] += 1
            tot = sum(counts.values())
            if tot % 1000 == 0:
                print(f"[voice-only] {tot} labelled  {counts}", flush=True)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(run_one, todo))
    finally:
        fh.close()
        proc.terminate()

    tot = sum(counts.values()) or 1
    print(f"[voice-only] done -> {out_path}")
    for lab, name in ((0, "clean/OUTSIDER"), (2, "STAKEHOLDER"), (-1, "unusable")):
        print(f"  label {lab:>2} {name:<16}: {counts[lab]:6d}  ({counts[lab] / tot:.2%})")
    if counts[-1] / tot > 0.05:
        print(f"[voice-only] WARNING: {counts[-1] / tot:.1%} unusable — check the "
              f"judge outputs before trusting the pairs.")


if __name__ == "__main__":
    main()
