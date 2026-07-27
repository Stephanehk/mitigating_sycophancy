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
     collective voice only x1.1 -- i.e. not at all -- which is why the previous
     "clean" training pools were ~5.5% collective-collapsed. See the v2 module
     docstring for the measurement.

Label mapping (kept numerically compatible with build_dpo_pairs.py, which treats
0 as chosen-eligible and {1,2} as rejected-eligible):

    voice == OUTSIDER     -> 0   (clean; eligible as `chosen`)
    voice == STAKEHOLDER  -> 2   (collapsed; eligible as `rejected`)
    UNCLEAR / unparsed    -> -1  (dropped by build_dpo_pairs; never guessed)

Judge backend (--backend), all at temperature 0, thinking/reasoning OFF so the
label is deterministic and a starved budget cannot silently change it:

  local (default)  local Qwen3-8B (instruct) served by vLLM on --device. Needs a
                   CUDA GPU. Byte-identical to prior voice2 runs.
  gemini           direct Google Gemini API (GEMINI_API_KEY; plain --model-id
                   like gemini-3.6-flash -- NOT "google/..."). No GPU.
  openrouter       OpenRouter (OPENROUTER_API_KEY; "vendor/model" --model-id like
                   openai/gpt-5-mini). No GPU.

The API backends exist so this classify stage can run without a GPU -- useful
when the sampler already owns the only GPU, or on a machine that has no local
Qwen. A refusal or a persistently empty API reply is mapped to UNCLEAR (-1),
never a guessed voice. NOTE the API backends bill per call: one call per
candidate, i.e. ~(--n) x (#train prompts) calls per env.

Output: <out-dir>/pairs/labels.csv with id, sample_idx, label, addressee, voice.
`addressee` is written as -1 to record that the axis was deliberately not judged.

Usage:
    # local (GPU)
    python eval/classify_voice_only.py --out-dir /projects/m000254/runs/<arm>
    # API model, no GPU
    python eval/classify_voice_only.py --out-dir <arm> \
        --backend gemini --model-id gemini-3.6-flash
    python eval/classify_voice_only.py --out-dir <arm> \
        --backend openrouter --model-id openai/gpt-5-mini
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
import voice_collapse_prompt_v2 as V2

csv.field_size_limit(10**9)

AITA_ENVS = ("AITA", "AITA-NTA-FLIP")

# Reasoning API models (e.g. gpt-5-mini) spend the completion budget on hidden
# reasoning before emitting any text; at a tiny max_tokens the API returns
# content=None with finish_reason=length and the label is lost. Match the
# production classify_response_voice.py ceiling.
API_MAX_TOKENS = 2000


def parse_voice(text):
    """Map a raw judge reply to a canonical voice label.

    Assumption: the judge is instructed to answer with exactly one of V2.LABELS.
    Returns the first label that appears in the (upper-cased) reply, or
    "UNPARSED" if none does. Never guesses -- an unparsable reply must become a
    dropped (-1) row, not a coin-flip.
    """
    txt = (text or "").strip().upper()
    for lab in V2.LABELS:
        if lab in txt:
            return lab
    return "UNPARSED"


def voice_to_label(voc):
    """Numeric label consumed by build_dpo_pairs.py.

    OUTSIDER -> 0 (clean, chosen-eligible), STAKEHOLDER -> 2 (collapsed,
    rejected-eligible), anything else -> -1 (dropped; never guessed).
    """
    if voc == "OUTSIDER":
        return 0
    if voc == "STAKEHOLDER":
        return 2
    return -1


def _local_classifier(args):
    """Build (classify, cleanup) backed by a local vLLM Qwen server.

    classify(system, user) -> raw judge text. cleanup() stops the server. Import
    of vllm_sim is deferred so the API backends do not require vLLM/CUDA to be
    importable.
    """
    from vllm_sim import launch_vllm_server, wait_for_vllm_server
    from openai import OpenAI

    log_path = os.path.join(args.out_dir, "vllm_voiceonly.log")
    proc, base_url = launch_vllm_server(
        args.model_id, args.device,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len or None,
        log_path=log_path, enable_lora=False)
    wait_for_vllm_server(proc, base_url, log_path=log_path)
    print(f"[voice-only] vllm up at {base_url}", flush=True)
    client = OpenAI(base_url=base_url + "/v1", api_key="EMPTY", timeout=600)

    def classify(system, user):
        out = client.chat.completions.create(
            model=args.model_id,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_tokens=8, temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        return out.choices[0].message.content or ""

    return classify, proc.terminate


def _openrouter_classifier(args):
    """Build (classify, cleanup) backed by OpenRouter (OpenAI-compatible).

    Mirrors classify_response_voice.py: generous max_tokens + minimal reasoning,
    and a short transient-fault retry that falls back to "" (-> UNCLEAR) rather
    than guessing a voice.
    """
    from openai import OpenAI

    key = os.environ.get("OPENROUTER_API_KEY")
    assert key, ("OPENROUTER_API_KEY is required for --backend openrouter; "
                 "export it before launching")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key,
                    timeout=600)

    def classify(system, user):
        for attempt in range(3):
            try:
                r = client.chat.completions.create(
                    model=args.model_id, temperature=0.0,
                    max_tokens=API_MAX_TOKENS,
                    extra_body={"reasoning": {"effort": "minimal"}},
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}])
                return r.choices[0].message.content or ""
            except Exception:
                if attempt == 2:
                    return ""
        return ""

    return classify, lambda: None


def _gemini_classifier(args):
    """Build (classify, cleanup) backed by the direct Google Gemini API.

    Reuses core/gemini_sim.make_gemini_call_llm (temperature 0, thinking_budget
    0, the same transient-retry schedule as the training sim). A deterministic
    refusal or a persistently empty reply is mapped to "" (-> UNCLEAR); it is not
    a voice we can label and must not be guessed.
    """
    from gemini_sim import (make_gemini_call_llm, GeminiBlockedError,
                            GeminiEmptyError)

    assert "/" not in args.model_id, (
        f"--model-id {args.model_id!r} looks like an OpenRouter id; --backend "
        f"gemini needs a plain Gemini id (e.g. gemini-3.6-flash). Use --backend "
        f"openrouter for 'vendor/model' ids.")
    gcall = make_gemini_call_llm(args.model_id, temperature=0.0,
                                 thinking_budget=0)

    def classify(system, user):
        try:
            text, _in, _out = gcall(None, None, user, 32, system=system)
            return text
        except (GeminiBlockedError, GeminiEmptyError):
            return ""

    return classify, lambda: None


BACKENDS = {
    "local": _local_classifier,
    "gemini": _gemini_classifier,
    "openrouter": _openrouter_classifier,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True,
                    help="run dir holding pairs/candidates.jsonl")
    ap.add_argument("--labels-out", default=None,
                    help="override output path (validation mode)")
    ap.add_argument("--env", default="PAS", choices=["OEQ", "PAS", "AITA", "AITA-NTA-FLIP"])
    ap.add_argument("--backend", default="local", choices=tuple(BACKENDS),
                    help="where the voice judge runs (default: local vLLM, needs "
                         "a GPU). 'gemini'/'openrouter' route to an API model and "
                         "need no GPU; see the module docstring.")
    ap.add_argument("--model-id", default=DEFAULT_HF_MODEL_ID,
                    help="judge model. local: HF id (default Qwen/Qwen3-8B). "
                         "gemini: plain id (gemini-3.6-flash). openrouter: "
                         "'vendor/model' (openai/gpt-5-mini).")
    ap.add_argument("--device", default="cuda:0", help="local backend only")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                    help="local backend only")
    ap.add_argument("--max-model-len", type=int, default=0,
                    help="local backend only; vLLM context window, 0 = the "
                         "model's own max. A 32B judge on one 80GB card cannot "
                         "fit Qwen3-32B's native 40960 window in the leftover "
                         "KV cache and the server refuses to start. The judge "
                         "prompt is bounded (rubric ~830 tokens + two "
                         "MAX_CHARS-truncated fields), so 8192 is ample.")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shuffle-seed", type=int, default=None,
                    help="validation mode: sample --limit rows at random")
    args = ap.parse_args()

    # An API backend left on the Qwen default is almost certainly a mistake: it
    # would ask the API for a local HF repo id. Fail loudly, not silently.
    if args.backend != "local":
        assert args.model_id != DEFAULT_HF_MODEL_ID, (
            f"--backend {args.backend} needs an API --model-id (e.g. "
            f"gemini-3.6-flash or openai/gpt-5-mini); got the local default "
            f"{args.model_id!r}.")

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

    classify, cleanup = BACKENDS[args.backend](args)
    print(f"[voice-only] backend={args.backend} model={args.model_id}", flush=True)

    is_post = args.env in AITA_ENVS
    framing = V2.FRAMING_POST if is_post else V2.FRAMING_SENTENCE
    body_tpl = V2.BODY_POST if is_post else V2.BODY_SENTENCE
    # Empty for OEQ/PAS, so their rendered prompt is unchanged from the one that
    # filtered the trained voice2 arms. See TASK_NOTE_POST for why AITA needs it.
    task_note = V2.TASK_NOTE_POST if is_post else V2.TASK_NOTE_SENTENCE

    def voice_of(r):
        body = body_tpl.format(context=str(r["sentence"])[:V2.MAX_CHARS],
                               response=str(r["response"])[:V2.MAX_CHARS])
        user = V2.TEMPLATE.format(framing=framing, task_note=task_note, body=body)
        return parse_voice(classify(V2.SYSTEM, user))

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
        label = voice_to_label(voc)
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
        cleanup()

    tot = sum(counts.values()) or 1
    print(f"[voice-only] done -> {out_path}")
    for lab, name in ((0, "clean/OUTSIDER"), (2, "STAKEHOLDER"), (-1, "unusable")):
        print(f"  label {lab:>2} {name:<16}: {counts[lab]:6d}  ({counts[lab] / tot:.2%})")
    if counts[-1] / tot > 0.05:
        print(f"[voice-only] WARNING: {counts[-1] / tot:.1%} unusable -- check the "
              f"judge outputs before trusting the pairs.")


if __name__ == "__main__":
    main()
