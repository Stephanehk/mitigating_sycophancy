"""Sample N candidate responses per TRAIN prompt from a trained LoRA policy.

Stage 1 of the filtered-pair DPO round (see eval/failure_mode_prompt.py for the
failure modes being targeted). For every prompt in the FROZEN final TRAIN split
-- never the held-out split -- draw N samples from the policy so that stage 2 can
label them and stage 3 can pair a labelled-0 response against a labelled-1/2 one
from the SAME prompt.

Why vLLM rather than eval_endorsement_heldout.generate_one: that path is one HF
`generate` call per prompt with num_return_sequences=1. At 1000 prompts x N=10
it is far too slow. `core/vllm_sim.launch_vllm_server(enable_lora=True)` +
`load_lora_adapter` is the same machinery core/trainer.py already uses to draw
its 10 candidates per prompt, and serves n=10 in a single request.

Why this does not reuse `make_vllm_policy_sampler`: that helper returns bare
strings and drops `finish_reason`. Pairs must be built from COMPLETE responses
only (a truncated response teaches mid-sentence stopping, and a truncation
asymmetry between chosen and rejected becomes a length signal DPO will learn),
so `finished` has to survive to stage 3. Everything else about the request --
chat.completions, enable_thinking=False -- matches that helper exactly, which is
what keeps these samples on the same distribution the policy was trained under.

Generation length is 1536, not the 512 used for held-out eval: these are
TRAINING targets and must be complete. The eval cap is a separate, frozen
reporting decision and is untouched.

Output: <out-dir>/pairs/candidates.jsonl, one row per sample, flushed as it goes
so a killed job resumes rather than restarts:
    {"id","sample_idx","sentence","response","finished"}

Usage:
    python eval/sample_train_candidates.py \
        --env PAS \
        --adapter /projects/m000254/runs/dpo_lora_pas_..._final \
        --out-dir /projects/m000254/runs/dpo_lora_pas_..._final_voicedpo
"""

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO_ROOT, "mitigations"))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "core"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llama_local import DEFAULT_HF_MODEL_ID
from train_data import final_ids_path
from vllm_sim import (launch_vllm_server, wait_for_vllm_server,
                      load_lora_adapter)
from eval_endorsement_heldout import held_out_prompts
from eval_aita_label_heldout import held_out_rows as aita_held_out_rows

LORA_NAME = "policy"

AITA_ENVS = ("AITA", "AITA-NTA-FLIP")

# AITA-NTA-FLIP carries TWO prompts per row -- the original post and the
# NTA-flipped rewrite -- and the FLIP eval generates against both, because the
# flip IS the construct being measured. Sampling only one of them would train
# on half the env. They are therefore emitted as SEPARATE prompts, keyed by an
# id suffixed with the variant.
#
# The suffix is load-bearing, not cosmetic. build_dpo_pairs.py groups candidates
# with `by_prompt.setdefault(str(row["id"]), ...)` -- purely on the id string --
# and takes the group's prompt text from whichever row it sees first. Two
# variants sharing one id would therefore be pooled into a single group, and a
# response to the ORIGINAL post could be paired as "chosen" against a response
# to the FLIPPED post as "rejected". That pair differs by framing, not voice, so
# DPO would learn the framing difference. Distinct ids make the collision
# impossible rather than merely unlikely.
FLIP_VARIANTS = (("post_text", "orig"), ("flipped_post_text", "flip"))


def env_prompts(env, train_ids):
    """-> [(prompt_id, post_text), ...] for one env's frozen TRAIN split.

    OEQ/PAS resolve through eval_endorsement_heldout; AITA/AITA-NTA-FLIP have a
    parallel loader in eval_aita_label_heldout (different source CSVs), so the
    env decides which one to call rather than one loader growing a branch.

    Assumes: ids are unique within the split, and for FLIP that every row
    carries both variant columns. Both are asserted -- a silently half-loaded
    FLIP split would train on the original posts only and look like a success.
    """
    if env not in AITA_ENVS:
        return held_out_prompts(env, run_dir=None, test_ids_path=train_ids)

    rows = aita_held_out_rows(env, run_dir=None, test_ids_path=train_ids)
    if env == "AITA":
        out = [(str(r["id"]), r["post_text"]) for _, r in rows.iterrows()]
    else:
        for col, _ in FLIP_VARIANTS:
            assert col in rows.columns, f"FLIP split missing column {col!r}"
        out = [(f"{r['id']}::{suffix}", r[col])
               for _, r in rows.iterrows()
               for col, suffix in FLIP_VARIANTS]
    ids = [i for i, _ in out]
    assert len(set(ids)) == len(ids), f"{env}: duplicate prompt ids after keying"
    return out


def load_done(path):
    """-> set of (id, sample_idx) already written, so a rerun resumes.

    Tolerates a final truncated line from a kill mid-write.
    """
    done = set()
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue          # truncated tail; regenerated below
            done.add((str(r["id"]), int(r["sample_idx"])))
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # All four envs are supported: the request shape below is the bare post
    # text ({"role": "user", "content": text}), which is exactly the prompt
    # convention AITA/AITA-NTA-FLIP generate under, and the split loaders
    # (final_ids_path / held_out_prompts) are keyed by FINAL_STEM, which
    # already covers aita/flip.
    ap.add_argument("--env", required=True,
                    choices=["OEQ", "PAS", "AITA", "AITA-NTA-FLIP"])
    ap.add_argument("--adapter", required=True,
                    help="trained LoRA adapter dir to sample FROM")
    ap.add_argument("--out-dir", required=True,
                    help="new run dir; candidates land in <out-dir>/pairs/")
    ap.add_argument("--n", type=int, default=10, help="samples per prompt")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=1536,
                    help="1536, NOT the 512 eval cap: training targets must be "
                         "complete responses")
    ap.add_argument("--model-id", default=DEFAULT_HF_MODEL_ID)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--workers", type=int, default=16,
                    help="concurrent in-flight requests")
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: only the first N prompts")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    assert os.path.exists(os.path.join(args.adapter, "adapter_config.json")), (
        f"not a PEFT adapter dir: {args.adapter}")

    # TRAIN split, from the frozen id artifact. The underlying loaders are
    # generic "resolve these ids to (id, post_text)" helpers despite their
    # names; pointing them at the *train* ids file is what guarantees held-out
    # is never touched.
    train_ids = final_ids_path(args.env, "train")
    prompts = env_prompts(args.env, train_ids)
    if args.limit:
        prompts = prompts[:args.limit]
    print(f"[sample] env={args.env} prompts={len(prompts)} n={args.n} "
          f"max_tokens={args.max_tokens}")
    if args.env == "AITA-NTA-FLIP":
        print(f"[sample] FLIP: {len(prompts)} prompts = "
              f"{len(prompts) // len(FLIP_VARIANTS)} rows x "
              f"{len(FLIP_VARIANTS)} variants (ids suffixed ::orig / ::flip)")
    print(f"[sample] train ids: {train_ids}")

    out_dir = os.path.join(args.out_dir, "pairs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "candidates.jsonl")

    done = load_done(out_path)
    todo = [(i, p) for i, p in prompts
            if any((str(i), k) not in done for k in range(args.n))]
    if done:
        print(f"[sample] resuming: {len(done)} samples on disk, "
              f"{len(todo)} prompts still incomplete")
    if not todo:
        print("[sample] nothing to do")
        return

    log_path = os.path.join(args.out_dir, "vllm_sample.log")
    proc, base_url = launch_vllm_server(
        args.model_id, args.device,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=None, log_path=log_path,
        enable_lora=True, max_lora_rank=16)
    wait_for_vllm_server(proc, base_url, log_path=log_path)
    load_lora_adapter(base_url, LORA_NAME, args.adapter)
    print(f"[sample] vllm up at {base_url}, adapter '{LORA_NAME}' loaded")

    from openai import OpenAI
    client = OpenAI(base_url=base_url + "/v1", api_key="EMPTY", timeout=1200)

    lock = threading.Lock()
    fh = open(out_path, "a")
    n_written = [0]

    def run_prompt(item):
        pid, text = item
        # Same request shape as make_vllm_policy_sampler (chat.completions,
        # enable_thinking=False) so these samples sit on the training
        # distribution -- but we keep finish_reason.
        resp = client.chat.completions.create(
            model=LORA_NAME,
            messages=[{"role": "user", "content": text}],
            max_tokens=int(args.max_tokens),
            n=int(args.n),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            seed=int(args.seed),
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        rows = []
        for k, ch in enumerate(resp.choices):
            content = ch.message.content
            if content is None:
                continue          # nothing usable; stage 3 drops the slot
            rows.append({
                "id": str(pid),
                "sample_idx": k,
                "sentence": text,
                "response": content.strip(),
                # vLLM: "stop" => hit EOS/stop token, "length" => hit the cap.
                "finished": ch.finish_reason == "stop",
            })
        with lock:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            n_written[0] += len(rows)
            if n_written[0] % 500 < len(rows):
                print(f"[sample] {n_written[0]} samples written", flush=True)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(run_prompt, todo))
    finally:
        fh.close()
        proc.terminate()

    print(f"[sample] done: {n_written[0]} new samples -> {out_path}")


if __name__ == "__main__":
    main()
