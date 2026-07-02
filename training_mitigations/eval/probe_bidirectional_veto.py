"""Probe: does a BIDIRECTIONAL (stance-mismatch) stakeholder veto ever veto a
NON-endorsing (challenging) response?

Motivation
----------
The current P5 veto is one-sided: it only ever fires on responses classified as
ENDORSING (Stage-1 YES). The proposed redesign makes Stage 2 symmetric — a
stakeholder vetoes a response whenever the stakeholder's own preferred stance on
the narrator's action (ENDORSE vs CHALLENGE) DISAGREES with the response's stance
(endorse-when-they'd-challenge, OR challenge-when-they'd-endorse).

The open question / my claim under test: because the whole v2 pipeline
(identify -> filter -> concerns) is built to SURFACE concerns, the simulated
stakeholders may (almost) always prefer to CHALLENGE the narrator. If so, the
"they'd endorse" branch is near-dead, no challenging response ever gets vetoed,
and the redesign degenerates back into the one-sided critical push.

This script tests that empirically. It:
  1. Loads 100 OEQ train-split prompts (seed 0 — same split family the _gp run
     trains on).
  2. For each prompt, generates candidates the DEFAULT `_gp` way:
     4 bare-post model samples + 0 critical + 6 stakeholder-voiced samples
     (STAKEHOLDER_RESPONSE_PROMPT_VOICE), round-robin across the identified
     stakeholders — identical to Trainer._sample_responses_mixed for that arm.
  3. Stage-1 classifies each candidate's stance with the EXACT P5 classifier
     (ENDORSE_CLASSIFY_PROMPT_P5) -> `endorses` bool.
  4. Runs the BIDIRECTIONAL mismatch veto per (prompt, stakeholder, response):
     the stakeholder states its own preferred stance and VETOes iff the response
     disagrees with it.
  5. Aggregates the veto with OR-over-stakeholders (a response is vetoed iff at
     least one stakeholder vetoes it) and reports, above all:

        of all NON-ENDORSING responses, how many end up VETOED.

Decision rule: if that fraction is ~0, the load-bearing-assumption claim holds
(stakeholders essentially never want to endorse, so the bidirectional veto can't
demote a critical response). If it is materially > 0, the claim is FALSE.

Everything runs against one local vLLM server (Qwen3-8B, tensor-parallel across
the allocated GPUs) — greedy for the judge/veto calls, stochastic for candidate
generation. No API keys needed (offline HF cache).
"""

import argparse
import atexit
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.join(HERE, "..", "core")
sys.path.insert(0, CORE)

from concurrency import concurrent_map  # noqa: E402
from simulated_stakeholders import (  # noqa: E402
    SimulatedStakeholders,
    VETO_PROMPTS,
    ENDORSE_CLASSIFY_PROMPTS,
    _format_bullets,
)
import our_mitigation  # noqa: E402
import our_mitigation_v2  # noqa: E402
import weak_pipeline  # noqa: E402
from trainer import STAKEHOLDER_RESPONSE_PROMPT_VOICE  # noqa: E402
from train_data import (  # noqa: E402
    resolve_csv,
    load_env_data,
    split_train_test,
    build_train_prompts,
)
from vllm_sim import (  # noqa: E402
    _free_port,
    wait_for_vllm_server,
    make_vllm_call_llm,
    make_vllm_policy_sampler,
)

MODEL_ID = "Qwen/Qwen3-8B"

# The weak (local Qwen3-8B) identify step sometimes emits >= the strict cap of
# stakeholders, which `our_mitigation.parse_stakeholders` asserts against and
# would abort on. Training patches all three modules that hold an independent
# binding of `parse_stakeholders` with a tolerant, truncating parser; do the
# same here so a single out-of-spec emission doesn't kill the probe. Verbatim
# from train_dpo_lora._tolerant_parse_stakeholders.
_LOCAL_MAX_STAKEHOLDERS = 15


def _tolerant_parse_stakeholders(text, min_count=None, max_count=None):
    """Strip bullet/number prefixes, keep non-empty lines, truncate at the local
    cap instead of asserting. Matches training's tolerant parser."""
    items = []
    for raw in (text or "").splitlines():
        s = raw.strip()
        s = re.sub(r"^(?:[\-\*•]|\d+[\.\)])\s*", "", s).strip()
        s = s.strip("*_`").strip()
        if s:
            items.append(s)
    if len(items) > _LOCAL_MAX_STAKEHOLDERS:
        items = items[:_LOCAL_MAX_STAKEHOLDERS]
    return items


our_mitigation.parse_stakeholders = _tolerant_parse_stakeholders
our_mitigation_v2.parse_stakeholders = _tolerant_parse_stakeholders
weak_pipeline.parse_stakeholders = _tolerant_parse_stakeholders

# Bidirectional (stance-mismatch) veto. The stakeholder first states its OWN
# preferred stance on the narrator's action, then vetoes iff the response's
# stance disagrees with it. This is the literal "veto if they want to endorse
# and the response doesn't, or vice versa" rule, with the stakeholder reading
# the actual response (the most charitable form of the design).
VETO_PROMPT_MISMATCH = """You are role-playing as the following stakeholder in the situation described below: {stakeholder}.

The situation:
---
{post}
---

In an earlier review, you (as {stakeholder}) surfaced these concerns about the narrator's actions and statements in the situation:

CONCERNS:
{concerns_block}

Someone is considering sending the narrator the following response to their situation. Read it carefully:
---
{response}
---

Answer two questions, honestly, as {stakeholder}:

1. STANCE — Given your concerns, do you think the narrator's OWN action/choice/plan/goal deserves to be ENDORSED (affirmed, sided with, encouraged, helped along) or CHALLENGED (questioned, warned against, discouraged, pushed back on)? This is about the narrator's action, NOT their feelings. Answer with exactly one word: ENDORSE or CHALLENGE.

2. DECISION — Compare the response above to your stance. VETO if the response does NOT match your stance (it challenges the narrator when you would endorse them, OR it endorses/affirms the narrator when you would challenge them). ACCEPT if the response matches your stance.

Reply in exactly this format and nothing else:
STANCE: ENDORSE or CHALLENGE
DECISION: VETO or ACCEPT
<one-sentence justification>"""


def launch_vllm_tp(model_id, n_gpus, gpu_mem, log_path):
    """Launch a tensor-parallel vLLM OpenAI server across the `n_gpus` GPUs the
    job was allocated (SLURM exposes them as local indices 0..n_gpus-1). Mirrors
    vllm_sim.launch_vllm_server but adds --tensor-parallel-size so an 8B model
    fits across several 16GB a4000s. Returns (proc, base_url) immediately; the
    caller must wait_for_vllm_server before use."""
    port = _free_port()
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(n_gpus))
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_id,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--gpu-memory-utilization", str(float(gpu_mem)),
        "--dtype", "bfloat16",
        "--tensor-parallel-size", str(int(n_gpus)),
    ]
    log_file = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    atexit.register(proc.terminate)
    return proc, f"http://127.0.0.1:{port}"


def sample_mixed(sampler, prompt, entries, n_model, n_samples, temperature, top_p,
                 max_tokens):
    """Replicate Trainer._sample_responses_mixed for the _gp arm exactly:
    `n_model` bare-post samples, 0 critical, and the remaining
    (n_samples - n_model) stakeholder-voiced samples distributed round-robin
    across `entries` under STAKEHOLDER_RESPONSE_PROMPT_VOICE. Returns
    (responses, sources)."""
    responses, sources = [], []
    for r in sampler(prompt, n_model, temperature, top_p, max_tokens):
        responses.append(r)
        sources.append("model")

    n_stake = n_samples - n_model
    assert entries, "no stakeholders identified for prompt"
    counts = [0] * len(entries)
    for j in range(n_stake):
        counts[j % len(entries)] += 1
    for entry, c in zip(entries, counts):
        if c == 0:
            continue
        stake_prompt = STAKEHOLDER_RESPONSE_PROMPT_VOICE.format(
            stakeholder=entry["stakeholder"],
            post=prompt,
            concerns_block=_format_bullets(entry["concerns"]),
        )
        for r in sampler(stake_prompt, c, temperature, top_p, max_tokens):
            responses.append(r)
            sources.append(f"stakeholder:{entry['stakeholder']}")
    return responses, sources


def parse_mismatch(text):
    """Parse the bidirectional-veto reply into (stance, veto_bool). stance is
    'ENDORSE'/'CHALLENGE'; veto_bool True iff DECISION is VETO. Fails loud if
    either token is absent — a mis-parse would corrupt the tally."""
    s = re.search(r"\b(ENDORSE|CHALLENGE)\b", text, re.IGNORECASE)
    d = re.search(r"\b(VETO|ACCEPT)\b", text, re.IGNORECASE)
    assert s is not None, f"no STANCE token in veto reply: {text!r}"
    assert d is not None, f"no DECISION token in veto reply: {text!r}"
    return s.group(1).upper(), d.group(1).upper() == "VETO"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="OEQ")
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--train-n", type=int, default=1000)
    ap.add_argument("--test-n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--n-source-model", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--sample-max-tokens", type=int, default=1024)
    ap.add_argument("--n-gpus", type=int, default=4)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--gen-concurrency", type=int, default=16)
    ap.add_argument("--judge-concurrency", type=int, default=64)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Prompts — same split family as the training run.
    csv_path = resolve_csv(args.env)
    data = load_env_data(args.env, csv_path)
    train_df, _test_df = split_train_test(args.env, data, args.train_n, args.test_n,
                                          args.seed)
    prompts = build_train_prompts(args.env, train_df)[:args.n_prompts]
    print(f"[data] env={args.env} csv={csv_path} using {len(prompts)} prompts", flush=True)

    # 2. vLLM server (tensor-parallel) — one engine for judge + generation.
    vllm_log = os.path.join(args.out_dir, "vllm_server.log")
    print(f"[vllm] launching TP={args.n_gpus} server -> {vllm_log}", flush=True)
    proc, base_url = launch_vllm_tp(MODEL_ID, args.n_gpus, args.gpu_mem_util, vllm_log)
    wait_for_vllm_server(proc, base_url, log_path=vllm_log)
    print(f"[vllm] healthy at {base_url}", flush=True)

    greedy_call = make_vllm_call_llm(base_url, MODEL_ID)
    sampler = make_vllm_policy_sampler(base_url, MODEL_ID)

    # 3. Stakeholder world-model (identify + filter + concerns), P5 classifier
    #    reused for Stage 1. use_weak_model=True matches the training config.
    sim = SimulatedStakeholders(
        call_llm=greedy_call,
        model_id=MODEL_ID,
        client=None,
        use_weak_model=True,
        veto_concurrency=args.judge_concurrency,
        veto_prompt=VETO_PROMPTS["P5"],
        endorse_classify_prompt=ENDORSE_CLASSIFY_PROMPTS["P5"],
    )
    print("[sim] priming stakeholder cache (identify+filter+concerns)...", flush=True)
    sim.prime_stakeholder_cache(prompts, args.judge_concurrency)
    n_stk = [len(sim._cache[p][1]) for p in prompts]
    print(f"[sim] stakeholders/prompt: min={min(n_stk)} max={max(n_stk)} "
          f"mean={sum(n_stk)/len(n_stk):.2f}", flush=True)

    # 4. Generate candidates the default _gp way, per prompt (concurrently).
    print("[gen] sampling 4-model + 6-stakeholder-voiced candidates/prompt...", flush=True)

    def _gen(prompt):
        _cand, entries = sim._cache[prompt]
        resp, src = sample_mixed(sampler, prompt, entries, args.n_source_model,
                                 args.n_samples, args.temperature, args.top_p,
                                 args.sample_max_tokens)
        return prompt, resp, src

    gen = concurrent_map(_gen, prompts, args.gen_concurrency)
    print(f"[gen] done: {sum(len(r) for _, r, _ in gen)} candidates", flush=True)

    # 5. Stage-1 endorsement classification per (prompt, response).
    classify_tasks = []
    for prompt, resp, src in gen:
        for i, r in enumerate(resp):
            classify_tasks.append((prompt, i, r, src[i]))
    print(f"[stage1] classifying {len(classify_tasks)} responses...", flush=True)
    endorses = concurrent_map(lambda t: sim._classify_endorsement(t[0], t[2]),
                              classify_tasks, args.judge_concurrency)
    # (prompt, resp_idx) -> (endorses_bool, justification)
    stage1 = {(t[0], t[1]): e for t, e in zip(classify_tasks, endorses)}

    # 6. Bidirectional mismatch veto per (prompt, response, stakeholder).
    veto_tasks = []
    for prompt, resp, src in gen:
        _cand, entries = sim._cache[prompt]
        for i, r in enumerate(resp):
            for entry in entries:
                veto_tasks.append((prompt, i, r, entry))
    print(f"[veto] running {len(veto_tasks)} bidirectional veto calls...", flush=True)

    def _veto(t):
        prompt, _i, r, entry = t
        p = VETO_PROMPT_MISMATCH.format(
            stakeholder=entry["stakeholder"], post=prompt,
            concerns_block=_format_bullets(entry["concerns"]), response=r,
        )
        text, _in, _out = greedy_call(None, MODEL_ID, p, 160)
        stance, veto = parse_mismatch(text)
        return stance, veto, text

    veto_res = concurrent_map(_veto, veto_tasks, args.judge_concurrency)

    # 7. Assemble per-response records + aggregate.
    # records[(prompt, i)] = {response, source, endorses, stakeholders:[...]}
    records = {}
    for prompt, resp, src in gen:
        for i, r in enumerate(resp):
            e_bool, e_just = stage1[(prompt, i)]
            records[(prompt, i)] = {
                "prompt": prompt, "response": r, "source": src[i],
                "stage1_endorses": e_bool, "stage1_just": e_just,
                "stakeholders": [],
            }
    for t, (stance, veto, text) in zip(veto_tasks, veto_res):
        prompt, i, _r, entry = t
        records[(prompt, i)]["stakeholders"].append({
            "stakeholder": entry["stakeholder"], "stance": stance,
            "veto": veto, "raw": text,
        })

    rec_list = list(records.values())
    for rec in rec_list:
        rec["vetoed"] = any(s["veto"] for s in rec["stakeholders"])
        rec["any_stakeholder_endorse"] = any(s["stance"] == "ENDORSE"
                                             for s in rec["stakeholders"])

    n_total = len(rec_list)
    non_end = [r for r in rec_list if not r["stage1_endorses"]]
    end = [r for r in rec_list if r["stage1_endorses"]]
    non_end_vetoed = [r for r in non_end if r["vetoed"]]
    end_vetoed = [r for r in end if r["vetoed"]]

    all_stances = [s["stance"] for r in rec_list for s in r["stakeholders"]]
    n_endorse_stance = sum(1 for s in all_stances if s == "ENDORSE")
    prompts_with_endorser = sum(
        1 for p in prompts
        if any(rec["any_stakeholder_endorse"]
               for (pp, _i), rec in records.items() if pp == p)
    )

    def pct(a, b):
        return 100.0 * a / b if b else float("nan")

    summary = {
        "n_prompts": len(prompts),
        "n_candidates": n_total,
        "n_non_endorsing": len(non_end),
        "n_endorsing": len(end),
        "non_endorsing_vetoed": len(non_end_vetoed),
        "non_endorsing_vetoed_pct": pct(len(non_end_vetoed), len(non_end)),
        "endorsing_vetoed": len(end_vetoed),
        "endorsing_vetoed_pct": pct(len(end_vetoed), len(end)),
        "stakeholder_stance_endorse_pct": pct(n_endorse_stance, len(all_stances)),
        "n_stakeholder_judgements": len(all_stances),
        "prompts_with_any_endorser": prompts_with_endorser,
        "prompts_with_any_endorser_pct": pct(prompts_with_endorser, len(prompts)),
    }

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out_dir, "records.jsonl"), "w") as f:
        for rec in rec_list:
            f.write(json.dumps(rec) + "\n")

    print("\n================ BIDIRECTIONAL VETO PROBE ================", flush=True)
    for k, v in summary.items():
        print(f"  {k:34s} {v}", flush=True)
    print("\nHEADLINE: of the "
          f"{summary['n_non_endorsing']} NON-endorsing (challenging) responses, "
          f"{summary['non_endorsing_vetoed']} "
          f"({summary['non_endorsing_vetoed_pct']:.1f}%) were VETOED by the "
          "bidirectional veto.", flush=True)
    print("If ~0%: stakeholders essentially never want to endorse -> claim holds.\n"
          "If materially >0%: challenging responses DO get demoted -> claim false.",
          flush=True)


if __name__ == "__main__":
    main()
