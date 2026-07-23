"""Score ASSISTANT_ROLE_VETO_PROMPT against the labelled validation set.

Runs the EXACT production classifier path — SimulatedStakeholders
._classify_assistant_role, same prompt, same 0/1 parse, same fail-open — over
eval/arole_validation_set.csv, on either judge backend:

  --backend gemini   the Gemini 3.1 Flash-Lite sim used by the *_arole_gemini
                     arms (temp 0, thinking_budget=0). Runs anywhere with
                     GEMINI_API_KEY; no GPU.
  --backend vllm     the local frozen Qwen3-8B sim used by the marlowe_*_arole
                     arms. Needs a GPU, so run it via
                     slurm_scripts/run_arole_validation.sbatch.

Reports, separately for positives and controls:
  recall on expected==1 (per sub-mode), false-positive rate on expected==0,
  unparseable-label rate, and the per-item table so disagreements can be read.

Assumption being tested: the prompt fires on responses that already-run judges
labelled as abandoning the assistant role, and does NOT fire on harsh-but-valid
assistant replies. The controls are load-bearing — an always-1 judge scores
perfect recall, so recall alone proves nothing.

Output: eval/arole_validation_scored_<backend>.csv (+ a printed summary).
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

from concurrency import concurrent_map
import simulated_stakeholders as S

HERE = os.path.dirname(__file__)
IN_CSV = os.path.join(HERE, "arole_validation_set.csv")


def build_sim(backend, model, concurrency, prompt_name="base"):
    """A SimulatedStakeholders carrying ONLY what _classify_assistant_role
    touches (veto backend + tallies), built via __new__ so no pipeline or model
    load happens. Keeps the classifier path identical to production."""
    sim = S.SimulatedStakeholders.__new__(S.SimulatedStakeholders)
    sim.assistant_role_veto = True
    sim.addressee_filter = False
    sim._arole_unparsed = 0
    sim._arole_total = 0
    sim.veto_max_tokens = 32
    sim.veto_model_id = model
    sim.veto_client = None
    sim.assistant_role_prompt = (
        S.ASSISTANT_ROLE_VETO_PROMPT_QWEN if prompt_name == "qwen"
        else S.ASSISTANT_ROLE_VETO_PROMPT)

    if backend == "gemini":
        from gemini_sim import make_gemini_call_llm
        assert os.environ.get("GEMINI_API_KEY"), "set GEMINI_API_KEY"
        sim.veto_call_llm = make_gemini_call_llm(model, thinking_budget=0)
    elif backend == "vllm":
        # Serve the frozen Qwen3-8B exactly as the training runs do, so the
        # judge path here is byte-identical to the marlowe_*_arole arms (greedy,
        # enable_thinking=False via the server's chat_template_kwargs).
        from vllm_sim import (launch_vllm_server, wait_for_vllm_server,
                              make_vllm_call_llm)
        proc, base_url = launch_vllm_server(model, device="cuda:0",
                                            log_path="/tmp/arole_vllm.log")
        wait_for_vllm_server(proc, base_url, log_path="/tmp/arole_vllm.log")
        sim.veto_call_llm = make_vllm_call_llm(base_url, model)
        sim._vllm_proc = proc  # kept so main() can tear the server down
    else:
        raise AssertionError(f"unknown backend {backend!r}")
    return sim


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", required=True, choices=("gemini", "vllm"))
    ap.add_argument("--model", default=None,
                    help="judge model id (default: the arm's model for the backend)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--prompt", default="base", choices=("base", "qwen"),
                    help="which assistant-role prompt to score: the base "
                         "prompt, or the Qwen-specific variant that restores "
                         "the sub-mode (b) epistemic test")
    ap.add_argument("--input", default=IN_CSV,
                    help="labelled set to score (default: the first set)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    model = args.model or ("gemini-3.1-flash-lite" if args.backend == "gemini"
                           else "Qwen/Qwen3-8B")
    tag = os.path.basename(args.input).replace(".csv", "")
    out_path = args.output or os.path.join(
        HERE, f"scored_{tag}_{args.backend}_{args.prompt}.csv")

    d = pd.read_csv(args.input)
    print(f"[arole-val] {len(d)} items  backend={args.backend}  "
          f"model={model}  prompt={args.prompt}  set={os.path.basename(args.input)}")
    sim = build_sim(args.backend, model, args.concurrency, args.prompt)

    items = list(zip(d["sentence"].astype(str), d["response"].astype(str)))
    try:
        results = concurrent_map(
            lambda t: sim._classify_assistant_role(t[0], t[1]),
            items, args.concurrency)
    finally:
        proc = getattr(sim, "_vllm_proc", None)
        if proc is not None:
            proc.terminate()

    d["predicted"] = [int(flag) for flag, _ok in results]
    d["parse_ok"] = [ok for _flag, ok in results]
    d["correct"] = d["predicted"] == d["expected"]
    d.to_csv(out_path, index=False)

    pos = d[d.expected == 1]
    ctrl = d[d.expected == 0]
    n_unparsed = (~d.parse_ok).sum()

    print(f"\n=== {args.backend} / {model} / prompt={args.prompt} ===")
    print(f"unparseable labels: {n_unparsed}/{len(d)} "
          f"(these FAIL OPEN -> counted as predicted 0)")
    print(f"\nRECALL on {len(pos)} known role-abandoning responses: "
          f"{pos.correct.mean():.3f}  ({pos.correct.sum()}/{len(pos)})")
    for sm, g in pos.groupby("submode"):
        print(f"   {sm:22s} {g.correct.sum():2d}/{len(g):2d}  {g.correct.mean():.3f}")
    print(f"\nFALSE-POSITIVE RATE on {len(ctrl)} known-good responses: "
          f"{1 - ctrl.correct.mean():.3f}  ({(~ctrl.correct).sum()}/{len(ctrl)})")

    fn = d[d.source.str.contains("FALSE NEGATIVE", na=False)]
    if len(fn):
        got = "CAUGHT" if fn.iloc[0].predicted == 1 else "MISSED"
        print(f"\nknown voice-gate false negative ({fn.iloc[0].id}): {got}")

    miss = pos[~pos.correct]
    if len(miss):
        print(f"\n--- {len(miss)} MISSED positives ---")
        for _, r in miss.iterrows():
            print(f"  [{r.submode}] {str(r.id)[:14]:14s} {str(r.response)[:110]}...")
    fp = ctrl[~ctrl.correct]
    if len(fp):
        print(f"\n--- {len(fp)} FALSE POSITIVES ---")
        for _, r in fp.iterrows():
            print(f"  {str(r.id)[:14]:14s} {str(r.response)[:110]}...")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
