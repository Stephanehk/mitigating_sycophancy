"""Generate baseline PAS/OEQ responses from a model for the sycoharm experiments.

Takes the canonical 500-prompt set per env (the `sentence` column of the
existing sycoharm input files inputs/{env}_gpt4o.csv -- PAS shares all 500
sentences with the deepseek baseline, OEQ shares 499, so the gpt4o file's
prompts are the common comparison set) and elicits one baseline response per
prompt from `--model`, exactly as the mitigation harness does for the
`baseline` condition: the bare post text sent as a single user turn with no
system prompt and no steering (mitigations/core/prompts.py PROMPT_TEMPLATE_OEQ
= "{post}"). Routing/clients are reused from mitigations/core so Claude goes
through the Anthropic SDK and everything else through OpenRouter.

Output: sycoharm_runs/inputs/{env}_{tag}.csv with columns (sentence, response),
the exact contract judge_sycophancy.py consumes. The judge is NOT run here --
run it separately on the produced file.

Resumable: if the output file already exists, rows whose response is present
and non-empty and not an [ERROR] marker are kept; only missing/failed rows are
(re)generated. A row whose generation raises after retries is written with a
"[ERROR] ..." response rather than crashing the whole run, so a run always
completes and failures are visible and re-runnable (matching the per-row error
convention in evaluate_action_endorsement_stephane.py).

Assumptions (asserted): the env input file exists and has a `sentence` column;
ANTHROPIC_API_KEY / OPENROUTER_API_KEY set as the routing family requires.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import tenacity
from tqdm import tqdm

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "mitigations"))

from core.models import make_client, model_pricing  # noqa: E402
from core.llm import call_llm  # noqa: E402

INPUTS = os.path.join(_HERE, "sycoharm_runs", "inputs")

# env -> canonical prompt-set file (its `sentence` column is the 500 prompts)
ENV_PROMPT_FILE = {
    "pas": os.path.join(INPUTS, "pas_gpt4o.csv"),
    "oeq": os.path.join(INPUTS, "oeq_gpt4o.csv"),
    "pas2": os.path.join(INPUTS, "pas2_gpt4o.csv"),
    "oeq2": os.path.join(INPUTS, "oeq2_gpt4o.csv"),
}


def _is_filled(resp):
    """True if `resp` is a usable generated response (present, non-empty, not
    an error marker) -- used to decide which rows to keep when resuming."""
    if resp is None or (isinstance(resp, float) and pd.isna(resp)):
        return False
    s = str(resp).strip()
    return bool(s) and not s.startswith("[ERROR]")


def generate_one(model, sentence):
    """Elicit one baseline response: bare post text, single user turn, no
    system prompt. Retries transient failures (tenacity, 4 attempts); on final
    failure returns an "[ERROR] ..." marker instead of raising so one bad row
    doesn't drop the other 499. Returns (response_text)."""

    @tenacity.retry(
        wait=tenacity.wait_random_exponential(multiplier=2, max=30),
        stop=tenacity.stop_after_attempt(4),
        reraise=True,
    )
    def _call():
        client = make_client(model)
        text, _in, _out = call_llm(client, model, sentence,
                                   max_tokens=2048, system=None)
        assert text.strip(), "empty generation"
        return text

    try:
        return _call()
    except Exception as e:  # noqa: BLE001 -- record, don't drop the whole run
        return f"[ERROR] {type(e).__name__}: {e}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True, choices=sorted(ENV_PROMPT_FILE))
    parser.add_argument("--model", required=True,
                        help="model id, e.g. openai/gpt-5.4 or claude-sonnet-4-6")
    parser.add_argument("--tag", required=True,
                        help="output-name tag -> inputs/{env}_{tag}.csv")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--checkpoint", type=int, default=50)
    args = parser.parse_args()

    src = ENV_PROMPT_FILE[args.env]
    assert os.path.exists(src), f"missing prompt file: {src}"
    prompts = pd.read_csv(src)["sentence"].reset_index(drop=True)
    n = len(prompts)
    assert n == 500, f"expected 500 prompts, got {n}"

    out_path = os.path.join(INPUTS, f"{args.env}_{args.tag}.csv")
    # Resume: start from any existing output, keep filled rows, regenerate rest.
    responses = [None] * n
    if os.path.exists(out_path):
        prev = pd.read_csv(out_path)
        assert (prev["sentence"].reset_index(drop=True) == prompts).all(), (
            f"{out_path} prompts differ from {src}; refusing to resume")
        responses = prev["response"].tolist()

    todo = [i for i in range(n) if not _is_filled(responses[i])]
    print(f"[{args.env}/{args.tag}] model={args.model} | {len(todo)}/{n} rows to generate "
          f"-> {out_path}")
    if not todo:
        print("  nothing to do (all rows filled)")
        return

    def save():
        pd.DataFrame({"sentence": prompts.values, "response": responses}).to_csv(
            out_path, index=False)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(generate_one, args.model, prompts[i]): i for i in todo}
        for k, fut in enumerate(tqdm(as_completed(futs), total=len(futs),
                                     desc=f"{args.env}/{args.tag}"), 1):
            responses[futs[fut]] = fut.result()
            if k % args.checkpoint == 0:
                save()
    save()

    n_err = sum(1 for r in responses if not _is_filled(r))
    pin, pout = model_pricing(args.model)
    print(f"[{args.env}/{args.tag}] done in {time.time()-t0:.0f}s; "
          f"{n_err}/{n} rows still failed (re-run to retry). wrote {out_path}")


if __name__ == "__main__":
    main()
