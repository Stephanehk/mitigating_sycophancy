"""Sample a trained policy's free-form responses on the held-out AITA /
AITA-NTA-FLIP set. This is the GENERATION stage only — it does NOT assign a
verdict.

WHAT THIS DOES
--------------
The policy answers each held-out post in free-form prose (the bare post, the
exact surface training sampled against — no output-format nudge). The YTA/NTA
verdict is NOT self-emitted or parsed here: an external gpt-5-mini response
classifier reads the verdict from the prose downstream
(`classify_aita_verdicts.py`), which also computes the metrics. This replaced
the old self-emitted `AITA_LABEL_SUFFIX` + deterministic `parse_yta_nta`, whose
forced trailing label was found to decouple from the response prose for trained
policies.

  AITA            : one response per held-out post; carries the `is_asshole`
                    ground truth so the classifier stage can compute FNR / FPR.
                    FNR (a true-YTA post judged NTA) is the sycophancy signal.
  AITA-NTA-FLIP   : two responses per held-out PAIR — one for the original post,
                    one for the POV-flipped retelling. The classifier stage's
                    per-pair sycophancy_score = 1 iff BOTH are judged NTA.

The held-out ids come from `<run-dir>/test_ids.json` (written by the trainer's
split), or an explicit `--test-ids-file`. Output filenames are tagged by adapter
/ decode / seed so reruns and shards don't clobber each other. Each shard writes
`aita_responses_<tag>.csv` (raw responses, no verdict) + `responses_aita_<tag>.jsonl`.

PROMPT-ONLY BASELINES (on the base model, --base). At most one applies; each is a
generation-time steer on the bare post:
  --be-critical       append trainer.CRITICAL_SUFFIX ("\n\nBe critical.")
  --explicit          prepend trainer.EXPLICIT_PREFIX (Cheng et al. 2026)
  --sharma-honesty    append trainer.SHARMA_HONESTY_SUFFIX (Sharma et al.)
  --gen-suffix/--gen-tag  append an arbitrary "\n\n<text>" (e.g. dontbesyco)
  --perspective-shift third-person rewrite then answer (Cheng et al. 2025)
The suffix/prefix constants are imported from the trainer so they can't drift
from training.

ASSUMPTIONS (asserted)
----------------------
- Every held-out id resolves in the env CSV.
- AITA rows carry a 0/1 `is_asshole`; AITA-NTA-FLIP rows carry
  `flipped_post_text`.

Run (generation), then the classifier stage:
    python training_mitigations/eval/eval_aita_label_heldout.py \
        --run-dir runs/dpo_lora_AITA_... --adapter runs/.../adapter_iter_3 \
        --temperature 1.0 --seed 0
    python training_mitigations/eval/classify_aita_verdicts.py \
        --run-dir runs/dpo_lora_AITA_... --env AITA --tag final_..._seed0 --num-shards N
"""

import argparse
import json
import os
import sys

import pandas as pd
import torch

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_HERE)
# mitigations/ for the harness loaders + metrics; training_mitigations/ and its
# core/ for llama_local + train_data; eval/ (this dir) for the sibling helpers.
sys.path.insert(0, os.path.join(_REPO_ROOT, "mitigations"))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "core"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from explore_mitigations import (
    AITA_DATA_CSV, AITA_NTA_FLIP_DATA_CSV,
    load_aita_clean, load_aita_nta_flip,
)
from llama_local import DEFAULT_HF_MODEL_ID, load_llama
from perspective_shift import perspective_shift_post
# Reuse the sibling eval's generation + adapter + shard helpers verbatim so the
# decoding path is identical to the OEQ/PAS eval (single source of truth).
from eval_endorsement_heldout import generate_one, attach_adapter, shard_slice
from train_data import final_ids_path


# --- env / data ------------------------------------------------------------
def resolve_csv(env):
    """Abs path to the env's source CSV, anchored to the repo root (mirrors the
    trainer's split so held-out ids re-resolve exactly)."""
    rel = {"AITA": AITA_DATA_CSV, "AITA-NTA-FLIP": AITA_NTA_FLIP_DATA_CSV}[env]
    path = rel if os.path.isabs(rel) else os.path.join(_REPO_ROOT, rel)
    assert os.path.exists(path), f"missing {env} csv: {path}"
    return path


def load_env_data(env):
    """Load the env DataFrame via the harness loader. AITA carries `is_asshole`;
    AITA-NTA-FLIP carries `flipped_post_text`."""
    csv_path = resolve_csv(env)
    if env == "AITA":
        return load_aita_clean(csv_path)
    if env == "AITA-NTA-FLIP":
        return load_aita_nta_flip(csv_path)
    raise AssertionError(f"unknown env: {env!r}")


def infer_env_from_run_dir(run_dir):
    """Pick AITA / AITA-NTA-FLIP from the run-dir name (trainer names dirs
    dpo_lora_<env>_...). Checks the FLIP token first since 'aita' is a substring
    of 'aita-nta-flip'. Fails loud if neither appears."""
    name = os.path.basename(os.path.normpath(run_dir)).lower()
    if "aita-nta-flip" in name or "aita_nta_flip" in name:
        return "AITA-NTA-FLIP"
    if "aita" in name:
        return "AITA"
    raise AssertionError(
        f"cannot infer env from run dir {name!r}; pass --env explicitly"
    )


def held_out_rows(env, run_dir, test_ids_path=None):
    """Return the held-out rows as a DataFrame indexed by the saved id order.

    Columns: `id`, `post_text`, plus `is_asshole` (AITA) or `flipped_post_text`
    (AITA-NTA-FLIP). Asserts every saved id resolves in the env CSV."""
    if test_ids_path is None:
        test_ids_path = os.path.join(run_dir, "test_ids.json")
    assert os.path.exists(test_ids_path), f"missing {test_ids_path}"
    with open(test_ids_path) as f:
        ids = [str(i) for i in json.load(f)]
    data = load_env_data(env).copy()
    data["id"] = data["id"].astype(str)
    by_id = data.set_index("id")
    missing = [i for i in ids if i not in by_id.index]
    assert not missing, (
        f"{len(missing)} test ids not found in {env} CSV (e.g. {missing[:3]})"
    )
    return by_id.loc[ids].reset_index()


# --- main ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-dir", required=True,
                        help="run dir holding test_ids.json (and, by default, "
                             "the final adapter to evaluate)")
    parser.add_argument("--env", choices=("AITA", "AITA-NTA-FLIP"), default=None,
                        help="dataset env; inferred from --run-dir name if omitted")
    parser.add_argument("--adapter", default=None,
                        help="adapter dir to evaluate (default: --run-dir, the "
                             "final saved adapter). Point at a per-iteration "
                             "adapter dir to evaluate that iteration.")
    parser.add_argument("--base", action="store_true",
                        help="evaluate the untrained base model (no adapter); "
                             "the iteration-0 baseline")
    parser.add_argument("--be-critical", action="store_true",
                        help="prompt-only baseline: append the trainer's "
                             "CRITICAL_SUFFIX ('\\n\\nBe critical.') to each "
                             "held-out post at GENERATION time. Tags outputs "
                             "'<adapter_tag>-becritical'.")
    parser.add_argument("--perspective-shift", action="store_true",
                        help="ELEPHANT (Cheng et al. 2025, App G.2) perspective-"
                             "shift baseline: first REWRITE each held-out post "
                             "from first- to third-person (greedy, via the "
                             "rewriter set by --rewrite-backend), then generate "
                             "the verdict on the rewrite. Tags '<adapter_tag>-"
                             "perspshift'. Mutually exclusive with --be-critical.")
    parser.add_argument("--assess-narrative", action="store_true",
                        help="with --perspective-shift, also append the paper's "
                             "OEQ addendum 'Assess this narrative.' after the "
                             "third-person rewrite (the paper adds it for long "
                             "narratives where the model keeps saying 'you'). "
                             "Tags '<adapter_tag>-perspshiftassess'.")
    parser.add_argument("--rewrite-max-tokens", type=int, default=1536,
                        help="max new tokens for the third-person rewrite call "
                             "(greedy). Long AITA posts need room or the rewrite "
                             "truncates (default: 1536).")
    parser.add_argument("--explicit", action="store_true",
                        help="prompt-only baseline (Cheng et al. 2026, EXPLICIT "
                             "accommodation): PREPEND the trainer's EXPLICIT_PREFIX "
                             "(misconception-correction instruction) to each post "
                             "at generation time. Tags '<adapter_tag>-explicit'.")
    parser.add_argument("--sharma-honesty", dest="sharma_honesty",
                        action="store_true",
                        help="prompt-only baseline (Sharma et al.): APPEND the "
                             "trainer's SHARMA_HONESTY_SUFFIX, BEFORE the YTA/NTA "
                             "label nudge. Tags '<adapter_tag>-sharma'.")
    parser.add_argument("--gen-suffix", default=None,
                        help="generic prompt-only steer: append '\\n\\n<text>' to "
                             "each post (BEFORE the label nudge), same mechanics as "
                             "--be-critical with an arbitrary instruction (e.g. "
                             "\"Don't be sycophantic.\"). Requires --gen-tag.")
    parser.add_argument("--gen-tag", default=None,
                        help="adapter-tag stem for --gen-suffix (e.g. 'dontbesyco' "
                             "-> tags '<adapter_tag>-dontbesyco').")
    parser.add_argument("--teacher-out", action="store_true",
                        help="ALSO write a teacher CSV 'responses_<tag>.csv' with "
                             "columns [id, sentence, response] for SFT distillation "
                             "(train_sft_lora.py). `sentence` is the BARE student "
                             "surface (the post, NO steer); "
                             "`response` is this run's generated target. For "
                             "AITA-NTA-FLIP, emits TWO rows per pair (original + "
                             "flipped). Pair with --be-critical on the TRAIN ids to "
                             "build a 'Be critical.' SFT teacher set.")
    parser.add_argument("--model-id", default=DEFAULT_HF_MODEL_ID,
                        help=f"base HF model id (default: {DEFAULT_HF_MODEL_ID})")
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HOME"),
                        help="HF cache dir (default: $HF_HOME)")
    parser.add_argument("--device", default="cuda:0", help="torch device "
                        "(single-GPU load; ignored when --device-map is set)")
    parser.add_argument("--device-map", default=None,
                        help="shard the base model across visible GPUs via "
                             "accelerate (e.g. 'auto') instead of a single "
                             "--device.")
    parser.add_argument("--tag-extra", default="",
                        help="extra token inserted into output filenames so a "
                             "run on a different held-out id set doesn't "
                             "overwrite the original eval outputs")
    parser.add_argument("--test-ids-file", default=None,
                        help="explicit JSON list of held-out ids to evaluate, "
                             "overriding <run-dir>/test_ids.json")
    parser.add_argument("--use-final-data", action="store_true",
                        help="evaluate on the frozen final_training_eval_data/ "
                             "split for the env (see --final-split) instead of "
                             "<run-dir>/test_ids.json. Mutually exclusive with "
                             "--test-ids-file.")
    parser.add_argument("--final-split", choices=("test", "train"), default="test",
                        help="which final split to load under --use-final-data: "
                             "'test' (default; the held-out eval) or 'train' "
                             "(used to generate SFT be-critical teacher targets "
                             "over the final train prompts).")
    parser.add_argument("--n", type=int, default=None,
                        help="limit to the first N held-out rows (default: all)")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="split the held-out rows into this many disjoint "
                             "contiguous shards and only evaluate --shard-id")
    parser.add_argument("--shard-id", type=int, default=0,
                        help="which shard (0..num_shards-1) this job evaluates")
    # sampling knobs (match training defaults so responses reflect the
    # distribution the policy was shaped under).
    parser.add_argument("--sample-max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--greedy", action="store_true",
                        help="deterministic decoding instead of sampling")
    parser.add_argument("--seed", type=int, default=0,
                        help="torch manual seed for the single-sample decoding")
    args = parser.parse_args()

    assert not (args.base and args.adapter), "--base and --adapter are mutually exclusive"
    env = args.env or infer_env_from_run_dir(args.run_dir)
    if args.use_final_data:
        assert args.test_ids_file is None, \
            "--use-final-data and --test-ids-file are mutually exclusive"
        args.test_ids_file = final_ids_path(env, args.final_split)
        print(f"[data] FINAL {args.final_split} split for {env}: "
              f"{args.test_ids_file}")

    # Tag for output filenames.
    if args.base:
        adapter_tag = "base"
    else:
        adapter_dir = args.adapter or args.run_dir
        adapter_tag = ("final" if os.path.normpath(adapter_dir) ==
                       os.path.normpath(args.run_dir)
                       else os.path.basename(os.path.normpath(adapter_dir)))
    # At most one generation-time steer. Each baseline either PREPENDS (explicit)
    # or APPENDS (be-critical / sharma / gen-suffix) an instruction, or rewrites
    # the post (perspective-shift); they don't compose.
    n_steers = sum(bool(x) for x in (args.be_critical, args.perspective_shift,
                                     args.explicit, args.sharma_honesty,
                                     args.gen_suffix))
    assert n_steers <= 1, (
        "--be-critical / --perspective-shift / --explicit / --sharma-honesty / "
        "--gen-suffix are mutually exclusive baselines"
    )
    assert not (args.assess_narrative and not args.perspective_shift), (
        "--assess-narrative only applies with --perspective-shift"
    )
    # Prompt-only steers: appended (gen_steer) or prepended (gen_prefix) to the
    # post at generation time. Suffix/prefix constants are pulled from the trainer
    # (single source of truth) so they can't drift from training.
    gen_steer = ""
    gen_prefix = ""
    if args.be_critical:
        from trainer import CRITICAL_SUFFIX
        gen_steer = CRITICAL_SUFFIX
        adapter_tag = f"{adapter_tag}-becritical"
    elif args.perspective_shift:
        adapter_tag = f"{adapter_tag}-perspshift" + ("assess" if args.assess_narrative else "")
    elif args.explicit:
        from trainer import EXPLICIT_PREFIX
        gen_prefix = EXPLICIT_PREFIX
        adapter_tag = f"{adapter_tag}-explicit"
    elif args.sharma_honesty:
        from trainer import SHARMA_HONESTY_SUFFIX
        gen_steer = SHARMA_HONESTY_SUFFIX
        adapter_tag = f"{adapter_tag}-sharma"
    elif args.gen_suffix:
        assert args.gen_tag, "--gen-suffix requires --gen-tag"
        gen_steer = "\n\n" + args.gen_suffix
        adapter_tag = f"{adapter_tag}-{args.gen_tag}"
    decode_tag = "greedy" if args.greedy else f"temp{args.temperature}"
    extra = f"_{args.tag_extra}" if args.tag_extra else ""
    tag = f"{adapter_tag}{extra}_{decode_tag}_seed{args.seed}"

    out_dir = os.path.join(args.run_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Held-out rows. Shard AFTER --n so (--n, --num-shards) compose on the
    # same fixed test_ids order, keeping shards disjoint and tiling exactly.
    rows = held_out_rows(env, args.run_dir, args.test_ids_file)
    if args.n is not None:
        rows = rows.iloc[:args.n].reset_index(drop=True)
    n_total = len(rows)
    if args.num_shards > 1:
        s, e = shard_slice(n_total, args.num_shards, args.shard_id)
        rows = rows.iloc[s:e].reset_index(drop=True)
        tag = f"{tag}_shard{args.shard_id}of{args.num_shards}"
        print(f"[eval] shard {args.shard_id}/{args.num_shards}: rows [{s}:{e}) "
              f"of {n_total}")
    print(f"[eval] env={env}  rows={len(rows)}  adapter={adapter_tag}  decode={decode_tag}")

    # 2. Load policy.
    placement = f"device_map={args.device_map}" if args.device_map else args.device
    print(f"[load] base {args.model_id} -> {placement}")
    model, tokenizer = load_llama(model_id=args.model_id, device=args.device,
                                  cache_dir=args.cache_dir, device_map=args.device_map)
    if args.base:
        print("[load] no adapter (base-model / iter-0 baseline)")
    else:
        print(f"[load] attaching adapter {adapter_dir}")
        model = attach_adapter(model, adapter_dir)

    def _rewrite(post):
        """Greedy first->third-person rewrite of `post` (perspective shift).
        Greedy so the rewrite is deterministic and not a sampling-noise source,
        independent of how the answer itself is decoded."""
        return perspective_shift_post(
            lambda p: generate_one(model, tokenizer, p, args.rewrite_max_tokens,
                                   args.temperature, args.top_p, greedy=True),
            post, assess_narrative=args.assess_narrative)

    def _respond(post):
        """Return (response, rewritten_or_None). With --perspective-shift the
        model answers the THIRD-person rewrite; otherwise the bare post. The
        response is free-form prose with NO output-format nudge — the YTA/NTA
        verdict is read from the prose downstream by classify_aita_verdicts.py,
        not self-emitted here. The optional be-critical steer is still appended."""
        rewritten = _rewrite(post) if args.perspective_shift else None
        body = rewritten if rewritten is not None else post
        resp = generate_one(model, tokenizer,
                            gen_prefix + body + gen_steer,
                            args.sample_max_tokens, args.temperature,
                            args.top_p, args.greedy)
        return resp, rewritten

    # 3. Generate. The policy answers in free-form prose; NO verdict is parsed
    # here. classify_aita_verdicts.py reads these responses and produces the
    # YTA/NTA/UNCLEAR verdict from the prose with the gpt-5-mini classifier.
    torch.manual_seed(args.seed)
    responses_path = os.path.join(out_dir, f"responses_aita_{tag}.jsonl")
    records = []
    with open(responses_path, "w") as rf:
        for k, row in rows.iterrows():
            if env == "AITA":
                resp, rewritten = _respond(row["post_text"])
                rec = {"id": row["id"], "post_text": row["post_text"],
                       "rewritten_post": rewritten,
                       "response": resp,
                       "is_asshole": int(row["is_asshole"])}
            else:  # AITA-NTA-FLIP
                resp_o, rw_o = _respond(row["post_text"])
                resp_f, rw_f = _respond(row["flipped_post_text"])
                rec = {"id": row["id"],
                       "post_text": row["post_text"],
                       "flipped_post_text": row["flipped_post_text"],
                       "rewritten_original": rw_o, "rewritten_flipped": rw_f,
                       "response_original": resp_o, "response_flipped": resp_f}
            records.append(rec)
            rf.write(json.dumps(rec) + "\n")
            rf.flush()
            if (k + 1) % 50 == 0:
                print(f"[gen] {k + 1}/{len(rows)} rows done")
    df = pd.DataFrame(records)
    responses_csv_path = os.path.join(out_dir, f"aita_responses_{tag}.csv")
    df.to_csv(responses_csv_path, index=False)
    print(f"[gen] wrote {responses_path}")
    print(f"[gen] wrote {responses_csv_path}")

    # Optional SFT teacher CSV: [id, sentence, response] with `sentence` the BARE
    # student surface (the post, the same prompt the SFT student and eval see --
    # NOT the steer-augmented gen prompt). FLIP -> 2 rows/pair.
    # train_sft_lora.py's default glob is responses_*becritical*shard*.csv.
    if args.teacher_out:
        t_rows = []
        for rec in records:
            if env == "AITA":
                t_rows.append({"id": rec["id"],
                               "sentence": rec["post_text"],
                               "response": rec["response"]})
            else:  # AITA-NTA-FLIP: original + flipped are two independent examples
                t_rows.append({"id": f"{rec['id']}_orig",
                               "sentence": rec["post_text"],
                               "response": rec["response_original"]})
                t_rows.append({"id": f"{rec['id']}_flip",
                               "sentence": rec["flipped_post_text"],
                               "response": rec["response_flipped"]})
        teacher_path = os.path.join(out_dir, f"responses_{tag}.csv")
        pd.DataFrame(t_rows)[["id", "sentence", "response"]].to_csv(
            teacher_path, index=False)
        print(f"[teacher] wrote {len(t_rows)} (sentence,response) pairs -> {teacher_path}")

    # 4. Metrics are NOT computed here. This job only samples responses; the
    # YTA/NTA/UNCLEAR verdict and the FNR/FPR (AITA) or both-NTA sycophancy
    # (FLIP) metrics are produced by classify_aita_verdicts.py, which reads the
    # per-shard aita_responses_*.csv, classifies the prose, and writes the
    # aita_scored_* + summary the aggregator/tables consume.
    summary = {
        "env": env,
        "run_dir": os.path.abspath(args.run_dir),
        "adapter_tag": adapter_tag,
        "adapter": None if args.base else os.path.abspath(adapter_dir),
        "be_critical": args.be_critical,
        "gen_steer": gen_steer,
        "perspective_shift": args.perspective_shift,
        "assess_narrative": args.assess_narrative,
        "decode": decode_tag,
        "seed": args.seed,
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "n_rows": len(df),
        "metrics": None,  # filled by classify_aita_verdicts.py
    }
    summary_path = os.path.join(out_dir, f"summary_aita_{tag}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"AITA gen  [{env} / {adapter_tag} / {decode_tag}]  N={len(df)}")
    print(f"  responses -> {responses_csv_path}")
    print(f"  summary   -> {summary_path}")
    print(f"  NEXT: classify_aita_verdicts.py --run-dir {args.run_dir} "
          f"--env {env} --tag {os.path.basename(tag)} to produce verdicts+metrics")
    print("=" * 70)


if __name__ == "__main__":
    main()
