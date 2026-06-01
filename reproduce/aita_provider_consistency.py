#!/usr/bin/env python3
"""Do Azure and OpenRouter give the same AITA verdicts for the SAME model?

For each of three models that are deployed on BOTH Azure (as a chat
deployment on the `stephhk-syceval-resource` resource) and OpenRouter, we run
the identical balanced AITA sample through both providers, parse each reply to
a YTA/NTA verdict, and report per-(model, provider) FPR and FNR against the
human `is_asshole` ground truth. The point is to isolate provider/serving
discrepancies for the SAME model from real model differences: same prompts,
same temperature, same parsing — only the serving backend changes.

Models (Azure deployment id  <->  OpenRouter model id):
  DeepSeek-V3-0324                 DeepSeek-V3-0324                deepseek/deepseek-chat-v3-0324
  Llama-4-Scout-17B-16E-Instruct   Llama-4-Scout-17B-16E-Instruct meta-llama/llama-4-scout
  gpt-4o-2024-11-20                 gpt-4o                         openai/gpt-4o-2024-11-20

Metric definitions are reused verbatim from the harness (`compute_fnr_fpr`):
  FNR = NTA-predictions / YTA-truth posts  (model letting true-YTA off the
        hook; higher = more sycophantic).
  FPR = YTA-predictions / NTA-truth posts  (model calling true-NTA "asshole";
        higher = blanket harshness).
Both arms use only rows that parsed to YTA/NTA; parser/API failures are dropped
and reported as `n_dropped_other` so a degraded provider never silently shrinks
a denominator.

The AITA prompt (`PROMPT_TEMPLATE_BINARY_BASE`), the data loader
(`load_aita_clean`), the balanced sampler (`sample_balanced`), the verdict
parser (`parse_yta_nta`), and the metric (`compute_fnr_fpr`) are all imported
from the mitigations harness so this script stays consistent with the main
pipeline rather than re-deriving any of them.

Both providers are driven through the OpenAI SDK chat-completions API: Azure
via `AzureOpenAI` (api_version 2024-02-15-preview, creds
AZURE_OPENAI_ENDPOINT/AZURE_OPENAI_API_KEY), OpenRouter via
`OpenAI(base_url=...openrouter..., api_key=OPENROUTER_API_KEY)`. `max_retries`
is raised above the SDK default so 429s back off internally instead of
surfacing as dropped rows (the Azure `gpt-4o` deployment shares quota with
other jobs on the resource, so 429s are expected and absorbed).

HOW TO RUN (credentials required; not run automatically):
    export AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
    export AZURE_OPENAI_API_KEY=...
    export OPENROUTER_API_KEY=...
    python reproduce/aita_provider_consistency.py --n 200 --temp 1

Requires: openai (with Azure support), pandas, tqdm, and the mitigations
harness importable (this script adds it to sys.path).
"""

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

# reproduce/ lives one level under the repo root; reuse the harness helpers so
# the prompt, loader, sampler, parser, and metric match the main pipeline.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "mitigations"))

from explore_mitigations import (
    PROMPT_TEMPLATE_BINARY_BASE,
    load_aita_clean,
    sample_balanced,
    parse_yta_nta,
    compute_fnr_fpr,
    AITA_DATA_CSV,
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# model key -> (Azure deployment id, OpenRouter model id). The keys are the
# user-facing model names; the values are how each provider addresses them.
MODELS = {
    "DeepSeek-V3-0324": (
        "DeepSeek-V3-0324", "deepseek/deepseek-chat-v3-0324"),
    "Llama-4-Scout-17B-16E-Instruct": (
        "Llama-4-Scout-17B-16E-Instruct", "meta-llama/llama-4-scout"),
    "gpt-4o-2024-11-20": (
        "gpt-4o", "openai/gpt-4o-2024-11-20"),
}


def build_client(provider):
    """Return an OpenAI-SDK chat client for `provider` ('azure' | 'openrouter').

    Both are OpenAI-compatible so the call site is identical; only the
    constructor differs. `max_retries` is raised so 429s back off inside the
    SDK rather than surfacing as dropped rows.
    """
    if provider == "openrouter":
        from openai import OpenAI
        assert os.environ.get("OPENROUTER_API_KEY"), "set OPENROUTER_API_KEY"
        return OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=os.environ["OPENROUTER_API_KEY"],
            max_retries=12,
        )
    assert provider == "azure", f"unknown provider {provider!r}"
    from openai import AzureOpenAI
    assert os.environ.get("AZURE_OPENAI_ENDPOINT"), "set AZURE_OPENAI_ENDPOINT"
    assert os.environ.get("AZURE_OPENAI_API_KEY"), "set AZURE_OPENAI_API_KEY"
    return AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version="2024-02-15-preview",
        max_retries=12,
    )


def classify_one(client, model_id, post_text, temp, max_tokens):
    """Run the binary AITA prompt on one post. Returns (raw, pred, served).

    `pred` is 'YTA'/'NTA'/'OTHER' via the harness `parse_yta_nta`. The per-row
    API failure is caught at this external-API boundary and surfaced as an
    'ERROR: ...' raw string with pred 'OTHER', so one content-filter trip or a
    hard failure drops that single row (via `compute_fnr_fpr`'s OTHER drop)
    instead of aborting the whole run.
    """
    prompt = PROMPT_TEMPLATE_BINARY_BASE.format(post=post_text)
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temp,
        )
        raw = (resp.choices[0].message.content or "").strip()
        served = getattr(resp, "model", None)
        return raw, parse_yta_nta(raw), served
    except Exception as e:  # external API boundary; see docstring
        return f"ERROR: {e}", "OTHER", None


def run_provider_model(client, provider, model_key, sample, temp, max_tokens,
                       workers, max_rps, out_path):
    """Classify every post in `sample` through one (provider, model) combo.

    Threaded with a request-rate throttle (semaphore + sleep at `max_rps`) to
    respect the shared Azure quota; checkpoints to `out_path` so a long, 429-y
    run is interruptible. Returns a DataFrame aligned to `sample.index` with
    columns: id, is_asshole, raw, pred, served_model.
    """
    azure_id, or_id = MODELS[model_key]
    model_id = azure_id if provider == "azure" else or_id
    sem = threading.Semaphore(max_rps)

    def _task(idx, post_text):
        with sem:
            time.sleep(1.0 / max_rps)
            raw, pred, served = classify_one(
                client, model_id, post_text, temp, max_tokens)
        return idx, raw, pred, served

    out = pd.DataFrame(
        index=sample.index,
        columns=["id", "is_asshole", "raw", "pred", "served_model"])
    out["id"] = sample["id"].values
    out["is_asshole"] = sample["is_asshole"].values

    # Resume: if a checkpoint for this combo exists, adopt its completed rows
    # and only re-run the ones with no verdict yet (pred is NaN). This makes
    # the run restartable from disk — re-invoking the same command picks up
    # exactly where an interrupted run stopped, losing at most the rows since
    # the last 50-row checkpoint. Asserts the checkpoint lines up with the
    # current sample so a stale/mismatched file fails loud instead of
    # silently scrambling verdicts.
    if os.path.exists(out_path):
        prev = pd.read_csv(out_path)
        assert len(prev) == len(out), (
            f"checkpoint {out_path} has {len(prev)} rows, expected {len(out)}; "
            f"delete it to restart this combo from scratch")
        assert (prev["id"].astype(str).values == out["id"].astype(str).values).all(), (
            f"checkpoint {out_path} id/order mismatch with current sample; "
            f"delete it to restart this combo from scratch")
        for col in ("raw", "pred", "served_model"):
            out[col] = prev[col].values

    todo = [(idx, row) for idx, row in sample.iterrows()
            if pd.isna(out.at[idx, "pred"])]
    if not todo:
        print(f"  [resume] {model_key}[{provider}] already complete "
              f"({len(out)}/{len(out)}); skipping")
        return out
    if len(todo) < len(out):
        print(f"  [resume] {model_key}[{provider}] {len(out)-len(todo)}/{len(out)} "
              f"already done; running remaining {len(todo)}")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(_task, idx, str(row["post_text"]))
            for idx, row in todo
        ]
        for count, fut in enumerate(
                tqdm(as_completed(futures), total=len(futures),
                     desc=f"{model_key[:18]}[{provider}]"), 1):
            idx, raw, pred, served = fut.result()
            out.at[idx, "raw"] = raw
            out.at[idx, "pred"] = pred
            out.at[idx, "served_model"] = served if served is not None else ""
            if count % 50 == 0:
                out.to_csv(out_path, index=False)
    out.to_csv(out_path, index=False)
    return out


def wald_ci(p, n):
    """95% Wald (normal-approx) half-width for a proportion `p` over `n` trials.

    Returns nan when n == 0. Used only for display alongside the point rate;
    matches the simple normal-approx CI the harness uses elsewhere.
    """
    if n == 0:
        return float("nan")
    return 1.96 * np.sqrt(p * (1.0 - p) / n)


def report_model(model_key, per_provider):
    """Print the FPR/FNR table for one model across providers, plus the
    Azure-vs-OpenRouter prediction agreement on the identical posts.

    `per_provider` maps provider -> (results_df, metrics_dict). Returns a list
    of flat summary rows (one per provider) for the combined metrics CSV.
    """
    print("\n" + "=" * 78)
    print(f"MODEL: {model_key}")
    print("=" * 78)
    print(f"{'provider':12s} {'n_yta':>6s} {'n_nta':>6s} {'drop':>5s} "
          f"{'FNR':>7s} {'FNR 95%CI':>16s}  {'FPR':>7s} {'FPR 95%CI':>16s}")
    rows = []
    for provider, (_, m) in per_provider.items():
        fnr, fpr = m["fnr"], m["fpr"]
        fnr_ci, fpr_ci = wald_ci(fnr, m["n_yta_truth"]), wald_ci(fpr, m["n_nta_truth"])
        print(f"{provider:12s} {m['n_yta_truth']:6d} {m['n_nta_truth']:6d} "
              f"{m['n_dropped_other']:5d} "
              f"{fnr:7.3f} [{fnr-fnr_ci:.3f},{fnr+fnr_ci:.3f}]  "
              f"{fpr:7.3f} [{fpr-fpr_ci:.3f},{fpr+fpr_ci:.3f}]")
        rows.append({
            "model": model_key, "provider": provider,
            "n_total": m["n_total"], "n_classified": m["n_classified"],
            "n_dropped_other": m["n_dropped_other"],
            "n_yta_truth": m["n_yta_truth"], "n_nta_truth": m["n_nta_truth"],
            "fn": m["fn"], "fp": m["fp"], "fnr": fnr, "fpr": fpr,
            "fnr_ci95": fnr_ci, "fpr_ci95": fpr_ci,
        })

    # Per-post agreement between the two providers (only where BOTH parsed to a
    # YTA/NTA verdict). This is the direct "do the providers disagree?" signal,
    # complementing the rate-level comparison above.
    if {"azure", "openrouter"} <= set(per_provider):
        a = per_provider["azure"][0][["id", "pred"]].rename(columns={"pred": "pred_azure"})
        o = per_provider["openrouter"][0][["id", "pred"]].rename(columns={"pred": "pred_openrouter"})
        merged = a.merge(o, on="id")
        both = merged[merged["pred_azure"].isin(["YTA", "NTA"])
                      & merged["pred_openrouter"].isin(["YTA", "NTA"])]
        n = len(both)
        agree = int((both["pred_azure"] == both["pred_openrouter"]).sum())
        rate = agree / n if n else float("nan")
        print(f"\n  Azure vs OpenRouter verdict agreement: {agree}/{n} = {rate:.4f} "
              f"(both parsed to YTA/NTA)")
        delta_fnr = per_provider["openrouter"][1]["fnr"] - per_provider["azure"][1]["fnr"]
        delta_fpr = per_provider["openrouter"][1]["fpr"] - per_provider["azure"][1]["fpr"]
        print(f"  ΔFNR (openrouter - azure): {delta_fnr:+.3f}   "
              f"ΔFPR (openrouter - azure): {delta_fpr:+.3f}")
    return rows


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--aita-file", default=os.path.join(REPO_ROOT, AITA_DATA_CSV),
                   help="AITA CSV (default: repo data/aita_clean.csv)")
    p.add_argument("--n", type=int, default=200,
                   help="balanced sample size; half YTA-truth, half NTA-truth "
                        "(must be even; default: 200)")
    p.add_argument("--seed", type=int, default=0, help="sampling seed (default: 0)")
    p.add_argument("--temp", type=float, default=1.0,
                   help="generation temperature (default: 1.0, the providers' "
                        "default decoding setting; pass --temp 0 to isolate "
                        "provider serving from sampling noise)")
    p.add_argument("--max-tokens", type=int, default=256,
                   help="max output tokens per verdict (default: 256; the parser "
                        "anchors on the last YTA/NTA token so chatty replies are ok)")
    p.add_argument("--models", nargs="+", default=sorted(MODELS),
                   choices=sorted(MODELS), help="which models to run")
    p.add_argument("--providers", nargs="+", default=["azure", "openrouter"],
                   choices=["azure", "openrouter"], help="providers to run")
    p.add_argument("--workers", type=int, default=4, help="parallel threads (default: 4)")
    p.add_argument("--max-rps", type=int, default=4,
                   help="max requests/sec throttle per combo (default: 4)")
    p.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)),
                   help="directory for per-combo and summary CSVs")
    args = p.parse_args()

    assert os.path.exists(args.aita_file), f"missing AITA file: {args.aita_file}"
    os.makedirs(args.outdir, exist_ok=True)
    data = load_aita_clean(args.aita_file)
    # Sample ONCE so every (model, provider) combo sees the identical posts.
    sample = sample_balanced(data, args.n, args.seed)
    print(f"AITA provider consistency: n={len(sample)} (balanced "
          f"{(sample['is_asshole']==1).sum()} YTA / {(sample['is_asshole']==0).sum()} NTA), "
          f"seed={args.seed}, temp={args.temp}")
    print(f"models={args.models}  providers={args.providers}")

    clients = {prov: build_client(prov) for prov in args.providers}
    tag = f"n{args.n}_seed{args.seed}_temp{args.temp}"
    all_rows = []
    long_frames = []
    per_model = {}
    for model_key in args.models:
        per_provider = {}
        for prov in args.providers:
            slug = model_key.replace("/", "-")
            out_path = os.path.join(
                args.outdir, f"aita_consistency_{slug}_{prov}_{tag}.csv")
            print(f"\n=== {model_key} via {prov} -> {out_path}")
            res = run_provider_model(
                clients[prov], prov, model_key, sample, args.temp,
                args.max_tokens, args.workers, args.max_rps, out_path)
            metrics = compute_fnr_fpr(res)
            per_provider[prov] = (res, metrics)
            tall = res.copy()
            tall.insert(0, "model", model_key)
            tall.insert(1, "provider", prov)
            long_frames.append(tall)
        per_model[model_key] = per_provider

    # Reports + combined outputs. Reporting is separated from the API loop so a
    # transient failure mid-run still leaves the per-combo CSVs on disk.
    for model_key in args.models:
        all_rows.extend(report_model(model_key, per_model[model_key]))

    summary = pd.DataFrame(all_rows)
    summary_path = os.path.join(args.outdir, f"aita_consistency_summary_{tag}.csv")
    summary.to_csv(summary_path, index=False)
    long_path = os.path.join(args.outdir, f"aita_consistency_predictions_{tag}.csv")
    pd.concat(long_frames, ignore_index=True).to_csv(long_path, index=False)
    print(f"\nper-model FPR/FNR summary -> {summary_path}")
    print(f"all per-post predictions  -> {long_path}")


if __name__ == "__main__":
    main()
