"""Audit the two-stage P5 veto on real OEQ mixed_P4 candidates.

Two parts, matching the two questions:

PART A — does Stage 1 (the stakeholder-INDEPENDENT endorsement classifier, run
  by the frozen Qwen3-8B) agree with the v5 gpt-5-mini endorse judge? Stage 1
  needs only (post, response), so we run it on a sample of training candidates
  that already carry a v5 label (from sycoharm_runs/endorse_gpt5mini_v5_temp0/
  scored_cvr_*.csv), stratified across the four v5 labels, and cross-tabulate
  Stage-1 YES/NO against the v5 label (0 challenge / 1 explicit-affirm /
  2 implicit-affirm / 3 neutral). "Endorses" for v5 = label in {1,2}.

PART B — when do the stakeholders veto (Stage 2)? On a few "inversion" prompts
  we run the FULL two-stage veto (regenerating the stakeholder concerns), and
  for each response record Stage-1 YES/NO + justification and, for the endorsing
  ones, each stakeholder's Stage-2 VETO/ACCEPT + justification. This shows what
  makes a stakeholder object to an endorsement vs accept it.

Greedy decoding => deterministic. Assumes the cvr scored CSVs and the run jsonl
exist. Outputs to <run>/eval/: p5_stage1_vs_v5.csv and p5_two_stage_audit.md.
"""

import argparse
import json
import os
import re
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO_ROOT, "mitigations"))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "core"))

from llama_local import DEFAULT_HF_MODEL_ID, load_llama, make_call_llm
import simulated_stakeholders as ss
from simulated_stakeholders import (SimulatedStakeholders, VETO_PROMPT_P5,
                                    ENDORSE_CLASSIFY_PROMPT_P5)
import our_mitigation, our_mitigation_v2, weak_pipeline

_LOCAL_MAX_STAKEHOLDERS = 8
V5_DIR = os.path.join(_HERE, "sycoharm_runs", "endorse_gpt5mini_v5_temp0")
V5_CELLS = [f"scored_cvr_{v}_{s}.csv"
            for v in ("chosen", "rejected")
            for s in ("model", "critical", "stakeholder")]
LABEL_NAME = {0: "challenge", 1: "explicit-affirm", 2: "implicit-affirm", 3: "neutral"}


def first_label(x):
    m = re.search(r"[0-3]", str(x))
    return int(m.group()) if m else None


def tolerant_parse_stakeholders(text, min_count=None, max_count=None):
    items = []
    for raw in (text or "").splitlines():
        s = raw.strip()
        s = re.sub(r"^(?:[\-\*•]|\d+[\.\)])\s*", "", s).strip()
        s = s.strip("*_`").strip()
        if s:
            items.append(s)
    return items[:_LOCAL_MAX_STAKEHOLDERS]


def load_v5_sample(n_per_label, seed):
    """Pool the v5-scored cvr cells, dedup on (sentence,response), and take up to
    n_per_label rows per v5 label. Returns a DataFrame [sentence,response,v5]."""
    frames = []
    for fn in V5_CELLS:
        p = os.path.join(V5_DIR, fn)
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)[["sentence", "response", "endorse_response"]]
        frames.append(d)
    pool = pd.concat(frames, ignore_index=True)
    pool["v5"] = pool["endorse_response"].map(first_label)
    pool = pool.dropna(subset=["v5"])
    pool["v5"] = pool["v5"].astype(int)
    pool = pool.drop_duplicates(subset=["sentence", "response"])
    parts = []
    for lab in (0, 1, 2, 3):
        sub = pool[pool.v5 == lab]
        if len(sub) > n_per_label:
            sub = sub.sample(n_per_label, random_state=seed)
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def select_inversion_prompts(jsonl_path, n_prompts, max_per_prompt):
    """Pick prompts where P4 inverted: >=1 stakeholder-source candidate vetoed
    AND >=1 model/critical candidate accepted. Returns list of
    {prompt, candidates:[(source, response, p4_vetoed)]}."""
    picked = []
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            vset = set(r["vetoed"])
            cands = [(src.split(":")[0], resp, resp in vset)
                     for resp, src in zip(r["responses"], r["sources"])]
            sv = [c for c in cands if c[0] == "stakeholder" and c[2]]
            ma = [c for c in cands if c[0] in ("model", "critical") and not c[2]]
            if not (sv and ma):
                continue
            sa = [c for c in cands if c[0] == "stakeholder" and not c[2]]
            subset = (ma[:2] + sv[:2] + sa[:1])[:max_per_prompt]
            picked.append({"prompt": r["prompt"], "candidates": subset})
            if len(picked) >= n_prompts:
                break
    return picked


def confusion_report(df):
    """df has columns v5 (0..3) and s1_yes (bool). Returns a list of report
    lines: per-v5-label Stage-1 YES rate + the agreement summary treating v5 in
    {1,2} as 'endorses'."""
    lines = ["### PART A — Stage-1 (Qwen3-8B) endorsement YES rate by v5 label"]
    lines.append(f"{'v5 label':18} {'n':>4} {'Stage1=YES':>11} {'Stage1=NO':>10}")
    for lab in (0, 1, 2, 3):
        sub = df[df.v5 == lab]
        n = len(sub)
        if n == 0:
            continue
        y = int(sub.s1_yes.sum())
        lines.append(f"{lab} {LABEL_NAME[lab]:15} {n:>4} {y:>6} ({y/n:>4.0%}) {n-y:>5} ({(n-y)/n:>4.0%})")
    endorse_true = df[df.v5.isin([1, 2])]
    endorse_false = df[df.v5.isin([0, 3])]
    tp = int(endorse_true.s1_yes.sum())
    fn = len(endorse_true) - tp
    fp = int(endorse_false.s1_yes.sum())
    tn = len(endorse_false) - fp
    tot = len(df)
    agree = (tp + tn) / tot if tot else float("nan")
    lines.append("")
    lines.append("Treating v5 in {1,2} (explicit+implicit affirm) as ENDORSES:")
    lines.append(f"  agreement (Stage1==v5-endorses) = {(tp+tn)}/{tot} = {agree:.1%}")
    lines.append(f"  TP (v5 affirm & Stage1 YES)   = {tp}")
    lines.append(f"  FN (v5 affirm & Stage1 NO)    = {fn}   <- Stage1 MISSES these endorsements")
    lines.append(f"  FP (v5 non-affirm & Stage1 YES)= {fp}   <- Stage1 over-flags these")
    lines.append(f"  TN (v5 non-affirm & Stage1 NO) = {tn}")
    expl = df[df.v5 == 1]
    impl = df[df.v5 == 2]
    if len(expl):
        lines.append(f"  explicit-affirm (v5=1) caught by Stage1: {int(expl.s1_yes.sum())}/{len(expl)}")
    if len(impl):
        lines.append(f"  implicit-affirm (v5=2) caught by Stage1: {int(impl.s1_yes.sum())}/{len(impl)}")
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir",
                   default=os.path.join(_HERE, "runs",
                                        "dpo_lora_oeq_qwen3-8b_n1000_mixed_P4"))
    p.add_argument("--iter-file", default="responses_iter_3.jsonl")
    p.add_argument("--n-per-label", type=int, default=40)
    p.add_argument("--n-prompts", type=int, default=5)
    p.add_argument("--max-per-prompt", type=int, default=5)
    p.add_argument("--model-id", default=DEFAULT_HF_MODEL_ID)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", default="hf", choices=("hf", "openrouter"),
                   help="hf = local Qwen3-8B (GPU); openrouter = API model (no GPU)")
    p.add_argument("--api-model", default="openai/gpt-5-mini",
                   help="model id for --backend openrouter")
    p.add_argument("--skip-part-a", action="store_true",
                   help="skip the Stage-1-vs-v5 sweep; only run Part B")
    args = p.parse_args()

    for mod in (our_mitigation, our_mitigation_v2, weak_pipeline, ss):
        if hasattr(mod, "parse_stakeholders"):
            mod.parse_stakeholders = tolerant_parse_stakeholders

    if args.backend == "hf":
        print(f"[load] frozen sim {args.model_id} (greedy) -> device_map={args.device_map}")
        model, tok = load_llama(model_id=args.model_id, device=args.device,
                                cache_dir=os.environ.get("HF_HOME"),
                                device_map=args.device_map)
        model.eval()
        sim_call_llm = make_call_llm(model, tok, do_sample=False)
        sim_model_id, sim_client = args.model_id, None
        sim_kwargs = {}
    else:
        # API backend (no GPU). Route through the harness's dispatching call_llm;
        # give a generous token budget since gpt-5 spends tokens on reasoning.
        from explore_mitigations import make_client, call_llm
        sim_client = make_client(args.api_model)
        sim_call_llm = call_llm
        sim_model_id = args.api_model
        sim_kwargs = dict(stakeholder_max_tokens=4000, veto_max_tokens=4000,
                          veto_concurrency=8)
        print(f"[backend] openrouter {args.api_model}")
    sim = SimulatedStakeholders(
        call_llm=sim_call_llm, model_id=sim_model_id, client=sim_client,
        use_weak_model=True,
        veto_prompt=VETO_PROMPT_P5,
        endorse_classify_prompt=ENDORSE_CLASSIFY_PROMPT_P5,
        **({"veto_concurrency": 1} if args.backend == "hf" else {}),
        **sim_kwargs,
    )

    out = ["# Two-stage P5 audit on OEQ mixed_P4 candidates",
           f"_sim={args.model_id} greedy; Stage-1 classifier vs v5 gpt-5-mini; "
           f"source={args.iter_file}_\n"]

    # ---- PART A: Stage 1 vs v5 ------------------------------------------------
    if not args.skip_part_a:
        sample = load_v5_sample(args.n_per_label, args.seed)
        print(f"[partA] classifying {len(sample)} v5-labeled responses with Stage 1")
        s1 = []
        for k, row in enumerate(sample.itertuples(index=False)):
            endorses, just = sim._classify_endorsement(row.sentence, row.response)
            s1.append((endorses, just))
            if (k + 1) % 25 == 0:
                print(f"[partA] {k+1}/{len(sample)}")
        sample = sample.copy()
        sample["s1_yes"] = [e for e, _ in s1]
        sample["s1_just"] = [j for _, j in s1]
        csv_path = os.path.join(args.run_dir, "eval", f"p5_stage1_vs_v5_{args.backend}.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        sample[["sentence", "response", "v5", "s1_yes", "s1_just"]].to_csv(csv_path, index=False)
        out += confusion_report(sample)
        out.append(f"\n_per-row table -> {csv_path}_")

        # disagreement examples: v5 implicit-affirm (2) that Stage1 said NO (the old failure)
        miss = sample[(sample.v5 == 2) & (~sample.s1_yes)]
        out.append("\n#### Sample v5=implicit-affirm that Stage-1 still calls NO (missed endorsements)")
        for r in miss.head(4).itertuples(index=False):
            out.append(f"- RESP: {str(r.response)[:200].replace(chr(10),' ')}")
            out.append(f"    Stage1 NO because: {r.s1_just}")

    # ---- PART B: Stage 2 — when do stakeholders veto -------------------------
    out.append("\n" + "=" * 90)
    out.append("## PART B — full two-stage on inversion prompts (when do stakeholders veto?)")
    prompts = select_inversion_prompts(
        os.path.join(args.run_dir, args.iter_file), args.n_prompts, args.max_per_prompt)
    veto_among_endorsing = {"veto": 0, "total": 0}
    for pi, item in enumerate(prompts):
        prompt = item["prompt"]
        sim.prime_stakeholder_cache([prompt], concurrency=1)
        _c, entries = sim._stakeholder_data_for(prompt)
        out.append("\n" + "=" * 90)
        out.append(f"### Prompt {pi+1}\n{prompt[:350]}\n")
        out.append("Stakeholders: " + "; ".join(e["stakeholder"] for e in entries))
        for src, resp, p4_veto in item["candidates"]:
            endorses, just = sim._classify_endorsement(prompt, resp)
            out.append(f"\n**source={src}  P4_vetoed={p4_veto}**")
            out.append(f"> {resp[:200].replace(chr(10),' ')}")
            out.append(f"  Stage1: {'ENDORSES' if endorses else 'no-endorse'} — {just}")
            if not endorses:
                out.append("  => not vetoed (Stage 1 found no endorsement to object to)")
                continue
            any_veto = False
            for e in entries:
                decision, raw = sim._stakeholder_endorsement_objection(prompt, resp, e, just)
                any_veto = any_veto or decision
                m = re.search(r"\b(VETO|ACCEPT)\b", raw, re.IGNORECASE)
                why = raw[m.end():].strip().replace("\n", " ") if m else raw
                out.append(f"    - [{'VETO' if decision else 'ACCEPT'}] {e['stakeholder']}: {why}")
            veto_among_endorsing["total"] += 1
            veto_among_endorsing["veto"] += int(any_veto)
            out.append(f"  => Stage2 veto (OR over stakeholders) = {any_veto}  (P4 said {p4_veto})")
        print(f"[partB] prompt {pi+1}/{len(prompts)} done")

    v = veto_among_endorsing
    out.append("\n" + "=" * 90)
    out.append(f"## PART B summary: of {v['total']} ENDORSING responses (Stage1=YES), "
               f"stakeholders vetoed {v['veto']} ({(v['veto']/v['total'] if v['total'] else 0):.0%})")

    md_path = os.path.join(args.run_dir, "eval", f"p5_two_stage_audit_{args.backend}.md")
    with open(md_path, "w") as f:
        f.write("\n".join(out))
    print(f"[write] {md_path}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
