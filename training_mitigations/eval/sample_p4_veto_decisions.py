"""Re-run the P4 stakeholder veto WITH its justifications on real training
candidates from the OEQ mixed_P4 run, to diagnose why the veto inverts
(rejects the least-endorsing stakeholder-voiced candidates, keeps the more
narrator-affirming model/critical ones).

The trainer discards the veto justification (only the VETO/ACCEPT bit is kept
in responses_iter_*.jsonl). The per-stakeholder concerns cache is in-memory
only. So we faithfully reconstruct the veto: load the SAME frozen sim model
(Qwen3-8B, greedy) used in training, rebuild SimulatedStakeholders with the P4
veto prompt, regenerate the stakeholder concerns (deterministic under greedy
decoding, matching training), and re-veto each candidate — this time capturing
the one-sentence justification the model emits.

For a few "inversion" prompts (>=1 stakeholder-voiced candidate vetoed in
training AND >=1 model/critical candidate accepted), we print, per candidate:
its source, its training veto bit, the freshly reproduced VETO/ACCEPT, the
model's justification, and a response snippet — alongside the stakeholder(s)
and their concerns. This makes the decision rule the veto actually applies
legible.

Assumptions (asserted): run jsonl exists; the sim reproduces a VETO/ACCEPT
token per call (inherited from _stakeholder_vetos). Greedy decoding makes the
reproduction deterministic; regenerated concerns may differ slightly from the
training-time cache but are produced by the identical pipeline+model.

Output: training_mitigations/runs/.../eval/p4_veto_decisions.md (+ stdout).
"""

import argparse
import json
import os
import re
import sys

import torch

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO_ROOT, "mitigations"))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "core"))

from llama_local import DEFAULT_HF_MODEL_ID, load_llama, make_call_llm
import simulated_stakeholders as ss
from simulated_stakeholders import SimulatedStakeholders, VETO_PROMPTS
import our_mitigation, our_mitigation_v2, weak_pipeline

_LOCAL_MAX_STAKEHOLDERS = 8


def tolerant_parse_stakeholders(text, min_count=None, max_count=None):
    """Inlined copy of train_dpo_lora._tolerant_parse_stakeholders: strip
    bullet/number prefixes, keep non-empty lines, cap, no strict assert."""
    items = []
    for raw in (text or "").splitlines():
        s = raw.strip()
        s = re.sub(r"^(?:[\-\*•]|\d+[\.\)])\s*", "", s).strip()
        s = s.strip("*_`").strip()
        if s:
            items.append(s)
    return items[:_LOCAL_MAX_STAKEHOLDERS]


def select_inversion_prompts(jsonl_path, n_prompts, max_per_prompt):
    """Pick prompts where the veto inverted in training: >=1 stakeholder-source
    candidate was vetoed AND >=1 model/critical candidate was accepted. Returns
    a list of dicts {prompt, candidates:[(source, response, train_vetoed)]},
    candidates trimmed to a readable subset that keeps the contrast."""
    picked = []
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            vset = set(r["vetoed"])
            cands = []
            for resp, src in zip(r["responses"], r["sources"]):
                cands.append((src.split(":")[0], resp, resp in vset))
            stake_vetoed = [c for c in cands if c[0] == "stakeholder" and c[2]]
            mc_accepted = [c for c in cands if c[0] in ("model", "critical") and not c[2]]
            if not (stake_vetoed and mc_accepted):
                continue
            # Keep a readable contrast set: 2 accepted model/critical + 2 vetoed
            # stakeholder (+ any accepted stakeholder, which is the rare case).
            stake_accepted = [c for c in cands if c[0] == "stakeholder" and not c[2]]
            subset = (mc_accepted[:2] + stake_vetoed[:2] + stake_accepted[:1])[:max_per_prompt]
            picked.append({"prompt": r["prompt"], "candidates": subset})
            if len(picked) >= n_prompts:
                break
    return picked


def justification_of(raw):
    """Pull the one-sentence justification (everything after the VETO/ACCEPT
    token) from the veto model's raw reply."""
    m = re.search(r"\b(VETO|ACCEPT)\b", raw, re.IGNORECASE)
    if not m:
        return raw.strip().replace("\n", " ")
    return raw[m.end():].strip().replace("\n", " ")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir",
                   default=os.path.join(_HERE, "runs",
                                        "dpo_lora_oeq_qwen3-8b_n1000_mixed_P4"))
    p.add_argument("--iter-file", default="responses_iter_3.jsonl",
                   help="which iteration's candidates to sample (latest policy)")
    p.add_argument("--n-prompts", type=int, default=5)
    p.add_argument("--max-per-prompt", type=int, default=5)
    p.add_argument("--veto-prompt", default="P5", choices=tuple(VETO_PROMPTS),
                   help="which veto prompt to apply (default: P5). The "
                        "train_vetoed bit in the jsonl is always the P4 "
                        "training decision, so this surfaces P4->X flips.")
    p.add_argument("--model-id", default=DEFAULT_HF_MODEL_ID)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--device-map", default="auto")
    args = p.parse_args()

    # Patch the tolerant parser onto every module holding an independent
    # binding (mirrors train_dpo_lora) so a stray identify emission can't crash.
    for mod in (our_mitigation, our_mitigation_v2, weak_pipeline, ss):
        if hasattr(mod, "parse_stakeholders"):
            mod.parse_stakeholders = tolerant_parse_stakeholders

    jsonl_path = os.path.join(args.run_dir, args.iter_file)
    assert os.path.exists(jsonl_path), f"missing {jsonl_path}"
    prompts = select_inversion_prompts(jsonl_path, args.n_prompts, args.max_per_prompt)
    print(f"[select] {len(prompts)} inversion-case prompts from {args.iter_file}")

    print(f"[load] frozen sim {args.model_id} (greedy) -> device_map={args.device_map}")
    model, tok = load_llama(model_id=args.model_id, device=args.device,
                            cache_dir=os.environ.get("HF_HOME"),
                            device_map=args.device_map)
    model.eval()
    sim_call_llm = make_call_llm(model, tok, do_sample=False)

    VP = args.veto_prompt
    sim = SimulatedStakeholders(
        call_llm=sim_call_llm, model_id=args.model_id, client=None,
        use_weak_model=True, veto_concurrency=1,
        veto_prompt=VETO_PROMPTS[VP],
    )

    # Flip tally vs the P4 training decision (the jsonl train_vetoed bit).
    tally = {"total": 0, "flips": 0,
             "mc_accept_total": 0, "mc_accept_now_veto": 0,
             "stake_veto_total": 0, "stake_veto_now_accept": 0}

    out_lines = [f"# {VP} veto decisions on real OEQ mixed_P4 candidates",
                 f"_sim={args.model_id} greedy, veto_prompt={VP}, source={args.iter_file}; "
                 f"train_vetoed bit = P4 training decision_\n"]
    for pi, item in enumerate(prompts):
        prompt = item["prompt"]
        sim.prime_stakeholder_cache([prompt], concurrency=1)
        _cands, entries = sim._stakeholder_data_for(prompt)

        out_lines.append("\n" + "=" * 100)
        out_lines.append(f"## Prompt {pi + 1}\n{prompt[:600]}\n")
        out_lines.append("### Stakeholders & regenerated concerns")
        for e in entries:
            out_lines.append(f"- **{e['stakeholder']}**")
            for c in e["concerns"]:
                out_lines.append(f"    - {c}")
        out_lines.append("\n### Candidate decisions (per stakeholder)")
        for src, resp, train_veto in item["candidates"]:
            out_lines.append(f"\n**source={src}  train_vetoed={train_veto}**")
            out_lines.append(f"> {resp[:260].replace(chr(10), ' ')}")
            any_veto = False
            for e in entries:
                decision, raw = sim._stakeholder_vetos(prompt, resp, e)
                any_veto = any_veto or decision
                verdict = "VETO" if decision else "ACCEPT"
                out_lines.append(
                    f"  - [{verdict}] as _{e['stakeholder']}_: {justification_of(raw)}")
            flipped = (any_veto != train_veto)
            tally["total"] += 1
            tally["flips"] += int(flipped)
            if src in ("model", "critical") and not train_veto:
                tally["mc_accept_total"] += 1
                tally["mc_accept_now_veto"] += int(any_veto)
            if src == "stakeholder" and train_veto:
                tally["stake_veto_total"] += 1
                tally["stake_veto_now_accept"] += int(not any_veto)
            flag = "   <-- FLIPPED vs P4" if flipped else ""
            out_lines.append(f"  => {VP} veto (OR over stakeholders) = {any_veto}"
                             f"  (P4/training veto = {train_veto}){flag}")
        print(f"[done] prompt {pi + 1}/{len(prompts)}")

    out_lines.append("\n" + "=" * 100)
    out_lines.append(f"## Summary: {VP} vs P4 on {tally['total']} audited candidates")
    out_lines.append(f"- total decisions flipped vs P4: {tally['flips']}/{tally['total']}")
    out_lines.append(f"- affirming model/critical that P4 ACCEPTED -> now VETOED by {VP}: "
                     f"{tally['mc_accept_now_veto']}/{tally['mc_accept_total']}  "
                     f"(want high: these are the sycophantic chosen we want to reject)")
    out_lines.append(f"- stakeholder voices that P4 VETOED -> now ACCEPTED by {VP}: "
                     f"{tally['stake_veto_now_accept']}/{tally['stake_veto_total']}  "
                     f"(want high: these are the de-centering responses we want to keep)")

    out_path = os.path.join(args.run_dir, "eval", f"{VP.lower()}_veto_decisions.md")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(out_lines))
    print(f"[write] {out_path}")
    print("\n".join(out_lines))


if __name__ == "__main__":
    main()
