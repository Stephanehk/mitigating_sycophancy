"""Regenerate representative stakeholder-simulation examples for the
300-prompt OEQ and PAS DPO runs, for showing to an advisor.

WHY THIS EXISTS
  The training run (`train_dpo_lora.py`) persists only the *vetoed
  response strings* per prompt (`runs/.../responses_iter_*.jsonl`).
  The identified stakeholders, their concerns, and the per-response
  veto justification are computed inside `SimulatedStakeholders` and
  then discarded. This script re-derives them.

WHY THE RE-DERIVATION IS FAITHFUL (not made-up data)
  The veto signal during training ran on the FROZEN base model
  (Qwen/Qwen3-8B) with GREEDY decoding (`make_call_llm(..., do_sample=
  False)`) and the `use_weak_model=True` pipeline. None of that depends
  on the trained policy. So loading the same base model and replaying
  identify -> filter -> concerns -> veto on the same prompts reproduces
  the exact stakeholders / concerns / veto decisions seen in iteration 1
  of training. We assert that the reproduced veto decision on each saved
  vetoed response is in fact VETO; a mismatch raises rather than being
  papered over.

WHAT IT EMITS (per selected prompt)
  (1) identified stakeholders (candidates -> kept set after filter)
  (2) each kept stakeholder's concerns
  (3) one VETOED response + the stakeholder's veto reasoning
  (4) for contrast, one NON-vetoed (chosen) response + an ACCEPT
      reasoning, so the advisor can see the judge discriminating.

Outputs a human-readable .txt and a machine-readable .json under
`runs/stakeholder_examples/`.
"""
import json
import os
import sys

import torch

THIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(THIS_DIR)
# Same path wiring as train_dpo_lora.py: the harness lives in mitigations/,
# the local helpers in training_mitigations/.
sys.path.insert(0, os.path.join(_REPO_ROOT, "mitigations"))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(THIS_DIR, "core"))

import our_mitigation
import our_mitigation_v2
import weak_pipeline
from llama_local import DEFAULT_HF_MODEL_ID, load_llama, make_call_llm
from simulated_stakeholders import SimulatedStakeholders
from train_dpo_lora import _tolerant_parse_stakeholders


# Representative prompts (by prompt_idx in responses_iter_1.jsonl).
# Chosen as PARTIAL-veto cases so each shows the judge discriminating
# between sycophantic and non-sycophantic responses, and spanning the
# kinds of stakeholders Qwen surfaces (legal/financial, medical safety,
# mental-health / self-worth).
SELECTED = {
    "oeq": [221, 24, 65, 229, 81],
    "pas": [2, 143, 259, 270, 85],
}

OUT_DIR = os.path.join(THIS_DIR, "runs", "stakeholder_examples")


def load_rows(env):
    path = os.path.join(THIS_DIR, "runs", f"dpo_lora_{env}_qwen3-8b",
                        "responses_iter_1.jsonl")
    rows = {}
    for line in open(path):
        r = json.loads(line)
        rows[r["prompt_idx"]] = r
    return rows


def build_sim():
    """Load Qwen3-8B once as the frozen sim and wire the pipeline exactly
    as train_dpo_lora.py does (greedy call_llm, tolerant parser, weak
    pipeline)."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[load] sim_base {DEFAULT_HF_MODEL_ID} -> {device}", flush=True)
    sim_model, sim_tok = load_llama(model_id=DEFAULT_HF_MODEL_ID, device=device)
    for p in sim_model.parameters():
        p.requires_grad = False
    sim_model.eval()

    sim_call_llm = make_call_llm(sim_model, sim_tok, do_sample=False)
    our_mitigation.call_llm = sim_call_llm
    our_mitigation_v2.call_llm = sim_call_llm
    weak_pipeline.call_llm = sim_call_llm
    our_mitigation.parse_stakeholders = _tolerant_parse_stakeholders
    our_mitigation_v2.parse_stakeholders = _tolerant_parse_stakeholders
    weak_pipeline.parse_stakeholders = _tolerant_parse_stakeholders

    sim = SimulatedStakeholders(
        call_llm=sim_call_llm,
        model_id=DEFAULT_HF_MODEL_ID,
        client=None,
        stakeholder_max_tokens=1024,
        use_weak_model=True,
    )
    return sim


def first_vetoing_stakeholder(sim, prompt, response, entries):
    """Replay the short-circuit veto loop: return (stakeholder, concerns,
    raw_reasoning) for the first stakeholder that vetoes `response`, or
    None if none do."""
    for entry in entries:
        decision, raw = sim._stakeholder_vetos(prompt, response, entry)
        if decision:
            return entry["stakeholder"], entry["concerns"], raw
    return None


def accept_reasoning(sim, prompt, response, entries):
    """Run the veto judge over all stakeholders for a non-vetoed response
    and return (stakeholder, raw) for the first one, as an illustrative
    ACCEPT justification. Asserts no stakeholder vetoes (it should be a
    saved non-vetoed response)."""
    for entry in entries:
        decision, raw = sim._stakeholder_vetos(prompt, response, entry)
        assert not decision, (
            f"expected ACCEPT on a saved non-vetoed response but got VETO "
            f"from {entry['stakeholder']!r}"
        )
    # return the first stakeholder's accept text for display
    entry = entries[0]
    _decision, raw = sim._stakeholder_vetos(prompt, response, entry)
    return entry["stakeholder"], raw


def process_prompt(sim, env, row):
    prompt = row["prompt"]
    vetoed = row["vetoed"]
    responses = row["responses"]
    non_vetoed = [r for r in responses if r not in set(vetoed)]

    print(f"\n[{env}] prompt_idx={row['prompt_idx']} "
          f"vetoed={len(vetoed)}/{len(responses)} — running pipeline...", flush=True)
    candidates, entries = sim._stakeholder_data_for(prompt)

    veto_ex = None
    if vetoed:
        got = first_vetoing_stakeholder(sim, prompt, vetoed[0], entries)
        assert got is not None, (
            f"saved vetoed response no longer vetoes on replay "
            f"(prompt_idx={row['prompt_idx']}) — pipeline not reproducing"
        )
        s, concerns, raw = got
        veto_ex = {"stakeholder": s, "stakeholder_concerns": concerns,
                   "response": vetoed[0], "veto_reasoning": raw}

    accept_ex = None
    if non_vetoed:
        s, raw = accept_reasoning(sim, prompt, non_vetoed[0], entries)
        accept_ex = {"stakeholder": s, "response": non_vetoed[0],
                     "accept_reasoning": raw}

    return {
        "env": env,
        "prompt_idx": row["prompt_idx"],
        "prompt": prompt,
        "num_vetoed": len(vetoed),
        "num_responses": len(responses),
        "identified_candidates": candidates,
        "kept_stakeholders": [e["stakeholder"] for e in entries],
        "stakeholder_concerns": {e["stakeholder"]: e["concerns"] for e in entries},
        "veto_example": veto_ex,
        "accept_example": accept_ex,
    }


def render_txt(results):
    L = []
    for r in results:
        L.append("=" * 100)
        L.append(f"{r['env'].upper()}  prompt_idx={r['prompt_idx']}  "
                 f"({r['num_vetoed']}/{r['num_responses']} responses vetoed)")
        L.append("=" * 100)
        L.append("\nPROMPT:")
        L.append("  " + r["prompt"].replace("\n", "\n  "))
        L.append("\n(1) STAKEHOLDERS IDENTIFIED BY QWEN:")
        L.append("  candidates (identify step): " + " | ".join(r["identified_candidates"]))
        L.append("  kept (after filter):        " + " | ".join(r["kept_stakeholders"]))
        L.append("\n    concerns per kept stakeholder:")
        for s, cs in r["stakeholder_concerns"].items():
            L.append(f"      [{s}]")
            for c in cs:
                L.append(f"         - {c}")
        if r["veto_example"]:
            v = r["veto_example"]
            L.append("\n(2)+(3) A VETOED RESPONSE AND THE STAKEHOLDER'S REASONING:")
            L.append(f"    vetoing stakeholder: {v['stakeholder']}")
            L.append("    VETOED RESPONSE:")
            L.append("      " + v["response"].replace("\n", "\n      "))
            L.append("    VETO REASONING:")
            L.append("      " + v["veto_reasoning"].replace("\n", "\n      "))
        if r["accept_example"]:
            a = r["accept_example"]
            L.append("\n(contrast) A NON-VETOED RESPONSE AND AN ACCEPT REASONING:")
            L.append(f"    stakeholder: {a['stakeholder']}")
            L.append("    NON-VETOED RESPONSE:")
            L.append("      " + a["response"].replace("\n", "\n      "))
            L.append("    ACCEPT REASONING:")
            L.append("      " + a["accept_reasoning"].replace("\n", "\n      "))
        L.append("")
    return "\n".join(L)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sim = build_sim()
    results = []
    for env, idxs in SELECTED.items():
        rows = load_rows(env)
        for idx in idxs:
            assert idx in rows, f"{env}: prompt_idx {idx} not in iter1 jsonl"
            results.append(process_prompt(sim, env, rows[idx]))

    json_path = os.path.join(OUT_DIR, "examples_batch2.json")
    txt_path = os.path.join(OUT_DIR, "examples_batch2.txt")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    with open(txt_path, "w") as f:
        f.write(render_txt(results))
    print(f"\n[done] wrote {json_path}")
    print(f"[done] wrote {txt_path}")


if __name__ == "__main__":
    main()
