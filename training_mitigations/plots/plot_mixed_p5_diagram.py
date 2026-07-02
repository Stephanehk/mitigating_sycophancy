"""Render a flow diagram of the `mixed_P5` training procedure, in the same
visual style as plot_training_diagram.py but specific to the mixed-sampling /
two-stage-P5-veto configuration.

This is a procedure schematic (not a data plot): every label is taken from the
actual code/config —
  - hyperparameters + flags: slurm_scripts/run_train_oeq_n1000_mixed_P5.sbatch
  - mixed sampling + pair-building: core/trainer.py
  - two-stage P5 veto: core/simulated_stakeholders.py
    (ENDORSE_CLASSIFY_PROMPT_P5 = Stage 1; VETO_PROMPT_P5 = Stage 2)
Run with the project interpreter; writes a PNG under plots_dpo_summary/.
"""
import os
import matplotlib.pyplot as plt

# Reuse the reference diagram's box/arrow helpers and palette so the two
# figures share one visual language.
from plot_training_diagram import (
    box, arrow, C_POLICY, C_POLICY_E, C_FROZEN, C_FROZEN_E,
    C_DATA, C_DATA_E, C_STEP, C_STEP_E,
)

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots_dpo_summary")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.5, 15))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(50, 98.5,
            "mixed_P5:  iterative DPO + LoRA with a two-stage simulated-stakeholder veto",
            ha="center", va="center", fontsize=14.5, weight="bold")
    ax.text(50, 95.8,
            "Qwen3-8B + LoRA (r=16, α=32)  ·  OEQ / PAS  ·  1000 train prompts  ·  "
            "3 iterations  ·  no self-preference",
            ha="center", va="center", fontsize=10, color="#333333")

    # --- two model copies (top banner) ---
    pol = box(ax, 27, 89, 38, 5.5, "POLICY\nQwen3-8B + LoRA",
              C_POLICY, C_POLICY_E, fontsize=11, weight="bold")
    sim = box(ax, 74, 89, 42, 5.5, "STAKEHOLDER SIMULATOR\nfrozen Qwen3-8B",
              C_FROZEN, C_FROZEN_E, fontsize=11, weight="bold")

    # --- Step A: mixed sampling ---
    a = box(ax, 27, 76.5, 46, 9.5,
            "A.  Sample N=10 per prompt  —  MIXED sources\n\n"
            "2 bare  ·  2 “Be critical.”  ·  6 stakeholder-voiced\n"
            "temp 1.0 · top-p 0.95   (from the CURRENT policy)",
            C_STEP, C_STEP_E, fontsize=10)
    arrow(ax, pol["bottom"], a["top"])

    # --- Step B: two-stage P5 veto pipeline ---
    b = box(ax, 74, 60, 46, 23,
            "B.  Veto pipeline  (P5, two-stage)\n\n"
            "identify ≈3 stakeholders + their concerns\n"
            "(policy-independent · cached once)\n"
            "______________________________\n\n"
            "Stage 1 — endorse-classify (per response):\n"
            "does it ENDORSE the narrator's action?  YES / NO\n\n"
            "Stage 2 — per stakeholder, only if Stage 1 = YES:\n"
            "do they OBJECT to that endorsement?  → veto",
            C_FROZEN, C_FROZEN_E, fontsize=9.5)
    arrow(ax, a["right"], (b["left"][0], 67), rad=-0.15)
    arrow(ax, sim["bottom"], b["top"])

    # --- Step C: veto rule ---
    c = box(ax, 74, 43, 46, 5.5,
            "C.  Response vetoed  ⇔  ≥ 1 stakeholder objects",
            C_STEP, C_STEP_E, fontsize=10)
    arrow(ax, b["bottom"], c["top"])

    # --- Step D: build pairs ---
    d = box(ax, 50, 30, 70, 8,
            "D.  Build DPO preference pairs\n"
            "chosen = survived (non-vetoed)      rejected = vetoed\n"
            "(every vetoed  ×  every surviving response)",
            C_DATA, C_DATA_E, fontsize=10)
    arrow(ax, c["bottom"], (d["right"][0] - 6, d["top"][1]), rad=-0.2)
    arrow(ax, a["bottom"], (d["left"][0] + 6, d["top"][1]), rad=0.2)

    # --- Step E: DPO update ---
    e = box(ax, 50, 17, 60, 6.5,
            "E.  DPO epoch\nβ=0.1 · lr 5e-5 · 1 epoch  →  snapshot adapter_iter_k",
            C_POLICY, C_POLICY_E, fontsize=10, weight="bold")
    arrow(ax, d["bottom"], e["top"])

    # --- loop back arrow E -> A (policy updated) ---
    arrow(ax, e["left"], (4, 17), style="-", color=C_POLICY_E, lw=2.2)
    arrow(ax, (4, 17), (4, 76.5), style="-", color=C_POLICY_E, lw=2.2)
    arrow(ax, (4, 76.5), (a["left"][0], 76.5), color=C_POLICY_E, lw=2.2,
          text="repeat ×3\nre-sample from\nupdated policy", toffx=0, toffy=10,
          tha="center", fontsize=9)

    # --- output ---
    out = box(ax, 50, 7, 46, 5, "OUTPUT:  LoRA adapter  (after 3 iterations)",
              C_DATA, C_DATA_E, fontsize=10, weight="bold")
    arrow(ax, e["bottom"], out["top"])

    ax.text(50, 2,
            "Iteration 1 samples from the base policy; iterations 2–3 sample from the updated "
            "policy.  Stage 1 is stakeholder-independent\nand cached per response; Stage 2 runs "
            "only on responses Stage 1 marks YES, so it cannot veto a clear push-back.",
            ha="center", va="center", fontsize=8.5, style="italic", color="#555555")

    out_path = os.path.join(OUT_DIR, "mixed_p5_procedure_diagram.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[diagram] wrote {out_path}")


if __name__ == "__main__":
    main()
