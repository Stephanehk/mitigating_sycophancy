"""Render a flow diagram of the iterative DPO+LoRA training procedure with a
simulated-stakeholder veto signal, for presentation.

This is a procedure schematic (not a data plot): every label is taken from the
actual code/config —
  - hyperparameters: training_mitigations/run_train_oeq.sbatch
  - sampling + pair-building: training_mitigations/trainer.py
  - stakeholder/veto pipeline: training_mitigations/simulated_stakeholders.py
Run with the project interpreter; writes a PNG under plots_dpo_summary/.
"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots_dpo_summary")

# colors
C_POLICY = "#cfe3ff"
C_POLICY_E = "#2f6fb3"
C_FROZEN = "#eee0c8"
C_FROZEN_E = "#b07d2b"
C_DATA = "#d6f0d6"
C_DATA_E = "#3a8a3a"
C_STEP = "#f2f2f2"
C_STEP_E = "#555555"


def box(ax, x, y, w, h, text, fc, ec, fontsize=10, weight="normal", align="center"):
    """Draw a rounded box centered at (x, y) and return useful anchors."""
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle="round,pad=0.3,rounding_size=1.2",
                       linewidth=1.6, facecolor=fc, edgecolor=ec, zorder=2)
    ax.add_patch(p)
    ha = {"center": "center", "left": "left"}[align]
    tx = x if align == "center" else x - w / 2 + 1.2
    ax.text(tx, y, text, ha=ha, va="center", fontsize=fontsize,
            weight=weight, zorder=3, color="#111111")
    return {"top": (x, y + h / 2), "bottom": (x, y - h / 2),
            "left": (x - w / 2, y), "right": (x + w / 2, y),
            "cx": x, "cy": y}


def arrow(ax, p0, p1, text=None, color="#333333", style="-|>", rad=0.0,
          fontsize=8.5, lw=1.8, toffx=0, toffy=0, tha="center"):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=16,
                        lw=lw, color=color,
                        connectionstyle=f"arc3,rad={rad}", zorder=1)
    ax.add_patch(a)
    if text:
        mx, my = (p0[0] + p1[0]) / 2 + toffx, (p0[1] + p1[1]) / 2 + toffy
        ax.text(mx, my, text, ha=tha, va="center", fontsize=fontsize,
                color=color, zorder=4,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.5, 14))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(50, 98.5, "Iterative DPO + LoRA with a simulated-stakeholder veto signal",
            ha="center", va="center", fontsize=15, weight="bold")
    ax.text(50, 95.6,
            "Qwen3-8B + LoRA (r=16, α=32)  ·  OEQ / PAS  ·  300 train prompts  ·  3 iterations",
            ha="center", va="center", fontsize=10.5, color="#333333")

    # --- two model copies (top banner) ---
    pol = box(ax, 27, 88, 38, 5.5, "POLICY\nQwen3-8B + LoRA",
              C_POLICY, C_POLICY_E, fontsize=11, weight="bold")
    sim = box(ax, 74, 88, 42, 5.5, "STAKEHOLDER SIMULATOR\nfrozen Qwen3-8B",
              C_FROZEN, C_FROZEN_E, fontsize=11, weight="bold")

    # --- Step A: sample ---
    a = box(ax, 27, 76, 44, 6.5,
            "A.  Sample 10 responses per prompt\ntemp 1.0 · top-p 0.95",
            C_STEP, C_STEP_E, fontsize=10.5)
    arrow(ax, pol["bottom"], a["top"])

    # --- Step B: stakeholder pipeline ---
    b = box(ax, 74, 62, 44, 12,
            "B.  Veto pipeline\n\n"
            "identify  →  filter  →\nconcerns  →  veto",
            C_FROZEN, "#b07d2b", fontsize=11)
    arrow(ax, a["right"], (b["left"][0], 70), rad=-0.15)
    arrow(ax, sim["bottom"], b["top"])

    # --- Step C: veto rule ---
    c = box(ax, 74, 47, 44, 5.5,
            "C.  Vetoed  ⇔  ≥ 1 stakeholder vetoes",
            C_STEP, C_STEP_E, fontsize=10.5)
    arrow(ax, b["bottom"], c["top"])

    # --- Step D: build pairs ---
    d = box(ax, 50, 34, 66, 7,
            "D.  Build DPO pairs\nchosen = non-vetoed     rejected = vetoed",
            C_DATA, C_DATA_E, fontsize=10.5)
    arrow(ax, c["bottom"], (d["right"][0] - 6, d["top"][1]), rad=-0.2)
    arrow(ax, a["bottom"], (d["left"][0] + 6, d["top"][1]), rad=0.2)

    # --- Step E: DPO update ---
    e = box(ax, 50, 21, 58, 6,
            "E.  DPO epoch\nβ=0.1 · lr 5e-5 · 1 epoch",
            C_POLICY, C_POLICY_E, fontsize=10.5, weight="bold")
    arrow(ax, d["bottom"], e["top"])

    # --- loop back arrow E -> A (policy updated) ---
    arrow(ax, e["left"], (5, 21), style="-", color=C_POLICY_E, lw=2.2)
    arrow(ax, (5, 21), (5, 76), style="-", color=C_POLICY_E, lw=2.2)
    arrow(ax, (5, 76), (a["left"][0], 76), color=C_POLICY_E, lw=2.2,
          text="repeat ×3\nupdated policy", toffx=0, toffy=8,
          tha="center", fontsize=9)

    # --- output ---
    out = box(ax, 50, 9, 44, 5, "OUTPUT:  LoRA adapter",
              C_DATA, C_DATA_E, fontsize=10.5, weight="bold")
    arrow(ax, e["bottom"], out["top"])

    ax.text(50, 3,
            "iteration 1 samples from the base model; iterations 2–3 sample from the updated policy.",
            ha="center", va="center", fontsize=9, style="italic", color="#555555")

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "training_procedure_diagram.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[diagram] wrote {out_path}")


if __name__ == "__main__":
    main()
