"""Cross-model combined OEQ and PAS figures for the GPT-4o-judged re-run.

The mitigations sweep was originally scored by a GPT-5.4 judge; those combined
figures live at `plots_combined/{oeq,pas}_combined.png`. This script renders the
same two combined views from the GPT-4o re-judge outputs
(`*_<gen>_gpt-4o.csv`, produced by `rerun_mitigations_judge_gpt4o.py`) and
writes them to *separate* filenames so the GPT-5.4 figures are not overwritten:

  - plots_combined/oeq_combined_gpt-4o.png
  - plots_combined/pas_combined_gpt-4o.png

Reuses the plotting functions from `plot_explore_mitigations` verbatim — only
the family list (judge slug `gpt-4o`) and output paths differ.

Scope: the GPT-4o re-judge covers only the four reported methods (baseline,
be_critical, no_syco_critical, stakeholder_concerns_v2), so each panel shows
those four. The `oeq_human_diff_combined` view is intentionally NOT reproduced:
it requires the `human_reference` condition, which was not re-judged with
GPT-4o.

Run from the repo root:

    python mitigations/plot_combined_gpt4o.py
"""

import os
import sys

# Moved to legacy/; plot_explore_mitigations lives in ../mitigations/.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mitigations"))

from plot_explore_mitigations import (
    COMBINED_OUT_DIR,
    OEQ_CONDITION_ORDER,
    csv_for_family,
    plot_combined_endorse,
    plot_combined_pas,
)

# Same generation models / display names as the GPT-5.4 COMBINED_FAMILIES, but
# pinned to the GPT-4o judge slug so `csv_for_family` resolves the re-judge
# outputs.
GPT4O_FAMILIES = [
    ("claude-sonnet-4-6_gpt-4o", "Claude Sonnet 4.6"),
    ("deepseek-v3.2_gpt-4o", "DeepSeek v3.2"),
    ("gpt-5.4_gpt-4o", "GPT-5.4"),
]


def main():
    os.makedirs(COMBINED_OUT_DIR, exist_ok=True)
    # Fail loud up front if any expected GPT-4o CSV is missing, rather than
    # halfway through rendering.
    for family, _ in GPT4O_FAMILIES:
        for env in ("OEQ", "PAS"):
            csv = csv_for_family(family, env)
            assert os.path.exists(csv), f"missing GPT-4o results csv: {csv}"

    oeq_out = os.path.join(COMBINED_OUT_DIR, "oeq_combined_gpt-4o.png")
    plot_combined_endorse("OEQ", OEQ_CONDITION_ORDER, GPT4O_FAMILIES, oeq_out)
    print(f"wrote {oeq_out}")

    pas_out = os.path.join(COMBINED_OUT_DIR, "pas_combined_gpt-4o.png")
    plot_combined_pas(GPT4O_FAMILIES, pas_out)
    print(f"wrote {pas_out}")


if __name__ == "__main__":
    main()
