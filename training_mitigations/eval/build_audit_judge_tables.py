"""Assemble the 2-audit-judge (J2 warrant / J3 actionability) tables for the
FINAL-DATASET (`finaldata` tag) conditions, per env.

Separate from build_finaldata_judge_tables.py on purpose: that builder reports
BINARY judges by averaging their normalized <metric>_score_ column. Neither audit
judge can be reported that way --
  warrant     is NOMINAL 0-3 (no-pushback / warranted / generic / unwarranted);
              the mean of its score column is meaningless.
  actionable  is ORDINAL 0-2 but its reportable quantity is a rate, not a mean.
Both are aggregated here with their own modules' rate helpers, so the numbers in
this table and the definitions in those modules cannot drift apart.

Columns and CI conventions:
  pushback         share with any pushback at the user (warrant in {1,2,3}).
                   Binary proportion -> 95% Wilson score interval.
  precision        #warranted / #pushback. The headline for J2: among responses
                   that push back at all, the share whose pushback is grounded
                   in this user's situation. Its denominator is JUDGE-SELECTED
                   and varies by arm, so n is reported per row (same convention
                   as the paper's endorsement rate over its {0,1} subset).
  generic /        shares of ALL rows at warrant levels 2 / 3. Wilson.
  unwarranted
  advice           #actionable / n. AS-DELIVERED (see judge_actionability): a
                   response truncated at the generation cap before its advice
                   section scores 0, identically to one that withheld advice.
                   Wilson over all rows.

POLARITY (do not read every column as "higher = better"):
  pushback     higher = challenges the user more often. Not a quality axis by
               itself -- that is what precision is for.
  precision    higher = better. Low precision with high pushback = harshness
               masquerading as criticism.
  unwarranted  higher = WORSE (manufactured blame, or blame on invented facts).
  advice       higher = more useful, but understates arms that defer advice to
               the end of the response (see the truncation caveat).

Every rate is asserted to have zero unparsable rows before it is reported: a NaN
means a failed API call, so the condition is incomplete and must be re-judged
rather than averaged over.

Run:
    python training_mitigations/eval/build_audit_judge_tables.py
    python training_mitigations/eval/build_audit_judge_tables.py --envs OEQ
"""

import argparse
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_TM = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

import judge_actionability  # noqa: E402
import judge_criticism_warrant  # noqa: E402
from build_finaldata_tables import (  # noqa: E402
    OEQ_METHODS, PAS_METHODS, wilson_ci, fmt)
from run_finaldata_judges import build_conditions  # noqa: E402

JUDGE_DIR = os.path.join(_TM, "sycoharm_runs", "judges_finaldata")
OUT_DIR = os.path.join(_TM, "sycoharm_runs")

ENVS = ("OEQ", "PAS")
LABELS = {"OEQ": [l for l, _ in OEQ_METHODS], "PAS": [l for l, _ in PAS_METHODS]}


def raw_labels(env, metric, slug, lo, hi):
    """-> the raw integer label vector for one (env, metric, condition), or None
    if that judge has not run yet.

    Asserts no unparsable cell: judge_common.expand_score writes NaN for a failed
    API call, and a rate computed over a short vector would under-report silently.
    """
    p = os.path.join(JUDGE_DIR, env, f"{metric}_{slug}.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    col = f"{metric}_{slug}" if f"{metric}_{slug}" in df.columns else f"{metric}_response"
    assert col in df.columns, f"{p}: no {col} (mid-run checkpoint?)"
    errored = df[col].astype(str).str.startswith("ERROR")
    labels = df[col].mask(errored).map(
        judge_common_parser(lo, hi)).astype("Float64")
    assert labels.notna().all(), \
        f"{p}: {int(labels.isna().sum())} unparsable -- re-judge this condition"
    return labels


def judge_common_parser(lo, hi):
    """-> judge_common's digit parser for [lo, hi]; wrapped so the import stays
    local to this module's one use."""
    import judge_common
    return judge_common.make_digit_parser(lo, hi)


def build_env(env):
    """-> (DataFrame, missing) for one env: one row per condition with the J2/J3
    rates and their CIs, plus the list of (metric, slug) pairs not yet judged.

    A condition missing EITHER judge is reported in `missing` and skipped, so a
    half-finished sweep cannot render as a finished table.
    """
    W, A = judge_criticism_warrant, judge_actionability
    rows, missing = [], []
    for (slug, _src), label in zip(build_conditions(env), LABELS[env]):
        w = raw_labels(env, W.METRIC, slug, W.LO, W.HI)
        a = raw_labels(env, A.METRIC, slug, A.LO, A.HI)
        if w is None:
            missing.append((W.METRIC, slug))
        if a is None:
            missing.append((A.METRIC, slug))
        if w is None or a is None:
            continue
        wr = W.warrant_rates(w)
        ar = A.actionability_rates(a)
        n_push = int(round(wr["pushback_rate"] * wr["n"]))
        n_warr = int(round(wr["rate_warranted"] * wr["n"]))
        row = {"method": label, "N": wr["n"]}
        row["pushback"] = fmt(*wilson_ci(n_push, wr["n"]))
        row["precision"] = (fmt(*wilson_ci(n_warr, n_push)) + f"  (n={n_push})"
                            if n_push else "-- (no pushback)")
        row["generic"] = fmt(*wilson_ci(int(round(wr["rate_generic"] * wr["n"])), wr["n"]))
        row["unwarranted"] = fmt(*wilson_ci(int(round(wr["rate_unwarranted"] * wr["n"])), wr["n"]))
        row["advice"] = fmt(*wilson_ci(int(round(ar["advice_rate"] * ar["n"])), ar["n"]))
        row["no-advice"] = f"{ar['rate_no-advice']:.3f}"
        row["vague"] = f"{ar['rate_vague-only']:.3f}"
        rows.append(row)
    return pd.DataFrame(rows), missing


def md(df, title):
    """-> a markdown table for one env, with the title as a heading."""
    return f"\n## {title}\n\n{df.to_markdown(index=False)}\n"


def main():
    p = argparse.ArgumentParser(description="Build the J2/J3 audit judge tables.")
    p.add_argument("--envs", nargs="+", choices=ENVS, default=list(ENVS))
    args = p.parse_args()

    out = []
    for env in args.envs:
        df, missing = build_env(env)
        if missing:
            print(f"[{env}] NOT YET JUDGED ({len(missing)}): "
                  f"{sorted(set(f'{m}/{s}' for m, s in missing))}")
        if not len(df):
            print(f"[{env}] no complete conditions -- skipping table")
            continue
        block = md(df, f"{env} -- audit judges (J2 warrant / J3 actionability), "
                       f"final-dataset held-out, 95% CIs")
        print(block)
        out.append(block)

    if out:
        dest = os.path.join(OUT_DIR, "audit_judge_tables.md")
        with open(dest, "w") as f:
            f.write("".join(out))
        print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
