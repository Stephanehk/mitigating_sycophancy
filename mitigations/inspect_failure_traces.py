"""Print full per-POV traces (stakeholders, concerns/supports, judge raw
output incl. verdict) for one or more posts from an AITA-NTA-FLIP run of
the multi-stakeholder mitigation, alongside the original AITA post text
for context.

Useful for debugging why the method fails on specific posts — typically
critical=1 cases (both POVs → YTA) or sycophancy=1 cases (both POVs →
NTA), since those are the AITA-NTA-FLIP failure modes.

Usage:

  # default trace file (gpt-5.4_gpt-5.4 v2 run), pick post ids:
  python -m mitigations.inspect_failure_traces aa6vne alnfoi

  # if no ids are given, dumps every critical=1 trace in the file:
  python -m mitigations.inspect_failure_traces

  # different family / different env: point --traces at the right file:
  python -m mitigations.inspect_failure_traces \\
      --traces mitigations/results/explore_mitigations_results_AITA-NTA-FLIP_deepseek-v3.2_gpt-5.4_traces.jsonl \\
      --filter critical

  # show sycophancy=1 cases instead of critical=1:
  python -m mitigations.inspect_failure_traces --filter sycophancy

The flipped POV's post text is NOT stored verbatim in traces — only the
original post body is shown for context. Stakeholders / concerns /
supports / judge output for both POVs are dumped from the trace file.
"""

import argparse
import json
import pandas as pd


DEFAULT_TRACES = (
    "mitigations/results/"
    "explore_mitigations_results_AITA-NTA-FLIP_gpt-5.4_gpt-5.4_traces.jsonl"
)
DEFAULT_AITA_CSV = "data/aita_clean.csv"


def load_traces(path):
    """Load AITA-NTA-FLIP traces JSONL into {id: row}. Asserts that ids
    are unique so we don't silently drop a duplicate row.
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    by_id = {}
    for r in rows:
        assert r["id"] not in by_id, f"duplicate id in trace file: {r['id']}"
        by_id[r["id"]] = r
    return by_id


def select_ids(traces, requested_ids, filter_mode):
    """Return the list of post ids to inspect.

    `requested_ids` (positional CLI args) wins if non-empty; otherwise
    `filter_mode` selects ('critical' for critical_score==1, 'sycophancy'
    for sycophancy_score==1, 'all' for every trace). Asserts that every
    requested id is present so a typo fails fast.
    """
    if requested_ids:
        missing = [pid for pid in requested_ids if pid not in traces]
        assert not missing, (
            f"requested ids not in trace file: {missing}. "
            f"Available (first 10): {sorted(traces)[:10]}"
        )
        return list(requested_ids)
    if filter_mode == "critical":
        return [pid for pid, r in traces.items() if r.get("critical_score") == 1]
    if filter_mode == "sycophancy":
        return [pid for pid, r in traces.items() if r.get("sycophancy_score") == 1]
    if filter_mode == "all":
        return list(traces.keys())
    raise AssertionError(f"unknown filter mode {filter_mode!r}")


def render_pov(label, t):
    """Print one POV's trace block (stakeholders + per-stakeholder
    concerns/supports + judge raw output)."""
    print(f"\n{'─' * 80}\n{label}\n{'─' * 80}")
    print(f"\nStakeholders ({len(t['stakeholders'])}):")
    for s in t["stakeholders"]:
        print(f"  - {s}")
    print("\nPer-stakeholder concerns/supports:")
    for e in t["concerns_entries"]:
        print(f"\n  [{e['stakeholder']}]")
        print("    CONCERNS:")
        for c in e["concerns"] or ["(none)"]:
            print(f"      - {c}")
        print("    SUPPORTS:")
        for s in e["supports"] or ["(none)"]:
            print(f"      - {s}")
    print("\nJudge raw output (incl. <verdict>):")
    print(t["judge_raw"])


def render_trace(pid, traces, aita):
    """Print the full inspection block for one post id."""
    r = traces[pid]
    post = aita.loc[pid] if (aita is not None and pid in aita.index) else None
    print("=" * 100)
    truth = post["is_asshole"] if post is not None else "?"
    print(
        f"id={pid}  truth_is_asshole={truth}  "
        f"pred_orig={r['pred_original']}  pred_flip={r['pred_flipped']}  "
        f"sycophancy={r['sycophancy_score']}  critical={r['critical_score']}"
    )
    print("=" * 100)
    if post is not None:
        print(f"\n--- ORIGINAL POST ---\n{post['title']}\n\n{post['body']}\n")
        print(
            "(The FLIPPED post is this same situation POV-rewritten by the "
            "harness; not stored verbatim in traces.)"
        )
    else:
        print(f"\n(post id {pid!r} not found in {DEFAULT_AITA_CSV} — original "
              "post text omitted.)")
    render_pov("ORIGINAL POV", r["trace_original"])
    render_pov("FLIPPED POV (other party as narrator)", r["trace_flipped"])


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "post_ids", nargs="*",
        help="post ids to inspect (positional). If omitted, --filter "
             "selects which traces to dump.",
    )
    p.add_argument(
        "--traces", default=DEFAULT_TRACES,
        help=f"trace JSONL path (default: {DEFAULT_TRACES})",
    )
    p.add_argument(
        "--aita-csv", default=DEFAULT_AITA_CSV,
        help=f"AITA dataset path used to fetch the original post body "
             f"(default: {DEFAULT_AITA_CSV})",
    )
    p.add_argument(
        "--filter", choices=["critical", "sycophancy", "all"],
        default="critical",
        help="when no post_ids are given, which traces to dump: 'critical' "
             "= critical_score==1 (both POVs YTA), 'sycophancy' = "
             "sycophancy_score==1 (both POVs NTA), 'all' = every trace. "
             "Default: critical.",
    )
    args = p.parse_args()

    traces = load_traces(args.traces)
    aita = pd.read_csv(args.aita_csv).set_index("id")
    ids = select_ids(traces, args.post_ids, args.filter)
    assert ids, (
        f"no traces selected (filter={args.filter!r}, requested={args.post_ids!r})"
    )
    print(f"# inspecting {len(ids)} trace(s) from {args.traces}\n")
    for pid in ids:
        render_trace(pid, traces, aita)
        print()


if __name__ == "__main__":
    main()
