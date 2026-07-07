"""Render the OEQ + PAS held-out results tables (built by build_results_tables.py)
into a single self-contained, theme-aware HTML page for the Artifact tool.

Reads sycoharm_runs/results_table_{oeq,pas}.csv VERBATIM and emits cells as a
bold point estimate over a muted 95% CI, so the page is a faithful view of the
computed numbers (no values are recomputed or invented here).
"""

import os
import re
import html

_HERE = os.path.dirname(os.path.abspath(__file__))
_RUNS = os.path.join(_HERE, "..", "sycoharm_runs")
import pandas as pd

CELL = re.compile(r"([-\d.]+)\s*\[([-\d.]+),\s*([-\d.]+)\]")

# metric column key -> (short header, direction note)
METRICS = [
    ("endorse_rate", "Endorsement", "↓ less sycophantic"),
    ("neutral_rate", "Neutral", "share, no stance"),
    ("addressee", "Addressee", "↑ message-as-artifact"),
    ("empathy: validating_emotions", "Validating", "↑ more empathy"),
    ("empathy: demonstrating_understanding", "Understanding", "↑ more empathy"),
    ("empathy: dismissing_emotions (rev)", "Dismissing⁻¹", "↑ less dismissive"),
    ("framing", "Framing", "↑ accepts premise"),
    ("validation", "Validation", "↑ more validating"),
]
GROUPS = [("(a) Action endorsement", 2), ("(b) Response-quality judges", 4),
          ("(c) ELEPHANT", 2)]
OUR = {"mixed_P5", "mixed_P5_nocrit_distill_gp_frozenstake",
       "mixed_P6_perspshift_distill_gp"}
OUR_LABELS = {"mixed_P5", "mixed_P5_nocrit_distill_gp_frozenstake",
              "mixed_P6_perspshift_distill_gp"}


def cell_html(raw):
    m = CELL.search(str(raw))
    if not m:
        return f'<td>{html.escape(str(raw))}</td>'
    pt, lo, hi = m.groups()
    return (f'<td><span class="pt">{pt}</span>'
            f'<span class="ci">{lo}, {hi}</span></td>')


def table_html(env, df):
    rows = []
    for _, r in df.iterrows():
        method = r["method"]
        cls = []
        if method in OUR_LABELS:
            cls.append("our")
        if method == "Human baseline":
            cls.append("human")
        tr_cls = f' class="{" ".join(cls)}"' if cls else ""
        cells = [f'<th scope="row" class="method">{html.escape(method)}'
                 f'<span class="nsub">n(0/1)={int(r["n(0/1)"])}</span></th>',
                 f'<td class="ncol">{int(r["N"])}</td>']
        for key, _, _ in METRICS:
            cells.append(cell_html(r[key]))
        rows.append(f'<tr{tr_cls}>' + "".join(cells) + "</tr>")

    group_ths = ['<th class="corner" rowspan="2">Method</th>',
                 '<th class="ncol" rowspan="2">N</th>']
    for gname, span in GROUPS:
        group_ths.append(f'<th class="grp" colspan="{span}">{gname}</th>')
    metric_ths = []
    for _, short, note in METRICS:
        metric_ths.append(f'<th>{short}<span class="dir">{note}</span></th>')

    return f"""
    <section class="tblwrap">
      <h2>{env} <span class="sub">— 1000-prompt held-out set</span></h2>
      <div class="scroll">
      <table>
        <thead>
          <tr class="grouprow">{''.join(group_ths)}</tr>
          <tr class="metricrow">{''.join(metric_ths)}</tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
      </div>
    </section>"""


def main():
    oeq = pd.read_csv(os.path.join(_RUNS, "results_table_oeq.csv"))
    pas = pd.read_csv(os.path.join(_RUNS, "results_table_pas.csv"))
    body = table_html("OEQ", oeq) + table_html("PAS", pas)

    page = f"""<title>Sycophancy mitigation — held-out results</title>
<style>
:root {{
  --bg:#f7f8fa; --panel:#ffffff; --ink:#161a22; --muted:#5c6472;
  --faint:#8a92a0; --line:#e4e7ec; --line2:#eef0f3;
  --accent:#0d7d76; --accent-soft:#e6f4f2; --human:#b45309; --human-soft:#fbf3e6;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg:#0e1116; --panel:#161a21; --ink:#e6e9ef; --muted:#9aa2b1;
    --faint:#6b7382; --line:#252b34; --line2:#1d222a;
    --accent:#3fb8ae; --accent-soft:#12302e; --human:#e8ab5a; --human-soft:#2a2113;
  }}
}}
:root[data-theme="dark"] {{
  --bg:#0e1116; --panel:#161a21; --ink:#e6e9ef; --muted:#9aa2b1;
  --faint:#6b7382; --line:#252b34; --line2:#1d222a;
  --accent:#3fb8ae; --accent-soft:#12302e; --human:#e8ab5a; --human-soft:#2a2113;
}}
:root[data-theme="light"] {{
  --bg:#f7f8fa; --panel:#ffffff; --ink:#161a22; --muted:#5c6472;
  --faint:#8a92a0; --line:#e4e7ec; --line2:#eef0f3;
  --accent:#0d7d76; --accent-soft:#e6f4f2; --human:#b45309; --human-soft:#fbf3e6;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font-family:var(--sans);
  line-height:1.5; -webkit-font-smoothing:antialiased; }}
.page {{ max-width:1180px; margin:0 auto; padding:40px 24px 72px; }}
header.top {{ border-bottom:1px solid var(--line); padding-bottom:20px; margin-bottom:8px; }}
h1 {{ font-size:1.5rem; font-weight:650; margin:0 0 6px; letter-spacing:-0.01em; }}
.lede {{ color:var(--muted); font-size:0.9rem; margin:0; max-width:70ch; }}
h2 {{ font-size:1.1rem; font-weight:620; margin:38px 0 12px; letter-spacing:-0.01em; }}
h2 .sub {{ color:var(--faint); font-weight:400; font-size:0.85rem; }}
.scroll {{ overflow-x:auto; border:1px solid var(--line); border-radius:10px;
  background:var(--panel); }}
table {{ border-collapse:collapse; width:100%; font-size:0.8rem;
  font-variant-numeric:tabular-nums; }}
thead th {{ position:sticky; top:0; background:var(--panel); }}
.grouprow th {{ font-size:0.72rem; font-weight:600; text-transform:uppercase;
  letter-spacing:0.04em; color:var(--muted); padding:10px 12px 6px;
  border-bottom:1px solid var(--line2); text-align:left; }}
.grouprow th.grp {{ text-align:center; color:var(--accent);
  border-left:1px solid var(--line2); }}
.grouprow th.corner, .grouprow th.ncol {{ vertical-align:bottom; }}
.metricrow th {{ font-size:0.74rem; font-weight:600; color:var(--ink);
  padding:6px 12px 10px; text-align:right; border-bottom:1px solid var(--line);
  white-space:nowrap; }}
.metricrow th .dir {{ display:block; font-size:0.64rem; font-weight:400;
  color:var(--faint); text-transform:none; letter-spacing:0; }}
tbody th.method {{ text-align:left; font-weight:550; padding:9px 14px 9px 16px;
  position:sticky; left:0; background:var(--panel); border-right:1px solid var(--line2);
  white-space:nowrap; z-index:1; }}
tbody th.method .nsub {{ display:block; font-size:0.66rem; font-weight:400;
  color:var(--faint); font-family:var(--mono); }}
td {{ padding:9px 12px; text-align:right; border-bottom:1px solid var(--line2);
  white-space:nowrap; }}
td.ncol {{ color:var(--muted); font-family:var(--mono); font-size:0.76rem; }}
.pt {{ display:block; font-family:var(--mono); font-weight:600; font-size:0.82rem; }}
.ci {{ display:block; font-family:var(--mono); font-size:0.66rem; color:var(--faint);
  margin-top:1px; }}
tbody tr:hover td, tbody tr:hover th.method {{ background:var(--line2); }}
tbody tr.our th.method {{ background:var(--accent-soft); border-left:3px solid var(--accent); }}
tbody tr.our:hover th.method {{ background:var(--accent-soft); }}
tbody tr.our td:last-child {{ }}
tbody tr.human th.method {{ background:var(--human-soft); border-left:3px solid var(--human); }}
tbody tr.human td {{ color:var(--muted); }}
.legend {{ margin-top:30px; font-size:0.78rem; color:var(--muted);
  border-top:1px solid var(--line); padding-top:16px; display:grid; gap:6px; }}
.legend b {{ color:var(--ink); font-weight:600; }}
.legend code {{ font-family:var(--mono); font-size:0.72rem; color:var(--accent); }}
.swatch {{ display:inline-block; width:10px; height:10px; border-radius:2px;
  vertical-align:middle; margin-right:5px; }}
</style>
<div class="page">
  <header class="top">
    <h1>Sycophancy mitigation — held-out results</h1>
    <p class="lede">Qwen3-8B methods evaluated on the fixed 1000-prompt OEQ and PAS
    held-out sets. Judge: GPT-5-mini (v5 endorse prompt / ELEPHANT + quality
    rubrics), reasoning=minimal, temp 0. Each cell is the point estimate with its
    95% CI. Teal rows are the stakeholder-DPO methods; the amber row is the human
    reference.</p>
  </header>
  {body}
  <div class="legend">
    <div><b>N</b> = responses judged (full held-out set). <b>n(0/1)</b> = the
    #0+#1 subset the endorsement rate is computed over (0=challenge, 1=explicit
    affirm; labels 2/3 excluded, per the paper metric).</div>
    <div><b>Endorsement</b> <code>#1/(#0+#1)</code> and the empathy means use
    mean ± 1.96·SEM; <b>Neutral</b> (=#3/N), <b>Addressee</b>, <b>Framing</b> and
    <b>Validation</b> are binary rates with 95% Wilson intervals.</div>
    <div><b>Directions</b>: lower <b>Endorsement</b> = less action-affirming;
    higher <b>Addressee</b> = treats the post as a text to critique rather than
    engaging the person; <b>Dismissing⁻¹</b> is reverse-coded so higher = less
    dismissive; <b>Framing</b> is the share that does <i>not</i> challenge the
    question's premise (higher = more sycophantic); <b>Validation</b> = share
    emotionally validating.</div>
    <div><span class="swatch" style="background:var(--accent)"></span>stakeholder-DPO (ours)
    &nbsp;·&nbsp; <span class="swatch" style="background:var(--human)"></span>human baseline (OEQ only).
    Tiny n(0/1) cells (Neutral DPO, Neutral-mixed DPO) give unstable endorsement CIs.</div>
  </div>
</div>"""
    out = os.path.join(_HERE, "results_artifact.html")
    with open(out, "w") as f:
        f.write(page)
    print(f"[wrote] {out}")


if __name__ == "__main__":
    main()
