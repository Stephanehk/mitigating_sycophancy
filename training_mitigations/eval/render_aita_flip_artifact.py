"""Render the AITA (FNR/FPR) and AITA-NTA-FLIP results tables into a
self-contained, theme-aware HTML page, matching the OEQ/PAS results artifact.
Reads sycoharm_runs/results_table_{aita,flip}.csv verbatim."""

import os
import re
import html

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_RUNS = os.path.join(_HERE, "..", "sycoharm_runs")
CELL = re.compile(r"([-\d.]+)\s*\[([-\d.]+),\s*([-\d.]+)\]")
OUR = {"mixed_P5", "mixed_P5_nocrit_distill_gp_frozenstake",
       "mixed_P6_perspshift_distill_gp"}


def cell(raw):
    m = CELL.search(str(raw))
    if not m:
        return f'<td>{html.escape(str(raw))}</td>'
    p, lo, hi = m.groups()
    return f'<td><span class="pt">{p}</span><span class="ci">{lo}, {hi}</span></td>'


def aita_rows(df):
    out = []
    for _, r in df.iterrows():
        cls = ' class="our"' if r["method"] in OUR else ""
        out.append(
            f'<tr{cls}><th scope="row" class="method">{html.escape(r["method"])}'
            f'<span class="nsub">YTA={int(r["n(YTA-truth)"])} · NTA={int(r["n(NTA-truth)"])}</span></th>'
            f'<td class="ncol">{int(r["N"])}</td>'
            f'<td class="ncol">{int(r["dropped"])}</td>'
            + cell(r["FNR (sycophancy)"]) + cell(r["FPR"]) + "</tr>")
    return "".join(out)


def flip_rows(df):
    out = []
    for _, r in df.iterrows():
        cls = ' class="our"' if r["method"] in OUR else ""
        out.append(
            f'<tr{cls}><th scope="row" class="method">{html.escape(r["method"])}</th>'
            f'<td class="ncol">{int(r["N (pairs)"])}</td>'
            f'<td class="ncol">{int(r["dropped"])}</td>'
            + cell(r["sycophancy (both-NTA)"]) + cell(r["critical (both-YTA)"]) + "</tr>")
    return "".join(out)


def main():
    aita = pd.read_csv(os.path.join(_RUNS, "results_table_aita.csv"))
    flip = pd.read_csv(os.path.join(_RUNS, "results_table_flip.csv"))

    page = f"""<title>AITA & FLIP — sycophancy held-out results</title>
<style>
:root {{
  --bg:#f7f8fa; --panel:#ffffff; --ink:#161a22; --muted:#5c6472; --faint:#8a92a0;
  --line:#e4e7ec; --line2:#eef0f3; --accent:#0d7d76; --accent-soft:#e6f4f2;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
}}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#0e1116; --panel:#161a21; --ink:#e6e9ef; --muted:#9aa2b1;
    --faint:#6b7382; --line:#252b34; --line2:#1d222a; --accent:#3fb8ae; --accent-soft:#12302e; }}
}}
:root[data-theme="dark"] {{ --bg:#0e1116; --panel:#161a21; --ink:#e6e9ef; --muted:#9aa2b1;
  --faint:#6b7382; --line:#252b34; --line2:#1d222a; --accent:#3fb8ae; --accent-soft:#12302e; }}
:root[data-theme="light"] {{ --bg:#f7f8fa; --panel:#ffffff; --ink:#161a22; --muted:#5c6472;
  --faint:#8a92a0; --line:#e4e7ec; --line2:#eef0f3; --accent:#0d7d76; --accent-soft:#e6f4f2; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font-family:var(--sans);
  line-height:1.5; -webkit-font-smoothing:antialiased; }}
.page {{ max-width:920px; margin:0 auto; padding:40px 24px 72px; }}
header.top {{ border-bottom:1px solid var(--line); padding-bottom:20px; }}
h1 {{ font-size:1.5rem; font-weight:650; margin:0 0 6px; letter-spacing:-0.01em; }}
.lede {{ color:var(--muted); font-size:0.9rem; margin:0; max-width:74ch; }}
h2 {{ font-size:1.1rem; font-weight:620; margin:38px 0 6px; letter-spacing:-0.01em; }}
.note {{ color:var(--faint); font-size:0.82rem; margin:0 0 12px; max-width:74ch; }}
.scroll {{ overflow-x:auto; border:1px solid var(--line); border-radius:10px; background:var(--panel); }}
table {{ border-collapse:collapse; width:100%; font-size:0.82rem; font-variant-numeric:tabular-nums; }}
thead th {{ position:sticky; top:0; background:var(--panel); font-size:0.75rem; font-weight:600;
  color:var(--ink); padding:10px 14px; text-align:right; border-bottom:1px solid var(--line);
  white-space:nowrap; }}
thead th .dir {{ display:block; font-size:0.66rem; font-weight:400; color:var(--faint); }}
thead th.method, thead th.ncol {{ text-align:left; }}
tbody th.method {{ text-align:left; font-weight:550; padding:9px 14px; position:sticky; left:0;
  background:var(--panel); border-right:1px solid var(--line2); white-space:nowrap; }}
tbody th.method .nsub {{ display:block; font-size:0.66rem; font-weight:400; color:var(--faint);
  font-family:var(--mono); }}
td {{ padding:9px 14px; text-align:right; border-bottom:1px solid var(--line2); white-space:nowrap; }}
td.ncol {{ color:var(--muted); font-family:var(--mono); font-size:0.76rem; text-align:left; }}
.pt {{ display:block; font-family:var(--mono); font-weight:600; }}
.ci {{ display:block; font-family:var(--mono); font-size:0.68rem; color:var(--faint); margin-top:1px; }}
tbody tr:hover td, tbody tr:hover th.method {{ background:var(--line2); }}
tbody tr.our th.method {{ background:var(--accent-soft); border-left:3px solid var(--accent); }}
tbody tr.our:hover th.method {{ background:var(--accent-soft); }}
.legend {{ margin-top:30px; font-size:0.78rem; color:var(--muted); border-top:1px solid var(--line);
  padding-top:16px; display:grid; gap:6px; max-width:78ch; }}
.legend b {{ color:var(--ink); font-weight:600; }}
.legend code {{ font-family:var(--mono); font-size:0.72rem; color:var(--accent); }}
</style>
<div class="page">
  <header class="top">
    <h1>AITA &amp; FLIP — binary-judgment sycophancy</h1>
    <p class="lede">Qwen3-8B methods on the AITA and AITA-NTA-FLIP held-out sets. The
    model free-forms a verdict and a parser extracts YTA / NTA. These are the same
    methods as the OEQ/PAS tables, restricted to those with eval data. Neutral DPO
    and Neutral-mixed DPO are absent by design (no neutral target for a binary
    label). Teal rows are the stakeholder-DPO methods.</p>
  </header>

  <h2>AITA — FNR / FPR</h2>
  <p class="note">1000 posts (≈500 true-YTA, ≈500 true-NTA). <b>FNR</b> = share of
  true-YTA posts the model lets off as NTA — the sycophancy axis (↑ = more
  sycophantic). <b>FPR</b> = share of true-NTA posts wrongly called YTA (blanket
  harshness). A low FNR bought by a high FPR is just always-YTA, not calibration.</p>
  <div class="scroll"><table>
    <thead><tr>
      <th class="method">Method</th><th class="ncol">N</th><th class="ncol">OTHER</th>
      <th>FNR<span class="dir">↑ more sycophantic</span></th>
      <th>FPR<span class="dir">↑ more harsh</span></th>
    </tr></thead>
    <tbody>{aita_rows(aita)}</tbody>
  </table></div>

  <h2>AITA-NTA-FLIP — POV-flip consistency</h2>
  <p class="note">1000 scenario pairs, each judged in its original and POV-flipped
  form; the flip is constructed so the non-sycophantic answer is NTA on both sides.
  FNR/FPR is undefined here (no per-item truth label), so the native metrics are
  shown: <b>sycophancy</b> = share of pairs answered NTA on <i>both</i> sides
  (always-nice; ↑ = more sycophantic); <b>critical</b> = share answered YTA on both
  (always-harsh).</p>
  <div class="scroll"><table>
    <thead><tr>
      <th class="method">Method</th><th class="ncol">N pairs</th><th class="ncol">dropped</th>
      <th>sycophancy<span class="dir">both-NTA · ↑ more sycophantic</span></th>
      <th>critical<span class="dir">both-YTA</span></th>
    </tr></thead>
    <tbody>{flip_rows(flip)}</tbody>
  </table></div>

  <div class="legend">
    <div>All rates carry a 95% <b>Wilson</b> interval over their own denominator
    (FNR over #true-YTA, FPR over #true-NTA, FLIP rates over #pairs).</div>
    <div><b>OTHER / dropped</b> = responses whose verdict did not parse to YTA or
    NTA; excluded from the rate denominators and reported for transparency.</div>
    <div>Metrics computed with the repo's own <code>compute_fnr_fpr</code> and
    <code>compute_aita_nta_flip_metrics</code> on the concatenated held-out shards.</div>
  </div>
</div>"""
    out = os.path.join(_HERE, "aita_flip_artifact.html")
    with open(out, "w") as f:
        f.write(page)
    print(f"[wrote] {out}")


if __name__ == "__main__":
    main()
