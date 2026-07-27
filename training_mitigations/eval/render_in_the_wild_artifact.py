"""Render an in-the-wild Study-3 dump as a side-by-side comparison page: each
participant prompt, then the response from each condition in its own column.

Two condition sets are available via --set:
  8b   the original four-way 8B page (untrained / “Be critical.” / neutral-mixed
       DPO / Our Model (3+1))
  32b  the two-way 32B page (untrained Qwen3-32B / Our Model (3+1) at 32B)

Reads the CSVs written by in_the_wild_response_analysis.py VERBATIM and asserts
the chosen set shares an identical prompt set (same utterances, same order), so
a row on the page is genuinely the same prompt across all columns. Writes
in_the_wild_artifact[_<set>].html next to this script.

The page deliberately shows nothing but the prompts and the responses — no
metrics, metadata, or commentary — so the comparison is read directly off the
text.

Assumptions (asserted): every CSV exists, carries `utterance` and `response`,
and the prompt sets match exactly.
"""

import argparse
import html
import os
import re

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ARMS = "/projects/m000254/runs/in_the_wild"
_VOICE2 = ("/projects/m000254/runs/dpo_lora_oeq_qwen3-8b_P6_voice2_rpo0p5/eval/"
           "in_the_wild_study3_ourmodel3plus1_oeq_temp1.0_seed0.csv")

# Column order in both sets: untrained first, ours last.
CONDITION_SETS = {
    "8b": [
        ("Qwen3-8B", "untrained",
         os.path.join(_ARMS, "in_the_wild_study3_base_temp1.0_seed0.csv")),
        ("Qwen3-8B", "“Be critical.”",
         os.path.join(_ARMS, "in_the_wild_study3_base-becritical_temp1.0_seed0.csv")),
        ("Neutral-mixed", "DPO",
         os.path.join(_ARMS, "in_the_wild_study3_neutralmixed_temp1.0_seed0.csv")),
        ("Our Model (3+1)", "OEQ", _VOICE2),
    ],
    "32b": [
        ("Qwen3-32B", "untrained",
         os.path.join(_ARMS, "in_the_wild_study3_base_32b_temp1.0_seed0.csv")),
        ("Our Model (3+1)", "32B / OEQ",
         os.path.join(_ARMS,
                      "in_the_wild_study3_ourmodel3plus1_oeq_32b_temp1.0_seed0.csv")),
    ],
}

TITLES = {
    "8b": "Four models, same prompts",
    "32b": "Qwen3-32B, trained and untrained, same prompts",
}
HEADLINES = {
    "8b": "Four models answering the same {n} people",
    "32b": "Qwen3-32B before and after our training, answering the same {n} people",
}


# --- markdown -> html ------------------------------------------------------
def _inline(text):
    """Escaped text -> inline markup. Bold before italic so **x** is not eaten
    by the single-asterisk rule; `code` first so markup inside it is left be."""
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*(?!\s)([^*]+?)(?<!\s)\*(?!\*)", r"<em>\1</em>", text)
    return text


def md_to_html(md):
    """Convert the subset of markdown these responses actually use: ## - ####
    headers, --- rules, - / * bullets (one nesting level), 1. ordered lists,
    **bold**, *italic*, `code`, blank-line-separated paragraphs. Verified
    against the corpus: no code fences, no tables."""
    # `stack` holds the open list tags; `li_open[i]` says whether an <li> is
    # open at that level. A nested list must sit INSIDE its parent's <li>.
    out, stack, li_open = [], [], []

    def close_lists(to_depth=0):
        while len(stack) > to_depth:
            if li_open[-1]:
                out.append("</li>")
            li_open.pop()
            out.append(f"</{stack.pop()}>")

    for raw in md.split("\n"):
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            close_lists()
            continue
        if re.match(r"^(-{3,}|\*{3,}|_{3,})$", stripped):
            close_lists()
            out.append("<hr />")
            continue
        h = re.match(r"^(#{1,6}) +(.*)$", stripped)
        if h:
            close_lists()
            level = min(len(h.group(1)) + 1, 6)
            out.append(f"<h{level}>{_inline(html.escape(h.group(2)))}</h{level}>")
            continue
        b = re.match(r"^([ \t]*)[-*] +(.*)$", line)
        o = re.match(r"^([ \t]*)\d+[.)] +(.*)$", line)
        m = b or o
        if m:
            depth = 1 + (1 if len(m.group(1).expandtabs(4)) >= 2 else 0)
            # Never skip a level: an indented bullet with no parent item above
            # would open a <ul> directly inside a <ul>.
            depth = min(depth, len(stack) + 1)
            tag = "ul" if b else "ol"
            close_lists(depth)
            if stack and stack[-1] != tag and len(stack) == depth:
                close_lists(depth - 1)
            while len(stack) < depth:
                out.append(f"<{tag}>")
                stack.append(tag)
                li_open.append(False)
            if li_open[-1]:
                out.append("</li>")
            out.append(f"<li>{_inline(html.escape(m.group(2)))}")
            li_open[-1] = True
            continue
        close_lists()
        out.append(f"<p>{_inline(html.escape(stripped))}</p>")
    close_lists()
    return "\n".join(out)


# --- data ------------------------------------------------------------------
def load_conditions(conditions):
    """Load the frames and assert they cover an identical prompt set in an
    identical order. Returns (utterances, [(name, sub, [response, ...]), ...])."""
    frames = []
    for name, sub, path in conditions:
        assert os.path.exists(path), f"missing condition CSV: {path}"
        df = pd.read_csv(path)
        for col in ("utterance", "response"):
            assert col in df.columns, f"{path} has no {col!r} column"
        frames.append((name, sub, df))

    ref = frames[0][2]["utterance"].tolist()
    for name, sub, df in frames[1:]:
        got = df["utterance"].tolist()
        assert got == ref, (
            f"prompt set for {name} / {sub} differs from {frames[0][0]} "
            f"({len(got)} vs {len(ref)} rows) — the columns would not be "
            f"comparable row by row"
        )
    cols = [(n, s, df["response"].astype(str).tolist()) for n, s, df in frames]
    return ref, cols


def entry_html(i, utterance, cols):
    cells = []
    for name, sub, responses in cols:
        cells.append(
            f'<div class="cell">'
            f'<h2 class="who"><b>{html.escape(name)}</b>'
            f'<span>{html.escape(sub)}</span></h2>'
            f'<div class="say">{md_to_html(responses[i])}</div>'
            f'</div>'
        )
    return f"""<article class="entry">
  <div class="ask"><span class="n">{i + 1}</span>
    <blockquote>{html.escape(utterance)}</blockquote></div>
  <div class="grid">{''.join(cells)}</div>
</article>"""


CSS = """
:root{
  --ground:#F3F5F3; --panel:#E7EAE6; --ink:#161A18; --muted:#5C6763;
  --rule:#D6DCD8; --accent:#35695A;
  --serif: ui-serif, "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
}
@media (prefers-color-scheme: dark){
  :root{--ground:#0F1312; --panel:#191F1C; --ink:#E6EBE8; --muted:#8E9A95;
        --rule:#28302C; --accent:#84C4A8;}
}
:root[data-theme="dark"]{--ground:#0F1312; --panel:#191F1C; --ink:#E6EBE8;
  --muted:#8E9A95; --rule:#28302C; --accent:#84C4A8;}
:root[data-theme="light"]{--ground:#F3F5F3; --panel:#E7EAE6; --ink:#161A18;
  --muted:#5C6763; --rule:#D6DCD8; --accent:#35695A;}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);
  font-size:16px;line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:var(--wrap);margin:0 auto;padding:0 28px}

.mast{padding:44px 0 22px;border-bottom:1px solid var(--rule)}
.mast h1{font-family:var(--serif);font-weight:600;font-size:clamp(26px,3.2vw,38px);
  line-height:1.1;letter-spacing:-.015em;margin:0;text-wrap:balance;max-width:24ch}

.entry{padding:44px 0;border-bottom:1px solid var(--rule)}
.ask{display:flex;gap:16px;align-items:flex-start;margin-bottom:28px}
.n{font-family:var(--serif);font-size:26px;line-height:1.2;color:var(--muted);
  font-variant-numeric:tabular-nums;flex:0 0 auto;min-width:2ch;text-align:right}
.ask blockquote{margin:0;flex:1 1 auto;background:var(--panel);
  border-left:2px solid var(--accent);padding:16px 20px;font-family:var(--mono);
  font-size:13.5px;line-height:1.72;white-space:pre-wrap;overflow-wrap:break-word;
  max-width:110ch}

.grid{display:grid;grid-template-columns:repeat(var(--cols),minmax(0,1fr));gap:0}
.cell{min-width:0;padding:0 22px;border-left:1px solid var(--rule)}
.cell:first-child{padding-left:0;border-left:0}
.cell:last-child{padding-right:0}
.who{margin:0 0 14px;font-family:var(--mono);font-weight:400;font-size:10.5px;
  letter-spacing:.1em;text-transform:uppercase;display:flex;flex-direction:column;gap:2px}
.who b{font-weight:600;color:var(--accent);letter-spacing:.08em}
.who span{color:var(--muted)}
.say{font-size:14px;line-height:1.62;overflow-wrap:break-word}
.say > :first-child{margin-top:0}
.say h3,.say h4,.say h5,.say h6{font-family:var(--serif);font-weight:600;
  line-height:1.25;margin:18px 0 7px;letter-spacing:-.005em;text-wrap:balance}
.say h3{font-size:16px}
.say h4{font-size:14.5px}
.say h5,.say h6{font-size:14px}
.say p{margin:0 0 11px}
.say ul,.say ol{margin:0 0 12px;padding-left:18px}
.say li{margin-bottom:5px}
.say li > ul,.say li > ol{margin:5px 0 0}
.say hr{border:0;border-top:1px solid var(--rule);margin:16px 0}
.say strong{font-weight:650}
.say code{font-family:var(--mono);font-size:.88em;background:var(--panel);
  padding:1px 4px;border-radius:2px}

@media (max-width:1180px){
  .grid{grid-template-columns:repeat(2,minmax(0,1fr));gap:28px 0}
  .cell:nth-child(odd){padding-left:0;border-left:0}
  .cell:nth-child(even){padding-right:0}
}
@media (max-width:700px){
  .grid{grid-template-columns:1fr;gap:26px}
  .cell{padding:0;border-left:0;border-top:1px solid var(--rule);padding-top:20px}
  .cell:first-child{border-top:0;padding-top:0}
  .ask{gap:10px}
}
"""


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="set_name", default="8b",
                    choices=sorted(CONDITION_SETS),
                    help="which condition set to render (default: 8b)")
    ap.add_argument("--out", default=None,
                    help="output HTML (default: in_the_wild_artifact[_<set>].html "
                         "next to this script)")
    args = ap.parse_args()

    utterances, cols = load_conditions(CONDITION_SETS[args.set_name])
    body = "\n".join(entry_html(i, u, cols) for i, u in enumerate(utterances))
    # Wide four-up needs the full 1640px; two-up reads better at a tighter
    # measure than half of it.
    wrap = "1640px" if len(cols) > 2 else "1180px"
    page = f"""<title>{html.escape(TITLES[args.set_name])}</title>
<style>:root{{--cols:{len(cols)};--wrap:{wrap}}}
{CSS}</style>
<header class="mast"><div class="wrap">
  <h1>{html.escape(HEADLINES[args.set_name].format(n=len(utterances)))}</h1>
</div></header>
<main class="wrap">
{body}
</main>
"""
    suffix = "" if args.set_name == "8b" else f"_{args.set_name}"
    out = args.out or os.path.join(_HERE, f"in_the_wild_artifact{suffix}.html")
    with open(out, "w") as f:
        f.write(page)
    print(f"[render] {len(utterances)} prompts x {len(cols)} conditions -> {out}")
    for name, sub, responses in cols:
        med = int(pd.Series([len(r.split()) for r in responses]).median())
        print(f"[render]   {name} / {sub}: {len(responses)} responses, "
              f"median {med} words")


if __name__ == "__main__":
    main()
