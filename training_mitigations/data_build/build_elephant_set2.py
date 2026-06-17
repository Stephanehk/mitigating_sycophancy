"""Build a SECOND disjoint 500-prompt ELEPHANT sample (set2) for tables 1/2.

The original sycoharm inputs use df.sample(500, random_state=0). We take the
SAME rs=0 permutation and walk positions from 500 onward, skipping any row whose
prompt text already appears in the original 500 (the OSF frames contain a few
duplicate prompt strings, so positional-disjointness is not text-disjointness),
until 500 fresh prompts are collected. The deepseek/gpt4o frames per env are
row-aligned (same length, same prompt order), so one position list keys both ->
the gpt4o and deepseek samples stay matched on prompt.

Writes (sentence, response): inputs/{pas2,oeq2}_{gpt4o,deepseek}.csv
and Myra's stored endorse labels: sycoharm_runs/myra_set2_labels.csv.
"""

import os
import re
import pandas as pd

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(_HERE)
OSF = os.path.join(_ROOT, "osfstorage-archive", "Data")
INP = os.path.join(_HERE, "sycoharm_runs", "inputs")


def first_label(cell):
    m = re.search(r"[0-3]", str(cell))
    return int(m.group(0)) if m else None


def fresh_positions(prompts, orig_set, want=500):
    """Walk the rs=0 permutation from position 500 on, returning the first
    `want` positions whose prompt text is not in orig_set."""
    perm = pd.Series(range(len(prompts))).sample(frac=1, random_state=0).tolist()
    pvals = prompts.values
    out = []
    for pos in perm[500:]:
        if pvals[pos] not in orig_set:
            out.append(pos)
            if len(out) == want:
                break
    assert len(out) == want, f"only found {len(out)} fresh prompts"
    return out


def save_input(name, sentence, response):
    out = pd.DataFrame({"sentence": list(sentence), "response": list(response)})
    assert len(out) == 500 and out.notna().all().all(), name
    orig = os.path.join(INP, f"{name.replace('2_', '_', 1)}.csv")
    prev = set(pd.read_csv(orig)["sentence"])
    assert len(set(out["sentence"]) & prev) == 0, f"{name}: overlaps original"
    out.to_csv(os.path.join(INP, f"{name}.csv"), index=False)
    print(f"wrote inputs/{name}.csv (500 rows, disjoint)")


labels = []
def collect(dataset, sentence, endorse):
    for s, e in zip(sentence, endorse):
        labels.append({"dataset": dataset, "sentence": s, "myra": first_label(e)})


# --- PAS: deepseek frame keys positions; gpt4o frame is row-aligned ---
d = pd.read_csv(os.path.join(OSF, "PAS_results", "endorse4_elephant_oeq_pas_deepseek.csv"))
g = pd.read_csv(os.path.join(OSF, "PAS_results", "endorse4_elephant_oeq_pas_gpt4o.csv"))
assert len(d) == len(g) and (d["sentences"].values == g["sentences"].values).all(), "PAS frames misaligned"
orig_pas = set(pd.read_csv(os.path.join(INP, "pas_deepseek.csv"))["sentence"])
pos = fresh_positions(d["sentences"], orig_pas)
save_input("pas2_deepseek", d["sentences"].iloc[pos], d["response"].iloc[pos])
save_input("pas2_gpt4o",   g["sentences"].iloc[pos], g["OEQ_sentences"].iloc[pos])
collect("pas_deepseek", d["sentences"].iloc[pos], d["endorse_response"].iloc[pos])
collect("pas_gpt4o",    g["sentences"].iloc[pos], g["endorse_OEQ_sentences"].iloc[pos])

# --- OEQ: deepseek frame has prompt; gpt4o frame row-aligned (both 3027) ---
d3 = pd.read_csv(os.path.join(OSF, "OEQ_results", "endorse4_elephant_oeq3k_deepseek.csv"))
g3 = pd.read_csv(os.path.join(OSF, "OEQ_results", "endorse4_elephant_oeq3k_gpt4o.csv"))
assert len(d3) == len(g3), "OEQ frames misaligned"
orig_oeq = set(pd.read_csv(os.path.join(INP, "oeq_deepseek.csv"))["sentence"])
pos = fresh_positions(d3["prompt"], orig_oeq)
save_input("oeq2_deepseek", d3["prompt"].iloc[pos], d3["deepseek"].iloc[pos])
save_input("oeq2_gpt4o",    d3["prompt"].iloc[pos], g3["GPT-4o"].iloc[pos])
collect("oeq_deepseek", d3["prompt"].iloc[pos], d3["endorse_deepseek"].iloc[pos])
collect("oeq_gpt4o",    d3["prompt"].iloc[pos], g3["endorse_GPT-4o"].iloc[pos])

pd.DataFrame(labels).to_csv(os.path.join(_HERE, "sycoharm_runs", "myra_set2_labels.csv"), index=False)
print(f"wrote sycoharm_runs/myra_set2_labels.csv ({len(labels)} rows)")
print(pd.DataFrame(labels).groupby("dataset")["myra"].apply(lambda x: f'{x.notna().sum()}/{len(x)} labeled').to_string())
