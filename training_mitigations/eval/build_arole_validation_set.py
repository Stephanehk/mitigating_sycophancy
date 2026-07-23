"""Build the labelled validation set for the assistant-role veto prompt.

WHY: ASSISTANT_ROLE_VETO_PROMPT (core/simulated_stakeholders.py) was written from
audit findings but never checked against data. Before spending Gemini budget on a
training arm, we need to know it fires on responses we ALREADY know abandon the
assistant role, and — just as important — that it does NOT fire on responses we
know are fine.

GROUND TRUTH comes from judges that already ran over the frozen `finaldata`
splits, so no new labelling is invented here:
  sub-mode (b) in-story voice  <- eval/classify_response_voice.py voice==STAKEHOLDER
  sub-mode (a) message-as-text <- eval/judge_addressee.py label==1
  sub-mode (c) third-person    <- base+perspshiftassess responses that are
                                  addressee==1 (its measured dominant mode)
  CONTROLS (must NOT fire)     <- voice==OUTSIDER AND addressee==0

The controls are the point of the design: a judge that answers "1" to everything
scores 100% on positives alone, so positives-only validation cannot distinguish a
working classifier from a broken one. Controls are deliberately drawn from the
HARSH arms (P6, be-critical) as well as base, because the specific false-positive
risk in this prompt is reading blunt second-person criticism, or an assistant
saying "I", as role abandonment.

Sampling is seeded (SEED) and the chosen ids are written out, so the set is fixed
and re-runnable rather than re-drawn per invocation.

Output: eval/arole_validation_set.csv with columns
  id, env, arm, sentence, response, expected, submode, source
where `expected` is 1 (should be auto-rejected) or 0 (must not be).
"""

import argparse
import os
import random

import pandas as pd

RUNS = "/projects/m000254/runs"
JUDGES = os.path.join(os.path.dirname(__file__), "..",
                      "sycoharm_runs", "judges_finaldata")
HERE = os.path.dirname(__file__)

SEED = 20260720
N_POSITIVE = 30
N_CONTROL = 15

# Run dirs + their response-file shard counts, per arm.
P6_PAS = "dpo_lora_pas_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_final"
BASE_PAS = "baseline_final_PAS"

# The known voice-gate FALSE NEGATIVE: P6 claimed first-hand presence ("what i
# saw", "not aligned with what i observed") yet classify_response_voice.py scored
# it OUTSIDER. Force-included as the hardest positive — if the new prompt catches
# it, it is already beating the gate it replaces.
VOICE_GATE_FALSE_NEGATIVE = "pas_443b80b2"


def load_responses(run_dir, tag, shards):
    """Concatenate an arm's sharded response CSVs in shard order.

    Assumes shard files are 0..shards-1 and that row order within the
    concatenation matches the `row` index used by voice_*.csv (both are built by
    concatenating the same shards in the same order).
    """
    parts = []
    for i in range(shards):
        p = os.path.join(RUNS, run_dir, "eval",
                         f"responses_{tag}_shard{i}of{shards}.csv")
        assert os.path.exists(p), f"missing shard: {p}"
        parts.append(pd.read_csv(p))
    return pd.concat(parts, ignore_index=True)


def load_voice(run_dir, tag):
    p = os.path.join(RUNS, run_dir, "eval", f"voice_{tag}.csv")
    assert os.path.exists(p), f"missing voice labels: {p}"
    v = pd.read_csv(p)
    assert set(v.columns) >= {"row", "voice"}, v.columns
    return v.set_index("row")["voice"]


def load_addressee(env, arm):
    """Addressee judge output for (env, arm). Keyed by `sentence`, since the
    judge CSVs carry the post text rather than the id."""
    p = os.path.join(JUDGES, env, f"addressee_{arm}.csv")
    assert os.path.exists(p), f"missing addressee labels: {p}"
    a = pd.read_csv(p)
    col = "addressee_response"
    assert col in a.columns, a.columns
    return a[["sentence", col]].rename(columns={col: "addressee"})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-positive", type=int, default=N_POSITIVE)
    ap.add_argument("--n-control", type=int, default=N_CONTROL)
    ap.add_argument("--exclude-from", default=None,
                    help="CSV whose `response` texts must NOT reappear here — "
                         "use to build a set DISJOINT from an earlier one, so a "
                         "prompt tuned on that one can be scored honestly.")
    ap.add_argument("--no-force-false-negative", action="store_true",
                    help="omit the forced pas_443b80b2 hard case (it belongs to "
                         "the first set; a held-out set should not reuse it)")
    ap.add_argument("--output", default=os.path.join(HERE, "arole_validation_set.csv"))
    args = ap.parse_args()

    seen = set()
    if args.exclude_from:
        prev = pd.read_csv(args.exclude_from)
        seen = set(prev["response"].astype(str))
        print(f"[excl] {len(seen)} responses excluded from {args.exclude_from}")

    rng = random.Random(args.seed)
    SEED_LOCAL = args.seed
    rows = []

    # --- sub-mode (b): in-story voice, from P6 PAS -------------------------
    tag = "final_finaldata_temp1.0_seed0"
    p6 = load_responses(P6_PAS, tag, 2)
    voice = load_voice(P6_PAS, tag)
    p6 = p6.join(voice.rename("voice"))
    assert p6["voice"].notna().all(), "voice labels must cover every response row"
    stake = p6[(p6.voice == "STAKEHOLDER") & (~p6.response.astype(str).isin(seen))]
    print(f"[b] P6 PAS STAKEHOLDER pool: {len(stake)}")

    # --- sub-mode (a): message-as-text -------------------------------------
    # PAS P6 has only 8 addressee==1, so pool across the arms where the mode was
    # actually measured (be-critical 15.5%, perspshiftassess 12.1%).
    addr_pool = []
    for env in ("PAS", "OEQ"):
        for arm in ("becritical", "perspshiftassess", "base", "P6perspshift_a05"):
            p = os.path.join(JUDGES, env, f"addressee_{arm}.csv")
            if not os.path.exists(p):
                continue
            a = pd.read_csv(p)
            hit = a[(a["addressee_response"] == 1)
                    & (~a["response"].astype(str).isin(seen))]
            for _, r in hit.iterrows():
                addr_pool.append({"env": env, "arm": arm,
                                  "sentence": r["sentence"],
                                  "response": r["response"]})
    print(f"[a] addressee==1 pool: {len(addr_pool)}")

    # --- sub-mode (c): third-person case-study -----------------------------
    # perspshiftassess responses that the addressee judge did NOT flag: those are
    # the ones whose failure is the third-person "the narrative/the author"
    # framing rather than explicit writing critique, i.e. exactly sub-mode (c)
    # and NOT already covered by pool (a).
    psa = load_addressee("PAS", "perspshiftassess")
    third = psa[(psa.addressee == 0)]
    psa_resp = pd.read_csv(os.path.join(JUDGES, "PAS", "addressee_perspshiftassess.csv"))
    marker = psa_resp["response"].str.contains(
        r"\bthe (narrative|author|narrator|passage|writer)\b", case=False, na=False)
    never_you = ~psa_resp["response"].str.contains(r"\byou\b", case=False, na=False)
    c_pool = psa_resp[marker & never_you & (psa_resp["addressee_response"] == 0)
                      & (~psa_resp["response"].astype(str).isin(seen))]
    print(f"[c] perspshiftassess third-person (not already addressee==1): {len(c_pool)}")

    # --- assemble 30 positives ---------------------------------------------
    # Weighted toward (b): it is the dominant measured mode in P6 (26.7%) and the
    # one this arm exists to remove.
    forced = 0 if args.no_force_false_negative else 1
    n_free = args.n_positive - forced
    n_b = round(n_free * 15 / 29); n_a = round(n_free * 8 / 29)
    n_c = n_free - n_b - n_a
    b_idx = rng.sample(list(stake.index), n_b)
    for i in b_idx:
        r = p6.loc[i]
        rows.append({"id": r["id"], "env": "PAS", "arm": "P6perspshift_a05",
                     "sentence": r["sentence"], "response": r["response"],
                     "expected": 1, "submode": "b_in_story_voice",
                     "source": "voice==STAKEHOLDER"})
    for r in rng.sample(addr_pool, n_a):
        rows.append({"id": "", "env": r["env"], "arm": r["arm"],
                     "sentence": r["sentence"], "response": r["response"],
                     "expected": 1, "submode": "a_message_as_text",
                     "source": "addressee==1"})
    for _, r in c_pool.sample(n=n_c, random_state=SEED_LOCAL).iterrows():
        rows.append({"id": "", "env": "PAS", "arm": "perspshiftassess",
                     "sentence": r["sentence"], "response": r["response"],
                     "expected": 1, "submode": "c_third_person",
                     "source": "psa third-person, addressee==0"})

    fn = p6[p6.id == VOICE_GATE_FALSE_NEGATIVE] if not args.no_force_false_negative else p6.iloc[:0]
    assert args.no_force_false_negative or len(fn) == 1, f"{VOICE_GATE_FALSE_NEGATIVE} not found in P6 PAS"
    for _, r in fn.iterrows():
         rows.append({"id": r["id"], "env": "PAS", "arm": "P6perspshift_a05",
                     "sentence": r["sentence"], "response": r["response"],
                     "expected": 1, "submode": "b_in_story_voice",
                     "source": "KNOWN voice-gate FALSE NEGATIVE (scored OUTSIDER)"})

    n_pos = sum(1 for x in rows if x["expected"] == 1)
    assert n_pos == args.n_positive, f"expected {args.n_positive}, built {n_pos}"

    # --- controls: must NOT fire -------------------------------------------
    # Drawn from P6 (harsh, blunt second person) and base. These are the
    # false-positive traps: a pronoun- or harshness-detector fails here.
    ok = p6[p6.voice == "OUTSIDER"]
    p6_addr = load_addressee("PAS", "P6perspshift_a05")
    clean_sent = set(p6_addr[p6_addr.addressee == 0]["sentence"])
    ok = ok[ok.sentence.isin(clean_sent) & (~ok.response.astype(str).isin(seen))]
    print(f"[ctrl] P6 OUTSIDER & addressee==0 pool: {len(ok)}")
    for i in rng.sample(list(ok.index), args.n_control):
        r = ok.loc[i]
        rows.append({"id": r["id"], "env": "PAS", "arm": "P6perspshift_a05",
                     "sentence": r["sentence"], "response": r["response"],
                     "expected": 0, "submode": "control",
                     "source": "voice==OUTSIDER & addressee==0"})

    out = pd.DataFrame(rows)
    assert out.response.notna().all() and (out.response.str.len() > 0).all()
    out.to_csv(args.output, index=False)
    print(f"\nwrote {args.output}  rows={len(out)}")
    print(out.groupby(["expected", "submode"]).size())


if __name__ == "__main__":
    main()
