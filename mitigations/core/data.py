"""Dataset loaders and samplers for the five envs (AITA, OEQ, PAS,
AITA-NTA-FLIP, OEQ_2), plus balanced/uniform/explicit-id sampling. Each
loader takes a path and returns a normalized DataFrame with a `post_text`
column and a stable `id`."""

import hashlib

import pandas as pd


def load_aita_clean(path):
    """Load data/aita_clean.csv. Returns DataFrame[id, title, body, verdict, is_asshole, post_text].

    `post_text` is built as `title\\n\\nbody`. Drops rows where any of
    title/body/verdict/is_asshole are null.

    Filters to rows where the community verdict is unambiguously YTA or NTA
    (i.e. `verdict in ('asshole', 'not the asshole')`). The dataset's binary
    `is_asshole` label collapses 'everyone sucks' (ESH) and 'no a-holes here'
    (NAH) into 1 and 0 respectively, but those verdicts are not clean YTA/NTA
    judgments — the model can be "right" while disagreeing with the binary
    label. Restricting to clean verdicts keeps FNR/FPR interpretable as a
    sycophancy signal rather than a label-noise signal.

    Asserts that within the filtered set, verdict='asshole' aligns with
    is_asshole==1 (and 'not the asshole' with 0) so a label flip in upstream
    data fails fast instead of silently corrupting metrics.
    """
    df = pd.read_csv(path, usecols=["id", "title", "body", "verdict", "is_asshole"])
    df = df.dropna(subset=["title", "body", "verdict", "is_asshole"]).copy()
    df = df[df["verdict"].isin(("asshole", "not the asshole"))].copy()
    assert set(df["is_asshole"].unique()) == {0, 1}, \
        "is_asshole must contain only 0/1"
    assert ((df["verdict"] == "asshole") == (df["is_asshole"] == 1)).all(), \
        "verdict 'asshole'/'not the asshole' must align with is_asshole 1/0"
    df["post_text"] = df["title"].astype(str).str.strip() + "\n\n" + df["body"].astype(str).str.strip()
    return df


def load_oeq(path):
    """Load an OSF OEQ_results CSV. Returns DataFrame[id, post_text, human_response].

    Reads only `sentence` (the OEQ prompt) and `Human` (the paired advice
    columnist / top Reddit comment). Dedupes on `sentence` since the OSF set
    has 3,027 rows / 3,011 unique prompts (a few cross-posted Reddit items).
    Generates a stable id as `oeq_<8-char md5 of sentence>` so
    --use_specific_posts works.

    Assumptions:
      - `sentence` and `Human` columns exist (true for every per-model
        OSF OEQ CSV by construction).
      - `Human` is non-null on the rows we care about. We assert this
        rather than dropping silently because the human_reference baseline
        depends on it.
    """
    df = pd.read_csv(path, usecols=["sentence", "Human"])
    assert "sentence" in df.columns, "OEQ csv missing 'sentence' column"
    assert "Human" in df.columns, "OEQ csv missing 'Human' column"
    df = df.dropna(subset=["sentence", "Human"]).copy()
    df = df.drop_duplicates(subset=["sentence"], keep="first").reset_index(drop=True)
    df["post_text"] = df["sentence"].astype(str).str.strip()
    df["human_response"] = df["Human"].astype(str).str.strip()
    df["id"] = df["sentence"].apply(
        lambda s: "oeq_" + hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
    )
    assert df["id"].is_unique, "OEQ id collisions; bump hash length"
    return df[["id", "post_text", "human_response"]]


def load_pas(path):
    """Load an OSF PAS_results CSV. Returns DataFrame[id, post_text].

    Reads only `sentences` (the PAS prompt — a single extracted problematic
    first-person sentence) and `problematic_to_affirm` (a 0/1 ground-truth
    flag from the paper's GPT-4o filter step). The released PAS subset is
    pre-filtered to `problematic_to_affirm==1`; we assert that rather than
    re-filter so any future PAS file shipped with mixed labels fails loud.

    Dedupes on `sentences` (the file has 6,433 rows / ~6,395 unique sentences;
    a small number of duplicates carry through the upstream pipeline). The id
    scheme `pas_<8-char md5(sentences)>` mirrors `load_oeq` so
    `--use_specific_posts` works.

    No `human_response` column is set: PAS prompts are isolated single
    sentences with no paired advice-columnist / top-comment response, so the
    `human_reference` condition is not available for PAS.

    Assumptions:
      - `sentences` and `problematic_to_affirm` columns exist (true for every
        per-model OSF PAS CSV by construction; the per-model files differ
        only in the model `response` column, which we ignore).
      - Every retained row has `problematic_to_affirm == 1`. The OSF release
        ships only the problematic subset; any other value indicates a
        different PAS file or a pipeline change.
    """
    df = pd.read_csv(path, usecols=["sentences", "problematic_to_affirm"])
    assert "sentences" in df.columns, "PAS csv missing 'sentences' column"
    assert "problematic_to_affirm" in df.columns, \
        "PAS csv missing 'problematic_to_affirm' column"
    df = df.dropna(subset=["sentences", "problematic_to_affirm"]).copy()
    unique_flags = set(df["problematic_to_affirm"].astype(int).unique())
    assert unique_flags == {1}, (
        f"PAS expects all rows to have problematic_to_affirm==1 (the OSF "
        f"release ships only the problematic subset); got values {unique_flags}"
    )
    df = df.drop_duplicates(subset=["sentences"], keep="first").reset_index(drop=True)
    df["post_text"] = df["sentences"].astype(str).str.strip()
    df["id"] = df["sentences"].apply(
        lambda s: "pas_" + hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
    )
    assert df["id"].is_unique, "PAS id collisions; bump hash length"
    return df[["id", "post_text"]]


def load_aita_nta_flip(path):
    """Load the AITA-NTA-FLIP CSV. Returns DataFrame[id, post_text,
    flipped_post_text].

    Each row carries two post variants: `original_post` (the canonical Reddit
    AITA post) and `flipped_story` (the same scenario rewritten from the
    other party's POV so the YTA/NTA verdict should flip). We rename
    `original_post`→`post_text` so the per-task plumbing in `main` and
    `run_one` can share the standard `post_text` field with the other envs;
    `flipped_post_text` is the FLIP-only extra.

    Assumptions:
      - `id`, `original_post`, `flipped_story` columns exist.
      - `id` values are non-null and unique.
    """
    df = pd.read_csv(path)
    for col in ("id", "original_post", "flipped_story"):
        assert col in df.columns, f"AITA-NTA-FLIP csv missing column: {col!r}"
    df = df.dropna(subset=["id", "original_post", "flipped_story"]).copy()
    df["id"] = df["id"].astype(str).str.strip()
    df["post_text"] = df["original_post"].astype(str).str.strip()
    df["flipped_post_text"] = df["flipped_story"].astype(str).str.strip()
    assert df["id"].is_unique, "AITA-NTA-FLIP id collisions in source CSV"
    return df[["id", "post_text", "flipped_post_text"]]


def load_oeq_2(path):
    """Load the OEQ_2 dataset (ELEPHANT_datasets/OEQ.csv).

    Returns DataFrame[id, post_text, validation_human, indirectness_human,
    framing_human]. The `human` (top-comment/columnist response) column is
    deliberately NOT loaded — OEQ_2's scoring uses only the precomputed
    binary human labels for comparison against the LLM-judge labels; the
    human response text is never used.

    Rows with NaN in any of the three human-label columns are dropped
    (~12 rows / 0.4% of the dataset) so downstream score arithmetic does
    not have to special-case missing labels. After dropna the three label
    columns are cast to int and asserted to be in {0, 1}.

    The `id` scheme `oeq2_<8-char md5(prompt)>` mirrors `load_oeq` so
    `--use_specific_posts` works the same way.

    Assumptions:
      - `prompt`, `validation_human`, `indirectness_human`, `framing_human`
        columns exist (verified at load).
      - Surviving label values are in {0, 1}.
    """
    df = pd.read_csv(path, usecols=[
        "prompt", "validation_human", "indirectness_human", "framing_human",
    ])
    for col in ("prompt", "validation_human", "indirectness_human",
                "framing_human"):
        assert col in df.columns, f"OEQ_2 csv missing {col!r} column"
    df = df.dropna(subset=[
        "prompt", "validation_human", "indirectness_human", "framing_human",
    ]).copy()
    df = df.drop_duplicates(subset=["prompt"], keep="first").reset_index(drop=True)
    df["post_text"] = df["prompt"].astype(str).str.strip()
    for col in ("validation_human", "indirectness_human", "framing_human"):
        df[col] = df[col].astype(int)
        unique = set(df[col].unique())
        assert unique.issubset({0, 1}), (
            f"OEQ_2: unexpected {col} values after dropna: {sorted(unique)}"
        )
    df["id"] = df["post_text"].apply(
        lambda s: "oeq2_" + hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
    )
    assert df["id"].is_unique, "OEQ_2 id collisions; bump hash length"
    return df[[
        "id", "post_text",
        "validation_human", "indirectness_human", "framing_human",
    ]]


def select_specific_posts(data, ids):
    """Select rows from `data` by post id, in the order given by `ids`.

    Used by the --use_specific_posts flag to run the mitigation set on a
    curated list of posts (e.g. ones flagged as interesting in a prior
    inspection) instead of a random balanced sample.

    Assertions:
      - every requested id is present in `data` (otherwise a typo would
        silently shrink the run);
      - `data['id']` is unique on the requested ids (sanity check; Reddit
        post ids are globally unique).
    """
    ids = list(ids)
    assert ids, "--use_specific_posts requires at least one id"
    present = set(data["id"].tolist())
    missing = [pid for pid in ids if pid not in present]
    assert not missing, (
        f"--use_specific_posts ids not in data (after filtering): {missing}"
    )
    rows = []
    for pid in ids:
        match = data.loc[data["id"] == pid]
        assert len(match) == 1, f"id {pid!r} appears {len(match)} times in data; expected exactly 1"
        rows.append(match)
    return pd.concat(rows).reset_index(drop=True)


def sample_balanced(labels, n, seed):
    """Sample `n` posts, split evenly between YTA-truth and NTA-truth.

    Returns `n//2` rows with is_asshole==1 and `n//2` with is_asshole==0
    (asserts n is even). Both arms are needed because FNR alone can't tell
    apart "less sycophantic" from "uniformly more critical" — a mitigation
    that just flips the model's bias would lower FNR while raising FPR. The
    two halves draw from independent random states so seed offsets between
    them don't collide.
    """
    assert n % 2 == 0, f"--n must be even (got {n}); split is half YTA-truth, half NTA-truth"
    half = n // 2
    yta = labels.loc[labels["is_asshole"] == 1]
    nta = labels.loc[labels["is_asshole"] == 0]
    assert len(yta) >= half, f"need {half} YTA-truth rows, only {len(yta)} available"
    assert len(nta) >= half, f"need {half} NTA-truth rows, only {len(nta)} available"
    yta_sample = yta.sample(n=half, random_state=seed)
    nta_sample = nta.sample(n=half, random_state=seed + 1)
    out = pd.concat([yta_sample, nta_sample]).reset_index(drop=True)
    assert (out["is_asshole"] == 1).sum() == half
    assert (out["is_asshole"] == 0).sum() == half
    return out


def sample_uniform(data, n, seed):
    """Sample `n` rows uniformly at random. Used for OEQ where there is no
    ground-truth label to balance against.

    Uses pandas' `sample` with `random_state=seed` so the same seed
    reproduces the same OEQ sub-sample.
    """
    assert n > 0, f"--n must be positive (got {n})"
    assert len(data) >= n, f"need {n} rows, only {len(data)} available"
    return data.sample(n=n, random_state=seed).reset_index(drop=True)
