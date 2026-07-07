"""Shared dataset / train-test-split helpers for the training entry points.

Both `train_dpo_lora.py` (the stakeholder-veto and neutrality-veto DPO baselines)
and `train_grpo_lora.py` (the GRPO neutrality baseline) need the exact same
dataset loading and train/test split logic so a DPO run and a GRPO run on the
same `--env`/`--seed`/`--test-n`/`--train-n` train and evaluate on identical
prompt splits. That logic lives here, in one place, so the two entry points
cannot drift.

The split contract (held constant across baselines):
  - `test_n` held-out prompts are sampled first with `sample_uniform(seed)`.
  - `train_n` train prompts are sampled from the disjoint complement with
    `sample_uniform(seed + 1000)` so the two samplers never collide.
  - disjointness is asserted, not assumed.

Nothing here calls an LLM or touches a GPU; it is pure pandas + the harness's
`sample_uniform`.
"""

import json
import os
import sys

# The harness (`mitigations/`) holds the per-env loaders + the uniform sampler.
# Anchor the import to the repo root so this module is importable from any cwd
# and from either entry point.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, "mitigations"))

from explore_mitigations import (
    OEQ_DATA_CSV, PAS_DATA_CSV, AITA_DATA_CSV, AITA_NTA_FLIP_DATA_CSV,
    load_oeq, load_pas, load_aita_clean, load_aita_nta_flip,
    sample_uniform, sample_balanced, select_specific_posts,
)

# Free-form AITA / AITA-NTA-FLIP elicitation: the policy answers in prose (like
# OEQ/PAS) but is asked to END with an explicit YTA/NTA verdict so the eval can
# parse a label without an LLM classifier. The SAME suffix is appended at both
# sampling time and eval time so the train and eval prompt surfaces match.
AITA_LABEL_SUFFIX = (
    "\n\nAt the end of your response you must output a label: NTA or YTA."
)

# Envs whose generation prompt carries the AITA label nudge.
_AITA_ENVS = ("AITA", "AITA-NTA-FLIP")


def resolve_csv(env_name):
    """Return the abs path to the env's CSV, anchored to the repo root so the
    caller runs identically from any cwd. Asserts the file exists.
    """
    rel = {
        "OEQ":  OEQ_DATA_CSV,
        "PAS":  PAS_DATA_CSV,
        "AITA": AITA_DATA_CSV,
        "AITA-NTA-FLIP": AITA_NTA_FLIP_DATA_CSV,
    }[env_name]
    path = rel if os.path.isabs(rel) else os.path.join(_REPO_ROOT, rel)
    assert os.path.exists(path), f"missing {env_name} csv: {path}"
    return path


def load_env_data(env_name, csv_path):
    """Dispatch to the per-env loader. Each returns a DataFrame with at least
    `id` and `post_text`.

    AITA additionally carries `is_asshole` (0/1 ground truth, used for the
    balanced split and the eval FNR/FPR). AITA-NTA-FLIP additionally carries
    `flipped_post_text` (the same scenario retold from the other party's POV).
    """
    if env_name == "OEQ":
        return load_oeq(csv_path)
    if env_name == "PAS":
        return load_pas(csv_path)
    if env_name == "AITA":
        return load_aita_clean(csv_path)
    if env_name == "AITA-NTA-FLIP":
        return load_aita_nta_flip(csv_path)
    raise AssertionError(f"unknown env_name: {env_name!r}")


def split_train_test(env_name, data, train_n, test_n, seed):
    """Pick `test_n` test rows and a disjoint `train_n` train rows from `data`.

    Disjointness is always enforced by dropping the test ids from the pool
    before sampling train; the train seed is `seed + 1000` so the two samplers
    never collide. Returns (train_df, test_df).

    Sampling differs by env:
      - OEQ / PAS / AITA-NTA-FLIP: `sample_uniform` (no per-row ground-truth
        label to balance against). For AITA-NTA-FLIP a "row" is a (original,
        flipped) PAIR, so test_n / train_n count pairs, not generations.
      - AITA: `sample_balanced` on `is_asshole` so BOTH the held-out and the
        train split are 50/50 YTA-truth / NTA-truth (test_n and train_n must
        be even). A balanced train keeps the DPO signal symmetric, and a
        balanced held-out makes FNR and FPR both meaningful.
    """
    if env_name == "AITA":
        test_df = sample_balanced(data, test_n, seed)
        pool = data[~data["id"].isin(set(test_df["id"]))].reset_index(drop=True)
        train_df = sample_balanced(pool, train_n, seed + 1000)
    else:
        test_df = sample_uniform(data, test_n, seed)
        pool = data[~data["id"].isin(set(test_df["id"]))].reset_index(drop=True)
        train_df = sample_uniform(pool, train_n, seed + 1000)
    assert set(test_df["id"]).isdisjoint(set(train_df["id"])), (
        "train/test id overlap — split logic is broken"
    )
    return train_df, test_df


def format_prompt(env_name, post_text, with_label_suffix=True):
    """Return the user message the model under training sees at both sampling
    time and train time.

    OEQ / PAS use the bare post (matching the harness's `PROMPT_TEMPLATE_OEQ =
    "{post}"`). AITA / AITA-NTA-FLIP keep that free-form surface but append
    `AITA_LABEL_SUFFIX` so the response ends with a parseable YTA/NTA verdict
    (no LLM classifier needed at eval). `post_text` is a single post string;
    for AITA-NTA-FLIP the caller passes either the original or the flipped post
    — see `build_train_prompts`.

    `with_label_suffix` controls whether the AITA nudge is appended. Default
    True is what eval / GRPO / probe callers want. The DPO trainer passes
    False to obtain the BARE post — it re-attaches `label_suffix_for(env)` at
    exactly the candidate-generation sites and the DPO prompt column, and keeps
    the bare post for the identify/concerns/veto/preference/rewrite stages (see
    core/trainer.py). OEQ/PAS ignore the flag (no suffix either way).
    """
    if env_name in ("OEQ", "PAS"):
        return post_text
    if env_name in _AITA_ENVS:
        return post_text + AITA_LABEL_SUFFIX if with_label_suffix else post_text
    raise AssertionError(f"unknown env_name: {env_name!r}")


def label_suffix_for(env_name):
    """The generation-time YTA/NTA nudge for `env_name`: `AITA_LABEL_SUFFIX`
    for the two AITA envs, else "" (OEQ/PAS). The DPO trainer stores this and
    appends it to every candidate-generation prompt and the DPO prompt column,
    so an empty string makes the whole mechanism a no-op for OEQ/PAS.
    """
    return AITA_LABEL_SUFFIX if env_name in _AITA_ENVS else ""


def build_train_prompts(env_name, df, with_label_suffix=True):
    """Flatten a split DataFrame into the ordered list of training prompts the
    trainer samples against.

    For OEQ / PAS / AITA there is one prompt per row (the formatted
    `post_text`). For AITA-NTA-FLIP each row is a (original, flipped) PAIR, so
    it expands to TWO prompts — the formatted `post_text` followed by the
    formatted `flipped_post_text` — fed INDEPENDENTLY through the veto
    pipeline. The order (every row's original, then its flipped) is
    deterministic given the row order, so a preempt-resume that rebuilds this
    list from the saved split ids keeps `prompt_idx` aligned with the partial
    responses file.

    `with_label_suffix` is forwarded to `format_prompt`. The DPO trainer passes
    False to get BARE posts (it owns the AITA nudge); eval / GRPO / probe use
    the default True. OEQ/PAS are unaffected by the flag.

    Assumes `df` rows are already in the desired order; returns a flat list.
    """
    if env_name in ("OEQ", "PAS", "AITA"):
        return [format_prompt(env_name, t, with_label_suffix)
                for t in df["post_text"].tolist()]
    if env_name == "AITA-NTA-FLIP":
        prompts = []
        for orig, flip in zip(df["post_text"].tolist(),
                              df["flipped_post_text"].tolist()):
            prompts.append(format_prompt(env_name, orig, with_label_suffix))
            prompts.append(format_prompt(env_name, flip, with_label_suffix))
        return prompts
    raise AssertionError(f"unknown env_name: {env_name!r}")


def subset_in_id_order(data, ids):
    """Return the rows of `data` whose `id` is in `ids`, in `ids` order.

    Used by the preempt-resume path to rebuild the exact training split from
    the saved `train_ids.json` (which holds row/pair ids) before flattening it
    with `build_train_prompts`. Delegates to the harness's
    `select_specific_posts`, which asserts every requested id is present.
    """
    return select_specific_posts(data, [str(i) for i in ids])


def write_ids(path, df):
    """Dump the `id` column of `df` to `path` as a JSON list. Used for the
    test_ids.json / train_ids.json split records a downstream eval reloads.
    """
    with open(path, "w") as f:
        json.dump([str(x) for x in df["id"].tolist()], f, indent=2)


# ---------------------------------------------------------------------------
# Preempt-resume detection.
#
# The trainer writes two kinds of durable per-iteration artifact under
# output_dir: `adapter_iter_<k>/` (the LoRA snapshot after iteration k's DPO
# step) and `responses_iter_<k>.jsonl` (per-prompt sampling+veto records,
# flushed per prompt). An `--allow-preempt` run uses these to restart itself
# idempotently after a kill — see `detect_resume_point`. A `.complete` sentinel
# distinguishes a fully-saved adapter from one left half-written by a kill
# mid-`save_pretrained`.
# ---------------------------------------------------------------------------

SENTINEL_NAME = ".complete"


def adapter_complete(adapter_dir):
    """True iff `adapter_dir` carries the `.complete` sentinel the trainer
    writes only AFTER `save_pretrained(adapter_dir)` returns. A dir without the
    sentinel (never written, or half-written when the job was killed mid-save)
    is treated as NOT a finished iteration.
    """
    return os.path.exists(os.path.join(adapter_dir, SENTINEL_NAME))


def write_complete_sentinel(adapter_dir):
    """Mark `adapter_dir` as fully saved by creating its `.complete` sentinel.
    Called immediately after `save_pretrained(adapter_dir)` returns, so the
    sentinel's presence implies the adapter on disk is complete.
    """
    open(os.path.join(adapter_dir, SENTINEL_NAME), "w").close()


def detect_resume_point(output_dir, num_iterations):
    """Scan `output_dir` to decide where an `--allow-preempt` run continues.

    Resume is keyed on the trainer's durable artifacts:
      - `adapter_iter_<k>/` + its `.complete` sentinel marks iteration k as
        FINISHED (sentinel written only after save_pretrained returns).
      - `responses_iter_<k>.jsonl` holds iteration k's per-prompt sampling+veto
        records; its presence past the last finished iteration means that
        iteration's SAMPLING phase was partially done and can be reloaded
        rather than restarted.

    Returns (start_iteration, resume_adapter, resume_partial, is_cold_start):
      start_iteration: 0-based iteration to begin at == number of finished
        iterations (k_max).
      resume_adapter:  path to `adapter_iter_<k_max>` to load as the trainable
        LoRA, or None when k_max == 0 (no finished iteration — start from base).
      resume_partial:  True iff `responses_iter_<k_max+1>.jsonl` exists, i.e.
        the next iteration's sampling was partially done.
      is_cold_start:   True iff `output_dir` has NO resume artifacts at all (no
        finished adapter, no partial responses, no train_ids.json) — a fresh run.

    Asserts the artifacts are self-consistent: finished adapters form a
    contiguous prefix `1..k_max` (no gap where a later iteration is complete but
    an earlier one is not).
    """
    k_max = 0
    for k in range(1, int(num_iterations) + 1):
        if adapter_complete(os.path.join(output_dir, f"adapter_iter_{k}")):
            k_max = k
        else:
            break
    # No finished adapter may exist beyond the contiguous prefix: a complete
    # adapter_iter_<k> for k > k_max+1, or a gap at k_max+1, means an earlier
    # sentinel went missing — inconsistent checkpoint state, fail fast.
    for k in range(k_max + 1, int(num_iterations) + 1):
        assert not adapter_complete(os.path.join(output_dir, f"adapter_iter_{k}")), (
            f"adapter_iter_{k} is complete but adapter_iter_{k_max + 1} is not; "
            f"checkpoint state in {output_dir} is inconsistent (gap in finished "
            f"iterations)"
        )
    resume_adapter = (os.path.join(output_dir, f"adapter_iter_{k_max}")
                      if k_max > 0 else None)
    partial_path = os.path.join(output_dir, f"responses_iter_{k_max + 1}.jsonl")
    resume_partial = os.path.exists(partial_path)
    train_ids_exists = os.path.exists(os.path.join(output_dir, "train_ids.json"))
    is_cold_start = (k_max == 0 and not resume_partial and not train_ids_exists)
    return k_max, resume_adapter, resume_partial, is_cold_start
