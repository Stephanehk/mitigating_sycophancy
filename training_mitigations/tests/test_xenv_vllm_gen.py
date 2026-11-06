"""Tests for eval/xenv_vllm_gen.py (no GPU / vLLM needed).

Covers the assumptions the vLLM rebuttal path relies on:
  1. the 12-condition table and shard counts match the HF submitter
     (submit_xenv_qwen3_gen.sh), so both backends describe the same experiment;
  2. assembled records carry exactly the HF drivers' CSV columns per env, in
     row order, with FLIP's two sides joined to the right pair;
  3. write_shards tiles the rows into the HF shard layout with no loss or
     overlap, and the files carry the HF names;
  4. make_container refuses a partial shard set and never re-points symlinks;
  5. per-request seeds are a pure function of the prompt text.

Run: /next/u/stephhk/miniconda3/envs/llm/bin/python -m pytest training_mitigations/tests/test_xenv_vllm_gen.py -q
"""

import os
import re
import sys

import pandas as pd
import pytest

_TM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_TM, "eval"))

import xenv_vllm_gen as X

HF_SUBMITTER = os.path.join(_TM, "slurm_scripts", "submit_xenv_qwen3_gen.sh")


def _rows(ev, n):
    d = {"id": [f"id{i}" for i in range(n)], "post_text": [f"post {i}" for i in range(n)]}
    if ev == "aita":
        d["is_asshole"] = [i % 2 for i in range(n)]
    if ev == "flip":
        d["flipped_post_text"] = [f"flipped {i}" for i in range(n)]
    return pd.DataFrame(d)


def _gens(reqs):
    return [{"response": f"resp {i}:{side}", "n_tokens": 10, "finish_reason": "stop",
             "seed": 1, "last_token": 151645} for i, side, _ in reqs]


def test_conditions_match_hf_submitter():
    sh = open(HF_SUBMITTER).read()
    for key, table in (("SRC", X.SRC), ("NSH", X.NSH), ("PREFIX", X.PREFIX)):
        block = re.search(rf"declare -A {key}=\((.*?)\)\n", sh, re.S).group(1)
        parsed = dict(re.findall(r"\[(\w+)\]=([\w.\-]+)", block))
        assert {k: str(v) for k, v in table.items()} == parsed, key
    conds = X.conditions()
    assert len(conds) == 12 and len(set(conds)) == 12
    assert all(tr != ev for tr, ev in conds)
    assert sum(X.NSH[ev] for _, ev in conds) == 138


HF_COLS = {
    "oeq": ["id", "sentence", "response", "winning_attempt", "gate_passed", "n_tried"],
    "aita": ["id", "post_text", "rewritten_post", "response", "winning_attempt",
             "gate_passed", "n_tried", "is_asshole"],
    "flip": ["id", "post_text", "flipped_post_text", "rewritten_original",
             "rewritten_flipped", "response_original", "response_flipped",
             "winning_attempt_original", "winning_attempt_flipped",
             "gate_passed_original", "gate_passed_flipped",
             "n_tried_original", "n_tried_flipped"],
}


@pytest.mark.parametrize("ev", ["oeq", "aita", "flip"])
def test_assemble_records_schema_and_join(ev):
    rows = _rows(ev, 5)
    reqs = X.requests_for(rows, ev)
    assert len(reqs) == (10 if ev == "flip" else 5)
    csv_recs, js_recs = X.assemble_records(rows, ev, reqs, _gens(reqs))
    assert [list(r) for r in csv_recs] == [HF_COLS[ev]] * 5
    assert [r["id"] for r in csv_recs] == list(rows["id"])
    if ev == "flip":
        assert csv_recs[3]["response_original"] == "resp 3:original"
        assert csv_recs[3]["response_flipped"] == "resp 3:flipped"
        assert js_recs[3]["last_token_flipped"] == 151645
    else:
        assert csv_recs[3]["response"] == "resp 3:None"


def test_assemble_rejects_missing_generations():
    rows = _rows("oeq", 3)
    reqs = X.requests_for(rows, "oeq")
    with pytest.raises(AssertionError):
        X.assemble_records(rows, "oeq", reqs, _gens(reqs)[:2])


@pytest.mark.parametrize("ev,n", [("pas", 1000), ("aita", 1000), ("flip", 1000), ("oeq", 13)])
def test_write_shards_tiles_rows(tmp_path, ev, n):
    rows = _rows(ev, n)
    reqs = X.requests_for(rows, ev)
    csv_recs, js_recs = X.assemble_records(rows, ev, reqs, _gens(reqs))
    os.makedirs(tmp_path / "eval")
    X.write_shards(str(tmp_path), ev, csv_recs, js_recs, {"k": 1})
    paths = X.shard_csv_paths(str(tmp_path), ev)
    assert all(os.path.exists(p) for p in paths)
    ids = pd.concat(pd.read_csv(p) for p in paths)["id"].tolist()
    assert ids == list(rows["id"])          # no loss, no overlap, order kept
    assert os.path.basename(paths[0]) == f"{X.PREFIX[ev]}_{X.TAG}_shard0of{X.NSH[ev]}.csv"


def _fake_adapter(tmp_path):
    src = tmp_path / "src"
    os.makedirs(src)
    for fn in ("adapter_config.json", "adapter_model.safetensors"):
        (src / fn).write_text("x")
    return str(src)


def test_make_container_partial_and_symlinks(tmp_path):
    src = _fake_adapter(tmp_path)
    d, done = X.make_container(str(tmp_path), "oeq", "pas", src, "vllm test")
    assert not done
    assert os.readlink(os.path.join(d, "adapter_config.json")) == os.path.join(src, "adapter_config.json")
    # rerun with no CSVs: fine, links unchanged
    assert X.make_container(str(tmp_path), "oeq", "pas", src, "vllm test") == (d, False)
    paths = X.shard_csv_paths(d, "pas")
    open(paths[0], "w").close()
    with pytest.raises(AssertionError, match="partial"):
        X.make_container(str(tmp_path), "oeq", "pas", src, "vllm test")
    for p in paths:
        open(p, "w").close()
    assert X.make_container(str(tmp_path), "oeq", "pas", src, "vllm test") == (d, True)


def test_make_container_refuses_repointing(tmp_path):
    src = _fake_adapter(tmp_path)
    X.make_container(str(tmp_path), "oeq", "pas", src, "vllm test")
    other = tmp_path / "other"
    os.makedirs(other)
    with pytest.raises(AssertionError, match="points elsewhere"):
        X.make_container(str(tmp_path), "oeq", "pas", str(other), "vllm test")


def test_seeds_are_pure_function_of_text():
    a = X.attempt_seed("post A", 1, 0)
    assert a == X.attempt_seed("post A", 1, 0)
    assert a != X.attempt_seed("post B", 1, 0)
    assert a != X.attempt_seed("post A", 1, 1)
