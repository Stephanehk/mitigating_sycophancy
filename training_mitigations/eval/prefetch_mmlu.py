"""Download all 57 MMLU subjects (cais/mmlu) into $HF_HOME so lm-eval MMLU jobs
can run with HF_DATASETS_OFFLINE=1 on compute nodes.

The subject list is read from the installed lm_eval's own task configs, so it
always matches what `--tasks mmlu` will request. Run with the lm-eval venv's
interpreter on a login node with internet:

    HF_HOME=<cache> <lmeval-venv>/bin/python training_mitigations/eval/prefetch_mmlu.py

Expect 14042 test questions.
"""

import glob
import os

import datasets
import lm_eval
import yaml


def main():
    d = os.path.join(os.path.dirname(lm_eval.__file__), "tasks", "mmlu", "default")
    groups = {"humanities", "other", "social_sciences", "stem"}
    subjects = sorted(
        os.path.basename(p)[len("mmlu_"):-len(".yaml")]
        for p in glob.glob(os.path.join(d, "mmlu_*.yaml"))
        if not os.path.basename(p).startswith("mmlu__"))
    subjects = [s for s in subjects if s not in groups]
    tmpl = os.path.join(d, "_default_template_yaml")
    path = yaml.safe_load(open(tmpl))["dataset_path"] if os.path.exists(tmpl) else "cais/mmlu"
    print(f"[mmlu] {len(subjects)} subjects from {path} -> HF_HOME={os.environ.get('HF_HOME')}")
    n = 0
    for s in subjects:
        n += len(datasets.load_dataset(path, s)["test"])
    print(f"[mmlu] cached; test questions = {n}")
    assert len(subjects) == 57 and n == 14042, "unexpected MMLU size"


if __name__ == "__main__":
    main()
