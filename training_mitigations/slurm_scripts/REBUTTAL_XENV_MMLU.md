# Rebuttal experiments: cross-dataset eval + MMLU (Qwen3-8B)

Two experiments on the four Qwen3-8B "Our Model (3+1)" adapters (the PlurPO
arms reported in the paper):

| env  | adapter run dir |
|------|-----------------|
| OEQ  | `dpo_lora_oeq_qwen3-8b_P6_voice2_rpo0p5` |
| PAS  | `dpo_lora_pas_qwen3-8b_P6_voice2_rpo0p5` |
| AITA | `dpo_lora_aita_qwen3-8b_P6_voice2R2_rpo0p5` |
| FLIP | `dpo_lora_flip_qwen3-8b_P6_voice2R2_rpo0p5` |

1. **Cross-dataset** - each adapter evaluated on the 3 envs it was not trained
   on (12 conditions), same protocol as the paper: frozen
   `final_training_eval_data/<env>_test` split, temp 1.0, seed 0, 512 tokens
   (OEQ/PAS) / 1536 (AITA/FLIP). No prompt leakage: zero exact or
   normalized-prefix overlap between any env's train split and any other env's
   test split (checked 2026-11-06; AITA was built excluding all FLIP ids).
2. **MMLU** - lm-eval 0.4.13, 5-shot loglikelihood, all 57 subjects (14,042
   questions), base + 4 adapters, paired McNemar vs base.

The eval-path files in this branch (`core/llama_local.py`,
`eval/eval_endorsement_heldout.py`, `eval/eval_aita_label_heldout.py`) are the
versions on the Marlowe filesystem as of 2026-11-06, previously uncommitted --
i.e. what the Marlowe runs of these experiments use. They add
`eos_ids_for_generate` (EOS handling), opt-in AITA shard resume, and the k-shot
baseline; the last two are inert here. Not verified to be byte-identical to the
code that produced the paper's in-domain responses.

## Files

| file | what |
|------|------|
| `slurm_scripts/submit_xenv_qwen3_gen.sh` | cross-dataset gen submitter, `CLUSTER=marlowe\|next` |
| `slurm_scripts/next_mmlu_qwen3.sbatch` | MMLU on next (array 0-4) |
| `slurm_scripts/run_mmlu_qwen3_preempt.sbatch` | MMLU on Marlowe paid preempt (array 0-4) |
| `eval/prefetch_mmlu.py` | cache all 57 subjects for offline jobs |
| `eval/mmlu_mcnemar.py` | accuracy + paired McNemar vs base (+ Holm) |
| `../requirements-mmlu.txt` | `lm_eval==0.4.13` + install recipe |

## Running on next

### 0. Get the four adapters onto next (laptop relay)

Marlowe's login node cannot auth to SC, so relay through the laptop. Only the
two adapter files are needed (~167 MB each):

```bash
N=/next/u/stephhk/sycophancy/training_mitigations/runs
for R in dpo_lora_oeq_qwen3-8b_P6_voice2_rpo0p5 dpo_lora_pas_qwen3-8b_P6_voice2_rpo0p5 \
         dpo_lora_aita_qwen3-8b_P6_voice2R2_rpo0p5 dpo_lora_flip_qwen3-8b_P6_voice2R2_rpo0p5; do
  mkdir -p "$R"
  scp stephhk@login.marlowe.stanford.edu:/projects/m000254/runs/$R/{adapter_config.json,adapter_model.safetensors} "$R/"
  ssh stephhk@scdt.stanford.edu "mkdir -p '$N/$R'"
  scp "$R"/adapter_config.json "$R"/adapter_model.safetensors "stephhk@scdt.stanford.edu:$N/$R/"
done
```
(Leave the `{a,b}` remote path unquoted - laptop scp is SFTP-mode.)

### 1. Code

```bash
cd /next/u/stephhk/sycophancy && git fetch origin && git checkout rebuttal-xenv-mmlu
```

### 2. MMLU (~1 h per model on an H100; expect ~2x on an A6000)

```bash
LLM=/next/u/stephhk/miniconda3/envs/llm
$LLM/bin/python -m venv --system-site-packages /next/u/stephhk/envs/lmeval
$LLM/bin/python -m pip freeze | grep -iE '^(torch|transformers|peft|accelerate|datasets)==' > /tmp/pins.txt
PYTHONNOUSERSITE=1 /next/u/stephhk/envs/lmeval/bin/pip install -r requirements-mmlu.txt -c /tmp/pins.txt
HF_HOME=/next/u/stephhk/.cache/huggingface PYTHONNOUSERSITE=1 \
  /next/u/stephhk/envs/lmeval/bin/python training_mitigations/eval/prefetch_mmlu.py   # expect 14042
mkdir -p training_mitigations/logs
sbatch --array=0-4 training_mitigations/slurm_scripts/next_mmlu_qwen3.sbatch
# when done:
/next/u/stephhk/envs/lmeval/bin/python training_mitigations/eval/mmlu_mcnemar.py \
  --root training_mitigations/runs/mmlu_qwen3-8b
```
If the next `llm` env's transformers predates Qwen3 / peft is missing, point
the venv at whichever env `next_final_gen.sbatch` uses (`_next_eval_env.sh`).

### 3. Cross-dataset generation (GPU only, no API)

```bash
CLUSTER=next bash training_mitigations/slurm_scripts/submit_xenv_qwen3_gen.sh --dry-run
CLUSTER=next bash training_mitigations/slurm_scripts/submit_xenv_qwen3_gen.sh
```
Creates `runs/xenv_qwen3-8b_<train>_on_<eval>/` (adapter symlinks +
`XENV_SOURCE.txt`) and submits one array per condition through
`next_final_gen.sbatch`. Idempotent: re-running skips finished conditions and
queued jobs, and the sbatch skips finished shards.

Cost (H100-hours, from past Marlowe fgen timings): OEQ/PAS ~5 per condition,
AITA ~6.5, FLIP ~13 -> ~90 for all 12 plus model-load overhead; A6000 is slower.

Completion check: every container must have all shards -
OEQ/PAS `responses_final_finaldata_temp1.0_seed0_shard{0..7}of8.csv`,
AITA `aita_responses_..._shard{0..9}of10.csv`, FLIP `..._shard{0..19}of20.csv`.
Count rows (1000 prompts / 1000 pairs per condition), don't trust job state.

### 4. Scoring (deferred - PAID API, ~30k gpt-5-mini calls, est. $25-40)

Not yet run. Same pipeline as the in-domain arms, with the xenv shard counts
and `--env` always explicit:
1. attempt-1 gate: `eval/judge_assistant_mode.py --run-dir <c> --env <E> --tag final_finaldata_temp1.0_seed0 --expected-shards <N>`
2. gate-loop resample of rejected ids (`--gate-loop`, up to 5 attempts; GPU + API)
3. OEQ/PAS: `eval/stage_heldout_inputs.py --run-dir <c> --tag final_finaldata_temp1.0_seed0 --num-shards 8 --name <c>` then `eval/run_endorse_gpt5mini.py`;
   AITA/FLIP: `eval/classify_aita_verdicts.py --run-dir <c> --env <E> --tag final_finaldata_temp1.0_seed0 --num-shards <N>`
4. gated tables: `eval/build_gated_finaldata_tables.py` (add the 12 rows).

## Cross-dataset generation on vLLM (next, 2026-11-06) -- what was actually run

The HF path above samples one prompt at a time (~40 min per 125-prompt PAS shard
on an A6000; ~19 h for all 138 shards on 6 GPUs). For the rebuttal the 12
conditions were instead generated with batched offline vLLM:

```bash
sbatch --array=0-11 training_mitigations/slurm_scripts/next_xenv_vllm_gen.sbatch
```

- `eval/xenv_vllm_gen.py`: one task per condition (index -> `conditions()`), whole
  1000-prompt / 1000-pair split in one `LLM.generate`, written into the HF shard
  layout/filenames/columns under `runs/xenv_vllm_qwen3-8b_<train>_on_<eval>/`, so
  the deferred scoring steps run unchanged (pass `--env`, shard counts 8/10/20).
- Parity settings: same prompt ids (`build_chat_input`), temp 1.0 / top_p 0.95 /
  top_k 20 (HF keeps Qwen3's generation_config top_k=20), stop only on
  `<|im_end|>` (asserted per generation), `generation_config="vllm"`. Seeds are
  per request (`attempt_seed(text, 1, 0)`), so samples differ from HF's draws.
- `eval/xenv_vllm_parity.py` (greedy, oeq adapter, 8 PAS prompts): teacher-forced
  |dlogp| HF+PEFT vs vLLM+LoRA mean 0.015 nats (max 0.25); greedy shared prefix
  median 149 tokens vs 36 for HF-base vs vLLM-base; vLLM base vs HF+PEFT diverges
  at token 0 on 7/8 (adapter is applied).
- HF reference `runs/xenv_qwen3-8b_oeq_on_pas/` (all 8 HF shards, 1000 rows) vs
  vLLM same condition: response-length KS p=0.37 (median 327 vs 354 tokens,
  ~capped at 512: 28.2% vs 31.7%).
- Note for scoring: AITA/FLIP adapters on OEQ hit the 512-token cap on ~97% of
  responses (OEQ/PAS protocol cap); PAS/OEQ adapters on AITA/FLIP almost never
  hit 1536. This is adapter behaviour, present on HF too.
