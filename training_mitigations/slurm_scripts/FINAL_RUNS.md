# Final paper runs — OEQ/PAS and AITA/AITA-NTA

Frozen, robust re-run of every method in the paper table on the **frozen
`final_training_eval_data/` split**: trained arms train on `<env>_train` and are
held out on `<env>_test`; prompt-only arms are evaluated on `<env>_test`. Decode
is **1 sample, temp 1.0, seed 0** (robustness from the N=1000 test sets). All
outputs are namespaced with the tag marker **`finaldata`**, so nothing collides
with the earlier tuning-era artifacts.

Every script here is new and dedicated to the final runs; the tuning scripts and
their outputs are untouched.

## Methods → scripts

### OEQ and PAS (10 methods each)

| Method | Kind | Script / condition |
|---|---|---|
| Qwen3-8B (base) | prompt-only | `marlowe_final_gen.sbatch` GEN_KIND=base STEER=none |
| + Cheng et al. | prompt-only | STEER=explicit (`--explicit`) |
| + Sharma et al. | prompt-only | STEER=sharma (`--sharma-honesty`) |
| + "Don't be sycophantic" | prompt-only | STEER=dontbesyco (`--gen-suffix "Don't be sycophantic."`) |
| + "Be critical" | prompt-only | STEER=becritical (`--be-critical`) |
| + perspective-shift | prompt-only | STEER=perspshift (`--perspective-shift --assess-narrative`) |
| SFT "be critical" | trained | `marlowe_final_train_becritical_sft.sbatch` (SFT_ENV=OEQ/PAS) |
| Neutral-mixed DPO | trained | `next_final_train_<env>_neutral_mixed_dpo.sbatch` |
| Neutral DPO | trained | `next_final_train_<env>_neutral_dpo.sbatch` |
| P6_persp α0.5 | trained | `marlowe_final_train_<env>_P6_perspshift_distill_gp_rpo0p5.sbatch` |

### AITA and AITA-NTA-FLIP (7 methods each)

Same six prompt-only baselines, **plus** P6_persp α0.5 only
(`marlowe_final_train_{aita,flip}_P6_perspshift_distill_gp_rpo0p5.sbatch`).
AITA metric = FNR/FPR; AITA-NTA-FLIP metric = per-pair both-NTA sycophancy.

## Where each stage runs (and why)

- **Marlowe (`batch`/free, H100, air-gapped compute):** all P6_persp + SFT
  training, ALL generation, ALL judging/classify (on the **login** node).
- **`next` (compute-node egress):** Neutral / Neutral-mixed DPO ONLY. Their veto
  is the gpt-5-mini endorse judge called **in the training loop** (OpenRouter
  egress), which an air-gapped Marlowe compute node cannot do. After training,
  relay the 4 adapters to Marlowe so generation/scoring stay single-cluster.

Prereqs: `HF_TOKEN` (gated Qwen3-8B) for training/gen; `OPENROUTER_API_KEY` for
the `next` neutral training and for all Marlowe login-node scoring.

---

## Phase 1 — Train (10 arms, seed 0)

**Marlowe** (from `/projects/m000254/sycophancy`):
```bash
export HF_TOKEN=...
SS=training_mitigations/slurm_scripts
# P6_persp alpha0.5 x4
sbatch $SS/marlowe_final_train_oeq_P6_perspshift_distill_gp_rpo0p5.sbatch
sbatch $SS/marlowe_final_train_pas_P6_perspshift_distill_gp_rpo0p5.sbatch
sbatch $SS/marlowe_final_train_aita_P6_perspshift_distill_gp_rpo0p5.sbatch
sbatch $SS/marlowe_final_train_flip_P6_perspshift_distill_gp_rpo0p5.sbatch
# SFT be-critical x2 (teacher-gen + SFT in one job each)
sbatch --export=ALL,SFT_ENV=OEQ $SS/marlowe_final_train_becritical_sft.sbatch
sbatch --export=ALL,SFT_ENV=PAS $SS/marlowe_final_train_becritical_sft.sbatch
```

**`next`** (from `/next/u/stephhk/sycophancy`):
```bash
export HF_TOKEN=...; export OPENROUTER_API_KEY=...
SS=training_mitigations/slurm_scripts
sbatch $SS/next_final_train_oeq_neutral_dpo.sbatch
sbatch $SS/next_final_train_oeq_neutral_mixed_dpo.sbatch
sbatch $SS/next_final_train_pas_neutral_dpo.sbatch
sbatch $SS/next_final_train_pas_neutral_mixed_dpo.sbatch
```

Each run writes `.complete` + the final adapter (`adapter_config.json` at the run
root) when finished.

## Phase 1b — Relay the 4 `next` neutral adapters to Marlowe

The Marlowe generation driver reads adapters from `/projects/m000254/runs/`, so
copy each finished neutral run dir over (whole dir: `adapter_config.json`,
`adapter_model.safetensors`, `.complete`). Names are IDENTICAL on both sides:
```
dpo_lora_oeq_qwen3-8b_n1000_neutral_final
dpo_lora_oeq_qwen3-8b_n1000_neutral_mixed_final
dpo_lora_pas_qwen3-8b_n1000_neutral_final
dpo_lora_pas_qwen3-8b_n1000_neutral_mixed_final
```
`next:.../training_mitigations/runs/<name>` → `marlowe:/projects/m000254/runs/<name>`
(use the usual two-hop relay via your laptop if the clusters can't reach each
other directly).

## Phase 2 — Generate (Marlowe login node)

Idempotent, gated on `.complete` (trained) and on existing gen shards. Dry-run
first to see the 34-condition plan:
```bash
export HF_TOKEN=...
bash training_mitigations/slurm_scripts/marlowe_final_gen_submit.sh --dry-run
bash training_mitigations/slurm_scripts/marlowe_final_gen_submit.sh
```
Shard regimes: OEQ/PAS = 2 shards (`--array=0-1`, `responses_<tag>_shard{0,1}of2.csv`);
AITA/FLIP = 1 unsharded task (`aita_responses_<tag>_shard0of1.csv`). Base/
prompt-only conditions write into `runs/baseline_final_<ENV>/eval/`.

## Phase 3 — Score (Marlowe login node, gpt-5-mini via OpenRouter)

```bash
source ~/.bashrc   # OPENROUTER_API_KEY
# OEQ/PAS: stage -> v5 endorse judge (+ OEQ role-play voice census)
bash training_mitigations/slurm_scripts/marlowe_final_endorse_all.sh all
# AITA/FLIP: verdict classify + FNR/FPR / both-NTA
bash training_mitigations/slurm_scripts/marlowe_final_classify_all.sh all
```

## Reading the numbers

- **OEQ / PAS** — action-endorsement rate `#1/(#0+#1)` per condition in
  `training_mitigations/sycoharm_runs/endorse_gpt5mini_v5_temp0/scored_<name>.csv`.
  OEQ role-play (stakeholder-voice) rate in `runs/<run>/eval/voice_<tag>.csv`.
  `marlowe_judge_watcher_p5.sh`'s `report()` shows the exact endorse-rate + CI
  read if you want a formatted table.
- **AITA / FLIP** — `classify_aita_verdicts.py` prints FNR/FPR (AITA) or
  `sycophancy_score_rate (both NTA)` (FLIP) per condition and writes
  `runs/<run>/eval/aita_scored_<tag>_shard0of1.csv`.

## Tags (for reference)

`tag = <adapter_tag>_finaldata_temp1.0_seed0`, `adapter_tag ∈ {base,
base-becritical, base-explicit, base-sharma, base-dontbesyco,
base-perspshiftassess, final}` (`final` = any trained adapter).

## Rough judge cost

~34 conditions scored by gpt-5-mini: endorse ≤ $6/condition (20 OEQ/PAS) + OEQ
voice census (10) + AITA/FLIP classify (14). Budget on the order of $150–250;
each stage is checkpointed/idempotent so re-runs don't re-spend.
