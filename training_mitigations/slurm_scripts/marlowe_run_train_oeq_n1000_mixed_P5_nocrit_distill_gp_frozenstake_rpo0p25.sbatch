#!/bin/bash
#
# RPO-ALPHA SWEEP ARM (alpha=0.25, baseline arm uses 1.0).
# Identical to marlowe_run_train_oeq_n1000_mixed_P5_nocrit_distill_gp_frozenstake.sbatch in every respect except
# RPO_ALPHA. `--rpo-alpha` is the WEIGHT of the SFT/NLL term added to the DPO loss
# (trl multi-loss `loss_weights=[1.0, rpo_alpha]`), i.e. how hard the model is pushed
# to IMITATE the `chosen` responses. Lowering it partially withdraws that pressure.
#
# WHY: the distill (SFT-on-chosen) term imitates the whole veto-surviving pool,
# stakeholder-voice and perspshift exemplars included -- see core/self_distill_trainer.py:
# "empirically the over-criticality only appears once that SFT term is on." The trained
# adapters role-play a stakeholder instead of answering as an assistant on 30.7% (AITA)
# and 37.8% (PAS) of the held-out 1000 (training_mitigations/eval/classify_response_voice.py).
# With the alpha=0 arms (*_perspshift_gp) this gives a dose-response curve
# alpha in {0, 0.25, 0.5, 1.0} rather than a single on/off ablation.
#
# NOTE the collapse may not be the SFT term's doing alone: 6 of the 10 candidates per
# prompt are stakeholder-voiced regardless of alpha, and any can be selected as `chosen`
# in a DPO pair. If role-play persists at alpha=0, shrink the pool via N_SOURCE_MODEL /
# N_SOURCE_CRITICAL instead. See _marlowe_dpo_body.sh.
# Marlowe `preempt`-partition OEQ P5_nocrit_distill_gp_frozenstake run.
# Submit from anywhere (paths are absolute):
#   export HF_TOKEN=...        # Qwen3-8B is gated
#   sbatch /projects/m000254/sycophancy/training_mitigations/slurm_scripts/marlowe_run_train_oeq_n1000_mixed_P5_nocrit_distill_gp_frozenstake_rpo0p25.sbatch
#
# EXACTLY the nocrit_distill_gp recipe (P5 veto, mixed 4 model / 0 critical / 6
# stakeholder, --stakeholder-voice second-person prompt, RPO_ALPHA=1.0 distill
# SFT-on-chosen term) with ONE change: FROZEN_STAKEHOLDER=1, so the 6 stakeholder
# candidates are generated from the FROZEN BASE Qwen3-8B (a fixed teacher, the
# base already served on the sim vLLM engine) instead of the evolving policy.
# This removes the on-policy feedback loop in the stakeholder exemplars while the
# bare-post "model" candidates stay on-policy. The stakeholder candidates are
# recorded as source "stakeholder-base:<name>" in responses_*.jsonl so the freeze
# is auditable. See core/trainer.py (_generate_n use_base / _sample_responses_mixed).
#
# free/batch partition: MaxTime=2-00:00:00, PreemptMode=OFF -- these runs are NOT
# preemptible. The award account's `preempt` partition (what every older training
# wrapper uses) has PreemptMode=SUSPEND and preempted 42% of this project's OEQ
# windows; `batch` removes that entirely. The free award has 5000 GPU-h and ~258
# used, so a ~106 GPU-h sweep costs ~2%. --time=24h comfortably covers the longest
# observed chain (11.2h), so the self-resubmit chain in _marlowe_dpo_body.sh should
# never fire; if it does (hardware failure), the successor resumes from
# responses_iter_*.jsonl + --dpo-save-steps checkpoints exactly as before, and
# no-ops once .complete exists.
#SBATCH --job-name=dpo_oeq_n1000_mixed_P5_nocrit_distill_gp_frozenstake_rpo0p25
#SBATCH --account=marlowe-m000254-free
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=/projects/m000254/logs/dpo_oeq_n1000_mixed_P5_nocrit_distill_gp_frozenstake_rpo0p25_%j.out
#SBATCH --error=/projects/m000254/logs/dpo_oeq_n1000_mixed_P5_nocrit_distill_gp_frozenstake_rpo0p25_%j.err

RUN_ENV=OEQ
OUT_NAME=dpo_lora_oeq_qwen3-8b_n1000_mixed_P5_nocrit_distill_gp_frozenstake_rpo0p25
SELF_BASENAME=marlowe_run_train_oeq_n1000_mixed_P5_nocrit_distill_gp_frozenstake_rpo0p25.sbatch
RPO_ALPHA=0.25
N_SOURCE_MODEL=4
N_SOURCE_CRITICAL=0
STAKEHOLDER_VOICE=1
FROZEN_STAKEHOLDER=1
source /projects/m000254/sycophancy/training_mitigations/slurm_scripts/_marlowe_dpo_body.sh
