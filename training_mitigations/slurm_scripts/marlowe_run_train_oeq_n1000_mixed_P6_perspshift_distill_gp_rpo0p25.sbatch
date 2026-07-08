#!/bin/bash
#
# RPO-ALPHA SWEEP ARM (alpha=0.25, baseline arm uses 1.0).
# Identical to marlowe_run_train_oeq_n1000_mixed_P6_perspshift_distill_gp.sbatch in every respect except
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
# Marlowe `preempt`-partition OEQ P6_perspshift_distill_gp run.
# Submit from anywhere (paths are absolute):
#   export HF_TOKEN=...        # Qwen3-8B is gated
#   sbatch /projects/m000254/sycophancy/training_mitigations/slurm_scripts/marlowe_run_train_oeq_n1000_mixed_P6_perspshift_distill_gp_rpo0p25.sbatch
#
# EXACT clone of marlowe_run_train_oeq_n1000_mixed_P5_perspshift_distill_gp.sbatch
# (same mixed 2 model / 2 perspshift / 6 stakeholder recipe, RPO_ALPHA=1.0 distill
# term, --stakeholder-voice "_gp" candidates, identical DPO hyperparameters) with
# the ONE method change under test — the veto panel:
#   VETO_PROMPT=P6       BIDIRECTIONAL (stance-mismatch) veto: each voter states
#                        its own preferred stance and vetoes on disagreement, so
#                        over-critical responses can be demoted too, not only
#                        affirming ones (vs. the one-sided P5 veto).
#   USER_STAKEHOLDER=1   adds the 4th "user"/narrator voice to the veto panel
#                        (veto-only; NOT sampled for candidates) to keep the panel
#                        from being one-sidedly critical. See
#                        core/simulated_stakeholders.py VETO_PROMPT_P6 /
#                        VETO_PROMPT_USER. Candidate generation is byte-identical to
#                        the P5 perspshift arm, so this isolates the veto change.
# Separate OUT_DIR (…_P6_perspshift_distill_gp). All site paths + the shared
# preempt/resubmit harness live in _marlowe_dpo_body.sh.
#
# preempt partition: MaxTime=04:00:00 (4h), GraceTime=900 (15min), preemptible.
# SELF-RESUBMITTING 4h chain (the body queues an afterany successor each window),
# resumable via responses_iter_*.jsonl + --dpo-save-steps checkpoints.
#SBATCH --job-name=dpo_oeq_n1000_mixed_P6_perspshift_distill_gp_rpo0p25
#SBATCH --account=marlowe-m000254
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=/projects/m000254/logs/dpo_oeq_n1000_mixed_P6_perspshift_distill_gp_rpo0p25_%j.out
#SBATCH --error=/projects/m000254/logs/dpo_oeq_n1000_mixed_P6_perspshift_distill_gp_rpo0p25_%j.err

RUN_ENV=OEQ
OUT_NAME=dpo_lora_oeq_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p25
SELF_BASENAME=marlowe_run_train_oeq_n1000_mixed_P6_perspshift_distill_gp_rpo0p25.sbatch
RPO_ALPHA=0.25
N_SOURCE_MODEL=2
N_SOURCE_CRITICAL=2
PERSPSHIFT=1
PERSPSHIFT_ASSESS=1
STAKEHOLDER_VOICE=1
VETO_PROMPT=P6
USER_STAKEHOLDER=1
source /projects/m000254/sycophancy/training_mitigations/slurm_scripts/_marlowe_dpo_body.sh
