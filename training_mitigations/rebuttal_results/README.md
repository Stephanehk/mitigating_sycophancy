# Rebuttal results (Qwen3-8B PlurPO adapters, 2026-11-06) -- plot-ready

Generated on next by `eval/xenv_score_table.py` and `eval/export_mmlu_plot_data.py`;
raw outputs stay on next under `training_mitigations/runs/` (not in git).

- `mmlu_summary.csv` -- lm-eval 0.4.13, 5-shot, all 57 subjects (14,042 q). Per model
  (base, oeq, pas, aita, flip): n_correct, acc + 95% Wilson [acc_lo, acc_hi],
  delta_vs_base, b = base right/model wrong, c = reverse, exact McNemar p, Holm p
  (over the 4 adapters).
- `mmlu_by_subject.csv` -- model, subject, n, n_correct, acc.
- `xenv_scores.csv` -- each adapter on the 3 datasets it was NOT trained on (12
  conditions), generated with batched vLLM (temp 1.0, top_p 0.95, top_k 20), scored
  with the in-domain judges (v5 gpt-5-mini endorse for OEQ/PAS, gpt-5-mini verdict
  classifier for AITA/FLIP). Two rows per condition: `view` = ungated (all 1000) or
  gated@1 (attempt-1 assistant-mode gate passers only; NO gate-loop resample).
  Columns per metric k in {a, b}: metric_k name, count_k / n_k, rate_k, 95% Wilson
  [lo_k, hi_k]. metric_a / metric_b = endorsement #1/(#0+#1) / neutral #3/all (OEQ,
  PAS); FNR / FPR (AITA); both-NTA / both-YTA over all pairs (FLIP).
  gate_pass1 / gate_keys = pass@1 (FLIP keys = 2 sides x 1000 pairs).
- `vllm_parity_oeq_on_pas.json` -- greedy HF+PEFT vs vLLM+LoRA check (8 PAS prompts).

Caveats: AITA/FLIP adapters on OEQ hit the 512-token cap on ~97% of responses and
their endorsement rate rests on n_a = 180-234; no base / in-domain rows are included
here (those live on Marlowe).
