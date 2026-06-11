# Experiment registry

Log every run here for quick comparison. W&B is the live source of truth; this file is the human-readable index.

**Base model:** `Qwen/Qwen3-4B-Instruct-2507` → `models/Qwen3-4B-Instruct/`

## Phase 0 — Pass@K baselines

| date | exp_name | model | k | pass@1 | pass@8 | pass@16 | temp | notes |
|------|----------|-------|---|--------|--------|---------|------|-------|
| — | `qwen3_rlvr_p0_passk_gsm8k_baseline` | Qwen3-4B-Instruct | 1,8,16 | — | — | — | 0.7 | pending |

## Phase 1 — GRPO training

| date | exp_name | steps | n_gen | lr | reward_mean_final | gsm8k_acc_final | notes |
|------|----------|-------|-------|-----|-------------------|-----------------|-------|
| — | `qwen3_rlvr_p1_grpo_gsm8k_debug` | 200 | 4 | 1e-6 | — | — | pending |

## Phase 3 — lm-eval-harness

| date | exp_name | tasks | gsm8k | mmlu | hellaswag | notes |
|------|----------|-------|-------|------|-----------|-------|
| — | — | — | — | — | — | — |

## Phase 4 — multi-domain

| date | exp_name | mixture | gsm8k | math | notes |
|------|----------|---------|-------|------|-------|
| — | — | — | — | — | — |

## Phase 5 — verifier

| date | exp_name | verifier_model | bon@16 | pass@16 | notes |
|------|----------|----------------|--------|---------|-------|
| — | — | — | — | — | — |
