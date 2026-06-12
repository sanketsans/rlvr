# Phase-by-phase guide

**Base model:** `Qwen/Qwen3-4B-Instruct-2507` at `models/Qwen3-4B-Instruct/`

## Phase 0 — Pass@K baseline

**Purpose:** Know the instruct-model starting point before any RL.

### Deliverables

- `scripts/pass_at_k.py`
- `configs/phase0_passk.yaml`
- W&B run `qwen3_rlvr_p0_passk_gsm8k_*`

### Procedure

1. Ensure model downloaded: `bash scripts/download_model.sh`
2. Run Pass@K on GSM8K `test` split.
3. Use Qwen3 chat template for prompts.
4. Extract answer after `####`.
5. Log `pass@1`, `pass@8`, `pass@16`.

### Local command

```bash
conda activate olmo
cd /home/coder/Projects/qwen3_rlvr
pip install -e .

export MODEL_PATH=/home/coder/Projects/qwen3_rlvr/models/Qwen3-4B-Instruct

# Full baseline (or use --config configs/phase0_passk.yaml)
python scripts/pass_at_k.py \
  --config configs/phase0_passk.yaml \
  --model $MODEL_PATH \
  --k 1,8,16 \
  --n-generations 16 \
  --temperature 0.7 \
  --max-samples 200 \
  --wandb-run qwen3_rlvr_p0_passk_gsm8k_debug

# Re-evaluate any checkpoint later (same script)
python scripts/pass_at_k.py \
  --model $OUT_ROOT/<exp>/checkpoints/step_500 \
  --k 1,8,16 \
  --n-generations 16 \
  --no-wandb \
  --output-dir $OUT_ROOT/<exp>/eval_passk_step_500
```

Outputs:
- `pass_at_k_summary.json` — aggregate metrics
- `pass_at_k_details.jsonl` — per-question completions and correctness
- `resource_monitor.json` — optional CPU/GPU time series (`--monitor-resources`)

### GPU / batch-size helpers

```bash
# Option A: monitor during pass@k eval
python scripts/pass_at_k.py --config configs/phase0_passk.yaml --monitor-resources

# Option B: standalone monitor in another terminal while a job runs
python scripts/monitor_resources.py --duration 300 --interval 2 \
  --output outputs/resource_watch.json

# Option C: probe largest generation batch size that fits on GPU
python scripts/probe_batch_size.py \
  --model $MODEL_PATH --max-new-tokens 128 --max-batch 32
```

---

## Phase 1 — Minimal GSM8K GRPO

**Purpose:** Smallest working RLVR loop — one domain, binary reward, GRPO only.

### Deliverables

- `src/qwen3_rlvr/rl/grpo.py`, `trainer.py`
- `scripts/train_grpo.py`
- `configs/phase1_gsm8k_grpo_debug.yaml`
- `skypilot/phase1_grpo_debug.yaml`

### Training loop

```
1. Sample B questions from GSM8K train
2. Generate N completions per question (temperature > 0)
3. reward = 1 if answer == gt else 0
4. Normalize rewards within each question group
5. advantages = normalized rewards
6. policy_loss = -advantage * log_prob + kl_coef * KL(π || π_ref)
7. optimizer.step()
```

### Local launch

```bash
export MODEL_PATH=/home/coder/Projects/qwen3_rlvr/models/Qwen3-4B-Instruct
torchrun --nproc_per_node=1 scripts/train_grpo.py \
  --config configs/phase1_gsm8k_grpo_debug.yaml
```

### SkyPilot launch

```bash
sky jobs launch skypilot/phase1_grpo_debug.yaml \
  --env WANDB_API_KEY=$WANDB_API_KEY \
  --env EXP_NAME=qwen3_rlvr_p1_grpo_gsm8k_debug
```

---

## Phase 2 — Verify and visualize

Built into the GRPO trainer (`scripts/train_grpo.py`):

- Periodic GSM8K eval every `eval_every_steps` → `eval_step_<N>.json`
- W&B sample tables tagged `early` / `mid` / `late`
- `samples.jsonl` appended during training

Offline HTML gallery:

```bash
python scripts/make_progress_report.py \
  --samples outputs/qwen3_rlvr_p1_grpo_gsm8k_debug/samples.jsonl \
  --output outputs/qwen3_rlvr_p1_grpo_gsm8k_debug/progress.html
```

Compare Phase 0 Pass@K vs post-RL using `scripts/pass_at_k.py` on `checkpoints/step_<N>/`.

---

## Phase 3 — Generalization benchmarks

**lm-evaluation-harness** on HF checkpoints directly:

| Tier | Tasks |
|------|-------|
| In-domain | `gsm8k` |
| Nearby | `mathqa`, `minerva_math` (subset) |
| Far | `mmlu` (5-shot), `hellaswag` |

```bash
lm_eval --model hf \
  --model_args pretrained=$CHECKPOINT,dtype=bfloat16 \
  --tasks gsm8k,mmlu,hellaswag \
  --num_fewshot 5 \
  --batch_size auto \
  --output_path $OUT_ROOT/eval_harness/<exp_name>
```

---

## Phase 4 — Multi-domain mixture

```yaml
mixture:
  gsm8k: 0.65
  hendrycks_math_algebra: 0.15
  hendrycks_math_num_theory: 0.10
  mathqa: 0.10
```

---

## Phase 5 — Verifier + best-of-N

Verifier candidates (same Qwen family):

| Model | VRAM | Notes |
|-------|------|-------|
| Qwen2.5-0.5B-Instruct | ~2 GB | Fast iteration |
| Qwen2.5-1.5B-Instruct | ~4 GB | Better accuracy |
| LoRA on Qwen3-0.6B | ~4 GB | Same tokenizer ecosystem |

---

## Experiment registry

Record runs in [experiments/REGISTRY.md](../experiments/REGISTRY.md).
