# Architecture

## System context

```
┌─────────────────────────────────────────────────────────────┐
│  qwen3_rlvr (this repo)                                      │
│  ┌─────────────┐  ┌──────────┐  ┌─────────┐  ┌────────────┐ │
│  │ GSM8K data  │→ │ GRPO     │→ │ W&B     │  │ lm-eval    │ │
│  │ + rewards   │  │ trainer  │  │ logger  │  │ (Phase 3+) │ │
│  └─────────────┘  └────┬─────┘  └─────────┘  └────────────┘ │
└────────────────────────┼────────────────────────────────────┘
                         │ transformers / accelerate
┌────────────────────────▼────────────────────────────────────┐
│  Qwen3-4B-Instruct (HF)                                      │
│  AutoModelForCausalLM · AutoTokenizer · model.generate()   │
└─────────────────────────────────────────────────────────────┘
```

**Source of truth:** All RLVR code lives in `qwen3_rlvr`. No dependency on the Molmo2 vision stack.

## Qwen3-4B-Instruct on GSM8K

- **Model:** `Qwen/Qwen3-4B-Instruct-2507` (same LLM backbone Molmo2-4B uses, without SigLIP).
- **Format:** Standard HuggingFace weights under `models/Qwen3-4B-Instruct/`.
- **Prompting:** Qwen3 chat template via `tokenizer.apply_chat_template()`.
- **Why text-only LLM:** GSM8K is pure math reasoning; a VLM adds complexity without benefit for this learning lab.

## Module layout (`src/qwen3_rlvr/`)

```
qwen3_rlvr/
├── config.py              # OmegaConf / dataclass experiment config
├── model/
│   └── load.py            # Load Qwen3 via transformers; ref policy copy
├── data/
│   ├── gsm8k.py           # HF load, chat prompt template, train/val split
│   └── mixture.py         # Phase 4 weighted multi-domain iterator
├── rewards/
│   ├── exact_match.py     # reward = 1 if extracted answer == gt else 0
│   └── extract.py         # Parse #### answer (GSM8K convention)
├── generation/
│   └── rollout.py         # N samples per question via model.generate()
├── rl/
│   ├── grpo.py            # Advantage computation, KL-regularized policy loss
│   └── trainer.py         # Main training loop
├── eval/
│   ├── gsm8k.py           # Fast in-loop accuracy / Pass@K
│   └── harness.py         # lm-eval wrapper (Phase 3)
├── logging/
│   ├── wandb_logger.py    # Metrics + sample tables + stage tagging
│   └── artifacts.py       # JSONL prediction dumps for offline viz
└── viz/
    └── progress_report.py # HTML gallery: early / mid / late samples
```

## GRPO algorithm (Phase 1)

For each training step, sample a batch of questions `q_1..q_B`. For each `q_i`, generate `N` completions `{y_i,j}_{j=1..N}`.

### 1. Rewards

```
r_i,j = 1[ extract(y_i,j) == gt_i ]
```

### 2. Group normalization (per question)

```
μ_i = mean_j(r_i,j)
σ_i = std_j(r_i,j) + ε
r̂_i,j = (r_i,j - μ_i) / σ_i
```

### 3. Advantages

```
A_i,j = r̂_i,j
```

### 4. Policy loss

```
L_i,j = -A_i,j * Σ_t log π_θ(y_t | q, y_<t)
        + β * KL(π_θ || π_ref)
```

**Reference model:** frozen copy of initial Qwen3-4B-Instruct weights.

### 5. Hyperparameters (starting point)

| Param | Debug | Production |
|-------|-------|--------------|
| `n_generations` (N) | 4 | 8–16 |
| `batch_size` (questions) | 2 | 4–8 |
| `lr` | 1e-6 | 5e-7 – 1e-6 |
| `kl_coef` (β) | 0.04 | 0.02 – 0.1 |
| `max_new_tokens` | 256 | 512 |
| `temperature` | 0.7 | 0.7 – 1.0 |
| `eval_every_steps` | 50 | 200–500 |
| `max_steps` | 200 | 5k – 20k |
| `dtype` | bfloat16 | bfloat16 |

## Phase 0: Pass@K

1. Load `models/Qwen3-4B-Instruct/`.
2. For each test question, generate `K' = max(K)` samples.
3. Compute `pass@1`, `pass@8`, `pass@16`.
4. Log to W&B.

## Phase 3: lm-eval-harness

RL checkpoints are already HF format — no conversion needed:

```bash
lm_eval --model hf \
  --model_args pretrained=$OUT_ROOT/<exp>/checkpoints/step_N,dtype=bfloat16 \
  --tasks gsm8k,mmlu,hellaswag \
  --batch_size auto \
  --output_path $OUT_ROOT/eval_harness/<run_id>
```

## Checkpoint I/O

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
```

Save RL checkpoints with `model.save_pretrained()` + `tokenizer.save_pretrained()`.

## SkyPilot job shape

```yaml
envs:
  MODEL_PATH: /workspace/.../qwen3_rlvr/models/Qwen3-4B-Instruct
  OUT_ROOT: /workspace/.../qwen3_rlvr/outputs
setup: |
  pip install -e $RLVR_ROOT/.[eval]
run: |
  torchrun --nproc_per_node=1 scripts/train_grpo.py \
    --config configs/phase1_gsm8k_grpo_debug.yaml
```

## Open decisions

1. **GRPO stack:** custom Transformers trainer (default) vs TRL `GRPOTrainer`?
2. **Precision:** bf16 policy + fp32 ref, or both bf16?
3. **GPU target:** 1× H200 debug, 4× H200 prod?
4. **Answer extraction:** GSM8K `####` + sympy fallback (default).

Default: (1) custom, (2) bf16 + fp32 ref, (3) 1/4 GPU, (4) `####` + sympy.
