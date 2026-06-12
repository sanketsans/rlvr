"""GRPO training loop for GSM8K RLVR."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
from torch.optim import AdamW

from qwen3_rlvr.data.gsm8k import Gsm8kExample, load_gsm8k
from qwen3_rlvr.eval.gsm8k import evaluate_gsm8k_quick
from qwen3_rlvr.generation.rollout import generate_rollouts
from qwen3_rlvr.logging.artifacts import SampleLogger, training_stage
from qwen3_rlvr.logging.wandb_grpo import GRPO_WandbLogger
from qwen3_rlvr.model.load import load_policy_and_reference
from qwen3_rlvr.rewards.exact_match import exact_match_rewards
from qwen3_rlvr.rewards.extract import extract_answer
from qwen3_rlvr.rl.grpo import compute_advantages, compute_policy_loss, GRPOBatch


@dataclass
class TrainerConfig:
    model_path: str
    output_dir: str
    split: str = "train"
    max_samples: Optional[int] = None
    max_steps: int = 200
    batch_size: int = 2
    grad_accum_steps: int = 1
    n_generations: int = 4
    lr: float = 1e-6
    kl_coef: float = 0.04
    temperature: float = 0.7
    max_new_tokens: int = 256
    grad_clip: float = 1.0
    dtype: str = "bfloat16"
    seed: int = 42
    eval_every_steps: int = 50
    eval_split: str = "test"
    eval_max_samples: int = 100
    eval_k: List[int] = field(default_factory=lambda: [1, 8])
    eval_n_generations: int = 8
    eval_max_new_tokens: int = 256
    reinforce: bool = False
    log_every_steps: int = 10
    log_samples_every: int = 50
    sample_table_size: int = 8
    save_every_steps: int = 50
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_tags: Optional[List[str]] = None


class GRPOTrainer:
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)

        random.seed(config.seed)
        torch.manual_seed(config.seed)

        self.train_examples = load_gsm8k(
            split=config.split,
            max_samples=config.max_samples,
            seed=config.seed,
        )
        self.policy, self.reference = load_policy_and_reference(
            config.model_path, dtype=config.dtype
        )
        self.optimizer = AdamW(
            (p for p in self.policy.model.parameters() if p.requires_grad),
            lr=config.lr,
        )
        self.sample_logger = SampleLogger(self.output_dir / "samples.jsonl")
        self.wandb: Optional[GRPO_WandbLogger] = None
        if config.wandb_project:
            self.wandb = GRPO_WandbLogger(
                project=config.wandb_project,
                name=config.wandb_name or "grpo_train" + f"_max_samples_{config.max_samples}",
                entity=config.wandb_entity,
                tags=config.wandb_tags,
                config=vars(config),
            )

    def _sample_batch(self, step: int) -> List[Gsm8kExample]:
        rng = random.Random(self.config.seed + step)
        return rng.sample(self.train_examples, k=min(self.config.batch_size, len(self.train_examples)))

    def _save_checkpoint(self, step: int) -> Path:
        ckpt = self.output_dir / "checkpoints" / f"step_{step}"
        ckpt.mkdir(parents=True, exist_ok=True)
        self.policy.model.save_pretrained(ckpt)
        self.policy.tokenizer.save_pretrained(ckpt)
        meta = {"step": step, "model_path": self.config.model_path}
        with (ckpt / "trainer_state.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return ckpt

    def train(self) -> None:
        cfg = self.config
        device = self.policy.device
        metrics_history: List[dict] = []
        self.optimizer.zero_grad()

        # eval_metrics = evaluate_gsm8k_quick(
        #     loaded=self.policy,
        #     split=cfg.eval_split,
        #     max_samples=cfg.eval_max_samples,
        #     n_generations=cfg.eval_n_generations,
        #     k_values=cfg.eval_k,
        #     max_new_tokens=cfg.eval_max_new_tokens,
        #     temperature=cfg.temperature,
        #     question_batch_size=cfg.batch_size,
        #     seed=cfg.seed,
        # )
        # print(f"  eval pass@1={eval_metrics.get('pass@1', 0):.4f}")
        # if self.wandb is not None:
        #     self.wandb.log_eval(eval_metrics, step=0)
        # eval_path = self.output_dir / f"eval_step_0.json"
        # with eval_path.open("w", encoding="utf-8") as f:
        #     json.dump(eval_metrics, f, indent=2)

        for step in range(1, cfg.max_steps + 1):
            batch = self._sample_batch(step)
            prompts, completions = generate_rollouts(
                loaded=self.policy,
                examples=batch,
                n_generations=cfg.n_generations,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                seed=cfg.seed + step,
            )

            reward_rows = [
                torch.tensor(exact_match_rewards(comp_list, ex.answer), dtype=torch.float32)
                for comp_list, ex in zip(completions, batch)
            ]
            rewards = torch.stack(reward_rows, dim=0)
            advantages = compute_advantages(rewards)

            grpo_batch = GRPOBatch(
                prompts=prompts,
                completions=completions,
                rewards=rewards,
                advantages=advantages,
            )

            loss, loss_metrics = compute_policy_loss(
                policy=self.policy.model,
                reference=self.reference.model,
                tokenizer=self.policy.tokenizer,
                grpo_batch=grpo_batch,
                kl_coef=cfg.kl_coef,
                device=device,
                reinforce=cfg.reinforce,
            )

            step_metrics = {
                "step": step,
                "loss": loss.item(),
                "reward_mean": rewards.mean().item(),
                "reward_std": rewards.std().item(),
                "frac_correct": rewards.mean().item(),
                "advantage_mean": advantages.mean().item(),
                "advantage_std": advantages.std().item(),
                "group_reward_spread": (rewards.max(dim=1).values - rewards.min(dim=1).values).mean().item(),
                **{f"loss/{k}": v for k, v in loss_metrics.items()},
            }

            if loss_metrics["num_loss_terms"] > 0:
                scaled_loss = loss / cfg.grad_accum_steps
                scaled_loss.backward()

            if step % cfg.grad_accum_steps == 0 or step == cfg.max_steps:
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), cfg.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()

            metrics_history.append(step_metrics)

            if step % cfg.log_every_steps == 0:
                print(
                    f"step {step}/{cfg.max_steps} loss={step_metrics['loss']:.4f} "
                    # f"reward={step_metrics['reward_mean']:.3f} kl={step_metrics.get('loss/kl_mean', 0):.4f} "
                    f"reward_std={step_metrics['reward_std']:.3f} reward_mean={step_metrics['reward_mean']:.3f} "
                    f"advantage_mean={step_metrics['advantage_mean']:.3f} advantage_std={step_metrics['advantage_std']:.3f} "
                    f"group_reward_spread={step_metrics['group_reward_spread']:.3f} "
                )

            if self.wandb is not None:
                self.wandb.log_train(step_metrics, step=step)

            if step == 1 or step % cfg.log_samples_every == 0:
                stage = training_stage(step, cfg.max_steps)
                records = []
                for ex, prompt, comp_list, rew_row in zip(batch, prompts, completions, reward_rows):
                    for comp, r in zip(comp_list, rew_row.tolist()):
                        records.append(
                            {
                                "step": step,
                                "stage": stage,
                                "example_id": ex.example_id,
                                "question": ex.question,
                                "ground_truth": extract_answer(ex.answer),
                                "completion": comp,
                                "reward": r,
                            }
                        )
                self.sample_logger.append_many(records[: cfg.sample_table_size])
                if self.wandb is not None:
                    self.wandb.log_samples(records[: cfg.sample_table_size], step=step, stage=stage)

            if step % cfg.eval_every_steps == 0:
                eval_metrics = evaluate_gsm8k_quick(
                    loaded=self.policy,
                    split=cfg.eval_split,
                    max_samples=cfg.eval_max_samples,
                    n_generations=cfg.eval_n_generations,
                    k_values=cfg.eval_k,
                    max_new_tokens=cfg.eval_max_new_tokens,
                    temperature=cfg.temperature,
                    seed=cfg.seed + step,
                    question_batch_size=cfg.batch_size,
                )
                print(f"  eval pass@1={eval_metrics.get('pass@1', 0):.4f}")
                if self.wandb is not None:
                    self.wandb.log_eval(eval_metrics, step=step)
                eval_path = self.output_dir / f"eval_step_{step}.json"
                with eval_path.open("w", encoding="utf-8") as f:
                    json.dump(eval_metrics, f, indent=2)

            # if step % cfg.save_every_steps == 0:
            #     ckpt = self._save_checkpoint(step)
            #     print(f"  saved checkpoint: {ckpt}")

        # self._save_checkpoint(cfg.max_steps)
        history_path = self.output_dir / "train_metrics.jsonl"
        with history_path.open("w", encoding="utf-8") as f:
            for row in metrics_history:
                f.write(json.dumps(row) + "\n")

        if self.wandb is not None:
            self.wandb.finish()
