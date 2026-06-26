"""GRPO training loop for GSM8K RLVR."""

from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm

from qwen3_rlvr.data.base import VerifiableExample
from qwen3_rlvr.data.recipe import load_dataset_by_name, load_recipe
from qwen3_rlvr.eval.recipe_eval import evaluate_recipe_quick
from qwen3_rlvr.generation.rollout import generate_rollouts
from qwen3_rlvr.logging.artifacts import SampleLogger, training_stage
from qwen3_rlvr.logging.logger import setup_logger
from qwen3_rlvr.logging.wandb_grpo import GRPO_WandbLogger
from qwen3_rlvr.model.load import load_policy_and_reference
from qwen3_rlvr.rewards.exact_match import exact_match_rewards
from qwen3_rlvr.rl.grpo import GRPOBatch, compute_advantages, compute_policy_loss

logger = setup_logger(__name__)


@dataclass
class TrainerConfig:
    model_path: str
    output_dir: str
    split: str = "train"
    dataset_name: Optional[str] = "gsm8k"
    recipe: Optional[str] = "gsm8k_train"
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
    eval_batch_size: int = 32
    eval_every_steps: int = 50
    eval_recipes: List[str] = field(default_factory=lambda: ["gsm8k_test"])
    eval_primary_recipe: str = "gsm8k_test"
    eval_split: str = "test"
    eval_max_samples: int = 100
    eval_k: List[int] = field(default_factory=lambda: [1, 8])
    eval_n_generations: int = 8
    eval_max_new_tokens: int = 256
    reinforce: bool = False  # whether to use reinforce training
    grpo_epochs: int = 1  # number of epochs to run GRPO policy updates / rollouts for each batch
    log_every_steps: int = 10
    log_samples_every: int = 50
    sample_table_size: int = 8
    save_every_steps: int = 50
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_tags: Optional[List[str]] = None


class Trainer(ABC):
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)

        random.seed(config.seed)
        torch.manual_seed(config.seed)

        self.train_examples = self._load_train_examples(config)
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

    def _load_train_examples(self, config: TrainerConfig) -> List[VerifiableExample]:
        if config.recipe:
            return load_recipe(
                recipe=config.recipe,
                max_samples=config.max_samples,
                seed=config.seed,
            )
        dataset_name = config.dataset_name or "gsm8k"
        return load_dataset_by_name(
            name=dataset_name,
            split=config.split,
            max_samples=config.max_samples,
            seed=config.seed,
        )

    def _sample_batch(self, step: int) -> List[VerifiableExample]:
        rng = random.Random(self.config.seed + step)
        return rng.sample(
            self.train_examples, k=min(self.config.batch_size, len(self.train_examples))
        )

    def _save_checkpoint(self, step: int) -> Path:
        ckpt = self.output_dir / "checkpoints" / f"step_{step}"
        ckpt.mkdir(parents=True, exist_ok=True)
        self.policy.model.save_pretrained(ckpt)
        self.policy.tokenizer.save_pretrained(ckpt)
        meta = {"step": step, "model_path": self.config.model_path}
        with (ckpt / "trainer_state.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return ckpt

    def log_samples(
        self,
        cfg: TrainerConfig,
        step_metrics: dict,
        step: int,
        batch: list[Any],
        completions: list[list[str]],
        grpo_batch: GRPOBatch,
    ) -> None:
        if step % cfg.log_every_steps == 0:
            log_string = ""
            for k, v in step_metrics.items():
                log_string += f"{k}={v:.5f} "
            logger.info(f"step {step}/{cfg.max_steps} loss={step_metrics['loss']:.4f} {log_string}")

        if self.wandb is not None:
            self.wandb.log_train(step_metrics, step=step)

        if step == 1 or step % cfg.log_samples_every == 0:
            stage = training_stage(step, cfg.max_steps)
            records = []
            for ex, comp_list, rew_row in zip(batch, completions, grpo_batch.rewards):
                num_correct = (rew_row > 0).sum().item()
                # Log both best and worst completions/rewards, but only best goes to wandb (by truncation below)
                records.append(
                    {
                        "step": step,
                        "stage": stage,
                        "example_id": ex.example_id,
                        "source": ex.source,
                        "question": ex.question,
                        "ground_truth": ex.answer,
                        "first_completion": comp_list[0],
                        "reward": rew_row[0].item(),
                        "num_correct": num_correct,
                    }
                )

            self.sample_logger.append_many(records[: cfg.sample_table_size])
            if self.wandb is not None:
                self.wandb.log_samples(records[: cfg.sample_table_size], step=step, stage=stage)

    @abstractmethod
    def train(self) -> None: ...

    def eval(self, cfg: TrainerConfig, last_eval_metrics: dict, step: int) -> dict:
        all_eval: dict = {}
        flat_wandb: dict = {}

        for recipe in cfg.eval_recipes:
            eval_metrics = evaluate_recipe_quick(
                loaded=self.policy,
                recipe=recipe,
                max_samples=cfg.eval_max_samples,
                n_generations=cfg.eval_n_generations,
                k_values=cfg.eval_k,
                max_new_tokens=cfg.eval_max_new_tokens,
                temperature=cfg.temperature,
                question_batch_size=cfg.eval_batch_size,
                seed=cfg.seed + step,
            )
            all_eval[recipe] = eval_metrics
            pass_at_1 = eval_metrics.get("pass@1", eval_metrics.get("accuracy", 0.0))
            logger.info(f"  eval [{recipe}] pass@1={pass_at_1:.4f}")
            for key, value in eval_metrics.items():
                if key in {"by_source", "k_values", "method"}:
                    continue
                if isinstance(value, (int, float)):
                    flat_wandb[f"{recipe}/{key}"] = value

        if self.wandb is not None:
            self.wandb.log_eval(flat_wandb, step=step)

        eval_path = self.output_dir / f"eval_step_{step}.json"
        with eval_path.open("w", encoding="utf-8") as f:
            json.dump(all_eval, f, indent=2)

        primary = all_eval.get(cfg.eval_primary_recipe, next(iter(all_eval.values())))
        primary_pass_at_1 = primary.get("pass@1", primary.get("accuracy", 0.0))

        if step == 0:
            last_eval_metrics = primary

        last_pass_at_1 = 0.0
        if last_eval_metrics is not None:
            last_pass_at_1 = last_eval_metrics.get("pass@1", last_eval_metrics.get("accuracy", 0.0))

        if last_eval_metrics is None or (step > 0 and primary_pass_at_1 > last_pass_at_1):
            ckpt = self._save_checkpoint(step)
            logger.info(
                f"  saved checkpoint: {ckpt} "
                f"with {cfg.eval_primary_recipe} pass@1={primary_pass_at_1:.4f}"
            )
            last_eval_metrics = primary

        return last_eval_metrics


class ReinforceTrainer(Trainer):
    def train(self) -> None:
        cfg = self.config
        metrics_history: List[dict] = []
        self.optimizer.zero_grad()

        last_eval_metrics = self.eval(cfg, {}, 0)
        # last_eval_metrics = None

        for step in range(1, cfg.max_steps + 1):
            batch = self._sample_batch(step)
            (
                _prompts,
                completions,
                tokenized_input_ids,
                tokenized_attention_mask,
                tokenized_completion_mask,
                _,
            ) = generate_rollouts(
                loaded=self.policy,
                examples=batch,
                n_generations=cfg.n_generations,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                seed=cfg.seed + step,
                return_logprobs=False,
            )

            reward_rows = [
                torch.tensor(exact_match_rewards(comp_list, ex.answer), dtype=torch.float32)
                for comp_list, ex in zip(completions, batch)
            ]
            rewards = torch.stack(reward_rows, dim=0)
            advantages = compute_advantages(rewards)

            grpo_batch = GRPOBatch(
                rewards=rewards,
                tokenized_input_ids=tokenized_input_ids,
                tokenized_attention_mask=tokenized_attention_mask,
                tokenized_completion_mask=tokenized_completion_mask,
                advantages=advantages,
            )

            loss, loss_metrics = compute_policy_loss(
                policy=self.policy.model,
                reference=self.reference.model,
                tokenized_input_ids=tokenized_input_ids,
                tokenized_attention_mask=tokenized_attention_mask,
                tokenized_completion_mask=tokenized_completion_mask,
                grpo_batch=grpo_batch,
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
                "group_reward_spread": (rewards.max(dim=1).values - rewards.min(dim=1).values)
                .mean()
                .item(),
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
            self.log_samples(cfg, step_metrics, step, batch, completions, grpo_batch)

            if step % cfg.eval_every_steps == 0:
                last_eval_metrics = self.eval(cfg, last_eval_metrics, step)

        # self._save_checkpoint(cfg.max_steps)
        history_path = self.output_dir / "train_metrics.jsonl"
        with history_path.open("w", encoding="utf-8") as f:
            for row in metrics_history:
                f.write(json.dumps(row) + "\n")

        if self.wandb is not None:
            self.wandb.finish()


class GRPOTrainer(Trainer):
    def train(self) -> None:
        cfg = self.config
        metrics_history: List[dict] = []
        self.optimizer.zero_grad()

        last_eval_metrics = self.eval(cfg, {}, 0)
        # last_eval_metrics = None
        for step in tqdm(range(1, cfg.max_steps + 1), desc="Training GRPO"):
            batch = self._sample_batch(step)
            (
                _prompts,
                completions,
                tokenized_input_ids,
                tokenized_attention_mask,
                tokenized_completion_mask,
                old_logprobs,
            ) = generate_rollouts(
                loaded=self.policy,
                examples=batch,
                n_generations=cfg.n_generations,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                seed=cfg.seed + step,
                return_logprobs=True,
            )

            reward_rows = [
                torch.tensor(exact_match_rewards(comp_list, ex.answer), dtype=torch.float32)
                for comp_list, ex in zip(completions, batch)
            ]
            rewards = torch.stack(reward_rows, dim=0)
            advantages = compute_advantages(rewards)

            grpo_batch = GRPOBatch(
                rewards=rewards,
                advantages=advantages,
                old_token_logp=old_logprobs,
                tokenized_input_ids=tokenized_input_ids,
                tokenized_attention_mask=tokenized_attention_mask,
                tokenized_completion_mask=tokenized_completion_mask,
            )
            step_metrics = {
                "step": step,
                "reward_mean": rewards.mean().item(),
                "reward_std": rewards.std().item(),
                "frac_correct": rewards.mean().item(),
                "advantage_mean": advantages.mean().item(),
                "advantage_std": advantages.std().item(),
                "group_reward_spread": (rewards.max(dim=1).values - rewards.min(dim=1).values)
                .mean()
                .item(),
            }

            grpo_epoch_metrics = defaultdict(list)
            for epoch_idx in range(cfg.grpo_epochs):
                loss, loss_metrics = compute_policy_loss(
                    policy=self.policy.model,
                    tokenizer=self.policy.tokenizer,
                    reference=self.reference.model,
                    grpo_batch=grpo_batch,
                    kl_coef=cfg.kl_coef,
                    reinforce=False,
                )
                if loss_metrics["num_loss_terms"] == 0:
                    if epoch_idx == 0:
                        logger.info(
                            f"  step {step}: skipping policy update "
                            "(all group advantages are ~0; uniform rewards within groups)"
                        )
                    continue
                grpo_epoch_metrics["loss"].append(loss.item())
                for k, v in loss_metrics.items():
                    grpo_epoch_metrics[k].append(v)

                loss.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), cfg.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()

            if grpo_epoch_metrics["loss"]:
                step_metrics.update(
                    {
                        "num_updates": len(grpo_epoch_metrics["loss"]),
                        "loss": float(np.mean(grpo_epoch_metrics["loss"])),
                        **{f"loss/{k}": float(np.mean(v)) for k, v in grpo_epoch_metrics.items()},
                    }
                )
            else:
                # Every epoch was skipped: all group advantages were ~0 (uniform
                # rewards within groups), so no policy update was applied. Record
                # that explicitly rather than calling np.mean([]), which emits a
                # "Mean of empty slice" RuntimeWarning and yields NaN.
                step_metrics.update({"num_updates": 0, "loss": float("nan")})
            metrics_history.append(step_metrics)
            self.log_samples(cfg, step_metrics, step, batch, completions, grpo_batch)

            if step % cfg.eval_every_steps == 0:
                last_eval_metrics = self.eval(cfg, last_eval_metrics, step)

        # self._save_checkpoint(cfg.max_steps)
        history_path = self.output_dir / "train_metrics.jsonl"
        with history_path.open("w", encoding="utf-8") as f:
            for row in metrics_history:
                f.write(json.dumps(row) + "\n")

        if self.wandb is not None:
            self.wandb.finish()
