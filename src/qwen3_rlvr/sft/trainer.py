"""Supervised fine-tuning on curated rejection-sampling data."""

from __future__ import annotations

import gc
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from qwen3_rlvr.model.load import load_model_and_tokenizer, LoadedModel
from qwen3_rlvr.logging.wandb_sft import SFTWandbLogger
from qwen3_rlvr.sft.curriculum import (
    build_problem_groups,
    curriculum_batch_indices,
    schedule_for_phase,
    CurriculumConfig,
    CurriculumSchedule,
)
from qwen3_rlvr.sft.dataset import load_gsm8k_sft, load_sft_from_manifest, SFTTokenDataset, _collate
from qwen3_rlvr.eval.recipe_eval import evaluate_recipe_quick
from qwen3_rlvr.logging.logger import setup_logger
from qwen3_rlvr.sft.scheduler import get_cosine_schedule_with_warmup
logger = setup_logger(__name__)


@dataclass
class SFTConfig:
    model_path: str
    output_dir: str
    jsonl_path: Optional[str] = None
    manifest_path: Optional[str] = None
    variant: str = "top2"
    processed_prompt_ids_path: Optional[str] = None
    curriculum: Optional[CurriculumConfig] = None
    include_original: bool = True
    max_samples: Optional[int] = None
    max_steps: int = 500
    batch_size: int = 4
    grad_accum_steps: int = 4
    lr: float = 2e-5
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    min_lr_ratio: float = 0.1
    max_seq_length: int = 2048
    grad_clip: float = 1.0
    dtype: str = "bfloat16"
    seed: int = 42
    log_every_steps: int = 10
    save_every_steps: int = 100
    problem_weighted_sampling: bool = True
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
    # eval config
    eval_every_steps: int = 25
    eval_recipes: List[str] = field(default_factory=lambda: ["gsm8k_test"])
    eval_primary_recipe: str = "gsm8k_test"
    eval_max_samples: int = 200
    eval_max_new_tokens: int = 512
    eval_n_generations: int = 5
    eval_k: List[int] = field(default_factory=lambda: [1, 3, 5])
    eval_temperature: float = 0.7
    eval_question_batch_size: int = 8


class SFTTrainer:
    def __init__(self, config: SFTConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)

        random.seed(config.seed)
        torch.manual_seed(config.seed)

        curriculum_cfg = config.curriculum or CurriculumConfig()
        if config.manifest_path:
            self.examples = load_sft_from_manifest(
                manifest_path=config.manifest_path,
                variant=config.variant,
                max_samples=config.max_samples,
                seed=config.seed,
                include_original=config.include_original,
                processed_prompt_ids_path=config.processed_prompt_ids_path
                or curriculum_cfg.processed_prompt_ids_path,
            )
        elif config.jsonl_path:
            self.examples = load_gsm8k_sft(
                jsonl_path=config.jsonl_path,
                max_samples=config.max_samples,
                seed=config.seed,
                variant=config.variant,
                include_original=config.include_original,
                processed_prompt_ids_path=config.processed_prompt_ids_path
                or curriculum_cfg.processed_prompt_ids_path,
            )
        else:
            raise ValueError("SFTConfig requires jsonl_path or manifest_path.")

        self.curriculum_cfg = curriculum_cfg
        self.curriculum_schedule = (
            curriculum_cfg.to_schedule()
            if curriculum_cfg.enabled
            else CurriculumSchedule(phases=[])
        )
        self.problem_groups = build_problem_groups(self.examples)
        problems_by_difficulty = self.problem_groups[0]
        num_problems = sum(len(groups) for groups in problems_by_difficulty.values())
        self.steps_per_phase = curriculum_cfg.steps_per_phase or max(
            1, num_problems // max(1, config.batch_size)
        )

        self.loaded = load_model_and_tokenizer(
            config.model_path,
            dtype=config.dtype,
            train=True,
        )
        self.tokenizer = self.loaded.tokenizer
        self.model = self.loaded.model
        self.device = self.loaded.device

        self.token_dataset = SFTTokenDataset(
            examples=self.examples,
            tokenizer=self.tokenizer,
            max_seq_length=config.max_seq_length,
        )
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        self.pad_token_id = pad_token_id
        self.collate_fn = lambda batch: _collate(batch, pad_token_id)
        self.use_curriculum_sampling = (
            config.problem_weighted_sampling or curriculum_cfg.enabled
        )
        if not self.use_curriculum_sampling:
            self.dataloader = DataLoader(
                self.token_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                collate_fn=self.collate_fn,
            )
        else:
            self.dataloader = None

        self.optimizer = AdamW(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=config.lr,
        )
        self.scheduler = None
        if config.lr_scheduler == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer, 
                warmup_ratio=config.warmup_ratio, 
                num_training_steps=config.max_steps, 
                grad_accum_steps=config.grad_accum_steps, 
                min_lr_ratio=config.min_lr_ratio)

        self.wandb: Optional[SFTWandbLogger] = None
        if config.wandb_project:
            self.wandb = SFTWandbLogger(
                project=config.wandb_project,
                name=config.wandb_name or f"sft_{config.variant}",
                entity=config.wandb_entity,
                tags=config.wandb_tags,
                config={
                    "model_path": config.model_path,
                    "variant": config.variant,
                    "max_steps": config.max_steps,
                    "batch_size": config.batch_size,
                    "lr": config.lr,
                    "lr_scheduler": config.lr_scheduler,
                    "warmup_ratio": config.warmup_ratio,
                    "min_lr_ratio": config.min_lr_ratio,
                    "curriculum_enabled": curriculum_cfg.enabled,
                    "problem_weighted_sampling": config.problem_weighted_sampling,
                },
            )
        dataset_stats = self._dataset_stats()
        with (self.output_dir / "dataset_stats.json").open("w", encoding="utf-8") as f:
            json.dump(dataset_stats, f, indent=2)
            # self._log_dataset_stats_to_wandb()

    @staticmethod
    def _cleanup_cuda() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def _dataset_stats(self) -> Dict[str, float]:
        rejection = [ex for ex in self.examples if ex.data_source == "rejection"]
        stats: Dict[str, float] = {
            "num_examples": len(self.examples),
            "num_rejection_examples": len(rejection),
            "steps_per_phase": self.steps_per_phase,
            "batch_size": self.config.batch_size,
            "grad_accum_steps": self.config.grad_accum_steps,
            "max_steps": self.config.max_steps,
        }
        for difficulty in ("easy", "mid", "hard", "very_hard"):
            stats[f"num_rejection_{difficulty}"] = sum(
                1 for ex in rejection if ex.difficulty == difficulty
            )
        for difficulty, groups in self.problem_groups[0].items():
            stats[f"num_rejection_problems_{difficulty}"] = len(groups)
        return stats

    def _log_dataset_stats_to_wandb(self) -> None:
        if self.wandb is not None:
            self.wandb.log_dataset(self._dataset_stats())

    def _batch_stats(self, indices: List[int], step: int) -> Dict[str, float]:
        phase = (step) // self.steps_per_phase
        stats: Dict[str, float] = {"phase": phase}
        for difficulty in ("easy", "mid", "hard", "very_hard"):
            stats[f"batch_frac_{difficulty}"] = sum(
                1 for i in indices if self.examples[i].difficulty == difficulty
            ) / len(indices)
        if self.curriculum_cfg.enabled:
            probs = schedule_for_phase(self.curriculum_schedule, phase)
            for difficulty, weight in probs.items():
                stats[f"curriculum_target_{difficulty}"] = weight
        return stats

    def _next_batch(self, step: int) -> tuple[dict, List[int]]:
        if not self.use_curriculum_sampling:
            assert self.dataloader is not None
            raise RuntimeError("_next_batch called without curriculum sampling iterator state")

        indices = curriculum_batch_indices(
            self.examples,
            schedule=self.curriculum_schedule,
            steps_per_phase=self.steps_per_phase,
            batch_size=self.config.batch_size,
            step=step,
            seed=self.config.seed,
            problem_groups=self.problem_groups,
        )
        items = [self.token_dataset[idx] for idx in indices]
        batch = self.collate_fn(items)
        del items
        return batch, indices

    def _save_checkpoint(self, step: int) -> Path:
        ckpt = self.output_dir / "checkpoints" / f"step_{step}"
        ckpt.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(ckpt)
        self.tokenizer.save_pretrained(ckpt)
        meta = {
            "step": step,
            "model_path": self.config.model_path,
            "num_examples": len(self.examples),
            "variant": self.config.variant,
            "curriculum_enabled": self.curriculum_cfg.enabled,
            "steps_per_phase": self.steps_per_phase,
        }
        with (ckpt / "trainer_state.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return ckpt

    def eval(self, cfg: SFTConfig, last_eval_metrics: dict, step: int) -> dict:
        all_eval: dict = {}
        flat_wandb: dict = {}
        self.model.eval()
        for recipe in cfg.eval_recipes:
            eval_metrics = evaluate_recipe_quick(
                loaded=LoadedModel(model=self.model, tokenizer=self.tokenizer, device=self.device),
                recipe=recipe,
                max_samples=cfg.eval_max_samples,
                n_generations=cfg.eval_n_generations,
                k_values=cfg.eval_k,
                max_new_tokens=cfg.eval_max_new_tokens,
                temperature=cfg.eval_temperature,
                question_batch_size=cfg.eval_question_batch_size,
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
        self.model.train()
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

        self._cleanup_cuda()
        return last_eval_metrics

    def train(self) -> None:
        cfg = self.config
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        data_iter = iter(self.dataloader) if self.dataloader is not None else None
        history_path = self.output_dir / "train_metrics.jsonl"
        history_path.write_text("")

        last_eval_metrics = self.eval(cfg, None, 0)
        self._cleanup_cuda()
        logger.info(f"Initial evaluation: {last_eval_metrics}")

        for step in tqdm(range(cfg.max_steps), desc="SFT training"):
            batch_indices: Optional[List[int]] = None
            if self.use_curriculum_sampling:
                batch, batch_indices = self._next_batch(step)
            else:
                assert data_iter is not None
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.dataloader)
                    batch = next(data_iter)

            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            outputs = self.model(**batch)
            loss = outputs.loss / cfg.grad_accum_steps
            loss_value = outputs.loss.item()
            loss.backward()

            del outputs, loss
            del batch

            if (step + 1) % cfg.grad_accum_steps == 0 or step == cfg.max_steps - 1:
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            phase = step // self.steps_per_phase
            step_metrics = {
                "step": step,
                "loss": loss_value,
                "phase": phase,
                "lr": self._current_lr(),
            }
            if batch_indices is not None:
                step_metrics.update(self._batch_stats(batch_indices, step))
            with history_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(step_metrics) + "\n")

            if step % cfg.log_every_steps == 0:
                logger.info(
                    f"Step {step}/{cfg.max_steps} loss={step_metrics['loss']:.4f} "
                    f"lr={step_metrics['lr']:.2e}"
                )
                if self.wandb is not None:
                    self.wandb.log_train(step_metrics, step=step)

                self._cleanup_cuda()

            if step > 0 and step % cfg.eval_every_steps == 0:
                last_eval_metrics = self.eval(cfg, last_eval_metrics, step)
                logger.info(f"Evaluation at step {step}: {last_eval_metrics}")

        curriculum_meta = {
            "enabled": self.curriculum_cfg.enabled,
            "steps_per_phase": self.steps_per_phase,
            "phases": self.curriculum_cfg.phases,
            "problem_counts": {
                difficulty: len(groups)
                for difficulty, groups in self.problem_groups[0].items()
            },
        }
        with (self.output_dir / "curriculum_state.json").open("w", encoding="utf-8") as f:
            json.dump(curriculum_meta, f, indent=2)

        self._cleanup_cuda()
        if self.wandb is not None:
            self.wandb.finish()
