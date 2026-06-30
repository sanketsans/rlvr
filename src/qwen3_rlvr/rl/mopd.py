"""MOPD — Multi-teacher On-Policy Distillation (dense token-level reverse-KL).

This is the on-policy distillation objective from the Nemotron-3-Ultra recipe,
implemented to slot into the same machinery as ``grpo.py`` (same tokenizer path,
same :func:`batched_sequence_log_probs`, same PPO-style clipped surrogate).

------------------------------------------------------------------------------
What MOPD changes vs GRPO (the whole point in one paragraph)
------------------------------------------------------------------------------
GRPO's advantage is a *sparse, sequence-level* scalar: a verifier scores the
whole completion (right/wrong), the group is mean/std-normalized, and that one
number is broadcast to every token. MOPD's advantage is a *dense, per-token*
signal that comes from a **teacher model** scoring the student's own rollout:

        Â_t  =  stop_grad[ log π_T(y_t | s_t)  −  log π_prox(y_t | s_t) ]

i.e. at each token, "how much more likely is this token under the teacher than
under the policy that sampled it." Maximizing the policy logprob weighted by Â_t
*maximizes the negative reverse-KL* and therefore *minimizes* the reverse-KL
D_KL(π_θ ‖ π_T) along trajectories the student actually visits. See
``post-training-kb/nemotron-3-ultra.md`` for the sign-convention note
(objective = teacher − student because J = −reverse_KL).

Everything else — the clipped ratio r_t, the optional stale-rollout correction
c_t — is reused verbatim from PPO/GRPO. So MOPD = GRPO's surrogate with the
verifier advantage swapped for a dense teacher reverse-KL advantage.

        GRPO   :  loss_t = −min( r_t · A_group ,  clip(r_t) · A_group )
        MOPD   :  loss_t = −c_t · min( r_t · Â_t ,  clip(r_t) · Â_t )

        with   r_t = π_θ / π_prox ,  Â_t = sg[logπ_T − logπ_prox] ,
               c_t = sg[π_prox / π_behav]   (=1 in the synchronous case)

------------------------------------------------------------------------------
⚠️  CAVEATS FOR A SINGLE-DATASET / SINGLE-DOMAIN SETUP (read this)
------------------------------------------------------------------------------
MOPD's headline result ("Multi-teacher") is about *merging many domain
specialists* into one student when a single unified RL stage dilutes per-domain
signal. With ONE dataset that motivation largely evaporates. Specifically:

1.  THE "MULTI" IN MOPD IS MOOT. One dataset ≈ one domain ≈ one teacher. The
    λ-weighted merge over teachers collapses to a single teacher term. You get
    plain on-policy distillation, not the multi-teacher capability transfer that
    makes the paper interesting.

2.  YOU NEED A TEACHER THAT IS ACTUALLY BETTER THAN THE STUDENT *on this data*.
    Distillation can only transfer what the teacher knows. If your only teacher
    is the same base model (or not meaningfully stronger on this task), Â_t ≈ 0
    on average and you mostly add noise. RLVR (verifier reward) is the better
    tool when you have no stronger teacher but you DO have a checkable answer —
    which, for GSM8K-style data, you do.

3.  MOPD ONLY HELPS WHERE THE TEACHER'S EDGE IS TOKEN-LEVEL PREFERENCES OVER
    TRAJECTORIES THE STUDENT ALREADY SAMPLES (formatting, tool/abstention
    decisions, step ordering, calibration). It does NOT inject *new
    capabilities* the student never explores — those rollouts are OOD for the
    teacher and its scoring is unreliable (the paper's HLE failure case).

4.  DISTRIBUTION MISMATCH ⇒ MOPD-WARMUP IS MANDATORY if the teacher came from a
    different SFT lineage than the student. Otherwise the student's rollouts are
    OOD for the teacher and the per-token supervision is garbage. Fix: a brief,
    light SFT on the teacher's own outputs to align supports *before* MOPD.

------------------------------------------------------------------------------
HOW TO ACTUALLY LEVERAGE MOPD IN A SINGLE-DATASET SETUP (pick one)
------------------------------------------------------------------------------
A.  RLVR → MOPD distillation back-end. Run your existing GRPO/RLVR to produce a
    strong checkpoint, then use THAT as the teacher and distill it on-policy into
    a fresh/smaller student. This is the legitimate single-dataset use: cheap,
    dense supervision to compress or stabilize an RLVR-trained model. (Set
    ``teacher_paths=[<your best RLVR ckpt>]``.)

B.  Sequence-level "specialists" instead of domain specialists. On one dataset
    you can still build distinct teachers by *recipe*: e.g. a high-temperature
    long-CoT teacher, a concise-answer teacher, a tool-use teacher — each RL'd
    with a different reward shaping — then MOPD-merge their token preferences
    into one student. This recreates the multi-teacher benefit without multiple
    datasets.

C.  Hybrid reward: blend the dense teacher advantage with the sparse verifier
    advantage (``verifier_coef`` below). Teacher gives shape/coverage, verifier
    keeps it grounded in correctness. Often the most robust single-dataset move.

D.  Self-distillation / iterated MOPD: teacher = the student's own best earlier
    checkpoint (Best-of-N filtered). Tightens the policy around its own good
    trajectories. Diminishing returns, but stabilizing.

If none of A–D apply (your only teacher is the base model and you have a
verifier), DON'T use MOPD — just run GRPO. MOPD is strictly extra compute
(teacher forward passes per rollout) for no signal in that case.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

import torch
from torch import nn

from qwen3_rlvr.rl.grpo import batched_sequence_log_probs


@dataclass
class MOPDBatch:
    """Tensors needed for one MOPD update. All token tensors are [B*N, L-1].

    They are produced by the exact same rollout/tokenize path GRPO uses, so the
    causal shift (logit at t predicts token t+1) and the completion masking
    already line up — non-completion tokens are zeroed by
    :func:`batched_sequence_log_probs`.
    """

    tokenized_input_ids: torch.Tensor  # [B*N, L]
    tokenized_attention_mask: torch.Tensor  # [B*N, L]
    tokenized_completion_mask: torch.Tensor  # [B*N, L]
    # log π_behav: the snapshot that SAMPLED the rollout (from generate_rollouts).
    behavior_token_logp: torch.Tensor  # [B*N, L-1]
    # log π_T: teacher scoring of the SAME tokens. Frozen ⇒ enters as stop-grad.
    # For multi-teacher this is already the λ-weighted combination (see
    # combine_teacher_logps): Σ_i λ_i log π_Ti.
    teacher_token_logp: torch.Tensor  # [B*N, L-1]
    # log π_prox: optional proximal trust-region snapshot for async/stale
    # rollouts (AReaL-style). If None we run fully synchronous: prox == behavior.
    prox_token_logp: Optional[torch.Tensor] = None  # [B*N, L-1]
    # Optional sparse verifier advantage (group-normalized scalar per sequence),
    # broadcast to tokens, for the hybrid objective (caveat C). [B*N] or None.
    verifier_advantages: Optional[torch.Tensor] = None
    # Verifier rewards [B, N] — not used by the loss; kept so the shared
    # Trainer.log_samples (which reads .rewards) works unchanged.
    rewards: Optional[torch.Tensor] = None


def combine_teacher_logps(
    teacher_logps: Sequence[torch.Tensor],
    weights: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """λ-weighted merge of per-token teacher logprobs: Σ_i λ_i · log π_Ti.

    This is the "multi-teacher" merge. With one teacher it is the identity.
    Weights are normalized to sum to 1 so the advantage stays on a log-ratio
    scale comparable to a single teacher. Each tensor is [B*N, L-1].
    """
    if len(teacher_logps) == 1:
        return teacher_logps[0]
    if weights is None:
        weights = [1.0 / len(teacher_logps)] * len(teacher_logps)
    total = float(sum(weights))
    weights = [w / total for w in weights]
    out = torch.zeros_like(teacher_logps[0])
    for logp, w in zip(teacher_logps, weights):
        out = out + w * logp
    return out


@torch.no_grad()
def score_with_teachers(
    teachers: Sequence[nn.Module],
    mopd_batch: MOPDBatch | dict,
    weights: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """Run each frozen teacher over the student's rollout and λ-merge the logps.

    Teachers never receive gradient; this is pure scoring. Returns [B*N, L-1].
    Accepts either a MOPDBatch (uses its tokenized_* fields) or a dict with the
    same keys, so it can be called before the batch is fully assembled.
    """
    if isinstance(mopd_batch, MOPDBatch):
        ids = mopd_batch.tokenized_input_ids
        attn = mopd_batch.tokenized_attention_mask
        comp = mopd_batch.tokenized_completion_mask
    else:
        ids = mopd_batch["tokenized_input_ids"]
        attn = mopd_batch["tokenized_attention_mask"]
        comp = mopd_batch["tokenized_completion_mask"]
    per_teacher = [batched_sequence_log_probs(t, ids, attn, comp).float() for t in teachers]
    return combine_teacher_logps(per_teacher, weights)


def compute_token_advantages(
    teacher_token_logp: torch.Tensor,
    prox_token_logp: torch.Tensor,
    valid_mask: torch.Tensor,
    normalize: bool = False,
    adv_clip: Optional[float] = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The dense reverse-KL advantage: Â_t = sg[log π_T − log π_prox].

    This is the negative per-token reverse-KL contribution (teacher − policy).
    Maximizing the policy logprob under this advantage minimizes
    D_KL(π_θ ‖ π_T). Both inputs are already zeroed outside completion tokens,
    so the difference is too; we still pass ``valid_mask`` for normalization
    statistics.

    Returns ``(advantages, reverse_kl_per_token)`` — the second is the *positive*
    reverse-KL (log π_prox − log π_T) for logging the actual divergence.
    """
    # stop-grad: the advantage is a fixed target wrt the policy. Teacher is
    # frozen anyway; prox is a snapshot. Detach to be explicit and safe.
    advantages = (teacher_token_logp - prox_token_logp).detach()
    reverse_kl = (prox_token_logp - teacher_token_logp).detach()  # = −advantages

    if normalize:
        # Whiten over valid (completion) tokens only — stabilizes the scale when
        # teacher/policy log-ratios are large. Optional; off by default to stay
        # faithful to the raw reverse-KL signal.
        m = valid_mask.bool()
        if m.any():
            vals = advantages[m]
            advantages = (advantages - vals.mean()) / (vals.std() + eps)
            advantages = advantages * valid_mask  # re-zero non-completion tokens

    if adv_clip is not None:
        # Guard against pathological tokens where the teacher is extremely
        # confident and the policy is not (log-ratio blows up). Symmetric clamp.
        advantages = advantages.clamp(-adv_clip, adv_clip)

    return advantages, reverse_kl


def compute_mopd_loss(
    policy: nn.Module,
    mopd_batch: MOPDBatch,
    tokenizer: Optional[Any] = None,
    clip_eps: float = 0.2,
    normalize_advantages: bool = False,
    adv_clip: Optional[float] = None,
    verifier_coef: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """MOPD policy loss: PPO-clipped surrogate driven by the dense teacher advantage.

    Mirrors :func:`grpo.compute_policy_loss` structurally. The two differences:
      * the advantage is per-token (teacher reverse-KL), not a broadcast scalar;
      * there is no separate KL-to-reference penalty — the teacher *is* the
        anchor (the objective already is a KL to the teacher).
    """
    policy.train()
    ids = mopd_batch.tokenized_input_ids
    attn = mopd_batch.tokenized_attention_mask
    comp = mopd_batch.tokenized_completion_mask

    # Completion-token mask aligned to the causal-shifted logp length [B*N, L-1].
    valid_mask = comp[:, 1:].float()
    n_valid = valid_mask.sum()
    if n_valid <= 0:
        return torch.zeros((), device=ids.device), {"num_completion_tokens": 0}

    # Current policy logprobs (WITH gradient). [B*N, L-1].
    policy_token_logp = batched_sequence_log_probs(policy, ids, attn, comp).float()

    behavior_logp = mopd_batch.behavior_token_logp.float()
    # Synchronous (on-policy) default: proximal == behavior, so c_t == 1 and the
    # ratio is exp(policy − behavior). Async/stale rollouts pass a real prox.
    prox_logp = (
        mopd_batch.prox_token_logp.float()
        if mopd_batch.prox_token_logp is not None
        else behavior_logp
    )
    teacher_logp = mopd_batch.teacher_token_logp.float()

    # Dense reverse-KL advantage (stop-grad target).
    advantages, reverse_kl = compute_token_advantages(
        teacher_logp, prox_logp, valid_mask, normalize=normalize_advantages, adv_clip=adv_clip
    )

    # Optional hybrid: add the sparse verifier advantage (broadcast to tokens).
    # Keeps the policy grounded in correctness while the teacher shapes tokens.
    if verifier_coef > 0.0 and mopd_batch.verifier_advantages is not None:
        v = mopd_batch.verifier_advantages.float().to(advantages.device).unsqueeze(-1)
        advantages = advantages + verifier_coef * v * valid_mask

    # PPO-clipped surrogate around the proximal policy.
    ratio = torch.exp(policy_token_logp - prox_logp)
    clip_adv = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    pg_token = -torch.min(ratio * advantages, clip_adv)

    # Stale-rollout correction c_t = sg[π_prox / π_behav]; == 1 when prox==behavior.
    correction = torch.exp(prox_logp - behavior_logp).detach()

    # Token-mean over completion tokens only.
    loss = (correction * pg_token * valid_mask).sum() / n_valid

    with torch.no_grad():
        clip_fraction = (((ratio - 1.0).abs() > clip_eps).float() * valid_mask).sum() / n_valid
        rkl_mean = (reverse_kl * valid_mask).sum() / n_valid
        metrics = {
            "loss": loss.item(),
            "num_completion_tokens": int(n_valid.item()),
            "reverse_kl_per_token": rkl_mean.item(),  # actual D_KL(π_θ‖π_T) estimate
            "advantage_mean": (advantages * valid_mask).sum().item() / n_valid.item(),
            "advantage_abs_mean": (advantages.abs() * valid_mask).sum().item() / n_valid.item(),
            "ratio_mean": (ratio * valid_mask).sum().item() / n_valid.item(),
            "clip_fraction": clip_fraction.item(),
            "correction_mean": (correction * valid_mask).sum().item() / n_valid.item(),
            "teacher_logp_mean": (teacher_logp * valid_mask).sum().item() / n_valid.item(),
            "policy_logp_mean": (policy_token_logp.detach() * valid_mask).sum().item()
            / n_valid.item(),
        }
    return loss, metrics


# ---------------------------------------------------------------------------
# Trainer — wires the loss into the existing rollout / data / eval infra.
# ---------------------------------------------------------------------------
# The pure algorithm above depends only on torch + grpo. The trainer below pulls
# in the heavy training/eval stack (transformers, datasets, eval, wandb), so we
# build it lazily via module __getattr__ — mirroring rl/trainer.py being deferred
# in rl/__init__.py. This keeps `from qwen3_rlvr.rl.mopd import compute_mopd_loss`
# dependency-light (no need for the full training environment just to use the math).


def _build_trainer():
    """Define and return (MOPDConfig, MOPDTrainer). Imports the heavy stack lazily."""
    import json

    from qwen3_rlvr.data.base import VerifiableExample
    from qwen3_rlvr.generation.rollout import generate_rollouts
    from qwen3_rlvr.logging.logger import setup_logger
    from qwen3_rlvr.model.load import load_model_and_tokenizer
    from qwen3_rlvr.rewards.exact_match import exact_match_rewards
    from qwen3_rlvr.rl.grpo import compute_advantages
    from qwen3_rlvr.rl.trainer import Trainer, TrainerConfig

    logger = setup_logger(__name__)

    @dataclass
    class MOPDConfig(TrainerConfig):
        """TrainerConfig + MOPD knobs. Inherits every base field (lr, batch_size, ...)."""

        teacher_paths: List[str] = field(default_factory=list)  # one path = single teacher
        teacher_weights: Optional[List[float]] = None  # λ_i; None ⇒ uniform
        clip_eps: float = 0.2
        normalize_advantages: bool = False
        adv_clip: Optional[float] = None  # e.g. 5.0 to clamp |Â_t|
        verifier_coef: float = 0.0  # >0 ⇒ hybrid teacher + verifier advantage
        mopd_epochs: int = 1  # PPO-style inner epochs per rollout batch

    class MOPDTrainer(Trainer):
        """On-policy distillation trainer. Student rollouts, teacher-scored advantage."""

        def __init__(self, config: MOPDConfig):
            super().__init__(config)  # loads policy + (unused) reference, optimizer, logging
            if not config.teacher_paths:
                raise ValueError(
                    "MOPDTrainer needs at least one teacher_path. With a single dataset, "
                    "the natural choice is your best RLVR/GRPO checkpoint (see caveat A "
                    "in mopd.py)."
                )
            self.teachers = []
            for path in config.teacher_paths:
                t = load_model_and_tokenizer(path, dtype=config.dtype, train=False)
                for p in t.model.parameters():
                    p.requires_grad = False
                self.teachers.append(t)
            logger.info(
                f"Loaded {len(self.teachers)} teacher(s) for MOPD "
                f"(weights={config.teacher_weights or 'uniform'})"
            )
            if len(self.teachers) == 1:
                logger.warning(
                    "Single teacher + (likely) single dataset: this is plain on-policy "
                    "distillation, NOT multi-teacher merging. Ensure the teacher is "
                    "meaningfully stronger than the student on this data, else prefer GRPO."
                )

        def train(self) -> None:
            cfg: MOPDConfig = self.config  # type: ignore[assignment]
            metrics_history: List[dict] = []
            self.optimizer.zero_grad()
            last_eval_metrics = self.eval(cfg, {}, 0)

            for step in range(1, cfg.max_steps + 1):
                batch: List[VerifiableExample] = self._sample_batch(step)

                # 1) Student samples its own rollouts (on-policy) and we keep the
                #    behavior logprobs (= π_behav, the sampling snapshot).
                (
                    _prompts,
                    completions,
                    ids,
                    attn,
                    comp,
                    behavior_logp,
                ) = generate_rollouts(
                    loaded=self.policy,
                    examples=batch,
                    n_generations=cfg.n_generations,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                    seed=cfg.seed + step,
                    return_logprobs=True,
                )

                # 2) Frozen teacher(s) score the SAME tokens → dense supervision.
                teacher_logp = score_with_teachers(
                    [t.model for t in self.teachers],
                    {
                        "tokenized_input_ids": ids,
                        "tokenized_attention_mask": attn,
                        "tokenized_completion_mask": comp,
                    },
                    weights=cfg.teacher_weights,
                )

                # 3) Verifier rewards — for logging, and for the optional hybrid term.
                reward_rows = [
                    torch.tensor(exact_match_rewards(c, ex.answer), dtype=torch.float32)
                    for c, ex in zip(completions, batch)
                ]
                rewards = torch.stack(reward_rows, dim=0)
                verifier_adv = None
                if cfg.verifier_coef > 0.0:
                    verifier_adv = compute_advantages(rewards).reshape(-1)

                mopd_batch = MOPDBatch(
                    tokenized_input_ids=ids,
                    tokenized_attention_mask=attn,
                    tokenized_completion_mask=comp,
                    behavior_token_logp=behavior_logp,
                    teacher_token_logp=teacher_logp,
                    prox_token_logp=None,  # synchronous: prox == behavior
                    verifier_advantages=verifier_adv,
                    rewards=rewards,
                )

                # 4) PPO-style inner epochs on the fixed rollout (behavior frozen).
                step_metrics = {
                    "step": step,
                    "reward_mean": rewards.mean().item(),
                    "frac_correct": rewards.mean().item(),
                }
                losses = []
                for _ in range(cfg.mopd_epochs):
                    loss, loss_metrics = compute_mopd_loss(
                        policy=self.policy.model,
                        mopd_batch=mopd_batch,
                        tokenizer=self.policy.tokenizer,
                        clip_eps=cfg.clip_eps,
                        normalize_advantages=cfg.normalize_advantages,
                        adv_clip=cfg.adv_clip,
                        verifier_coef=cfg.verifier_coef,
                    )
                    if loss_metrics.get("num_completion_tokens", 0) == 0:
                        continue
                    loss.backward()
                    if cfg.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.policy.model.parameters(), cfg.grad_clip
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    losses.append(loss.item())
                    step_metrics.update({f"loss/{k}": v for k, v in loss_metrics.items()})

                step_metrics["loss"] = float(sum(losses) / len(losses)) if losses else float("nan")
                metrics_history.append(step_metrics)
                self.log_samples(cfg, step_metrics, step, batch, completions, mopd_batch)

                if step % cfg.eval_every_steps == 0:
                    last_eval_metrics = self.eval(cfg, last_eval_metrics, step)

            with (self.output_dir / "train_metrics.jsonl").open("w", encoding="utf-8") as f:
                for row in metrics_history:
                    f.write(json.dumps(row) + "\n")
            if self.wandb is not None:
                self.wandb.finish()

    return MOPDConfig, MOPDTrainer


_TRAINER_CACHE: dict = {}


def __getattr__(name: str):
    """Lazily build MOPDConfig / MOPDTrainer on first access (PEP 562)."""
    if name in {"MOPDConfig", "MOPDTrainer"}:
        if not _TRAINER_CACHE:
            cfg, trainer = _build_trainer()
            _TRAINER_CACHE["MOPDConfig"] = cfg
            _TRAINER_CACHE["MOPDTrainer"] = trainer
        return _TRAINER_CACHE[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
