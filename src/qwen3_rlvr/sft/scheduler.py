import math
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(optimizer, warmup_ratio, num_training_steps, grad_accum_steps, min_lr_ratio=0.1):
    num_optimizer_steps = max(1, (num_training_steps + grad_accum_steps - 1) // grad_accum_steps)
    warmup_steps = int(warmup_ratio * num_optimizer_steps)
    
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, num_optimizer_steps - warmup_steps))
        progress = min(1.0, progress)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
        
    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)