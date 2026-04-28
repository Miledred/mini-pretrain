import torch
from torch.optim import Optimizer


class MyAdamW(Optimizer):
    def __init__(
        self,
        params,
        lr=3e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 value: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 value: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("MyAdamW does not support sparse gradients")

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                step = state["step"]

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step

                exp_avg_hat = exp_avg / bias_correction1
                exp_avg_sq_hat = exp_avg_sq / bias_correction2

                denom = exp_avg_sq_hat.sqrt().add_(eps)

                if weight_decay != 0.0:
                    p.add_(p, alpha=-lr * weight_decay)

                p.addcdiv_(exp_avg_hat, denom, value=-lr)

        return loss


def build_optimizer(model, cfg):
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or "bias" in name or "ln_" in name or "layernorm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {
            "params": decay_params,
            "lr": cfg.learning_rate,
            "betas": (cfg.adam_beta1, cfg.adam_beta2),
            "eps": cfg.adam_eps,
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": no_decay_params,
            "lr": cfg.learning_rate,
            "betas": (cfg.adam_beta1, cfg.adam_beta2),
            "eps": cfg.adam_eps,
            "weight_decay": 0.0,
        },
    ]

    if cfg.optimizer_name == "my_adamw":
        return MyAdamW(param_groups)

    if cfg.optimizer_name == "torch_adamw":
        return torch.optim.AdamW(param_groups)

    raise ValueError(f"Unknown optimizer_name: {cfg.optimizer_name}")