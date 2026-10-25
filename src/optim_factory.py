import math

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
            optimizer_type="adamw",
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            _adamw_step_group(self, group)

        return loss


def _adamw_step_group(optimizer: Optimizer, group: dict):
    lr = group["lr"]
    beta1, beta2 = group["betas"]
    eps = group["eps"]
    weight_decay = group["weight_decay"]

    for p in group["params"]:
        if p.grad is None:
            continue

        grad = p.grad
        if grad.is_sparse:
            raise RuntimeError("AdamW does not support sparse gradients")

        state = optimizer.state[p]

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


def _matrix_shape_scale(shape, mode: str) -> float:
    if mode == "none":
        return 1.0

    if mode == "sqrt_out_in":
        rows, cols = shape[-2], shape[-1]
        return math.sqrt(max(1.0, rows / max(1, cols)))

    raise ValueError(f"Unknown muon_update_scale_mode: {mode}")


@torch.no_grad()
def zeropower_via_newtonschulz5(
    matrix: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    use_bfloat16: bool = True,
) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError(f"Muon orthogonalization expects 2D tensor, got {matrix.ndim}D")

    original_dtype = matrix.dtype

    if use_bfloat16 and matrix.is_cuda and torch.cuda.is_bf16_supported():
        x = matrix.to(torch.bfloat16)
    else:
        x = matrix.float()

    transposed = False
    if x.shape[0] > x.shape[1]:
        x = x.T
        transposed = True

    x = x / (x.norm() + eps)

    # Practical quintic Newton-Schulz coefficients used for Muon-style updates.
    a, b, c = 3.4445, -4.7750, 2.0315

    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x

    if transposed:
        x = x.T

    return x.to(original_dtype)


def _muon_step_group(optimizer: Optimizer, group: dict):
    lr = group["lr"]
    momentum = group["momentum"]
    weight_decay = group["weight_decay"]
    nesterov = group["nesterov"]
    ns_steps = group["ns_steps"]
    eps = group["eps"]
    use_bfloat16 = group["use_bfloat16"]
    update_scale = group["update_scale"]
    update_scale_mode = group["update_scale_mode"]

    for p in group["params"]:
        if p.grad is None:
            continue

        grad = p.grad
        if grad.is_sparse:
            raise RuntimeError("Muon does not support sparse gradients")
        if p.ndim != 2:
            raise RuntimeError(
                f"Muon expects only 2D matrix parameters, got shape={tuple(p.shape)}"
            )

        state = optimizer.state[p]

        if len(state) == 0:
            state["step"] = 0
            state["momentum_buffer"] = torch.zeros_like(p)

        state["step"] += 1
        buf = state["momentum_buffer"]

        buf.mul_(momentum).add_(grad)

        if nesterov:
            update = grad.add(buf, alpha=momentum)
        else:
            update = buf

        update = zeropower_via_newtonschulz5(
            update,
            steps=ns_steps,
            eps=eps,
            use_bfloat16=use_bfloat16,
        )

        shape_scale = _matrix_shape_scale(p.shape, update_scale_mode)
        effective_lr = lr * update_scale * shape_scale

        if weight_decay != 0.0:
            p.add_(p, alpha=-lr * weight_decay)

        p.add_(update, alpha=-effective_lr)


class MyMuonAdamW(Optimizer):
    """Single Optimizer containing both Muon and AdamW groups.

    This avoids scheduler-sync issues. LambdaLR updates this optimizer's actual
    param_groups, and step() directly uses those same groups.
    """

    def __init__(self, param_groups):
        defaults = dict(lr=1e-4)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            optimizer_type = group.get("optimizer_type", None)

            if optimizer_type == "muon":
                _muon_step_group(self, group)
            elif optimizer_type == "adamw":
                _adamw_step_group(self, group)
            else:
                raise RuntimeError(f"Unknown optimizer_type in param group: {optimizer_type}")

        return loss


def _is_norm_or_bias(name: str, param: torch.nn.Parameter) -> bool:
    lower = name.lower()
    return (
        param.ndim <= 1
        or lower.endswith(".bias")
        or "ln_" in lower
        or "layernorm" in lower
        or ".ln" in lower
    )


def _is_embedding_like(name: str) -> bool:
    lower = name.lower()
    return (
        "wte" in lower
        or "wpe" in lower
        or "embed" in lower
        or "embedding" in lower
        or "lm_head" in lower
    )


def _split_adamw_params(model):
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if _is_norm_or_bias(name, param):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return decay_params, no_decay_params


def _split_muon_hybrid_params(model):
    muon_params = []
    aux_decay_params = []
    aux_no_decay_params = []

    counts = {
        "muon_params": 0,
        "aux_decay_params": 0,
        "aux_no_decay_params": 0,
        "muon_numel": 0,
        "aux_decay_numel": 0,
        "aux_no_decay_numel": 0,
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim == 2 and not _is_embedding_like(name):
            muon_params.append(param)
            counts["muon_params"] += 1
            counts["muon_numel"] += param.numel()

        elif _is_norm_or_bias(name, param):
            aux_no_decay_params.append(param)
            counts["aux_no_decay_params"] += 1
            counts["aux_no_decay_numel"] += param.numel()

        else:
            aux_decay_params.append(param)
            counts["aux_decay_params"] += 1
            counts["aux_decay_numel"] += param.numel()

    return muon_params, aux_decay_params, aux_no_decay_params, counts


def _attach_optimizer_summary(optimizer, summary: dict):
    optimizer._optimizer_summary = summary
    return optimizer


def build_optimizer(model, cfg):
    if cfg.optimizer_name in {"my_adamw", "torch_adamw"}:
        decay_params, no_decay_params = _split_adamw_params(model)

        param_groups = [
            {
                "params": decay_params,
                "lr": cfg.learning_rate,
                "betas": (cfg.adam_beta1, cfg.adam_beta2),
                "eps": cfg.adam_eps,
                "weight_decay": cfg.weight_decay,
                "optimizer_type": "adamw",
            },
            {
                "params": no_decay_params,
                "lr": cfg.learning_rate,
                "betas": (cfg.adam_beta1, cfg.adam_beta2),
                "eps": cfg.adam_eps,
                "weight_decay": 0.0,
                "optimizer_type": "adamw",
            },
        ]

        summary = {
            "optimizer_name": cfg.optimizer_name,
            "decay_params": len(decay_params),
            "no_decay_params": len(no_decay_params),
            "decay_numel": sum(p.numel() for p in decay_params),
            "no_decay_numel": sum(p.numel() for p in no_decay_params),
        }

        if cfg.optimizer_name == "my_adamw":
            return _attach_optimizer_summary(MyAdamW(param_groups), summary)

        # torch.optim.AdamW does not need optimizer_type, but harmless extra keys
        # are not accepted by all PyTorch versions, so remove them here.
        torch_param_groups = []
        for group in param_groups:
            copied = dict(group)
            copied.pop("optimizer_type", None)
            torch_param_groups.append(copied)

        return _attach_optimizer_summary(torch.optim.AdamW(torch_param_groups), summary)

    if cfg.optimizer_name == "my_muon":
        muon_params, aux_decay_params, aux_no_decay_params, counts = _split_muon_hybrid_params(model)

        if len(muon_params) == 0:
            raise ValueError("No parameters were selected for Muon. Check grouping rules.")

        aux_lr = cfg.learning_rate if cfg.aux_adamw_lr is None else cfg.aux_adamw_lr

        param_groups = [
            {
                "params": muon_params,
                "lr": cfg.learning_rate,
                "momentum": cfg.muon_momentum,
                "weight_decay": cfg.weight_decay,
                "nesterov": cfg.muon_nesterov,
                "ns_steps": cfg.muon_ns_steps,
                "eps": cfg.muon_eps,
                "use_bfloat16": cfg.muon_use_bfloat16,
                "update_scale": cfg.muon_update_scale,
                "update_scale_mode": cfg.muon_update_scale_mode,
                "optimizer_type": "muon",
            },
            {
                "params": aux_decay_params,
                "lr": aux_lr,
                "betas": (cfg.adam_beta1, cfg.adam_beta2),
                "eps": cfg.adam_eps,
                "weight_decay": cfg.weight_decay,
                "optimizer_type": "adamw",
            },
            {
                "params": aux_no_decay_params,
                "lr": aux_lr,
                "betas": (cfg.adam_beta1, cfg.adam_beta2),
                "eps": cfg.adam_eps,
                "weight_decay": 0.0,
                "optimizer_type": "adamw",
            },
        ]

        optimizer = MyMuonAdamW(param_groups)

        summary = {
            "optimizer_name": "my_muon",
            **counts,
            "muon_lr": cfg.learning_rate,
            "aux_adamw_lr": aux_lr,
            "muon_momentum": cfg.muon_momentum,
            "muon_nesterov": cfg.muon_nesterov,
            "muon_ns_steps": cfg.muon_ns_steps,
            "muon_update_scale": cfg.muon_update_scale,
            "muon_update_scale_mode": cfg.muon_update_scale_mode,
        }

        return _attach_optimizer_summary(optimizer, summary)

    raise ValueError(f"Unknown optimizer_name: {cfg.optimizer_name}")