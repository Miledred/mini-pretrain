import math
import torch


class Trainer:
    def __init__(self, model, optimizer, train_loader, val_loader, cfg, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.scheduler = scheduler

        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Internal step counter for low-frequency statistics
        self.global_step = 0

        # Update-stat control:
        # compute heavy stats every N steps instead of every step
        self.track_update_stats = getattr(cfg, "track_update_stats", True)
        self.update_stats_interval = max(1, int(getattr(cfg, "update_stats_interval", 100)))

        # Cache the last computed heavy stats so train.py does not need to change
        self.last_param_norm = 0.0
        self.last_update_norm = 0.0
        self.last_num_updated_params = 0
        self.last_update_ratio = 0.0

    @torch.no_grad()
    def _compute_grad_norm(self):
        total = torch.zeros(1, device=self.device)
        for p in self.model.parameters():
            if p.grad is not None:
                total += torch.sum(p.grad.detach().float() ** 2)
        return total.sqrt().item()

    @torch.no_grad()
    def _compute_param_norm(self):
        total = torch.zeros(1, device=self.device)
        for p in self.model.parameters():
            if p.requires_grad:
                total += torch.sum(p.detach().float() ** 2)
        return total.sqrt().item()

    @torch.no_grad()
    def _snapshot_trainable_params(self):
        snapshots = []
        for p in self.model.parameters():
            if p.requires_grad:
                snapshots.append((p, p.detach().clone()))
        return snapshots

    @torch.no_grad()
    def _compute_update_stats(self, snapshots):
        total = torch.zeros(1, device=self.device)
        num_updated_params = 0

        for p, before in snapshots:
            delta = p.detach() - before
            total += torch.sum(delta.float() ** 2)

            if torch.count_nonzero(delta).item() > 0:
                num_updated_params += 1

        update_norm = total.sqrt().item()
        return update_norm, num_updated_params

    def _should_compute_heavy_stats(self):
        if not self.track_update_stats:
            return False

        if self.global_step == 1:
            return True

        return (self.global_step % self.update_stats_interval) == 0

    def train_step(self, batch):
        self.global_step += 1
        self.model.train()

        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        outputs = self.model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        loss.backward()

        if self.cfg.grad_clip_norm is not None and self.cfg.grad_clip_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.cfg.grad_clip_norm,
            )
            if isinstance(grad_norm, torch.Tensor):
                grad_norm = grad_norm.item()
        else:
            grad_norm = self._compute_grad_norm()

        do_heavy_stats = self._should_compute_heavy_stats()
        params_before_step = None

        if do_heavy_stats:
            params_before_step = self._snapshot_trainable_params()

        # log lr used for this step
        lr = self.optimizer.param_groups[0]["lr"]

        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        if do_heavy_stats:
            update_norm, num_updated_params = self._compute_update_stats(params_before_step)
            param_norm = self._compute_param_norm()
            update_ratio = update_norm / (param_norm + 1e-12)

            self.last_update_norm = update_norm
            self.last_num_updated_params = num_updated_params
            self.last_param_norm = param_norm
            self.last_update_ratio = update_ratio

        metrics = {
            "loss": loss.item(),
            "grad_norm": float(grad_norm),
            "param_norm": float(self.last_param_norm),
            "update_norm": float(self.last_update_norm),
            "num_updated_params": int(self.last_num_updated_params),
            "update_ratio": float(self.last_update_ratio),
            "lr": float(lr),
        }
        return metrics

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()

        total_loss = 0.0
        total_batches = 0

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            total_loss += loss.item()
            total_batches += 1

        if total_batches == 0:
            return None

        return total_loss / total_batches