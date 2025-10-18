"""
trainer.py
==========

Robust training driver for EAAN that can run unmodified on CPU *or* GPU.

Highlights
----------
* **Dataset handling** via ``dataset.get_datasets``.
* **Mixed precision** (AMP + GradScaler) and `torch.compile` are enabled only
  when available *and* a CUDA device is present.
* **Cosine-annealing LR** and **Lookahead + RAdam** as the optimization setup.
* **Early stopping** on validation accuracy.
* Automatic _shape fix_ if features arrive flattened
  (e.g., ``(N, 1920, 1)``) so they match the architecture
  without changing saved weights.
* Logs to ``runs/<run_name>/metrics.txt`` and saves the best
  model as ``best_model.pt`` in the same folder.

Expected modules
----------------
``config.py``   : contains the ``Config`` dataclass.
``dataset.py``  : provides the ``get_datasets`` function.
``model.py``    : defines ``EAAN``.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_optimizer import Lookahead, RAdam
from tqdm import tqdm

from config import Config
from dataset import get_datasets
from model import EAAN

# -----------------------------------------------------------------------------


class Trainer:
    """Class that encapsulates *all* training logic for the EAAN model."""

    # -----------------------------------------------------------------
    def __init__(self, cfg: Config) -> None:
        """
        Parameters
        ----------
        cfg : Config
            Hyperparameters and file paths (see ``config.py``).

        Attributes created
        ------------------
        ``self.dev``          : current `torch.device`.
        ``self.cuda``         : bool, whether we have a GPU.
        ``self.model``        : EAAN instance (optionally compiled).
        ``self.train_loader`` / ``self.val_loader``
        ``self.optimizer``    : Lookahead(RAdam).
        ``self.scheduler``    : Cosine-annealing LR.
        ``self.autocast``     : context manager for AMP (or ``nullcontext``).
        ``self.scaler``       : ``GradScaler`` if GPU-AMP is active.
        ``self.log_file``     : open file handle for epoch logs.
        """
        self.cfg = cfg
        self.dev = torch.device(cfg.device)
        self.cuda = self.dev.type == "cuda"

        # ---- 1) Backend flags (GPU only) ----------------------------
        if self.cuda:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True

        # ---- 2) DataLoaders ----------------------------------------
        train_ds, val_ds, _ = get_datasets(
            cfg.train_path,
            cfg.val_path,
            cfg.test_path,
            pad_len=cfg.pad_len,
            data_format=cfg.data_format,
            stream=cfg.stream,
        )

        self.train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=self.cuda,
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=self.cuda,
        )

        # ---- 3) Model (optionally torch.compile) --------------------
        net = EAAN(cfg.input_dims, cfg.num_classes, conv_pooling="attention").to(
            self.dev
        )
        self.model = (
            torch.compile(net, mode="max-autotune")
            if self.cuda and hasattr(torch, "compile")
            else net
        )

        # ---- 4) Optimizer / scheduler -------------------------------
        base_opt = RAdam(
            self.model.parameters(),
            lr=cfg.lr,
            betas=(0.95, 0.999),
            eps=1e-5,
            weight_decay=cfg.weight_decay,
        )
        self.optimizer = Lookahead(base_opt, k=6, alpha=0.5)
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr
        )

        # ---- 5) Loss function --------------------------------------
        self.criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smooth)

        # ---- 6) AMP / GradScaler setup ------------------------------
        if self.cuda and hasattr(torch.cuda.amp, "GradScaler"):
            from torch.cuda.amp import GradScaler, autocast

            self.amp_enabled = True
            self.scaler = GradScaler()
            self.autocast = lambda: autocast(dtype=torch.float16)
        else:
            self.amp_enabled = False
            self.scaler = None
            self.autocast = nullcontext  # type: ignore

        # ---- 7) Logging ---------------------------------------------
        self.run_dir = Path("runs") / cfg.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = (self.run_dir / "metrics.txt").open("w")
        self.log_file.write(
            "epoch\tlr\ttrain_loss\ttrain_acc\tval_loss\tval_acc\n"
        )

    # -----------------------------------------------------------------
    @staticmethod
    def _prep_batch(
        batch: dict,
        dev: torch.device,
        use_cuda: bool,
        cfg: Config,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare a minibatch for model input.

        * Moves tensors to the correct device.
        * Fixes feature shapes if they are flattened or permuted.

        Accepted feature shapes
        -----------------------
        ``(N, P, C)``  : already correct.
        ``(N, C, P)``  : will be permuted.
        ``(N, flat, 1)``, ``(N, 1, flat)``, ``(N, flat)``  : will be reshaped,
        where ``flat == pad_len * input_dims``.

        Returns
        -------
        pts   : (N, P, D)   – relative coordinates.
        fts   : (N, P, C)   – particle features (corrected).
        mask  : (N, P, 1)   – 1 = valid particle.
        lbl   : (N,)        – class index.
        """
        nb = dict(non_blocking=use_cuda)

        pts = batch["X"]["points"].float().to(dev, **nb)  # (N, P, D)
        fts = batch["X"]["features"].float().to(dev, **nb)  # (N, ?, ?)

        # ---------- shape sanitation --------------------------------
        P, C = cfg.pad_len, cfg.input_dims
        flat_len = P * C

        if fts.dim() == 3 and fts.size(-1) == 1 and fts.size(1) == flat_len:
            # (N, flat, 1)  ->  (N, flat)  ->  (N, P, C)
            fts = fts.squeeze(-1).view(-1, P, C)
        elif fts.dim() == 3 and fts.size(1) == 1 and fts.size(2) == flat_len:
            # (N, 1, flat)  ->  (N, flat) ->  (N, P, C)
            fts = fts.squeeze(1).view(-1, P, C)
        elif fts.dim() == 2 and fts.size(1) == flat_len:
            # (N, flat) -> (N, P, C)
            fts = fts.view(-1, P, C)
        elif fts.dim() == 3 and fts.size(1) == C and fts.size(2) == P:
            # (N, C, P) -> (N, P, C)
            fts = fts.permute(0, 2, 1).contiguous()

        mask = (fts.abs().sum(2, keepdim=True) != 0).float()

        lbl = batch["y"]
        if lbl.ndim == 2:  # one-hot
            lbl = lbl.argmax(1)
        lbl = lbl.long().to(dev, **nb)

        return pts, fts, mask, lbl

    # -----------------------------------------------------------------
    def _run_epoch(self, loader, training: bool) -> Tuple[float, float]:
        """
        Run one epoch (train *or* validation).

        Returns
        -------
        loss : float
            Average cross-entropy loss per example.
        acc  : float
            Classification accuracy.
        """
        self.model.train() if training else self.model.eval()
        tot_loss = correct = seen = 0

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for batch in tqdm(
                loader,
                desc="Train" if training else "Val",
                leave=False,
                ncols=120,
            ):
                pts, fts, mask, lbl = self._prep_batch(
                    batch, self.dev, self.cuda, self.cfg
                )

                self.optimizer.zero_grad(set_to_none=True)

                with self.autocast():
                    logits = self.model(pts, fts, mask)
                    loss = self.criterion(logits, lbl)

                # -------------- backward / step --------------------
                if training:
                    if self.amp_enabled:
                        self.scaler.scale(loss).backward()  # type: ignore
                        self.scaler.step(self.optimizer)  # type: ignore
                        self.scaler.update()  # type: ignore
                    else:
                        loss.backward()
                        self.optimizer.step()

                # -------------- stats ------------------------------
                B = lbl.size(0)
                tot_loss += loss.item() * B
                correct += (logits.argmax(1) == lbl).sum().item()
                seen += B

        return tot_loss / seen, correct / seen

    # -----------------------------------------------------------------
    def train(self) -> None:
        """Main training loop with early stopping."""
        best_acc = epochs_no_improve = 0

        for epoch in range(1, self.cfg.epochs + 1):
            tr_loss, tr_acc = self._run_epoch(self.train_loader, True)
            val_loss, val_acc = self._run_epoch(self.val_loader, False)

            self.scheduler.step()
            lr_now = self.optimizer.param_groups[0]["lr"]

            print(
                f"[{epoch:02d}] lr {lr_now:.2e} | "
                f"train {tr_acc:.4f}/{tr_loss:.4f}  "
                f"val {val_acc:.4f}/{val_loss:.4f}"
            )

            self.log_file.write(
                f"{epoch}\t{lr_now:.6e}\t"
                f"{tr_loss:.5f}\t{tr_acc:.5f}\t"
                f"{val_loss:.5f}\t{val_acc:.5f}\n"
            )
            self.log_file.flush()

            # --------- early stopping ----------------------------
            if val_acc > best_acc:
                best_acc, epochs_no_improve = val_acc, 0
                torch.save(
                    self.model.state_dict(), self.run_dir / "best_model.pt"
                )
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.cfg.patience:
                    print("⏹️  Early stop (no validation improvement).")
                    break

        self.log_file.close()


# ---------------------------------------------------------------------
if __name__ == "__main__":
    Trainer(Config()).train()
