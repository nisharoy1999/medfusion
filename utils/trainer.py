import os, json, time, math
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast


def cosine_warmup(optimizer, warmup, total):
    def lr_fn(step):
        if step < warmup:
            return step / max(1, warmup)
        p = (step - warmup) / max(1, total - warmup)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * p)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)


def eval_metrics(logits, labels, sev, sev_pred, num_classes=5):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_absolute_error
    import numpy as np
    probs  = torch.softmax(logits, -1).numpy()
    preds  = probs.argmax(-1)
    lb     = labels.numpy()
    acc    = accuracy_score(lb, preds)
    f1     = f1_score(lb, preds, average="macro", zero_division=0)
    mae    = mean_absolute_error(sev.numpy(), sev_pred.numpy())
    try:
        auroc = roc_auc_score(lb, probs, multi_class="ovr", average="macro")
    except Exception:
        auroc = 0.5
    conf    = probs.max(-1)
    correct = (preds == lb).astype(float)
    bins    = np.linspace(0, 1, 16)
    ece     = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi)
        if mask.sum() > 0:
            ece += abs(correct[mask].mean() - conf[mask].mean()) * mask.sum() / len(lb)
    return {"acc": acc, "f1": f1, "auroc": auroc, "mae": mae, "ece": float(ece)}


class Trainer:
    def __init__(self, model, config, train_loader, val_loader, save_dir="outputs"):
        self.model        = model
        self.config       = config
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.save_dir     = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from model.losses import MedFusionLoss
        self.criterion = MedFusionLoss(config.get("num_classes", 5))

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.get("lr", 3e-4), weight_decay=1e-4)
        total = len(train_loader) * config.get("epochs", 20)
        self.scheduler  = cosine_warmup(self.optimizer, total // 10, total)
        self.scaler     = GradScaler(enabled=self.device.type == "cuda")
        self.best_auroc = 0.0
        self.history    = []

    def _mb(self, batch):
        result = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(self.device)
            else:
                result[k] = v
        return result

    def train_epoch(self):
        self.model.train()
        total = 0.0
        for batch in self.train_loader:
            b = self._mb(batch)
            with autocast(enabled=self.device.type == "cuda"):
                out  = self.model(b["image"], b["input_ids"],
                                  b["attention_mask"], b["struct_feat"])
                loss = self.criterion(
                    out, {"labels": b["label"], "severity": b["severity"]})["total"]
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad()
            total += loss.item()
        return total / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        all_l, all_lb, all_s, all_sp = [], [], [], []
        total = 0.0
        for batch in loader:
            b   = self._mb(batch)
            out = self.model(b["image"], b["input_ids"],
                             b["attention_mask"], b["struct_feat"])
            loss = self.criterion(
                out, {"labels": b["label"], "severity": b["severity"]})["total"]
            total   += loss.item()
            all_l.append(out["logits"].cpu().float())
            all_lb.append(b["label"].cpu())
            all_s.append(b["severity"].cpu().float())
            all_sp.append(out["severity"].cpu().float())
        m = eval_metrics(torch.cat(all_l), torch.cat(all_lb),
                         torch.cat(all_s),  torch.cat(all_sp))
        m["val_loss"] = total / len(loader)
        return m

    def fit(self):
        epochs  = self.config.get("epochs", 20)
        patience = self.config.get("patience", 5)
        pat_ct  = 0
        print("\n" + "=" * 60)
        print("  MedFusion Training  |  device={}  |  params={:,}".format(
            self.device, self.model.get_param_count()))
        print("=" * 60)
        for ep in range(1, epochs + 1):
            t0      = time.time()
            tr_loss = self.train_epoch()
            vm      = self.evaluate(self.val_loader)
            elapsed = time.time() - t0
            flag    = ""
            if vm["auroc"] > self.best_auroc:
                self.best_auroc = vm["auroc"]
                pat_ct = 0
                flag   = " CHECK BEST"
                torch.save(self.model.state_dict(),
                           os.path.join(self.save_dir, "best_model.pt"))
            else:
                pat_ct += 1
            self.history.append({"epoch": ep, "train_loss": tr_loss, **vm})
            print("Ep {:02d}/{} | loss={:.4f} | val_loss={:.4f} | "
                  "acc={:.3f} | auroc={:.3f} | f1={:.3f} | "
                  "mae={:.4f} | {:.1f}s{}".format(
                      ep, epochs, tr_loss, vm["val_loss"],
                      vm["acc"], vm["auroc"], vm["f1"],
                      vm["mae"], elapsed, flag))
            if pat_ct >= patience:
                print("Early stopping.")
                break
        with open(os.path.join(self.save_dir, "history.json"), "w") as f:
            json.dump(self.history, f, indent=2)
        print("\nBest AUROC: {:.4f}".format(self.best_auroc))
        return self.history
