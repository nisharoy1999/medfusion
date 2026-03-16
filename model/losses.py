import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, num_classes=5):
        super().__init__()
        self.alpha, self.gamma, self.num_classes = alpha, gamma, num_classes

    def forward(self, logits, targets):
        probs     = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        oh        = F.one_hot(targets, self.num_classes).float()
        pt        = (probs * oh).sum(-1)
        loss      = -self.alpha * (1 - pt)**self.gamma * (log_probs * oh).sum(-1)
        return loss.mean()


class EvidentialLoss(nn.Module):
    def __init__(self, lam=0.01):
        super().__init__()
        self.lam = lam

    def forward(self, preds, targets):
        mu, v, alpha, beta = preds[:,0], preds[:,1], preds[:,2]+1, preds[:,3]
        t2bl = 2*beta*(1+v)
        nll  = (0.5*(torch.log(torch.pi/v)-torch.log(v))
                - alpha*torch.log(t2bl)
                + (alpha+0.5)*torch.log(v*(targets-mu)**2+t2bl)
                + torch.lgamma(alpha) - torch.lgamma(alpha+0.5))
        reg  = torch.abs(targets-mu)*(2*v+alpha)
        return (nll + self.lam*reg).mean()


class MedFusionLoss(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.focal = FocalLoss(num_classes=num_classes)
        self.huber = nn.HuberLoss(delta=1.0)
        self.evid  = EvidentialLoss()

    def forward(self, outputs, targets):
        l_cls = self.focal(outputs["logits"],   targets["labels"])
        l_sev = self.huber(outputs["severity"], targets["severity"].float())
        l_unc = self.evid(outputs["uncertainty"],targets["severity"].float())
        total = l_cls + 0.5*l_sev + 0.3*l_unc
        return {"total": total, "cls": l_cls, "sev": l_sev, "unc": l_unc}
