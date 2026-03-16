"""
MedFusion — Entry Point
Usage: python train.py --epochs 5
"""
import os, sys, argparse, json
sys.path.insert(0, os.path.dirname(__file__))

import torch
from model.medfusion import MedFusion
from data.dataset    import get_loaders, LABELS
from utils.trainer   import Trainer

CONFIG = dict(embed_dim=128, num_heads=4, num_classes=5,
              vocab_size=10000, struct_dim=64,
              epochs=10, batch_size=32, lr=3e-4, patience=5)

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=CONFIG["epochs"])
    p.add_argument("--batch_size", type=int,   default=CONFIG["batch_size"])
    p.add_argument("--lr",         type=float, default=CONFIG["lr"])
    return p.parse_args()

def ablation(config, device):
    from data.dataset import get_loaders
    print("\n" + "="*60)
    print("  ABLATION STUDY — Modality Contributions")
    print("="*60)
    _, val_loader, _ = get_loaders(config["batch_size"])
    results = {}
    for name, mask in [
        ("Full  (V+T+S)", (True,True,True)),
        ("V+T only",      (True,True,False)),
        ("V+S only",      (True,False,True)),
        ("Vision only",   (True,False,False)),
        ("Text only",     (False,True,False)),
    ]:
        m = MedFusion(config).to(device).eval()
        correct=total=0
        with torch.no_grad():
            for b in val_loader:
                img = b["image"].to(device)   if mask[0] else torch.zeros_like(b["image"]).to(device)
                ids = b["input_ids"].to(device) if mask[1] else torch.zeros_like(b["input_ids"]).to(device)
                msk = b["attention_mask"].to(device)
                stf = b["struct_feat"].to(device) if mask[2] else torch.zeros_like(b["struct_feat"]).to(device)
                out = m(img, ids, msk, stf)
                correct += (out["logits"].argmax(-1)==b["label"].to(device)).sum().item()
                total   += len(b["label"])
        acc = correct/total
        results[name] = round(acc,4)
        bar = "█"*int(acc*30) + "░"*(30-int(acc*30))
        print(f"  {name:20s} [{bar}] {acc:.3f}")
    return results

def run_demo(model, test_loader, device):
    print("\n" + "─"*60)
    print("  EXPLAINABILITY REPORT — Single Sample")
    print("─"*60)
    model.eval()
    batch = next(iter(test_loader))
    with torch.no_grad():
        out = model(
            batch["image"][:1].to(device), batch["input_ids"][:1].to(device),
            batch["attention_mask"][:1].to(device), batch["struct_feat"][:1].to(device))
    probs = torch.softmax(out["logits"][0], -1).cpu().numpy()
    pred  = probs.argmax()
    unc   = out["uncertainty"][0].cpu().numpy()
    mu, v, alpha, beta = unc
    aleatoric = beta / max(alpha-1, 1e-8)
    epistemic = beta / max(v*(alpha-1), 1e-8)

    print(f"  Primary Diagnosis : {LABELS[pred]}")
    print(f"  Confidence        : {probs[pred]*100:.1f}%")
    print(f"  Severity Score    : {out['severity'][0].item():.3f}")
    print(f"  Aleatoric Uncert  : {aleatoric:.4f}  (data noise)")
    print(f"  Epistemic Uncert  : {epistemic:.4f}  (model uncertainty)")
    print()
    print("  Differential Diagnosis:")
    for lbl, p in sorted(zip(LABELS,probs), key=lambda x:-x[1]):
        bar = "█"*int(p*25)+"░"*(25-int(p*25))
        print(f"    {lbl:22s} [{bar}] {p*100:5.1f}%")
    print("─"*60)

def main():
    args = parse()
    cfg  = {**CONFIG, "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = get_loaders(cfg["batch_size"])
    model = MedFusion(cfg)

    trainer = Trainer(model, cfg, train_loader, val_loader, save_dir="outputs")
    history = trainer.fit()

    print("\n" + "─"*60)
    print("  FINAL TEST SET EVALUATION")
    print("─"*60)
    test_m = trainer.evaluate(test_loader)
    for k,v in test_m.items():
        print(f"  {k:15s}: {v:.4f}")

    run_demo(model, test_loader, device)
    abl = ablation(cfg, device)

    os.makedirs("outputs", exist_ok=True)
    json.dump(test_m, open("outputs/test_metrics.json","w"), indent=2)
    json.dump(abl,    open("outputs/ablation.json","w"),     indent=2)
    print("\n✓ All done! Results saved to outputs/")

if __name__ == "__main__":
    main()
