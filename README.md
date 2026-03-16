# MedFusion 🏥🧠

> **Hierarchical Cross-Modal Attention Fusion for Clinical Decision Support**  
> *A novel multimodal AI architecture fusing medical imaging, clinical NLP, and structured EHR data with evidential uncertainty quantification*

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Research](https://img.shields.io/badge/Target-NeurIPS%20%7C%20MICCAI-purple.svg)]()

---

## 🔬 Overview

MedFusion is a PhD-level research project introducing a **novel hierarchical cross-modal attention architecture** that simultaneously processes three clinical data modalities:

| Modality | Input | Encoder |
|---|---|---|
| 🩻 Medical Imaging | Chest X-rays (224×224) | CNN + Adapter layers |
| 📝 Clinical Notes | Doctor's free-text notes | Transformer (BioClinicalBERT-ready) |
| 🧪 Structured EHR | Lab values, vitals, demographics | Gated MLP with feature-wise attention |

Unlike existing work (MedCLIP, BioViL) that handles at most two modalities, MedFusion introduces **three novel contributions**:

1. **Asymmetric Cross-Modal Attention Gating** — a learnable sigmoid gate controls how much cross-modal signal each modality absorbs, enabling asymmetric information flow
2. **Hierarchical 3-Level Fusion** — V↔T → (V,T)↔S → Global self-attention, with principled information flow and full ablation support
3. **Evidential Multi-Task Head** — jointly predicts diagnosis + severity + aleatoric/epistemic uncertainty via Normal Inverse-Gamma distribution

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        MedFusion                            │
├─────────────┬──────────────────┬───────────────────────────-┤
│  X-ray      │  Clinical Notes  │  EHR Features (labs/vitals)│
│  (3×224×224)│  (token seq)     │  (64-dim vector)           │
└──────┬──────┴────────┬─────────┴───────────┬───────────────-┘
       ▼               ▼                     ▼
  VisionEncoder   TextEncoder          StructuredEncoder
  (CNN+Adapter)   (Transformer)        (Gated MLP)
       │               │                     │
       └───────────────┼─────────────────────┘
                       ▼
        ┌──────────────────────────────┐
        │  HierarchicalFusionModule    │
        │  Level 1: Vision ↔ Text      │
        │  Level 2: (V,T) ↔ Structured │
        │  Level 3: Global Transformer │
        └──────────────┬───────────────┘
                       ▼
               CLS Token (256-dim)
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   Diagnosis      Severity       Uncertainty
   (5-class)     (regression)   (NIG: μ,v,α,β)
```

---

## 📊 Results

> Trained on synthetic MIMIC-CXR-style data. Replace with real MIMIC-CXR + MIMIC-III for publication results.

### Test Set Performance

| Metric | Value |
|---|---|
| AUROC (macro OvR) | 0.452 |
| F1-macro | 0.141 |
| Severity MAE | 0.256 |
| ECE (calibration) | 0.083 |
| Parameters | 3.96M |

### Ablation Study — Modality Contributions

| Configuration | Accuracy | Notes |
|---|---|---|
| **Full Model (V+T+S)** | **0.180** | All three modalities |
| Vision + Text (V+T) | 0.170 | Without lab data |
| Vision + Struct (V+S) | 0.270 | Without clinical notes |
| Vision only (V) | 0.190 | Baseline |
| Text only (T) | 0.170 | Language only |

> Note: Low absolute accuracy is expected with random synthetic data and 5 epochs. With real MIMIC-CXR data + 50 epochs, target AUROC > 0.85.

### Sample Explainability Output

```
Primary Diagnosis  :  Pneumonia
Confidence         :  23.4%
Severity Score     :  0.614
Aleatoric Uncert   :  0.2141  (data noise)
Epistemic Uncert   :  0.0145  (model uncertainty)

Differential Diagnosis:
  Pneumonia         [█████░░░░░░░░░░░░░░░]  23.4%
  Pneumothorax      [████░░░░░░░░░░░░░░░░]  21.6%
  Normal            [████░░░░░░░░░░░░░░░░]  20.6%
  Pleural Effusion  [███░░░░░░░░░░░░░░░░░]  17.7%
  Cardiomegaly      [███░░░░░░░░░░░░░░░░░]  16.7%
```

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/nisharoy1999/medfusion.git
cd medfusion
pip install torch torchvision scikit-learn numpy matplotlib
```

### Train

```bash
# Quick demo (CPU, 5 epochs)
python train.py --epochs 5

# Full training (GPU recommended, 50 epochs)
python train.py --epochs 50 --batch_size 64 --lr 1e-4
```

### What you'll see

```
============================================================
  MedFusion Training  |  device=cpu  |  params=3,960,394
============================================================
Ep 01/05 | loss=1.682 | val_loss=0.142 | acc=0.190 | auroc=0.506 | ...
Ep 02/05 | loss=0.095 | val_loss=0.039 | acc=0.240 | auroc=0.519 | ... CHECK BEST
...
  ABLATION STUDY — Modality Contributions
  Full  (V+T+S)   [████░░░░░░░░░░░░░░░░░░░░░░░░░░] 0.280
  V+T only        [███░░░░░░░░░░░░░░░░░░░░░░░░░░░] 0.210
  Vision only     [██░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 0.180
```

---

## 📁 Project Structure

```
medfusion/
├── model/
│   ├── medfusion.py        # Full architecture
│   │                       #   VisionEncoder, TextEncoder,
│   │                       #   StructuredEncoder,
│   │                       #   HierarchicalFusionModule,
│   │                       #   MultiTaskHead, MedFusion
│   └── losses.py           # FocalLoss, EvidentialLoss,
│                           #   MedFusionLoss
├── data/
│   └── dataset.py          # SyntheticMedDataset
│                           #   (MIMIC-CXR integration ready)
├── utils/
│   ├── trainer.py          # Trainer with AMP, cosine LR,
│   │                       #   early stopping, ECE, AUROC
│   └── explainability.py   # GradCAM, AttentionRollout,
│                           #   IntegratedGradients
├── outputs/
│   ├── best_model.pt       # Best checkpoint
│   ├── history.json        # Training curves
│   ├── test_metrics.json   # Final evaluation
│   └── ablation.json       # Ablation results
├── train.py                # Entry point
└── requirements.txt
```

---

## 🔗 Connecting to Real Data

### MIMIC-CXR (Chest X-rays + Radiology Reports)
```python
# Request access at: https://physionet.org/content/mimic-cxr/
# Replace SyntheticMedDataset with your DICOM loader
```

### BioClinicalBERT (Clinical NLP)
```python
# pip install transformers
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
# Plug into TextEncoder backbone
```

### Pretrained ResNet-50 (Vision)
```python
import torchvision.models as models
backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
# Plug into VisionEncoder backbone
```

---

## 📐 Key Design Decisions

### Why Hierarchical Fusion?
Flat concatenation loses the structure of inter-modal relationships. Our 3-level hierarchy mirrors clinical reasoning: first align visual findings with textual descriptions, then ground both in objective lab data.

### Why Evidential Uncertainty?
Clinical AI must know what it doesn't know. Standard softmax confidence is poorly calibrated. Our NIG-based evidential head separately quantifies:
- **Aleatoric uncertainty** — irreducible noise in the data
- **Epistemic uncertainty** — model uncertainty from lack of training examples

### Why Asymmetric Gating?
Different modalities contribute unevenly depending on the case. A pneumothorax is primarily visual; sepsis is primarily lab-driven. The adaptive gate learns these case-specific weights automatically.

---

## 📄 References

1. Amini et al. "Deep Evidential Regression." *NeurIPS 2020*
2. Lin et al. "Focal Loss for Dense Object Detection." *ICCV 2017*
3. Wang et al. "MedCLIP: Contrastive Learning from Unpaired Medical Images and Text." *EMNLP 2022*
4. Bannur et al. "Learning to Exploit Temporal Structure for Biomedical Vision-Language Processing." *CVPR 2023*
5. Abnar & Zuidema. "Quantifying Attention Flow in Transformers." *ACL 2020*

---

## 📜 License

MIT License — free to use for research and academic purposes.

---

## 👩‍💻 Author

**Nisha Roy** — M.Tech 2nd Year, AI & ML  
GitHub: [@nisharoy1999](https://github.com/nisharoy1999)

---

*Built as an M.Tech final year project targeting PhD admission. Architecture designed for publication at NeurIPS / MICCAI / ICLR.*
