import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

LABELS = ["Normal","Pneumonia","Pleural Effusion","Cardiomegaly","Pneumothorax"]

class SyntheticMedDataset(Dataset):
    def __init__(self, size=1000, img_size=64, max_len=64, struct_dim=64,
                 num_classes=5, seed=42):
        rng = np.random.RandomState(seed)
        self.images   = torch.tensor(rng.randn(size,3,img_size,img_size), dtype=torch.float32)
        self.ids      = torch.tensor(rng.randint(1,10000,(size,max_len)), dtype=torch.long)
        self.masks    = torch.ones(size, max_len, dtype=torch.long)
        for i in range(size):
            l = rng.randint(10, max_len)
            self.masks[i,l:] = 0; self.ids[i,l:] = 0
        self.struct   = torch.tensor(rng.randn(size, struct_dim).astype(np.float32))
        self.labels   = torch.tensor(rng.randint(0, num_classes, size), dtype=torch.long)
        self.severity = torch.tensor(rng.uniform(0,1,size), dtype=torch.float32)

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        return {"image": self.images[i], "input_ids": self.ids[i],
                "attention_mask": self.masks[i], "struct_feat": self.struct[i],
                "label": self.labels[i], "severity": self.severity[i]}

def get_loaders(batch_size=32):
    tr = DataLoader(SyntheticMedDataset(800, seed=42),  batch_size, shuffle=True)
    va = DataLoader(SyntheticMedDataset(100, seed=123), batch_size)
    te = DataLoader(SyntheticMedDataset(100, seed=456), batch_size)
    return tr, va, te
