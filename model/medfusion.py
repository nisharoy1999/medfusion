"""
MedFusion: Hierarchical Cross-Modal Attention Fusion
for Clinical Decision Support
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class VisionEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.adapter = nn.Sequential(
            nn.Flatten(start_dim=2),
            nn.Linear(16, embed_dim // 2), nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )
        self.patch_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        feat = self.backbone(x)
        tokens = self.adapter(feat)
        return self.patch_norm(tokens)


class TextEncoder(nn.Module):
    def __init__(self, vocab_size=10000, max_len=256, embed_dim=256,
                 num_heads=4, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim*4, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, input_ids, attention_mask=None):
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.embedding(input_ids) + self.pos_embedding(pos)
        mask = (attention_mask == 0) if attention_mask is not None else None
        return self.norm(self.transformer(x, src_key_padding_mask=mask))


class StructuredEncoder(nn.Module):
    def __init__(self, input_dim=64, embed_dim=256, hidden_dim=512):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gate_net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim*2, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        h = F.gelu(self.input_proj(x))
        h = h * self.gate_net(h)
        return self.norm(self.ffn(h)).unsqueeze(1)


class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Sequential(nn.Linear(embed_dim*2, embed_dim), nn.Sigmoid())

    def forward(self, query_tokens, kv_tokens):
        B, Nq, D = query_tokens.shape
        Nkv = kv_tokens.shape[1]
        Q = self.q_proj(query_tokens).view(B,Nq,self.num_heads,self.head_dim).transpose(1,2)
        K = self.k_proj(kv_tokens).view(B,Nkv,self.num_heads,self.head_dim).transpose(1,2)
        V = self.v_proj(kv_tokens).view(B,Nkv,self.num_heads,self.head_dim).transpose(1,2)
        attn = self.dropout(F.softmax((Q @ K.transpose(-2,-1)) * self.scale, dim=-1))
        out = (attn @ V).transpose(1,2).reshape(B, Nq, D)
        out = self.out_proj(out)
        g = self.gate(torch.cat([query_tokens, out], dim=-1))
        return query_tokens + g * out, attn


class HierarchicalFusionModule(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4):
        super().__init__()
        self.v2t = CrossModalAttention(embed_dim, num_heads)
        self.t2v = CrossModalAttention(embed_dim, num_heads)
        self.vt2s = CrossModalAttention(embed_dim, num_heads)
        self.s2vt = CrossModalAttention(embed_dim, num_heads)
        gl = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim*4, dropout=0.1, batch_first=True)
        self.global_attn = nn.TransformerEncoder(gl, num_layers=2)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, v, t, s):
        B = v.shape[0]
        v2, v2t_attn = self.v2t(v, t)
        t2, t2v_attn = self.t2v(t, v)
        vt = torch.cat([v2, t2], dim=1)
        vt2, vt2s_attn = self.vt2s(vt, s)
        s2, _ = self.s2vt(s, vt)
        cls = self.cls_token.expand(B, -1, -1)
        all_tokens = self.global_attn(torch.cat([cls, vt2, s2], dim=1))
        cls_repr = self.norm(all_tokens[:, 0, :])
        attn_maps = {"v2t": v2t_attn.detach(), "t2v": t2v_attn.detach(),
                     "vt2s": vt2s_attn.detach()}
        return cls_repr, attn_maps


class MultiTaskHead(nn.Module):
    def __init__(self, embed_dim=256, num_classes=5):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(embed_dim, 512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.GELU())
        self.classifier = nn.Linear(256, num_classes)
        self.severity_head = nn.Linear(256, 1)
        self.uncertainty_head = nn.Sequential(nn.Linear(256, 4), nn.Softplus())

    def forward(self, x):
        h = self.shared(x)
        return {"logits": self.classifier(h),
                "severity": self.severity_head(h).squeeze(-1),
                "uncertainty": self.uncertainty_head(h)}


class MedFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        D  = config.get("embed_dim", 256)
        self.vision_enc  = VisionEncoder(D)
        self.text_enc    = TextEncoder(config.get("vocab_size",10000), embed_dim=D,
                                       num_heads=config.get("num_heads",4))
        self.struct_enc  = StructuredEncoder(config.get("struct_dim",64), D)
        self.fusion      = HierarchicalFusionModule(D, config.get("num_heads",4))
        self.head        = MultiTaskHead(D, config.get("num_classes",5))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, image, input_ids, attention_mask, struct_feat):
        v = self.vision_enc(image)
        t = self.text_enc(input_ids, attention_mask)
        s = self.struct_enc(struct_feat)
        cls, attn = self.fusion(v, t, s)
        out = self.head(cls)
        out["attention_maps"] = attn
        out["cls_repr"] = cls
        return out

    def get_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
