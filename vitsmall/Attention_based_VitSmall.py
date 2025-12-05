import torch
import torch.nn as nn
import timm


# ---- Modify attention forward to expose attn ----
def attention_forward_with_attn(self, x):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)

    out = (attn @ v).transpose(1, 2).reshape(B, N, C)
    out = self.proj(out)
    return out, attn


class AttentionPruneViTSmall(nn.Module):
    def __init__(self, prune_ratio_map=None, num_classes=100):
        super().__init__()

        self.model = timm.create_model(
            "vit_small_patch16_224",
            pretrained=True
        )
        self.model.head = nn.Linear(384, num_classes)

        # monkey patch all attention layers
        for blk in self.model.blocks:
            blk.attn.forward = attention_forward_with_attn.__get__(blk.attn, blk.attn.__class__)

        if prune_ratio_map is None:
            prune_ratio_map = {2: 0.10, 4: 0.15, 6: 0.20}

        self.prune_ratio_map = prune_ratio_map

    # ---- Attention-based Pruning (CLS → token) ----
    def prune(self, x, attn, ratio):
        """
        x: [B, N, C]
        attn: [B, heads, N, N]
        """
        cls_attn = attn[:, :, 0, 1:]          # CLS → tokens
        scores = cls_attn.mean(dim=1)         # [B, patch_N]

        B, N, C = x.shape
        cls = x[:, :1, :]
        tokens = x[:, 1:, :]

        keep = int(tokens.shape[1] * (1 - ratio))
        idx = scores.topk(keep, dim=1).indices
        idx = idx.unsqueeze(-1).expand(-1, -1, C)
        tokens = torch.gather(tokens, 1, idx)

        x = torch.cat([cls, tokens], dim=1)
        return x

    def forward(self, x):
        B = x.shape[0]

        x = self.model.patch_embed(x)
        cls = self.model.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        x = x + self.model.pos_embed
        x = self.model.pos_drop(x)

        for i, blk in enumerate(self.model.blocks):

            if i in self.prune_ratio_map:
                # print(f"\n[ATTN] Layer {i} prune ratio = {self.prune_ratio_map[i]}")
                # print(f" Before: {x.shape[1]} tokens")

                # Forward MSA and get attn
                y, attn = blk.attn(blk.norm1(x))
                x = x + y

                # prune based on attention
                x = self.prune(x, attn, self.prune_ratio_map[i])
                # print(f" After : {x.shape[1]} tokens")

                # continue block (MLP)
                x = x + blk.mlp(blk.norm2(x))

            else:
                # regular forward
                y, _ = blk.attn(blk.norm1(x))
                x = x + y
                x = x + blk.mlp(blk.norm2(x))

        x = self.model.norm(x)
        cls = x[:, 0]
        return self.model.head(cls)
