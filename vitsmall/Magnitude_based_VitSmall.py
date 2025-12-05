import torch
import torch.nn as nn
import timm


class MagnitudePruneViTSmall(nn.Module):
    def __init__(self, prune_ratio_map=None, num_classes=100):
        super().__init__()

        # Load ViT-Small pretrained on ImageNet
        self.model = timm.create_model(
            "vit_small_patch16_224",
            pretrained=True
        )
        self.model.head = nn.Linear(384, num_classes)

        # Default pruning configuration
        if prune_ratio_map is None:
            prune_ratio_map = {2: 0.10, 4: 0.15, 6: 0.20}

        self.prune_ratio_map = prune_ratio_map

    # ---- Magnitude-based pruning (L2 norm) ----
    def prune(self, x, ratio):
        """
        x: [B, N, C]
        """
        B, N, C = x.shape
        cls = x[:, :1, :]
        tokens = x[:, 1:, :]

        keep_tokens = int(tokens.shape[1] * (1 - ratio))

        scores = tokens.norm(dim=-1)       # [B, patch_N]
        idx = scores.topk(keep_tokens, dim=1).indices
        idx = idx.unsqueeze(-1).expand(-1, -1, C)

        tokens = torch.gather(tokens, 1, idx)
        x = torch.cat([cls, tokens], dim=1)
        return x

    def forward(self, x):
        B = x.shape[0]

        # Patch embedding
        x = self.model.patch_embed(x)

        # CLS
        cls_token = self.model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # Positional embedding
        x = x + self.model.pos_embed
        x = self.model.pos_drop(x)

        # Blocks
        for i, blk in enumerate(self.model.blocks):
            if i in self.prune_ratio_map:
                # print(f"\n[MAG] Layer {i} prune ratio = {self.prune_ratio_map[i]}")
                # print(f" Before: {x.shape[1]} tokens")
                x = self.prune(x, self.prune_ratio_map[i])
                # print(f" After : {x.shape[1]} tokens")

            x = blk(x)

        x = self.model.norm(x)
        cls = x[:, 0]
        return self.model.head(cls)
