import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=224):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels,
                              emb_size,
                              kernel_size=patch_size,
                              stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.position_embedding = nn.Parameter(torch.randn(
            (img_size // patch_size) ** 2 + 1, emb_size))

    def forward(self, x):
        # x shape: [batch_size, 3, 224, 224]
        b, c, h, w = x.shape
        x = self.proj(x)  # [batch_size, emb_size, h/patch_size, w/patch_size]
        x = x.flatten(2)  # [batch_size, emb_size, #patches]
        x = x.transpose(1, 2)  # [batch_size, #patches, emb_size]

        # prepend cls token
        cls_tokens = self.cls_token.repeat(b, 1, 1)
        x = torch.cat([cls_tokens, x], dim=1)

        # add positional embedding
        x = x + self.position_embedding[:x.size(1), :]
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size=768, num_heads=8):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads

        self.query = nn.Linear(emb_size, emb_size)
        self.key = nn.Linear(emb_size, emb_size)
        self.value = nn.Linear(emb_size, emb_size)
        self.out = nn.Linear(emb_size, emb_size)

    def forward(self, x):
        # x shape: [batch_size, tokens, emb_size]
        b, t, e = x.shape
        q = self.query(x).reshape(b, t, self.num_heads, self.head_dim)
        k = self.key(x).reshape(b, t, self.num_heads, self.head_dim)
        v = self.value(x).reshape(b, t, self.num_heads, self.head_dim)

        # transpose for multi-head attention
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # scaled dot product
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.permute(0, 2, 1, 3).reshape(b, t, e)
        out = self.out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, emb_size=768, num_heads=8, forward_expansion=4, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_size)
        self.mha = MultiHeadAttention(emb_size, num_heads)
        self.ln2 = nn.LayerNorm(emb_size)
        self.ff = nn.Sequential(
            nn.Linear(emb_size, forward_expansion * emb_size),
            nn.GELU(),
            nn.Linear(forward_expansion * emb_size, emb_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_norm = self.ln1(x)
        attention = self.mha(x_norm)
        x = x + self.dropout(attention)
        x_norm = self.ln2(x)
        forward = self.ff(x_norm)
        x = x + self.dropout(forward)
        return x

class VisionTransformer(nn.Module):
    def __init__(self,
                 in_channels=3,
                 patch_size=16,
                 emb_size=768,
                 img_size=224,
                 depth=6,
                 num_heads=8,
                 num_classes=1000):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(emb_size, num_heads) for _ in range(depth)
        ])
        self.ln = nn.LayerNorm(emb_size)
        self.fc = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        # x: [batch_size, 3, 224, 224]
        x = self.patch_embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        # take cls token
        x = x[:, 0]
        x = self.fc(x)
        return x
