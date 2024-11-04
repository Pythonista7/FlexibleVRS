import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, context, mask):
        b, n, _ = x.shape
        h = self.num_heads

        q = self.to_q(x).reshape(b, n, h, -1).permute(0, 2, 1, 3)
        k, v = self.to_kv(context).reshape(b, context.shape[1], 2, h, -1).permute(2, 0, 3, 1, 4)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            mask = mask.flatten(2, 3).unsqueeze(1)  # Add head dimension
            dots = dots.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(dots, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Mask2FormerDecoderLayer(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads=8, dropout=0.):
        super().__init__()
        self.masked_attn = MaskedAttention(dim, num_heads)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.ffn = FeedForward(dim, ffn_dim, dropout)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, mask):
        # Masked Attention
        residual = x
        x = self.norm1(x)
        x = self.masked_attn(x, context, mask)
        x = self.dropout(x)
        x = residual + x

        # Self Attention
        residual = x
        x = self.norm2(x)
        x = self.self_attn(x, x, x)[0]
        x = self.dropout(x)
        x = residual + x

        # FFN
        residual = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = residual + x

        return self.norm4(x)


class Mask2FormerDecoder(nn.Module):
    """
    The one that works!
    """
    def __init__(self, dim, ffn_dim, num_layers=9, num_heads=8, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([
            Mask2FormerDecoderLayer(dim, ffn_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.mask_embed = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, queries, multi_scale_features, masks=None):
        b, n, _ = queries.shape
        if masks is None:
            masks = torch.ones(b, n, multi_scale_features[0].shape[2], multi_scale_features[0].shape[3]).to(
                queries.device)

        for i, layer in enumerate(self.layers):
            level_index = i % 3  # Cycle through feature levels
            context = multi_scale_features[level_index].flatten(2).transpose(1, 2)
            mask_embeddings = self.mask_embed(queries)
            mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embeddings, multi_scale_features[level_index])
            mask = (mask_pred > 0).float()
            queries = layer(queries, context, mask)

        return queries


# Example usage
if __name__ == "__main__":
    dim = 256
    ffn_dim = 2048
    num_queries = 100
    batch_size = 2

    queries = torch.randn(batch_size, num_queries, dim)
    multi_scale_features = [
        torch.randn(batch_size, dim, 20, 20),  # 1/32 resolution
        torch.randn(batch_size, dim, 40, 40),  # 1/16 resolution
        torch.randn(batch_size, dim, 80, 80)  # 1/8 resolution
    ]

    decoder = Mask2FormerDecoder(dim, ffn_dim)
    output = decoder(queries, multi_scale_features)
    print(output.shape)  # Should be [batch_size, num_queries, dim]
