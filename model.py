import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=1)

# Residual Block
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()

        self.time_mlp = nn.Linear(time_dim, out_ch)

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(),
        )

        self.residual = (
            nn.Conv2d(in_ch, out_ch, 1)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x, t):
        h = self.block(x)
        time_emb = self.time_mlp(t)[:, :, None, None]
        return h + time_emb + self.residual(x)

# Self-Attention Block
class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_norm = self.norm(x)

        q = self.q(x_norm).reshape(B, C, H * W).permute(0, 2, 1)
        k = self.k(x_norm).reshape(B, C, H * W)
        v = self.v(x_norm).reshape(B, C, H * W)

        attn = torch.bmm(q, k) / math.sqrt(C)
        attn = attn.softmax(dim=-1)

        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.reshape(B, C, H, W)

        return x + self.proj(out)


# Diffusion U-Net WITH Attention
class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()

        image_channels = 3
        time_dim = 128
        chs = [64, 128, 256, 512]

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )

        # Input projection
        self.conv0 = nn.Conv2d(image_channels, chs[0], 3, padding=1)

        # Down path
        self.down1 = ResBlock(chs[0], chs[1], time_dim)
        self.down2 = ResBlock(chs[1], chs[2], time_dim)
        self.down3 = ResBlock(chs[2], chs[3], time_dim)

        self.attn = SelfAttention(chs[2])   

        self.pool = nn.AvgPool2d(2)

        # Middle
        self.mid = ResBlock(chs[3], chs[3], time_dim)

        # Up path
        self.up3 = ResBlock(chs[3] + chs[3], chs[2], time_dim)
        self.up2 = ResBlock(chs[2] + chs[2], chs[1], time_dim)
        self.up1 = ResBlock(chs[1] + chs[1], chs[0], time_dim)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # Output
        self.out = nn.Conv2d(chs[0], image_channels, 1)

    def forward(self, x, t):
        t = self.time_mlp(t)

        # Down
        x0 = self.conv0(x)                    # 64
        x1 = self.down1(x0, t)                # 128
        x2 = self.down2(self.pool(x1), t)     # 256
        x2 = self.attn(x2)                    # Attention
        x3 = self.down3(self.pool(x2), t)     # 512

        # Middle
        mid = self.mid(self.pool(x3), t)

        # Up
        u3 = self.upsample(mid)
        u3 = self.up3(torch.cat([u3, x3], dim=1), t)

        u2 = self.upsample(u3)
        u2 = self.up2(torch.cat([u2, x2], dim=1), t)

        u1 = self.upsample(u2)
        u1 = self.up1(torch.cat([u1, x1], dim=1), t)

        return self.out(u1)
