import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_, to_3tuple


class DyT(nn.Module):
    """Dynamic Tanh activation with learnable parameters"""

    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias


class EfficientAttention(nn.Module):
    """Memory-efficient attention implementation using PyTorch's scaled_dot_product_attention"""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Separate projections for better memory efficiency
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Lightweight normalization (using your DyT module)
        self.norm = DyT(dim)

        # Spatial reduction for memory efficiency
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv3d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_norm = DyT(dim)

    def forward(self, x, D, H, W):
        B, N, C = x.shape
        # Compute queries
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # Apply spatial reduction for keys and values if needed
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, D, H, W)
            x_ = self.sr(x_)
            x_ = x_.reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.sr_norm(x_)
            k = self.k(x_).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = self.v(x_).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        else:
            k = self.k(x).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = self.v(x).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # Scale queries
        q = q * self.scale
        # Use built-in scaled_dot_product_attention for improved efficiency.
        attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p, is_causal=False)

        # Reassemble output
        x = attn_output.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    """MLP with Depthwise Convolution"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # Reduce hidden dimension for memory efficiency
        reduced_features = max(hidden_features // 2, 32)

        self.fc1 = nn.Linear(in_features, reduced_features)
        self.dwconv = DWConv(reduced_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(reduced_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, D, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, D, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """Transformer block with memory-efficient implementations"""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, sr_ratio=1):
        super().__init__()

        # Use lightweight normalization
        self.norm1 = DyT(dim)

        # Use efficient attention
        self.attn = EfficientAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = DyT(dim)

        # Reduced MLP ratio for memory efficiency
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, D, H, W):
        # Pre-norm architecture with residual connections
        x = x + self.drop_path(self.attn(self.norm1(x), D, H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), D, H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding with overlapping patches"""

    def __init__(self, block3d_size=224, patch_size=7, stride=4, in_chans=1, embed_dim=768):
        super().__init__()
        block3d_size = to_3tuple(block3d_size)
        patch_size = to_3tuple(patch_size)

        self.block3d_size = block3d_size
        self.patch_size = patch_size

        self.D = block3d_size[0] // stride
        self.H = block3d_size[1] // stride
        self.W = block3d_size[2] // stride
        self.num_patches = self.D * self.H * self.W

        # Memory-efficient projection
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2, patch_size[2] // 2)
        )

        self.norm = DyT(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, D, H, W


class LinearMLP(nn.Module):
    """Memory-efficient Linear Projection"""

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        # Use a single linear projection
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class DWConv(nn.Module):
    """Depthwise Convolution"""

    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, D, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, D, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class EfficientSkipConnection(nn.Module):
    """Memory-efficient skip connection"""

    def __init__(self, decoder_dim):
        super().__init__()
        # Efficient fusion with reduced channels
        self.fusion = nn.Sequential(
            nn.Conv3d(decoder_dim * 2, decoder_dim, 1),
            nn.GELU()
        )

    def forward(self, x, skip):
        combined = torch.cat([x, skip], dim=1)
        out = self.fusion(combined)
        return out


class Segformer(nn.Module):
    """Memory-optimized SegFormer3D for semantic segmentation"""

    def __init__(
            self,
            num_classes=3,
            block3d_size=1024,
            patch_size=3,
            in_chans=1,
            embed_dims=[32, 64, 128, 256],  # Reduced dimensions
            num_heads=[1, 2, 4, 8],
            mlp_ratios=[2, 2, 2, 2],  # Reduced MLP ratios
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            depths=[2, 2, 8, 2],  # Reduced depths
            sr_ratios=[8, 4, 2, 1],
            decoder_dim=128,  # Reduced decoder dimension
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # Patch embeddings
        self.patch_embed1 = OverlapPatchEmbed(block3d_size=block3d_size, patch_size=patch_size, stride=2,
                                              in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(block3d_size=block3d_size // 4, patch_size=patch_size, stride=2,
                                              in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(block3d_size=block3d_size // 8, patch_size=patch_size, stride=2,
                                              in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(block3d_size=block3d_size // 16, patch_size=patch_size, stride=2,
                                              in_chans=embed_dims[2], embed_dim=embed_dims[3])

        # Configure stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        # Transformer blocks
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
            sr_ratio=sr_ratios[0]) for i in range(depths[0])])
        self.norm1 = DyT(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
            sr_ratio=sr_ratios[1]) for i in range(depths[1])])
        self.norm2 = DyT(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
            sr_ratio=sr_ratios[2]) for i in range(depths[2])])
        self.norm3 = DyT(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
            sr_ratio=sr_ratios[3]) for i in range(depths[3])])
        self.norm4 = DyT(embed_dims[3])

        # Lightweight decoder
        self.linear_c4 = LinearMLP(input_dim=embed_dims[3], embed_dim=decoder_dim)
        self.linear_c3 = LinearMLP(input_dim=embed_dims[2], embed_dim=decoder_dim)
        self.linear_c2 = LinearMLP(input_dim=embed_dims[1], embed_dim=decoder_dim)
        self.linear_c1 = LinearMLP(input_dim=embed_dims[0], embed_dim=decoder_dim)

        # Memory-efficient skip connections
        self.skip_fusions = nn.ModuleList([
            EfficientSkipConnection(decoder_dim) for _ in range(3)
        ])

        # Final prediction layers
        self.linear_fuse = nn.Conv3d(decoder_dim, decoder_dim, 1)
        self.dropout = nn.Dropout3d(drop_rate)

        # Two-stage prediction for memory efficiency
        self.linear_pred = nn.Sequential(
            nn.Conv3d(decoder_dim, decoder_dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(decoder_dim // 2, num_classes, 1)
        )

        self.loss = nn.CrossEntropyLoss()

        # Initialize weights
        self.max_accu = 0.0

    def _forward_block(self, x, blocks, norm, D, H, W):
        """Helper function for gradient checkpointing"""
        for blk in blocks:
            x = blk(x, D, H, W)
        return norm(x)

    def forward_features(self, x):
        B = x.shape[0]
        features = []

        # Stage 1
        x, D, H, W = self.patch_embed1(x)
        x = self._forward_block(x, self.block1, self.norm1, D, H, W)
        x_reshaped = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        features.append(x_reshaped)

        # Stage 2
        # x = x_reshaped.permute(0, 2, 3, 4, 1).reshape(B, D * H * W, -1)
        x, D, H, W = self.patch_embed2(x_reshaped)
        x = self._forward_block(x, self.block2, self.norm2, D, H, W)
        x_reshaped = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        features.append(x_reshaped)

        # Stage 3
        # x = x_reshaped.permute(0, 2, 3, 4, 1).reshape(B, D * H * W, -1)
        x, D, H, W = self.patch_embed3(x_reshaped)
        x = self._forward_block(x, self.block3, self.norm3, D, H, W)
        x_reshaped = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        features.append(x_reshaped)

        # Stage 4
        # x = x_reshaped.permute(0, 2, 3, 4, 1).reshape(B, D * H * W, -1)
        x, D, H, W = self.patch_embed4(x_reshaped)
        x = self._forward_block(x, self.block4, self.norm4, D, H, W)
        x_reshaped = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        features.append(x_reshaped)

        return features

    def forward(self, x):
        """Forward pass with memory optimizations"""
        d_out, h_out, w_out = x.size()[2], x.size()[3], x.size()[4]
        features = self.forward_features(x)
        c1, c2, c3, c4 = features

        # Process features through decoder
        n = c4.shape[0]

        # Process c4 with reduced memory operations
        # Process in chunks if needed to save memory
        x = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3], c4.shape[4])

        # Memory-efficient upsampling and skip connection fusion
        # C4 -> C3
        x = F.interpolate(x, size=c3.size()[2:], mode='trilinear', align_corners=False)
        c3_feat = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3], c3.shape[4])
        x = self.skip_fusions[0](x, c3_feat)

        # C3 -> C2
        x = F.interpolate(x, size=c2.size()[2:], mode='trilinear', align_corners=False)
        c2_feat = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3], c2.shape[4])
        x = self.skip_fusions[1](x, c2_feat)

        # C2 -> C1
        x = F.interpolate(x, size=c1.size()[2:], mode='trilinear', align_corners=False)
        c1_feat = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3], c1.shape[4])
        x = self.skip_fusions[2](x, c1_feat)

        # Final prediction with reduced operations
        x = self.linear_fuse(x)
        x = self.dropout(x)
        x = self.linear_pred(x)

        # Final upsampling to original size
        x = F.interpolate(input=x, size=(d_out, h_out, w_out), mode='trilinear', align_corners=False)
        x = x.type(torch.float32)

        return x
