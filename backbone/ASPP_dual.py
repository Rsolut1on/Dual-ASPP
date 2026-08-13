import torch
import torch.nn as nn
import backbone
from .net import ASPP_Module, V1Filters


# 
ABLATION_PRESETS = {
    'full': dict(
        use_dual=True, use_mask=True, use_aspp_branches=True,
        use_cross_attn=True, use_pos_enc=True, use_gate=True,
    ),
    'no_dual': dict(
        use_dual=False, use_mask=True, use_aspp_branches=True,
        use_cross_attn=True, use_pos_enc=True, use_gate=True,
    ),
    'no_mask': dict(
        use_dual=True, use_mask=False, use_aspp_branches=True,
        use_cross_attn=True, use_pos_enc=True, use_gate=True,
    ),
    'no_aspp': dict(
        use_dual=True, use_mask=True, use_aspp_branches=False,
        use_cross_attn=True, use_pos_enc=True, use_gate=True,
    ),
    'no_cross_attn': dict(
        use_dual=True, use_mask=True, use_aspp_branches=True,
        use_cross_attn=False, use_pos_enc=True, use_gate=True,
    ),
    'no_pos': dict(
        use_dual=True, use_mask=True, use_aspp_branches=True,
        use_cross_attn=True, use_pos_enc=False, use_gate=True,
    ),
    'no_gate': dict(
        use_dual=True, use_mask=True, use_aspp_branches=True,
        use_cross_attn=True, use_pos_enc=True, use_gate=False,
    ),
}


def resolve_ablation(variant='full', **overrides):
    if variant not in ABLATION_PRESETS:
        raise ValueError(
            'unknown ablation variant: %s, choose from %s'
            % (variant, list(ABLATION_PRESETS.keys()))
        )
    cfg = dict(ABLATION_PRESETS[variant])
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


class PlainConvBlock(nn.Module):
    """Single-branch substitute for ASPP_Module (ablation: w/o ASPP branches)."""

    def __init__(self, in_channel, out_channel, stride):
        super(PlainConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)
# 


class ASPP(nn.Module):
    # 
    def __init__(self, use_mask=True, use_aspp_branches=True):
        super(ASPP, self).__init__()
        self.use_mask = use_mask
        self.use_aspp_branches = use_aspp_branches
        Branch = ASPP_Module if use_aspp_branches else PlainConvBlock
        # 
        # image process
        self.conv1 = V1Filters(out_channel=64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.blocka_1 = Branch(in_channel=64, out_channel=128, stride=1)
        self.blocka_2 = Branch(in_channel=128, out_channel=256, stride=2)
        self.blocka_3 = Branch(in_channel=256, out_channel=256, stride=2)
        self.blocka_4 = Branch(in_channel=256, out_channel=256, stride=2)

        # mask process
        downsample1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
        )
        downsample2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2, padding=0),
            nn.BatchNorm2d(256),
        )
        self.blockb_1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3), nn.ReLU(True))
        self.blockb_2 = backbone.BasicBlock(inplanes=64, planes=128, stride=1, downsample=downsample1)
        self.blockb_3 = backbone.BasicBlock(inplanes=128, planes=256, stride=2, downsample=downsample2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, *img_msk, return_spatial=False):
        x, mask = img_msk[0], img_msk[1]

        # stage1
        x = self.conv1(x)
        x = self.maxpool(x)
        # 
        if self.use_mask:
            mask = self.blockb_1(mask)
            mask = self.maxpool(mask)
            x = x * mask + x

        # stage2
        x = self.blocka_1(x)
        if self.use_mask:
            mask = self.blockb_2(mask)
            x = x * mask + x

        # stage3
        x = self.blocka_2(x)
        if self.use_mask:
            mask = self.blockb_3(mask)
            x = x * mask + x
        # 
        # stage4
        x = self.blocka_3(x)
        x = self.blocka_4(x)
        if return_spatial:
            return x
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class EnhancedCrossModalFusion(nn.Module):
    # 
    def __init__(
        self,
        num_classes,
        dim=1024,
        num_heads=8,
        expansion_ratio=4,
        use_dual=True,
        use_mask=True,
        use_aspp_branches=True,
        use_cross_attn=True,
        use_pos_enc=True,
        use_gate=True,
        ablation_variant=None,
    ):
        super().__init__()
        if ablation_variant is not None:
            cfg = resolve_ablation(ablation_variant)
            use_dual = cfg['use_dual']
            use_mask = cfg['use_mask']
            use_aspp_branches = cfg['use_aspp_branches']
            use_cross_attn = cfg['use_cross_attn']
            use_pos_enc = cfg['use_pos_enc']
            use_gate = cfg['use_gate']

        assert dim % num_heads == 0, "dim必须能被num_heads整除"
        self.dim = dim
        if use_pos_enc and dim % 4 != 0:
            raise ValueError('2D position encoding requires dim divisible by 4')
        self.use_dual = use_dual
        self.use_mask = use_mask
        self.use_aspp_branches = use_aspp_branches
        self.use_cross_attn = use_cross_attn
        self.use_pos_enc = use_pos_enc
        self.use_gate = use_gate

        self.encode_a = ASPP(use_mask=use_mask, use_aspp_branches=use_aspp_branches)
        self.encode_b = ASPP(use_mask=use_mask, use_aspp_branches=use_aspp_branches) if use_dual else None
        self.feat_proj = nn.Identity() if dim == 256 else nn.Linear(256, dim)

        self.fusion_block = nn.ModuleList([
            CrossAttentionBlock(dim, num_heads, expansion_ratio)
            for _ in range(2)
        ]) if use_cross_attn and use_dual else None

        if use_aspp_branches:
            self.aspp_agg = nn.Sequential(
                nn.Conv1d(1, 256, kernel_size=1),
                *[nn.Sequential(
                    nn.Conv1d(256, 256, kernel_size=k, padding=k // 2, dilation=d),
                    nn.GELU(),
                ) for k, d in [(3, 1), (5, 2), (7, 4)]],
                nn.AdaptiveAvgPool1d(64),
                nn.Flatten(),
                nn.Linear(256 * 64, dim * 2),
            )
        else:
            self.aspp_agg = nn.Sequential(
                nn.Linear(dim * 2, dim * 2),
                nn.GELU(),
            )

        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.Sigmoid(),
        ) if use_gate else None

        self.classifier = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim, num_classes),
        )
    # 

    @staticmethod
    def _build_2d_sincos_position(height, width, dim, device, dtype):
        """Build deterministic 2D sine-cosine encoding for HxW texture tokens."""
        if dim % 4 != 0:
            raise ValueError('position-encoding dimension must be divisible by 4')
        quarter_dim = dim // 4
        omega = torch.arange(quarter_dim, device=device, dtype=torch.float32)
        omega = 1.0 / (10000.0 ** (omega / max(quarter_dim - 1, 1)))
        y_grid, x_grid = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing='ij',
        )
        x_phase = x_grid.reshape(-1, 1) * omega.reshape(1, -1)
        y_phase = y_grid.reshape(-1, 1) * omega.reshape(1, -1)
        position = torch.cat(
            (x_phase.sin(), x_phase.cos(), y_phase.sin(), y_phase.cos()),
            dim=1,
        )
        return position.unsqueeze(0).to(dtype=dtype)

    def forward(self, x_a, x_b=None):
        # 
        def _unpack_pair(pair, name):
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                img, msk = pair[0], pair[1]
                if torch.is_tensor(img) and img.dim() == 4 and torch.is_tensor(msk) and msk.dim() == 4:
                    return img, msk
            raise ValueError(
                '%s must be [images, masks] with 4D NCHW tensors, got %s.'
                % (name, type(pair))
            )

        img_a, msk_a = _unpack_pair(x_a, 'x_a')
        map_a = self.encode_a(img_a, msk_a, return_spatial=True)
        tokens_a = self.feat_proj(map_a.flatten(2).transpose(1, 2))

        if self.use_dual:
            if x_b is None:
                raise ValueError('dual-modality mode requires x_b as [images, masks]')
            img_b, msk_b = _unpack_pair(x_b, 'x_b')
            map_b = self.encode_b(img_b, msk_b, return_spatial=True)
            tokens_b = self.feat_proj(map_b.flatten(2).transpose(1, 2))
        else:
            map_b = map_a
            tokens_b = tokens_a

        if self.use_pos_enc:
            pos_a = self._build_2d_sincos_position(
                map_a.size(2), map_a.size(3), self.dim,
                tokens_a.device, tokens_a.dtype,
            )
            tokens_a = tokens_a + pos_a
            if self.use_dual:
                pos_b = self._build_2d_sincos_position(
                    map_b.size(2), map_b.size(3), self.dim,
                    tokens_b.device, tokens_b.dtype,
                )
                tokens_b = tokens_b + pos_b

        if self.use_cross_attn and self.use_dual and self.fusion_block is not None:
            for block in self.fusion_block:
                tokens_a, tokens_b = block(tokens_a, tokens_b)

        feat_a = tokens_a.mean(dim=1)
        feat_b = tokens_b.mean(dim=1)

        if self.use_dual:
            combined = torch.cat([feat_a, feat_b], dim=1)
        else:
            combined = torch.cat([feat_a, feat_a], dim=1)

        if self.use_gate and self.gate is not None:
            combined = combined * self.gate(combined)

        if self.use_aspp_branches:
            spatial_feat = self.aspp_agg(combined.unsqueeze(1))
        else:
            spatial_feat = self.aspp_agg(combined)

        return self.classifier(spatial_feat)
    # 


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, expansion_ratio):
        super().__init__()
        self.attn_ab = MultiHeadCrossAttention(dim, num_heads)
        self.attn_ba = MultiHeadCrossAttention(dim, num_heads)
        self.ffn = FeedForward(dim, expansion_ratio)

    def forward(self, a, b):
        delta_a = self.attn_ab(a, b)
        delta_b = self.attn_ba(b, a)
        a = a + delta_a
        b = b + delta_b
        a = a + self.ffn(a)
        b = b + self.ffn(b)
        return a, b


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(dim, dim * 2)
        self.scale = self.head_dim ** -0.5

        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, context):
        batch_size, query_len, dim = x.shape
        context_len = context.size(1)
        x = self.norm(x)
        context = self.norm(context)

        q = self.to_q(x).view(
            batch_size, query_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        k = k.view(
            batch_size, context_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = v.view(
            batch_size, context_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(
            batch_size, query_len, dim
        )
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, dim, expansion_ratio=4):
        super().__init__()
        hidden_dim = dim * expansion_ratio
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(0.1),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.net(self.norm(x))


if '__main__' == __name__:
    # 
    for name in ABLATION_PRESETS:
        model = EnhancedCrossModalFusion(num_classes=2, dim=256, ablation_variant=name)
        x_a = [torch.randn(2, 3, 224, 224), torch.randn(2, 1, 224, 224)]
        x_b = [torch.randn(2, 3, 224, 224), torch.randn(2, 1, 224, 224)]
        out = model(x_a, x_b)
        print(name, out.shape)
    # 
