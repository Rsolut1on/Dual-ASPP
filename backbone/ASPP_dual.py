import torch
import torch.nn as nn
import backbone
import numpy as np
import math
import torch.nn.functional as F
from .net import ASPP_Module, V1Filters
class ASPP(nn.Module):
    def __init__(self):
        super(ASPP, self).__init__()
        # image process
        self.conv1 = V1Filters(out_channel=64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.blocka_1 = ASPP_Module(in_channel=64, out_channel=128, stride=1)
        self.blocka_2 = ASPP_Module(in_channel=128, out_channel=256, stride=2)
        self.blocka_3 = ASPP_Module(in_channel=256, out_channel=256, stride=2)
        self.blocka_4 = ASPP_Module(in_channel=256, out_channel=256, stride=2)

        # mask process
        downsample1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
        )
        downsample2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2, padding=0),
            nn.BatchNorm2d(256),
        )
        # self.blocka_4 = backbone.BasicBlock(inplanes=512, planes=512, stride=2, downsample=downsample2)
        self.blockb_1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3), nn.ReLU(True))
        # self.blockb_2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(True))
        # self.blockb_3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), nn.ReLU(True))
        self.blockb_2 = backbone.BasicBlock(inplanes=64, planes=128, stride=1, downsample=downsample1)
        self.blockb_3 = backbone.BasicBlock(inplanes=128, planes=256, stride=2, downsample=downsample2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, *img_msk):
        x, mask = img_msk[0], img_msk[1]

        # stage1
        x = self.conv1(x)
        x = self.maxpool(x)
        # x = x * torch.nn.functional.adaptive_max_pool2d(mask, (x.shape[2], x.shape[3]))
        mask = self.blockb_1(mask) 
        mask = self.maxpool(mask)# b, 64,56,56
        x = x * mask + x

        # stage2
        x = self.blocka_1(x)# ->128
        # x = x * torch.nn.functional.adaptive_max_pool2d(mask, (x.shape[2], x.shape[3]))
        mask = self.blockb_2(mask)
        x = x * mask + x

        # stage3
        x = self.blocka_2(x) #->256
        # x = x * torch.nn.functional.adaptive_max_pool2d(mask, (x.shape[2], x.shape[3]))
        mask = self.blockb_3(mask)
        x = x * mask + x
        # stage4
        x = self.blocka_3(x)
        x = self.blocka_4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
class EnhancedCrossModalFusion(nn.Module):
    def __init__(self, num_classes, dim=1024, num_heads=8, expansion_ratio=4):
        super().__init__()
        assert dim % num_heads == 0, "dim必须能被num_heads整除"
        
        self.conv1 = V1Filters(out_channel=64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.encode_a = ASPP()
        self.encode_b = ASPP()

        # 自适应位置编码（考虑ASPP特性）
        self.pos_proj = nn.Sequential(
            nn.Linear(2, dim//8),  # 假设保留2D位置信息
            nn.GELU(),
            nn.Linear(dim//8, dim)
        )
        
        # 双向交叉注意力增强模块
        self.fusion_block = nn.ModuleList([
            CrossAttentionBlock(dim, num_heads, expansion_ratio) 
            for _ in range(2)  # 堆叠2个融合块
        ])
        
        # 多尺度特征聚合
        self.aspp_agg = nn.Sequential(
            nn.Conv1d(1, 256, kernel_size=1),  # 通道扩展
            *[nn.Sequential(  # 多分支并行
                nn.Conv1d(256, 256, kernel_size=k, padding=k//2, dilation=d),
                nn.GELU()
            ) for k, d in [(3,1), (5,2), (7,4)]],  # 不同膨胀率
            nn.AdaptiveAvgPool1d(64),  # 保留部分空间信息
            nn.Flatten(),
            nn.Linear(256*64, dim*2)  # 特征重整
        )
        
        # 动态特征门控
        self.gate = nn.Sequential(
            nn.Linear(dim*2, dim*2),
            nn.Sigmoid()
        )
        
        # 分类头增强
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim*2),
            nn.Linear(dim*2, dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x_a, x_b):
        # ASPP特征提取

        x_a = self.encode_a(x_a[0], x_a[1])  # [B, D] 32, 256
        x_b = self.encode_b(x_b[0], x_b[1])  # [B, D]
        # print('x_a:>>>>', x_a.shape)
        # print('x_b:>>>>', x_b.shape)
        # 生成相对位置编码
        pos = self.pos_proj(torch.randn(x_a.size(0), 2).to(x_a.device))  # 模拟位置坐标
        
        # 特征增强
        x_a = x_a + pos
        x_b = x_b + pos
        
        # 深度双向融合
        for block in self.fusion_block:
            x_a, x_b = block(x_a, x_b)
        
        # 门控融合
        combined = torch.cat([x_a, x_b], dim=1)
        gate = self.gate(combined)
        gated_combined = combined * gate
        
        # 多尺度聚合
        spatial_feat = self.aspp_agg(gated_combined.unsqueeze(1))
        
        return self.classifier(spatial_feat)

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, expansion_ratio):
        super().__init__()
        self.attn_ab = MultiHeadCrossAttention(dim, num_heads)
        self.attn_ba = MultiHeadCrossAttention(dim, num_heads)
        self.ffn = FeedForward(dim, expansion_ratio)
        
    def forward(self, a, b):
        # 双向交叉注意力
        a = self.attn_ab(a, b) + a
        b = self.attn_ba(b, a) + b
        
        # 特征增强
        a = self.ffn(a) + a
        b = self.ffn(b) + b
        return a, b

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(dim, dim*2)
        self.scale = self.head_dim ** -0.5
        
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, context):
        B, D = x.shape
        
        # 投影变换
        q = self.to_q(x).view(B, self.num_heads, self.head_dim)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        k = k.view(B, self.num_heads, self.head_dim)
        v = v.view(B, self.num_heads, self.head_dim)
        
        # 注意力计算
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).transpose(1,2).reshape(B, D)
        return self.norm(self.proj(out) + x)

class FeedForward(nn.Module):
    def __init__(self, dim, expansion_ratio=4):
        super().__init__()
        hidden_dim = dim * expansion_ratio
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(0.1)
        )
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        return self.norm(self.net(x) + x)