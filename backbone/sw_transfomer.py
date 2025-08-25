import torch
import torch.nn as nn


class SwinTransformer(nn.Module):
    def __init__(
        self,pretrain_img_size=224, patch_size=4,in_chans=3, embed_dim=96,
        depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],window_size=7,mlp_ratio=4.0, qkv_bias=True,qk_scale=None, drop_rate=0.0,attn_drop_rate=0.0,drop_path_rate=0.2,norm_layer=nn.LayerNorm, ape=False,patch_norm=True,out_indices=(0, 1, 2, 3), frozen_stages=-1, dilation=False,use_checkpoint=False,):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.dilation = dilation
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(patch_size=patch_size,in_chans=in_chans,embed_dim=embed_dim,norm_layer=norm_layer if self.patch_norm else None,)
        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0],pretrain_img_size[1] // patch_size[1]]
            self.absolute_pos_embed = nn.Parameter( torch.zeros(1, embed_dim, patches_resolution[0],patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        # build layers
        self.layers = nn.ModuleList()
        # prepare downsample list
        downsamplelist = [PatchMerging for i in range(self.num_layers)]
        downsamplelist[-1] = None
        num_features = [int(embed_dim * 2**i) for i in range(self.num_layers)]
        if self.dilation:
            downsamplelist[-2] = None
            num_features[-1] = int(embed_dim * 2 ** (self.num_layers -1)) // 2
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=num_features[i_layer],depth=depths[i_layer],num_heads=num_heads[i_layer],window_size=window_size,mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,qk_scale=qk_scale,drop=drop_rate,attn_drop=attn_drop_rate,drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],norm_layer=norm_layer,downsample=downsamplelist[i_layer],use_checkpoint=use_checkpoint,)
            self.layers.append(layer)
        # num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features
        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)
        self._freeze_stages()

    def _freeze_stages(self):
        ...
    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        x = self.patch_embed(x)
        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode="bicubic")
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)
        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x_out = norm_layer(x_out)
                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3,1,2).contiguous()
                outs.append(out)
        # collect for nesttensors
        outs_dict = {}
        for idx, out_i in enumerate(outs):
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(),size=out_i.shape[-2:]).to(torch.bool[0])
            outs_dict[idx] = NestedTensor(out_i, mask)
        return outs_dict

if __name__ == '__main__':
    SwinTransformer()