import torch
from torch import nn
from model import SwinTransformer_2
from cross_attention import CrossAttentionBlock
from model2 import MobileViT
from model_config import get_config



#patches和cls_token的交互
class Fusion_cls_patches(nn.Module):
    def __init__(self):
        super(Fusion_cls_patches, self).__init__()
        self.model1 = SwinTransformer_2(in_chans=3,
                                        patch_size=4,
                                        window_size=7,
                                        embed_dim=96,
                                        depths=(2, 2, 6, 2),
                                        num_heads=(3, 6, 12, 24),
                                        num_classes=3)
        self.model2 = MobileViT(get_config("x_small"), num_classes=3)
        self.cross_attn1 = CrossAttentionBlock(dim=384, num_heads=8, mlp_ratio=4, qkv_bias=True,
                                               drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm)
        self.cross_attn2 = CrossAttentionBlock(dim=768, num_heads=8, mlp_ratio=4, qkv_bias=True,
                                               drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm)
        self.projs1 = nn.Sequential(
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Linear(96, 384),
        )
        self.projs2 = nn.Sequential(
            nn.LayerNorm(48),
            nn.GELU(),
            nn.Linear(48, 768),
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(1152, 3)

    def forward(self, x1, x2):
        x1 = self.model1(x1)        #[8,3136,96],[8,1,768]
        x2 = self.model2(x2)        #[8,3136,48],[8,1,384]

        cls_tokens = []
        cls_tokens.append(x1[1])        #[8,1,768]
        cls_tokens.append(x2[1])        #[8,1,384]

        x11 = self.projs1(x1[0])       #[8,3136,384]
        x22 = self.projs2(x2[0])       #[8,3136,768]

        patches = []
        patches.append(x11)       #[8,3136,384]
        patches.append(x22)       #[8,3136,768]

        fusion1 = torch.cat((patches[0], cls_tokens[1]), dim=1)         #[8,1,384]
        fusion1 = self.cross_attn1(fusion1)
        fusion2 = torch.cat((patches[1], cls_tokens[0]), dim=1)         #[8,1,768]
        fusion2 = self.cross_attn2(fusion2)

        x = torch.cat((fusion1, fusion2), dim=2)  # [8,1,768]+[8,1,384]=[8,1,1152]
        x = self.avgpool(x.transpose(1, 2))  # [8,1152,1]
        x = torch.flatten(x, 1)  # [8,1152]
        x = self.head(x)

        return x

class Fusion_patches(nn.Module):
    def __init__(self):
        super(Fusion_patches, self).__init__()
        self.model1 = SwinTransformer_2(in_chans=3,
                                        patch_size=4,
                                        window_size=7,
                                        embed_dim=96,
                                        depths=(2, 2, 6, 2),
                                        num_heads=(3, 6, 12, 24),
                                        num_classes=3)
        self.model2 = MobileViT(get_config("x_small"), num_classes=3)
        self.cross_attn1 = CrossAttentionBlock(dim=384, num_heads=32, mlp_ratio=4, qkv_bias=True,
                                               drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm)

        self.projs = nn.Sequential(nn.LayerNorm(144),
                                   nn.GELU(),
                                   nn.Linear(144, 384))

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(384, 3)


    def forward(self, x1, x2):
        x1 = self.model1(x1)        #[8,3136,96],[8,1,768]
        x2 = self.model2(x2)        #[8,3136,48],[8,1,384]

        patches = []
        patches.append(x1[0])       #[8,3136,96]
        patches.append(x2[0])       #[8,3136,48]

        fusion = torch.cat((patches[0], patches[1]), dim=2)
        x = self.projs(fusion)
        fusion = self.cross_attn1(x)

        x = fusion
        x = self.avgpool(x.transpose(1, 2))  # [8,384,1]
        x = torch.flatten(x, 1)  # [8,384]
        x = self.head(x)

        return x

class Fusion_cls(nn.Module):
    def __init__(self):
        super(Fusion_cls, self).__init__()
        self.model1 = SwinTransformer_2(in_chans=3,
                                        patch_size=4,
                                        window_size=7,
                                        embed_dim=96,
                                        depths=(2, 2, 6, 2),
                                        num_heads=(3, 6, 12, 24),
                                        num_classes=3)
        self.model2 = MobileViT(get_config("x_small"), num_classes=3)
        self.cross_attn1 = CrossAttentionBlock(dim=768, num_heads=8, mlp_ratio=4, qkv_bias=True,
                                               drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm)
        self.cross_attn2 = CrossAttentionBlock(dim=384, num_heads=8, mlp_ratio=4, qkv_bias=True,
                                               drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm)
        self.projs1 = nn.Sequential(nn.LayerNorm(768),
                                    nn.GELU(),
                                    nn.Linear(768, 384),
                                    )

        self.projs2 = nn.Sequential(nn.LayerNorm(384),
                                    nn.GELU(),
                                    nn.Linear(384, 768),
                                    )

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(1152, 3)

    def forward(self, x1, x2):
        x1 = self.model1(x1)        #[8,3136,96],[8,1,768]
        x2 = self.model2(x2)        #[8,3136,48],[8,1,384]

        cls_tokens = []
        cls_tokens.append(x1[1])        #[8,1,768]
        cls_tokens.append(x2[1])        #[8,1,384]

        x1_proj = self.projs1(x1[1])
        x2_proj = self.projs2(x2[1])

        proj_cls_tokens = []
        proj_cls_tokens.append(x1_proj)        #[8,1,384]
        proj_cls_tokens.append(x2_proj)        #[8,1,768]

        fusion1 = torch.cat((cls_tokens[0], proj_cls_tokens[1]), dim=1)         #[8,1,768]
        fusion1 = self.cross_attn1(fusion1)
        fusion2 = torch.cat((cls_tokens[1], proj_cls_tokens[0]), dim=1)         #[8,1,384]
        fusion2 = self.cross_attn2(fusion2)

        x = torch.cat((fusion1, fusion2), dim=2)  # [8,1,768]+[8,1,384]=[8,1,1152]
        x = self.avgpool(x.transpose(1, 2))  # [8,1152,1]
        x = torch.flatten(x, 1)  # [8,1152]
        x = self.head(x)

        return x
