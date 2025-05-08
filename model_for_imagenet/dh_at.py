import torch
import torch.nn as nn
from .deit import deit_tiny_patch16_224, deit_base_patch16_224


class DeiT_Attach(nn.Module):
    def __init__(self, crop_size=224, patch_size=16, num_classes=10, size='tiny', args=None):
        super(DeiT_Attach, self).__init__()
        self.num_classes = num_classes

        if size == 'base':
            self.backbone1 = deit_base_patch16_224(pretrained=True, img_size=crop_size, num_classes=10,
                                                   patch_size=patch_size, args=args).cuda()
            self.backbone2 = deit_base_patch16_224(pretrained=True, img_size=crop_size, num_classes=10,
                                                   patch_size=patch_size, args=args).cuda()
            # self.backbone1 = nn.DataParallel(self.backbone1)
            # self.backbone2 = nn.DataParallel(self.backbone2)
            # self.backbone1.load_state_dict(torch.load(
            #     'trades_vanilla_imagenette_deit_base_patch16_224_TRADES_warmup/seed0/weight_decay_0.000100/drop_rate_1.000000/nw_10.000000/checkpoint_40')[
            #                                    'state_dict'])
            # self.backbone2.load_state_dict(torch.load(
            #     'trades_vanilla_imagenette_deit_base_patch16_224_TRADES_warmup/seed0/weight_decay_0.000100/drop_rate_1.000000/nw_10.000000/checkpoint_40')[
            #                                    'state_dict'])
            dim, num_heads = 768, 12
        else: # size == 'tiny'
            self.backbone1 = deit_tiny_patch16_224(pretrained=True, img_size=crop_size, num_classes=10,
                                                  patch_size=patch_size, args=args).cuda()
            self.backbone2 = deit_tiny_patch16_224(pretrained=True, img_size=crop_size, num_classes=10,
                                                   patch_size=patch_size, args=args).cuda()
            # self.backbone1 = nn.DataParallel(self.backbone1)
            # self.backbone2 = nn.DataParallel(self.backbone2)
            # self.backbone1.load_state_dict(torch.load(
            #     'trades_vanilla_imagenette_deit_tiny_patch16_224_TRADES_warmup/seed0/weight_decay_0.000100/drop_rate_1.000000/nw_10.000000/checkpoint_40')[
            #                                   'state_dict'])
            # self.backbone2.load_state_dict(torch.load(
            #     'trades_vanilla_imagenette_deit_tiny_patch16_224_TRADES_warmup/seed0/weight_decay_0.000100/drop_rate_1.000000/nw_10.000000/checkpoint_40')[
            #                                   'state_dict'])
            dim, num_heads = 192, 3
        # self.backbone1 = self.backbone1.module
        # self.backbone2 = self.backbone2.module
        self.backbone1.head = nn.Identity()
        self.backbone2.head = nn.Identity()
        self.backbone1.forward_head = lambda x: x[:, 0]
        self.backbone2.forward_head = lambda x: x[:, 0]

        for param in self.backbone2.parameters():
            param.data = param.data + torch.randn_like(param.data) * 1e-3

        self.head = nn.Linear(dim * 2, num_classes)

    def forward(self, x):
        out1 = self.backbone1(x)
        out2 = self.backbone2.forward_from_block2(self.backbone1.forward_block1(x))
        return self.head(torch.cat((out1, out2), 1))