import os
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import Mlp
from .deit import deit_tiny_patch16_224, deit_base_patch16_224
import torch.nn.functional as F


class RobustQuoteAttention(nn.Module):
    def __init__(self, num_classes, dim, num_heads):
        super(RobustQuoteAttention, self).__init__()
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qv = nn.Linear(dim, dim * 2, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, anchors):
        B, N, C = x.shape
        qv = self.qv(anchors).reshape(self.num_classes, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, v = qv.unbind(0)  # -> (num_classes, num_heads, N, C // num_heads)
        k = self.k(x).reshape(1, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)

        attn = (q.unsqueeze(1) @ k.transpose(-2, -1)) * self.scale  # -> (num_classes, B, num_heads, N, N)
        attn_cls = attn[:, :, :, 0, 0].permute(1, 2, 0).softmax(dim=-1)  # -> (B, num_heads, num_classes)
        attn = attn.softmax(dim=-1)
        x = (attn @ v.unsqueeze(1))  # -> (num_classes, B, num_heads, N, C//num_heads)
        x = (x.permute(1, 2, 3, 4, 0) @ attn_cls.unsqueeze(2).unsqueeze(-1)).squeeze(-1)  # -> (B, num_heads, N, C//num_heads)
        x = x.permute(0, 2, 1, 3).reshape(B, N, C)  # -> (B, N, C)
        x = self.proj(x)
        return x, attn_cls, attn


class RobustQuoteModule(nn.Module):
    def __init__(self, num_classes, dim, num_heads):
        super(RobustQuoteModule, self).__init__()
        self.num_classes = num_classes
        self.num_heads = num_heads

        self.attention = RobustQuoteAttention(num_classes, dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * 4.), act_layer=nn.GELU)

        self.extra_outputs = None

    def forward(self, x):
        x, anchors = x[:-self.num_classes], x[-self.num_classes:].detach()
        x_orig = x.clone()
        x, attn_cls, attn = self.attention(self.norm1(x), self.norm1(anchors))
        x = x + self.mlp(self.norm2(x))
        self.extra_outputs = [attn_cls, attn]
        return torch.cat((x, x_orig), dim=0)


class RobustQuoteNet(nn.Module):
    def __init__(self, blocks, alpha=0.5, crop_size=224, patch_size=16, size='tiny', args=None):
        super(RobustQuoteNet, self).__init__()
        assert len(blocks) == 1
        self.robust_blocks_ids = blocks
        self.alpha = alpha
        if size == 'base':
            self.backbone = deit_base_patch16_224(pretrained=True, img_size=crop_size, num_classes=10, patch_size=patch_size, args=args).cuda()
            # self.backbone = nn.DataParallel(self.backbone)
            # self.backbone.load_state_dict(torch.load(
            #     'trades_vanilla_cifar_deit_base_patch16_224_TRADES_warmup/seed0/weight_decay_0.000100/drop_rate_1.000000/nw_10.000000/checkpoint_40')[
            #                                   'state_dict'])
            dim, num_heads = 768, 12
        else: # size == 'tiny'
            self.backbone = deit_tiny_patch16_224(pretrained=True, img_size=crop_size, num_classes=10, patch_size=patch_size, args=args).cuda()
            # self.backbone = nn.DataParallel(self.backbone)
            # self.backbone.load_state_dict(torch.load(
            #     'trades_vanilla_cifar_deit_tiny_patch16_224_TRADES_warmup/seed0/weight_decay_0.000100/drop_rate_1.000000/nw_10.000000/checkpoint_40')[
            #                                   'state_dict'])
            dim, num_heads = 192, 3
        # self.backbone = self.backbone.module

        for list_index, i in enumerate(self.robust_blocks_ids):
            self.backbone.blocks[i] = nn.Sequential(self.backbone.blocks[i], RobustQuoteModule(10, dim, num_heads))

    def make_internal_anchor_loader(self, dataset, nb_classes=10):
        per_class_dataset = [[] for _ in range(nb_classes)]
        for i in range(len(dataset)):
            class_idx = dataset[i][1]
            per_class_dataset[class_idx].append(dataset[i])
        min_class_size = min([len(per_class_dataset[i]) for i in range(nb_classes)])
        new_set = []
        for i in range(min_class_size):
            for j in range(nb_classes):
                new_set.append(per_class_dataset[j][i])
        loader = torch.utils.data.DataLoader(new_set, batch_size=nb_classes, shuffle=False)
        self.anchor_ds = cycle(iter(loader))

    def update_anchors(self):
        assert hasattr(self, 'anchor_ds')
        self.anchors = next(self.anchor_ds)[0].cuda()

    def extra_loss(self, adv_extras, nat_extras, y):
        # extras = [[attn_cls, attn, x, x_orig, pred], ...]

        x_orig_nat = nat_extras[0][3]
        x_orig_adv = adv_extras[0][3]
        B = x_orig_nat.shape[0]
        criterion_kl = nn.KLDivLoss(size_average=False)
        loss_natural = F.cross_entropy(x_orig_nat, y)
        loss_robust = (1.0 / B) * criterion_kl(F.log_softmax(x_orig_adv, dim=1), F.softmax(x_orig_nat, dim=1))
        loss = loss_natural + 6.0 * loss_robust
        return loss

    def process_and_save_extras(self, input, labels, path=None):
        pass

    def forward(self, x):
        B = x.shape[0]
        x = torch.cat((x, self.anchors), dim=0)
        x = self.backbone(x)
        x, quotes = x[B:], x[:B]
        self.extra_outputs = []
        for i in self.robust_blocks_ids:
            extras = self.backbone.blocks[i][1].extra_outputs
            extras.append(quotes)
            extras.append(x)
            self.extra_outputs.append(extras)
        return quotes
