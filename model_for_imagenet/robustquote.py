import random
import torch
import torch.nn as nn
from timm.models.layers import Mlp
from torch.utils.data import Sampler
from .deit import deit_tiny_patch16_224, deit_base_patch16_224
import torch.nn.functional as F


class Quoter(nn.Module):
    def __init__(self, num_classes, dim, num_heads):
        super(Quoter, self).__init__()
        self.num_classes = num_classes
        self.num_heads = num_heads

        self.scale = (dim // num_heads) ** -0.5
        self.qv = nn.Linear(dim, dim * 2, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x_cls_token, anchors_cls_token):
        B, N, C = x_cls_token.shape
        qv = self.qv(anchors_cls_token).reshape(self.num_classes, 2, 1, self.num_heads, C // self.num_heads).permute(1, 0, 3, 2, 4)
        q, v = qv.unbind(0)
        q = q.unsqueeze(0)
        k = self.k(x_cls_token).unsqueeze(1).reshape(B, 1, self.num_heads, 1, C // self.num_heads)

        attn_cls = ((q @ k.transpose(-2, -1)) * self.scale).view(B, self.num_classes, self.num_heads).softmax(1)
        x_cls = self.proj((attn_cls.permute(2, 0, 1) @ v.squeeze(2).permute(1, 0, 2)).permute(1, 0, 2).reshape(B, 1, C))
        return x_cls, attn_cls


class Rectifier(nn.Module):
    def __init__(self, num_classes, dim, num_heads):
        super(Rectifier, self).__init__()
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qk = nn.Linear(dim, dim, bias=False)
        self.qk2 = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, anchors, weights):
        B, N, C = x.shape
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = self.qk(x).reshape(B, 1, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)
        k = self.qk(anchors).reshape(1, self.num_classes, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)
        q2 = self.qk2(x).reshape(B, 1, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)
        k2 = self.qk2(anchors).reshape(1, self.num_classes, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2,
                                                                                                            4)
        # [B, 1, H, N, C] @ [1, num_classes, H, C, N'] -> [B, num_classes, H, N, N']
        left_attn = -1 * (q @ k.transpose(-2, -1)) * self.scale
        right_attn = (k2 @ q2.transpose(-2, -1)) * self.scale
        left_attn[left_attn < 0] = 0
        right_attn[right_attn < 0] = 0
        # [B, num_classes, H, N, N'] @ [B, num_classes, H, N', N] -> [B, num_classes, H, N, N]
        attn = (left_attn @ right_attn).permute(0, 2, 3, 1, 4)
        # [B, H, 1, 1, num_classes] @ [B, H, N, num_classes, N] -> [B, H, N, N]
        tmp = (weights.reshape(B, self.num_heads, 1, 1, self.num_classes) @ attn).squeeze(-2)
        # [B, H, N, N] @ [B, H, N, C] -> [B, H, N', C] -> [B, N, C]
        x = self.proj((tmp @ v).reshape(B, N, C))
        return x, left_attn


class RobustQuoteModule(nn.Module):
    def __init__(self, num_classes, dim, num_heads, keep_anchors=True):
        super(RobustQuoteModule, self).__init__()
        self.num_classes = num_classes
        self.keep_anchors = keep_anchors
        self.num_heads = num_heads

        self.quoter = Quoter(num_classes, dim, num_heads)
        self.rectifier = Rectifier(num_classes, dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * 4.), act_layer=nn.GELU)

        self.extra_outputs = None

    def forward(self, x):
        x, anchors = x[:-self.num_classes], x[-self.num_classes:].detach()
        x_im_tokens, anchors_im_tokens = x[:, 1:], anchors[:, 1:]
        x_cls_token, anchors_cls_token = x[:, 0].unsqueeze(1), anchors[:, 0].unsqueeze(1)

        x_cls, attn_cls = self.quoter(self.norm1(x_cls_token), self.norm1(anchors_cls_token))
        rectified_x_im_tokens, attn_im = self.rectifier(self.norm2(x_im_tokens), self.norm2(anchors_im_tokens), attn_cls)
        self.extra_outputs = [attn_cls, rectified_x_im_tokens]

        x_im_tokens = x_im_tokens + rectified_x_im_tokens
        x = torch.cat((x_cls, x_im_tokens), dim=1)
        x = x + self.mlp(self.norm3(x))

        if self.keep_anchors:
            x = torch.cat((x, anchors), dim=0)
        return x


class RobustQuoteNet(nn.Module):
    def __init__(self, blocks, alpha=0.1, tau=0.5, crop_size=224, patch_size=16, size='tiny', args=None):
        super(RobustQuoteNet, self).__init__()
        self.robust_blocks_ids = blocks
        self.alpha = alpha
        self.tau = tau

        if size == 'base':
            self.backbone = deit_base_patch16_224(pretrained=True, img_size=crop_size, num_classes=10, patch_size=patch_size, args=args).cuda()
            dim, num_heads = 768, 12
        else: # size == 'tiny'
            self.backbone = deit_tiny_patch16_224(pretrained=True, img_size=crop_size, num_classes=10, patch_size=patch_size, args=args).cuda()
            dim, num_heads = 192, 3

        self.backbone = nn.DataParallel(self.backbone)
        self.backbone.load_state_dict(torch.load(
            f'trades_vanilla_imagenette_deit_{size}_patch16_224_TRADES_warmup/seed0/weight_decay_0.000100/drop_rate_1.000000/nw_10.000000/checkpoint_40')[
                                          'state_dict'])
        self.backbone = self.backbone.module

        for list_index, i in enumerate(self.robust_blocks_ids):
            self.backbone.blocks[i] = nn.Sequential(self.backbone.blocks[i], RobustQuoteModule(10, dim, num_heads, keep_anchors=(list_index != len(self.robust_blocks_ids) - 1)))

    def make_internal_anchor_loader(self, dataset, nb_classes=10):
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=nb_classes,
            shuffle=False,
            sampler=OneImagePerClassSampler(dataset),
            pin_memory=True,
            num_workers=10,
        )
        self.anchor_ds = loader

    def update_anchors(self):
        assert hasattr(self, 'anchor_ds')
        a, y = self.anchor_ds.__iter__().__next__()
        self.anchors = a.cuda()

    def extra_loss(self, adv_extras, nat_extras, y):
        # extras = [[attn_cls, rectified_x_im_tokens, pred], ...]
        # attn_cls: (B, num_classes, num_heads) with mean(attn_cls, dim=1) = 1
        # rectified_x_im_tokens: (B, N, C)
        loss = 0.0
        for (_, rectif_x_adv, pred_adv), (_, rectif_x_nat, pred_nat), id in zip(adv_extras, nat_extras, self.robust_blocks_ids):
            rectif_x_adv, rectif_x_nat = rectif_x_adv.norm(p=2, dim=2), rectif_x_nat.norm(p=2, dim=2)
            loss_im = torch.maximum(torch.zeros_like(rectif_x_nat).cuda(), (rectif_x_nat - (self.tau * rectif_x_adv)) / rectif_x_nat).mean()
            if id < len(self.backbone.blocks): loss += loss_im

        return loss

    def process_saved_extras(self, file):
        #TODO
        pass

    def forward(self, x):
        x = torch.cat((x, self.anchors), dim=0)
        B = x.shape[0]
        x = self.backbone(x)
        if x.shape[0] != B - 10:
            x = x[:-10]
        self.extra_outputs = []
        for i in self.robust_blocks_ids:
            extras = self.backbone.blocks[i][1].extra_outputs
            extras.append(x)
            self.extra_outputs.append(extras)
        return x

class OneImagePerClassSampler(Sampler):
    def __init__(self, dataset, num_classes=10):
        self.num_classes = num_classes
        train_vals = [963, 955, 993, 858, 941, 956, 961, 931, 951, 960]
        val_vals = [387, 395, 357, 386, 409, 394, 389, 419, 399, 390]
        sum_train, sum_val = sum(train_vals), sum(val_vals)
        if len(dataset) == sum_train:
            self.samples_for_each_class = train_vals
        else:
            self.samples_for_each_class = val_vals

    def __iter__(self):
        batch = []
        for cls in range(self.num_classes):
            start_idx = 0
            for i in range(cls):
                start_idx += self.samples_for_each_class[i]
            idx = start_idx + random.randint(0, self.samples_for_each_class[cls] - 1)
            batch.append(idx)
        return iter(batch)

    def __len__(self):
        return self.num_classes * self.samples_per_class