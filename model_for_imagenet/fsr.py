import torch
import torch.nn as nn
import numpy as np
from .deit import deit_tiny_patch16_224, deit_base_patch16_224


class GumbelSigmoid(nn.Module):
    def __init__(self, tau=1.0):
        super(GumbelSigmoid, self).__init__()

        self.tau = tau
        self.softmax = nn.Softmax(dim=1)
        self.p_value = 1e-8

    def forward(self, x, is_eval=False):
        r = 1 - x

        x = (x + self.p_value).log()
        r = (r + self.p_value).log()

        if not is_eval:
            x_N = torch.rand_like(x)
            r_N = torch.rand_like(r)
        else:
            x_N = 0.5 * torch.ones_like(x)
            r_N = 0.5 * torch.ones_like(r)

        x_N = -1 * (x_N + self.p_value).log()
        r_N = -1 * (r_N + self.p_value).log()
        x_N = -1 * (x_N + self.p_value).log()
        r_N = -1 * (r_N + self.p_value).log()

        x = x + x_N
        x = x / (self.tau + self.p_value)
        r = r + r_N
        r = r / (self.tau + self.p_value)

        x = torch.cat((x, r), dim=1)
        x = self.softmax(x)

        return x


class Separation(torch.nn.Module):
    def __init__(self, size, num_channel=64, tau=0.1):
        super(Separation, self).__init__()
        C, H = size
        self.C, self.H = C, H
        self.tau = tau

        self.sep_net = nn.Sequential(
            nn.Conv2d(C, num_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, C, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, feat, is_eval=False):
        rob_map = self.sep_net(feat)

        mask = rob_map.reshape(rob_map.shape[0], 1, -1)
        mask = torch.nn.Sigmoid()(mask)
        mask = GumbelSigmoid(tau=self.tau)(mask, is_eval=is_eval)
        mask = mask[:, 0].reshape(mask.shape[0], self.C, self.H, self.H)

        r_feat = feat * mask
        nr_feat = feat * (1 - mask)

        return r_feat, nr_feat, mask


class Recalibration(nn.Module):
    def __init__(self, size, num_channel=64):
        super(Recalibration, self).__init__()
        C, H = size
        self.rec_net = nn.Sequential(
            nn.Conv2d(C, num_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, C, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, nr_feat, mask):
        rec_units = self.rec_net(nr_feat)
        rec_units = rec_units * (1 - mask)
        rec_feat = nr_feat + rec_units

        return rec_feat


class FSR(nn.Module):
    def __init__(self, crop_size=224, patch_size=16, classes=1000, size='tiny', args=None):
        super(FSR, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.lambda_sep = 1.0
        self.lambda_rec = 1.0
        self.alpha = 0.5

        ds_code = 'imagenet' if classes == 1000 else 'imagenette'
        if size == 'base':
            self.backbone = deit_base_patch16_224(pretrained=True, img_size=crop_size, num_classes=classes,
                                                  patch_size=patch_size, args=args).cuda()
            # self.backbone = nn.DataParallel(self.backbone)
            # self.backbone.load_state_dict(torch.load(
            #     f'trades_vanilla_{ds_code}_deit_base_patch16_224_TRADES_warmup/seed0/weight_decay_0.000100/drop_rate_1.000000/nw_10.000000/checkpoint_40')[
            #                                   'state_dict'])
            dim, num_heads = 768, 12
        else: # size == 'tiny'
            self.backbone = deit_tiny_patch16_224(pretrained=True, img_size=crop_size, num_classes=classes,
                                                  patch_size=patch_size, args=args).cuda()
            # self.backbone = nn.DataParallel(self.backbone)
            # self.backbone.load_state_dict(torch.load(
            #     f'trades_vanilla_{ds_code}_deit_tiny_patch16_224_TRADES_warmup/seed0/weight_decay_0.000100/drop_rate_1.000000/nw_10.000000/checkpoint_40')[
            #                                   'state_dict'])
            dim, num_heads = 192, 3
        # self.backbone = self.backbone.module
        self.backbone.head = nn.Identity()
        self.backbone.forward_head = lambda x: x[:, 1:]

        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        side_dim = int(np.sqrt(self.backbone.patch_embed.num_patches))
        self.separation = Separation(size=(dim, side_dim), tau=0.1)
        self.recalibration = Recalibration(size=(dim, side_dim))
        self.aux = nn.Sequential(nn.Linear(dim, classes))

        self.linear = nn.Linear(dim, classes)

    def extra_loss(self, adv_extras, nat_extras, y):
        adv_logits, adv_r_logits, adv_nr_logits, adv_rec_logits = adv_extras

        logits = adv_logits.sort(dim=-1, descending=True)[1][:, 0]
        second_logits = adv_logits.sort(dim=-1, descending=True)[1][:, 1]
        adv_y = torch.where(logits == y, second_logits, logits)

        r_loss = torch.tensor(0.).cuda()
        if not len(adv_r_logits) == 0:
            for r_out in adv_r_logits:
                r_loss += self.lambda_sep * self.loss_fn(r_out, y)
            r_loss /= len(adv_r_logits)

        nr_loss = torch.tensor(0.).cuda()
        if not len(adv_nr_logits) == 0:
            for nr_out in adv_nr_logits:
                nr_loss += self.lambda_sep * self.loss_fn(nr_out, adv_y)
            nr_loss /= len(adv_nr_logits)
        sep_loss = r_loss + nr_loss

        rec_loss = torch.tensor(0.).cuda()
        if not len(adv_rec_logits) == 0:
            for rec_out in adv_rec_logits:
                rec_loss += self.lambda_rec * self.loss_fn(rec_out, y)
            rec_loss /= len(adv_rec_logits)

        return sep_loss + rec_loss

    def forward(self, x):
        r_outputs = []
        nr_outputs = []
        rec_outputs = []

        out = self.backbone(x)
        B, N, C = out.shape
        out = out.permute(0, 2, 1).view(B, C, int(np.sqrt(N)), int(np.sqrt(N)))

        r_feat, nr_feat, mask = self.separation(out, is_eval=not self.training)
        r_out = self.aux(torch.nn.AdaptiveAvgPool2d(1)(r_feat).reshape(r_feat.shape[0], -1))
        r_outputs.append(r_out)
        nr_out = self.aux(torch.nn.AdaptiveAvgPool2d(1)(nr_feat).reshape(nr_feat.shape[0], -1))
        nr_outputs.append(nr_out)

        rec_feat = self.recalibration(nr_feat, mask)
        rec_out = self.aux(torch.nn.AdaptiveAvgPool2d(1)(rec_feat).reshape(rec_feat.shape[0], -1))
        rec_outputs.append(rec_out)

        out = r_feat + rec_feat

        out = nn.AdaptiveAvgPool2d(1)(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        self.extra_outputs = (out, r_outputs, nr_outputs, rec_outputs)
        return out
