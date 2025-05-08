import os
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import Mlp
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


class No_Rectifier(nn.Module):
    def __init__(self, num_classes, dim, num_heads):
        super(No_Rectifier, self).__init__()
        self.num_heads = num_heads
        print("No_Rectifier is used")

    def forward(self, x, anchors, weights):
        B, N, C = x.shape
        attn = torch.zeros(B, self.num_heads, N, N).to(x.device)
        x = torch.zeros_like(x).to(x.device)
        return x, attn


class Self_Rectifier(nn.Module):
    def __init__(self, num_classes, dim, num_heads):
        super(Self_Rectifier, self).__init__()
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        print("Self_Rectifier is used")

    def forward(self, x, anchors, weights):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # [B, H, N, C] @ [B, H, C, N] -> [B, H, N, N]
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-2)
        # [B, H, N, N] @ [B, H, N, C] -> [B, H, N, C] -> [B, N, C]
        x = self.proj((attn @ v).reshape(B, N, C))
        return x, attn


class Random_Rectifier(nn.Module):
    def __init__(self, num_classes, dim, num_heads):
        super(Random_Rectifier, self).__init__()
        self.scale = (dim // num_heads) ** -0.5

        self.v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        print("Random_Rectifier is used")

    def forward(self, x, anchors, weights):
        B, N, C = x.shape
        v = self.v(x)
        attn = torch.rand(B, N, N).to(x.device) * self.scale
        # [B, N, N] @ [B, N, C] -> [B, N, C]
        return self.proj((attn @ v)), attn


class References_Rectifier(nn.Module):
    def __init__(self, num_classes, dim, num_heads):
        super(References_Rectifier, self).__init__()
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qk = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        print("References_Rectifier is used")

    def forward(self, x, anchors, weights):
        B, N, C = x.shape
        v = self.v(anchors).reshape(1, self.num_classes, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)
        q = self.qk(x).reshape(B, 1, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)
        k = self.qk(anchors).reshape(1, self.num_classes, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)
        # [B, 1, H, N, C] @ [1, num_classes, H, C, N'] -> [B, num_classes, H, N, N']
        attn = (-1 * (q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-2)
        # [B, num_classes, H, N, N'] @ [1, num_classes, H, N', C] -> [B, num_classes, H, N, C] -> [B, H, N, num_classes, C]
        tmp = (attn @ v).permute(0, 2, 3, 1, 4)
        # [B, H, 1, 1, num_classes] @ [B, H, N, num_classes, C] -> [B, H, N, C]
        x = (weights.reshape(B, self.num_heads, 1, 1, self.num_classes) @ tmp).squeeze(-2)
        # [B, H, N, C] -> [B, N, C]
        x = self.proj(x.permute(0, 2, 1, 3).reshape(B, N, C))
        return x, attn


class Shared_Conjugated_Rectifier(nn.Module):
    def __init__(self, num_classes, dim, num_heads):
        super(Shared_Conjugated_Rectifier, self).__init__()
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qk = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        print("Shared_Conjugated_Rectifier is used")

    def forward(self, x, anchors, weights):
        B, N, C = x.shape
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = self.qk(x).reshape(B, 1, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)
        k = self.qk(anchors).reshape(1, self.num_classes, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)
        # [B, 1, H, N, C] @ [1, num_classes, H, C, N'] -> [B, num_classes, H, N, N']
        a0 = (q @ k.transpose(-2, -1)) * self.scale
        left_attn, right_attn = -1 * a0, a0.transpose(-2, -1)
        left_attn[left_attn < 0] = 0
        right_attn[right_attn < 0] = 0
        # [B, num_classes, H, N, N'] @ [B, num_classes, H, N', N] -> [B, num_classes, H, N, N]
        attn = (left_attn @ right_attn).softmax(dim=-2).permute(0, 2, 3, 1, 4)
        # [B, H, 1, 1, num_classes] @ [B, H, N, num_classes, N] -> [B, H, N, N]
        tmp = (weights.reshape(B, self.num_heads, 1, 1, self.num_classes) @ attn).squeeze(-2)
        # [B, H, N', N] @ [B, H, N, C] -> [B, H, N', C] -> [B, N', C]
        x = self.proj((tmp @ v).reshape(B, N, C))
        return x, a0


class Conjugated_Rectifier(nn.Module):
    def __init__(self, num_classes, dim, num_heads):
        super(Conjugated_Rectifier, self).__init__()
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qk = nn.Linear(dim, dim, bias=False)
        self.qk2 = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        print("Conjugated_Rectifier is used")

    def forward(self, x, anchors, weights):
        B, N, C = x.shape
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = self.qk(x).reshape(B, 1, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)
        k = self.qk(anchors).reshape(1, self.num_classes, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)
        q2 = self.qk2(x).reshape(B, 1, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)
        k2 = self.qk2(anchors).reshape(1, self.num_classes, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)
        # [B, 1, H, N, C] @ [1, num_classes, H, C, N'] -> [B, num_classes, H, N, N']
        left_attn = -1 * (q @ k.transpose(-2, -1)) * self.scale
        right_attn = (k2 @ q2.transpose(-2, -1)) * self.scale
        left_attn[left_attn<0] = 0
        right_attn[right_attn<0] = 0
        # [B, num_classes, H, N, N'] @ [B, num_classes, H, N', N] -> [B, num_classes, H, N, N]
        attn = (left_attn @ right_attn).permute(0, 2, 3, 1, 4)
        # [B, H, 1, 1, num_classes] @ [B, H, N, num_classes, N] -> [B, H, N, N]
        tmp = (weights.reshape(B, self.num_heads, 1, 1, self.num_classes) @ attn).squeeze(-2)
        # [B, H, N, N] @ [B, H, N, C] -> [B, H, N', C] -> [B, N, C]
        x = self.proj((tmp @ v).reshape(B, N, C))
        return x, left_attn


class RobustQuoteModule(nn.Module):
    def __init__(self, num_classes, dim, num_heads, test=False, rectifier='Conjugated_', keep_anchors=True):
        super(RobustQuoteModule, self).__init__()
        if (rectifier + 'Rectifier') not in globals():
            print(f"Rectifier {rectifier} not found, using Conjugated_Rectifier")
            rectifier = 'Conjugated_'
        self.num_classes = num_classes
        self.test = test
        self.keep_anchors = keep_anchors
        self.num_heads = num_heads

        self.quoter = Quoter(num_classes, dim, num_heads)
        rectifier_fn = globals()[rectifier + 'Rectifier']
        self.rectifier = rectifier_fn(num_classes, dim, num_heads)
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
        x_im_rectifications, attn_im = self.rectifier(self.norm2(x_im_tokens), self.norm2(anchors_im_tokens), attn_cls)
        self.extra_outputs = [attn_cls, attn_im, x_im_rectifications]

        x = torch.cat((x_cls, x_im_tokens + x_im_rectifications), dim=1)
        x = x + self.mlp(self.norm3(x))

        if self.keep_anchors:
            x = torch.cat((x, anchors), dim=0)
        return x


class RobustQuoteNet(nn.Module):
    def __init__(self, blocks, alpha=0.5, tau=0.9, test=False, crop_size=224,
                 patch_size=16, size='tiny', rectifier='Conjugated_', args=None):
        super(RobustQuoteNet, self).__init__()
        self.robust_blocks_ids = blocks
        self.alpha = alpha
        if rectifier == 'No_':
            self.alpha = 0.0
        self.tau = tau
        self.test = test
        if size == 'base':
            self.backbone = deit_base_patch16_224(pretrained=True, img_size=crop_size, num_classes=10, patch_size=patch_size, args=args).cuda()
            dim, num_heads = 768, 12
        else: # size == 'tiny'
            self.backbone = deit_tiny_patch16_224(pretrained=True, img_size=crop_size, num_classes=10, patch_size=patch_size, args=args).cuda()
            dim, num_heads = 192, 3

        self.backbone = nn.DataParallel(self.backbone)
        self.backbone.load_state_dict(torch.load(
            f'trades_vanilla_cifar_deit_{size}_patch16_224_TRADES_warmup/seed0/weight_decay_0.000100/drop_rate_1.000000/nw_10.000000/checkpoint_40')[
                                          'state_dict'])
        self.backbone = self.backbone.module

        for list_index, i in enumerate(self.robust_blocks_ids):
            self.backbone.blocks[i] = nn.Sequential(self.backbone.blocks[i], RobustQuoteModule(10, dim, num_heads, test=test, rectifier=rectifier, keep_anchors=(list_index != len(self.robust_blocks_ids) - 1)))

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
        # extras = [[attn_cls, attn_im, rectified_x_im_tokens, pred], ...]
        # attn_cls: (B, num_classes, num_heads) with mean(attn_cls, dim=1) = 1
        # rectified_x_im_tokens: (B, N, C)
        loss = 0.0
        if self.alpha == 0.0:
            return loss
        for (_, _ , rectif_x_adv, pred_adv), (_, _ , rectif_x_nat, pred_nat), id in zip(adv_extras, nat_extras, self.robust_blocks_ids):
            rectif_x_adv, rectif_x_nat = rectif_x_adv.norm(p=2, dim=2), rectif_x_nat.norm(p=2, dim=2)
            loss_im = torch.maximum(torch.zeros_like(rectif_x_nat).cuda(), (rectif_x_nat - (self.tau * rectif_x_adv)) / rectif_x_nat).mean()
            if id < len(self.backbone.blocks): loss += loss_im

        return loss

    def process_and_save_extras(self, input, labels, path=None):
        img_id = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
        std_cifar, mean_cifar = torch.tensor([0.2471, 0.2435, 0.2616]), torch.tensor([0.4914, 0.4822, 0.4465])

        # look at the attn_cls and attn_im for each block as well as the predicted class
        for blk_id, vals in enumerate(self.extra_outputs):
            attn_cls, attn_im , rectif_x, pred = vals
            input, attn_cls, attn_im, rectif_x, pred = input[0].detach().cpu(), attn_cls[0].detach().cpu(), attn_im[0].detach().cpu(), rectif_x[0].detach().cpu(), pred[0].detach().cpu()
            label = labels[0].item()
            attn_im, rectif_x = attn_im.permute(2, 1, 3, 0), rectif_x.norm(p=2, dim=1)
            attn_im = (attn_im @ attn_cls.unsqueeze(1).unsqueeze(-1)).squeeze() # -> (num_classes, N, N)
            side, true_side = int(np.sqrt(attn_im.shape[1])), input.shape[-1]
            tmp = attn_im.max(1)[0] # -> (num_classes, N)
            attn_im_input = tmp.reshape(1, -1, side, side)
            tmp = attn_im.permute(0, 2, 1).max(1)[0] # -> (num_classes, N)
            attn_im_anchors = tmp.reshape(1, -1, side, side)
            # rescale the attention maps to the input size true_side
            attn_im_input = F.interpolate(attn_im_input, size=true_side, mode='bilinear', align_corners=False)
            attn_im_anchors = F.interpolate(attn_im_anchors, size=true_side, mode='bilinear', align_corners=False)
            # attn_im_input, attn_im_anchors -> (1, num_classes, true_side, true_side)
            attn_im_input, attn_im_anchors = attn_im_input.permute(1, 2, 3, 0), attn_im_anchors.permute(1, 2, 3, 0)
            # create a plot with two rows, one for the input and one for the anchors
            fig, axs = plt.subplots(2, self.anchors.shape[0], figsize=(24, 6))
            pred_cls = pred.argmax().item()
            for i in range(attn_im_input.shape[0]):
                # for each class, we apply as heatmap attn_im_anchor on the corresponding anchor and attn_im_input on the input
                anchor = self.anchors[i].permute(1, 2, 0).detach().cpu()
                axs[0, i].imshow(input.permute(1, 2, 0) * std_cifar + mean_cifar)
                axs[1, i].imshow(anchor * std_cifar + mean_cifar)
                axs[0, i].axis('off'), axs[1, i].axis('off'), axs[0, i].set_xticks([]), axs[1, i].set_xticks([]), axs[0, i].set_yticks([]), axs[1, i].set_yticks([])
                tmp = (attn_im_input[i] - attn_im_input.min()) / (attn_im_input.max() - attn_im_input.min())
                axs[0, i].imshow(tmp, cmap='jet', alpha=0.3)
                tmp = (attn_im_anchors[i] - attn_im_anchors.min()) / (attn_im_anchors.max() - attn_im_anchors.min())
                axs[1, i].imshow(tmp, cmap='jet', alpha=0.3)
                title = ""
                if i == pred_cls: title += "-Predicted-"
                if i == label: title += "(True label)"
                axs[0, i].set_title(title)
            fig.suptitle(f"Attention maps for the input and the anchors, correction={rectif_x.mean().item()}")
            plt.savefig(f"{path}/{img_id}_blk_{self.robust_blocks_ids[blk_id]}.png")
            plt.close()

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
