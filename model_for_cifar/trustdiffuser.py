from itertools import cycle
import torch
import torch.nn as nn
from timm.models.layers import Mlp
import torch.nn.functional as F
from .deit import deit_tiny_patch16_224


def piecewise_step_fn(x, delta=0.25):
    # Piecewise step function: for each value in the tensor x apply the following rule:
    # if x < -delta, x <- 0.0
    # if x > delta, x <- 1.0
    # otherwise, x <- (x/2*delta) + 0.5
    return torch.where(x < -delta, torch.zeros_like(x), torch.where(x > delta, torch.ones_like(x), (x / (2 * delta)) + 0.5))


class Diffuser(nn.Module):
    def __init__(self, D=768, num_heads=12, diffusion_steps=3, nb_ref=10):
        super(Diffuser, self).__init__()
        self.diffusion_steps = diffusion_steps
        self.nb_ref = nb_ref
        self.num_heads = num_heads
        self.scale = (D // num_heads) ** -0.5

        self.qv = nn.Linear(D, 2 * D, bias=False)
        self.k = nn.Linear(D, D, bias=False)
        self.proj = nn.Linear(D, D)
        self.norm = nn.LayerNorm(D)

    def forward(self, x, ref):
        for i in range(self.diffusion_steps):
            B, N, C = x.shape
            qv = self.qv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, v = qv.unbind(0)
            k = self.k(ref).reshape(self.nb_ref, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            # q, v: [B, 1, H, N, C], k: [1, nb_ref, H, N', C]
            q, v, k = q.unsqueeze(1), v.unsqueeze(1), k.unsqueeze(0)

            # attn: [B, nb_ref, H, N, N']
            attn = (q @ k.transpose(-2, -1)) * self.scale
            # attn: [B, nb_ref, H, N, N]
            attn = attn @ attn.transpose(-2, -1)
            # x: [B, nb_ref, H, N, C], [B, nb_ref, N, C], [B, N, C, nb_ref], [B, N, C]
            x = (attn @ v).reshape(B, self.nb_ref, N, C).permute(0, 2, 3, 1).mean(dim=-1)
            x = self.norm(self.proj(x))
        return x


class Projector(nn.Module):
    def __init__(self, D=768, tau=0.5):
        super(Projector, self).__init__()
        self.tau = tau
        self.encoder = Mlp(D, D//4, D//8, nn.GELU)

    def forward(self, x, ref):
        x = self.encoder(x).unsqueeze(2)
        ref = self.encoder(ref).permute(1, 0, 2).unsqueeze(0)
        correct_m = (torch.cosine_similarity(x, ref, dim=3).mean(dim=2) + 1) / 2
        correct_m = 0.5 + 0.5 * torch.tanh(25 * (correct_m - self.tau)).unsqueeze(2)
        return correct_m, [x.squeeze(2), ref.permute(0, 2, 1, 3).squeeze(0)]


class TrustDiffuserModule(nn.Module):
    def __init__(self, in_dim, num_heads, nb_ref=10, diffusion_steps=3, last_layer=False):
        super(TrustDiffuserModule, self).__init__()
        self.nb_ref = nb_ref
        self.projector = Projector(in_dim, tau=0.5)
        self.diffuser = Diffuser(D=in_dim, num_heads=num_heads, diffusion_steps=diffusion_steps, nb_ref=nb_ref)
        self.rectified_vects = None
        self.last_layer = last_layer
        self.extras = [None, None, None]

    def forward(self, x):
        x, ref = x[:-self.nb_ref], x[-self.nb_ref:]
        ref = ref.detach()
        correct_m, embed_x_ref = self.projector(x, ref)
        keep_m = 1 - correct_m
        corrected = self.diffuser(x * correct_m, ref)
        x = correct_m * corrected + keep_m * x
        embed_corrected = self.projector.encoder(x.clone().detach())
        self.extras = [embed_x_ref[1], embed_x_ref[0], embed_corrected]
        return x if self.last_layer else torch.cat((x, ref), dim=0)


class TrustDiffuserNet(nn.Module):
    def __init__(self, blocks, nb_anchors=5, crop_size=224, patch_size=16, args=None):
        super(TrustDiffuserNet, self).__init__()
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        self.nb_anchors = nb_anchors
        self.robust_blocks_ids = blocks
        self.alpha = 0.5

        self.backbone = deit_tiny_patch16_224(pretrained=True, img_size=crop_size, num_classes=10, patch_size=patch_size, args=args).cuda()
        self.backbone = nn.DataParallel(self.backbone)
        self.backbone.load_state_dict(torch.load('trades_vanilla_cifar_deit_tiny_patch16_224_TRADES_warmup/seed0/weight_decay_0.000100/drop_rate_1.000000/nw_10.000000/checkpoint_40')['state_dict'])
        self.backbone = self.backbone.module
        for id, i in enumerate(self.robust_blocks_ids):
            self.backbone.blocks[i] = nn.Sequential(self.backbone.blocks[i], TrustDiffuserModule(192, 3, nb_ref=self.nb_anchors, last_layer= id+1 == len(self.robust_blocks_ids)))

        for param in self.backbone.parameters():
            param.requires_grad = False
        for id, i in enumerate(self.robust_blocks_ids):
            for param in self.backbone.blocks[i][1].parameters():
                param.requires_grad = True

    def make_internal_anchor_loader(self, dataset):
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.nb_anchors, shuffle=True)
        self.anchor_ds = cycle(iter(loader))

    def update_anchors(self):
        assert hasattr(self, 'anchor_ds')
        self.anchors = next(self.anchor_ds)[0].cuda()

    # def extra_loss(self, adv_extras, nat_extras, y):
    #     loss = 0.0
    #     for i in range(len(adv_extras)):
    #         x_nat_rectified, x_adv, x_adv_rectified = nat_extras[i][2], nat_extras[i][1], adv_extras[i][2]
    #         loss += self.triplet_loss(x_nat_rectified, x_adv_rectified, x_adv)
    #     return loss / len(self.robust_blocks_ids)
    #
    # def forward(self, x):
    #     x = torch.cat((x, self.anchors), dim=0)
    #     x = self.backbone(x)
    #     self.extra_outputs = []
    #     for i in self.robust_blocks_ids:
    #         self.extra_outputs.append(self.backbone.blocks[i][1].extras)
    #     return x

    def extra_loss(self, adv_extras, nat_extras, y):
        criterion_kl, beta, batch_size = nn.KLDivLoss(reduction='sum'), 6.0, len(y)
        nat_in, adv_rect_out = nat_extras[0], adv_extras[1]
        for i in self.robust_blocks_ids:
            self.backbone.blocks[i][1].set_do_diffusion(False)
        nat_out = self.backbone(torch.cat((nat_in, self.anchors), dim=0))
        for i in self.robust_blocks_ids:
            self.backbone.blocks[i][1].set_do_diffusion(True)

        return (beta / batch_size) * criterion_kl(F.log_softmax(adv_rect_out, dim=1), F.softmax(nat_out, dim=1))

    def forward(self, x):
        self.extra_outputs = [x, None]
        x = torch.cat((x, self.anchors), dim=0)
        x = self.backbone(x)
        self.extra_outputs[1] = x
        return x
