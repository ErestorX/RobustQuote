import timm.models
import torch
import torch.nn as nn
import torch.nn.functional as F
from .deit import deit_tiny_patch16_224, deit_base_patch16_224


def aggregate(A, X, C):
    assert X.shape[-1] == C.shape[-1], "input, codeword feature dim mismatch"
    assert A.shape[:2] == X.shape[:2], "weight, input dim mismatch"
    X = X.unsqueeze(2)  # [b, n, d] -> [b, n, 1, d]
    C = C[None, None, ...]  # [k, d] -> [1, 1, k, d]
    A = A.unsqueeze(-1)  # [b, n, k] -> [b, n, k, 1]
    R = (X - C) * A  # [b, n, k, d]
    E = R.sum(dim=1)  # [b, k, d]
    return E


def scaled_l2(X, C, S):
    assert X.shape[-1] == C.shape[-1], "input, codeword feature dim mismatch"
    assert S.numel() == C.shape[0], "scale, codeword num mismatch"

    b, n, d = X.shape
    X = X.reshape(b*n, d)  # [bn, d]
    Ct = C.t()  # [d, k]
    X2 = X.pow(2.0).sum(-1, keepdim=True)  # [bn, 1]
    C2 = Ct.pow(2.0).sum(0, keepdim=True)  # [1, k]
    norm = X2 + C2 - 2.0 * X.mm(Ct)  # [bn, k]
    scaled_norm = S * norm
    D = scaled_norm.view(b, n, -1)  # [b, n, k]
    return D


class SACNet(nn.Module):
    def __init__(self, crop_size=224, patch_size=16, num_classes=10, conv_features=64, trans_features=32, K=48, D=32, size='tiny', args=None):
        super(SACNet, self).__init__()

        if size == 'base':
            self.backbone = deit_base_patch16_224(pretrained=True, img_size=crop_size, num_classes=10,
                                                  patch_size=patch_size, args=args).cuda()
            self.backbone = nn.DataParallel(self.backbone)
            self.backbone.load_state_dict(torch.load(
                'trades_vanilla_imagenette_deit_base_patch16_224_TRADES_warmup/seed0/weight_decay_0.000100/drop_rate_1.000000/nw_10.000000/checkpoint_40')[
                                              'state_dict'])
            # dim_red_mlp = timm.models.Mlp(in_features=768, hidden_features=768, out_features=384, act_layer=nn.GELU, drop=0.)
            # self.backbone = torch.nn.Sequential(self.backbone, dim_red_mlp)
            dim, num_heads = 768, 12
        else: # size == 'tiny'
            self.backbone = deit_tiny_patch16_224(pretrained=True, img_size=crop_size, num_classes=10,
                                                  patch_size=patch_size, args=args).cuda()
            self.backbone = nn.DataParallel(self.backbone)
            self.backbone.load_state_dict(torch.load(
                'trades_vanilla_imagenette_deit_tiny_patch16_224_TRADES_warmup/seed0/weight_decay_0.000100/drop_rate_1.000000/nw_10.000000/checkpoint_40')[
                                              'state_dict'])
            dim, num_heads = 192, 3
        self.backbone = self.backbone.module
        self.backbone.head = nn.Identity()
        self.backbone.forward_head = lambda x: x[:, 1:]

        self.conv0 = nn.Conv2d(dim, conv_features, kernel_size=3, stride=1, padding='same', dilation=1, bias=True)
        self.conv1 = nn.Conv2d(conv_features, conv_features, kernel_size=3, stride=1, padding='same', dilation=2, bias=True)
        self.conv2 = nn.Conv2d(conv_features, conv_features, kernel_size=3, stride=1, padding='same', dilation=3, bias=True)

        self.alpha3 = nn.Conv2d(conv_features, trans_features, kernel_size=1, stride=1, padding='same', bias=False)
        self.beta3 = nn.Conv2d(conv_features, trans_features, kernel_size=1, stride=1, padding='same', bias=False)
        self.gamma3 = nn.Conv2d(conv_features, trans_features, kernel_size=1, stride=1, padding='same', bias=False)
        self.deta3 = nn.Conv2d(trans_features, conv_features, kernel_size=1, stride=1, padding='same', bias=False)

        self.encoding = nn.Conv2d(conv_features, D, kernel_size=1, stride=1, padding='same', bias=False)

        self.codewords = nn.Parameter(torch.Tensor(K, D), requires_grad=True)
        self.scale = nn.Parameter(torch.Tensor(K), requires_grad=True)
        self.attention = nn.Linear(D, conv_features)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.conv_cls = nn.Conv2d(conv_features * 3, num_classes, kernel_size=1, stride=2, padding=0, bias=True)
        self.mlp_cls = nn.Linear(9 * num_classes, num_classes)

        self.drop = nn.Dropout(0.5)
        self.conv_features = conv_features
        self.trans_features = trans_features
        self.K = K
        self.D = D

        std1 = 1. / ((self.K * self.D) ** (1 / 2))
        self.codewords.data.uniform_(-std1, std1)
        self.scale.data.uniform_(-1, 0)
        self.BN = nn.BatchNorm1d(K)

    def forward(self, x):
        x = self.backbone(x)
        B, N, C = x.size()
        # reshape x from [B, N, C] to [B, C, sqrt(N), sqrt(N)]
        x = x.permute(0, 2, 1).view(B, C, int(N ** 0.5), int(N ** 0.5))

        interpolation = nn.UpsamplingBilinear2d(size=x.shape[2:4])
        x = self.relu(self.conv0(x))
        conv1 = x

        x = self.relu(self.conv1(x))
        conv2 = x
        # x = self.avgpool(x)

        x = self.relu(self.conv2(x))
        n, c, h, w = x.size()
        interpolation_context3 = nn.UpsamplingBilinear2d(size=x.shape[2:4])

        x_half = x  # self.avgpool(x)
        n, c, h, w = x_half.size()
        alpha_x = self.alpha3(x_half)
        beta_x = self.beta3(x_half)
        gamma_x = self.relu(self.gamma3(x_half))

        alpha_x = alpha_x.squeeze().permute(0, 2, 3, 1)
        # h*w x c
        alpha_x = alpha_x.view(B, -1, self.trans_features)
        # c x h*w
        beta_x = beta_x.view(B, self.trans_features, -1)
        gamma_x = gamma_x.view(B, self.trans_features, -1)

        context_x = torch.matmul(alpha_x, beta_x)
        context_x = F.softmax(context_x)

        context_x = torch.matmul(gamma_x, context_x)
        context_x = context_x.view(n, self.trans_features, h, w)
        context_x = interpolation_context3(context_x)

        deta_x = self.relu(self.deta3(context_x))
        x = deta_x + x

        Z = self.relu(self.encoding(x)).view(B, self.D, -1).permute(0, 2, 1)  # n,h*w,D

        A = F.softmax(scaled_l2(Z, self.codewords, self.scale), dim=2)  # b,n,k
        E = aggregate(A, Z, self.codewords)  # b,k,d
        E_sum = torch.sum(self.relu(self.BN(E)), 1)  # b,d
        gamma = self.sigmoid(self.attention(E_sum))  # b,num_conv
        gamma = gamma.view(-1, self.conv_features, 1, 1)
        x = x + x * gamma

        conv1 = interpolation(conv1)
        conv2 = interpolation(conv2)
        context3 = interpolation(x)
        x = torch.cat((conv1, conv2, context3), 1)
        x = self.conv_cls(x)  # b,num_cls,h,w
        x = self.avgpool(self.sigmoid(x)).view(B, -1)
        x = self.mlp_cls(x)  # b,num_cls

        return x
