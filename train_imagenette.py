import numpy as np
import tqdm
from functools import partial
import random
from timm.loss import  SoftTargetCrossEntropy
from timm.data import Mixup
from parser_imagenette import get_args
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from auto_LiRPA.utils import MultiAverageMeter
from utils import *
from torch.autograd import Variable
from pgd import evaluate_pgd,evaluate_CW
from evaluate import evaluate_aa
from auto_LiRPA.utils import logger
args = get_args()


args.out_dir = args.out_dir+"_"+args.dataset+"_"+args.model+"_"+args.method+"_warmup"
args.out_dir = args.out_dir +"/seed"+str(args.seed)
if args.ARD:
    args.out_dir = args.out_dir + "_ARD"
if args.PRM:
    args.out_dir = args.out_dir + "_PRM"
if args.scratch:
    args.out_dir = args.out_dir + "_no_pretrained"
if args.load:
    args.out_dir = args.out_dir + "_load"

args.out_dir = args.out_dir + "/weight_decay_{:.6f}/".format(
        args.weight_decay)+ "drop_rate_{:.6f}/".format(args.drop_rate)+"nw_{:.6f}/".format(args.n_w)

print(args.out_dir)
os.makedirs(args.out_dir,exist_ok=True)
logfile = os.path.join(args.out_dir, 'log_{:.4f}.log'.format(args.weight_decay))
prev_logfiles = []
if os.path.exists(logfile):
    prev_logfiles.append(logfile)
    logfile = os.path.join(args.out_dir, 'log_{:.4f}_v2.log'.format(args.weight_decay))
    version = 2
    while os.path.exists(logfile):
        version += 1
        prev_logfiles.append(logfile)
        logfile = os.path.join(args.out_dir, 'log_{:.4f}_v{}.log'.format(args.weight_decay, version))

file_handler = logging.FileHandler(logfile)
file_handler.setFormatter(logging.Formatter('%(levelname)-8s %(asctime)-12s %(message)s'))
logger.addHandler(file_handler)

logger.info(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

resize_size = args.resize
crop_size = args.crop
# if args.model == "vit_small_patch16_224":
#     from model_for_imagenet.vit import vit_small_patch16_224
#     model = vit_small_patch16_224(pretrained= (not args.scratch),num_classes =10).cuda()
#     model = nn.DataParallel(model)
#     logger.info('Model{}'.format(model))
# elif args.model == "vit_base_patch16_224":
#     from model_for_imagenet.vit import vit_base_patch16_224
#     model = vit_base_patch16_224(pretrained= (not args.scratch),num_classes =10).cuda()
#     model = nn.DataParallel(model)
#     logger.info('Model{}'.format(model))
# elif args.model == "vit_base_patch16_224_in21k":
#     from model_for_imagenet.vit import vit_base_patch16_224_in21k
#     model = vit_base_patch16_224_in21k(pretrained=(not args.scratch), num_classes=10).cuda()
#     model = nn.DataParallel(model)
#     logger.info('Model{}'.format(model))
# elif args.model == "deit_small_patch16_224":
#     from model_for_imagenet.deit import deit_small_patch16_224
#     model = deit_small_patch16_224(pretrained = (not args.scratch),num_classes =10 ).cuda()
#     model = nn.DataParallel(model)
#     logger.info('Model{}'.format(model))
if args.model == "deit_tiny_patch16_224":
    from model_for_imagenet.deit import deit_tiny_patch16_224
    model = deit_tiny_patch16_224(pretrained = (not args.scratch),num_classes =10).cuda()
    model = nn.DataParallel(model)
    # logger.info('Model{}'.format(model))
elif args.model == "deit_base_patch16_224":
    from model_for_imagenet.deit import deit_base_patch16_224
    model = deit_base_patch16_224(pretrained = (not args.scratch),num_classes =10).cuda()
    model = nn.DataParallel(model)
    # logger.info('Model{}'.format(model))
# elif args.model == "convit_base":
#     from model_for_imagenet.convit import convit_base
#     model = convit_base(pretrained=(not args.scratch), num_classes=10).cuda()
#     model = nn.DataParallel(model)
#     logger.info('Model{}'.format(model))
# elif args.model == "convit_small":
#     from model_for_imagenet.convit import convit_small
#     model = convit_small(pretrained=(not args.scratch), num_classes=10).cuda()
#     model = nn.DataParallel(model)
#     logger.info('Model{}'.format(model))
# elif args.model == "convit_tiny":
#     from model_for_imagenet.convit import convit_tiny
#     model = convit_tiny(pretrained = (not args.scratch),num_classes =10).cuda()
#     model = nn.DataParallel(model)
#     logger.info('Model{}'.format(model))
# elif args.model  == "swin_tiny_patch4_window7_224":
#     args.momentum = 0.5
#     from model_for_imagenet.swin import swin_tiny_patch4_window7_224
#     model = swin_tiny_patch4_window7_224(pretrained = (not args.scratch),num_classes =10).cuda()
#     model = nn.DataParallel(model)
#     logger.info('Model{}'.format(model))
# elif args.model  == "swin_small_patch4_window7_224":
#     args.momentum = 0.5
#     from model_for_imagenet.swin import swin_small_patch4_window7_224
#     model = swin_small_patch4_window7_224(pretrained = (not args.scratch), num_classes =10).cuda()
#     model = nn.DataParallel(model)
#     logger.info('Model{}'.format(model))
# elif args.model  == "swin_base_patch4_window7_224":
#     args.momentum = 0.5
#     from model_for_imagenet.swin import swin_base_patch4_window7_224
#     model = swin_base_patch4_window7_224(pretrained = (not args.scratch),num_classes =10).cuda()
#     model = nn.DataParallel(model)
#     logger.info('Model{}'.format(model))
elif args.model == 'fsr_tiny_patch16_224':
    from model_for_imagenet.fsr import FSR
    model = FSR(crop_size=crop_size, patch_size=args.patch, args=args, classes=10)
    model = nn.DataParallel(model)
    # logger.info('Model{}'.format(model))
    # args.epochs = args.epochs // 2
elif args.model == 'sacnet_tiny_patch16_224':
    from model_for_imagenet.sacnet import SACNet
    model = SACNet(crop_size=crop_size, patch_size=args.patch, args=args)
    model = nn.DataParallel(model)
#    logger.info('Model{}'.format(model))
    args.epochs = args.epochs // 2
elif args.model == 'dh_at_tiny_patch16_224':
    from model_for_imagenet.dh_at import DeiT_Attach
    model = DeiT_Attach(crop_size=crop_size, patch_size=args.patch, args=args)
    model = nn.DataParallel(model)
#    logger.info('Model{}'.format(model))
#     args.epochs = args.epochs // 2
elif args.model == 'robustquote6_tiny_patch16_224':
    from model_for_imagenet.robustquote import RobustQuoteNet
    model = RobustQuoteNet(blocks=[6], crop_size=crop_size, patch_size=args.patch, args=args)
    model = nn.DataParallel(model)
    # logger.info('Model{}'.format(model))
    args.epochs = args.epochs // 2
elif args.model == 'fsr_base_patch16_224':
    from model_for_imagenet.fsr import FSR
    model = FSR(crop_size=crop_size, patch_size=args.patch, args=args, classes=10, size='base')
    model = nn.DataParallel(model)
    # logger.info('Model{}'.format(model))
    # args.epochs = args.epochs // 2
elif args.model == 'sacnet_base_patch16_224':
    from model_for_imagenet.sacnet import SACNet
    model = SACNet(crop_size=crop_size, patch_size=args.patch, args=args, size='base')
    model = nn.DataParallel(model)
#    logger.info('Model{}'.format(model))
    args.epochs = args.epochs // 2
elif args.model == 'dh_at_base_patch16_224':
    from model_for_imagenet.dh_at import DeiT_Attach
    model = DeiT_Attach(crop_size=crop_size, patch_size=args.patch, args=args, size='base')
    model = nn.DataParallel(model)
#    logger.info('Model{}'.format(model))
#     args.epochs = args.epochs // 2
elif args.model == 'robustquote6_base_patch16_224':
    from model_for_imagenet.robustquote import RobustQuoteNet
    model = RobustQuoteNet(blocks=[6], crop_size=crop_size, patch_size=args.patch, args=args, size='base')
    model = nn.DataParallel(model)
    args.batch_size = int(args.batch_size * 0.33)
    args.AA_batch = int(args.AA_batch * 0.33)
    # logger.info('Model{}'.format(model))
    args.epochs = args.epochs // 2
else:
    raise ValueError("Model doesn't exist！")

train_loader, test_loader, train_dataset, test_dataset = get_loaders(args)

model.train()
if args.load:
    checkpoint = torch.load(args.load_path)
    model.load_state_dict(checkpoint['state_dict'])

def evaluate_natural(args, model, test_loader, verbose=False):
    model.eval()
    with torch.no_grad():
        meter = MultiAverageMeter()
        def test_step(step, X_batch, y_batch):
            X, y = X_batch.cuda(), y_batch.cuda()
            output = model(X)
            loss = F.cross_entropy(output, y)
            meter.update('test_loss', loss.item(), y.size(0))
            meter.update('test_acc', (output.max(1)[1] == y).float().mean(), y.size(0))
        for step, (X_batch, y_batch) in enumerate(test_loader):
            test_step(step, X_batch, y_batch)
            if hasattr(model.module, 'update_anchors'):
                model.module.update_anchors()
        logger.info('Evaluation {}'.format(meter))


def train_adv(args, model, ds_train, ds_test, logger):
    mu = torch.tensor(imagenet_mean).view(3, 1, 1).cuda()
    std = torch.tensor(imagenet_std).view(3, 1, 1).cuda()

    upper_limit = ((1 - mu) / std).cuda()
    lower_limit = ((0 - mu) / std).cuda()

    epsilon_base = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std

    train_loader, test_loader = ds_train, ds_test

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active :
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.labelsmoothvalue, num_classes=10)

    if mixup_active:
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = nn.CrossEntropyLoss()

    steps_per_epoch = len(train_loader)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.delta_init == 'previous':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
    lr_steps = args.epochs * steps_per_epoch
    def lr_schedule(t):
        if t< args.epochs-5:
            return args.lr_max
        elif t< args.epochs-2:
            return args.lr_max*0.1
        else:
            return args.lr_max * 0.01
    epoch_s = 0
    evaluate_natural(args, model, test_loader, verbose=False)
    for epoch in tqdm.tqdm(range(epoch_s + 1, args.epochs + 1)):
        train_loss = 0
        train_acc = 0
        train_n = 0

        def train_step(X, y,t,mixup_fn):
            model.train()
            def attn_drop_mask_grad(module, grad_in, grad_out, drop_rate):
                new = np.random.rand()
                if new > drop_rate:
                    gamma = 0
                else:
                    gamma = 1
                if len(grad_in) == 1:
                    mask = torch.ones_like(grad_in[0]) * gamma
                    return (mask * grad_in[0][:],)
                else:
                    mask = torch.ones_like(grad_in[0]) * gamma
                    mask_1 = torch.ones_like(grad_in[1]) * gamma
                    return (mask * grad_in[0][:], mask_1 * grad_in[1][:])
            if t < args.n_w:
                drop_rate = t / args.n_w * args.drop_rate
            else:
                drop_rate = args.drop_rate
            drop_hook_func = partial(attn_drop_mask_grad, drop_rate=drop_rate)
            model.eval()
            handle_list = list()
            if args.model in ["vit_base_patch16_224", "vit_large_patch16_224", "vit_small_patch16_224"]:
                if args.ARD:
                    from model_for_imagenet.vit import Block
                    for name, module in model.named_modules():
                        if isinstance(module, Block):
                            handle_list.append(module.drop_path.register_backward_hook(drop_hook_func))
            elif args.model in ["deit_small_patch16_224", "deit_tiny_patch16_224"]:
                if args.ARD:
                    from model_for_imagenet.deit import Block
                    for name, module in model.named_modules():
                        if isinstance(module, Block):
                            handle_list.append(
                                module.drop_path.register_backward_hook(drop_hook_func))
            elif args.model in ["convit_base", "convit_small", "convit_tiny"]:
                if args.ARD:
                    from model_for_imagenet.convit import Block
                    for name, module in model.named_modules():
                        if isinstance(module, Block):
                            handle_list.append(
                                module.drop_path.register_backward_hook(drop_hook_func))
            elif args.model in ["swin_tiny_patch4_window7_224", "swin_small_patch4_window7_224",
                                "swin_base_patch4_window7_224"]:
                if args.ARD:
                    from model_for_imagenet.swin import SwinTransformerBlock
                    for name, module in model.named_modules():
                        if isinstance(module, SwinTransformerBlock):
                            handle_list.append(
                                module.drop_path.register_backward_hook(drop_hook_func))
            model.train()
            if args.method == 'AT':
                X = X.cuda()
                y = y.cuda()
                if mixup_fn is not None:
                    X, y = mixup_fn(X, y)
                def pgd_attack():
                    model.eval()
                    epsilon = epsilon_base.cuda()
                    delta = torch.zeros_like(X).cuda()
                    if args.delta_init == 'random':
                        for i in range(len(epsilon)):
                            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
                        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
                    delta.requires_grad = True
                    for _ in range(args.attack_iters):
                        # patch drop
                        add_noise_mask = torch.ones_like(X)
                        grid_num_axis = int(args.resize / args.patch)
                        max_num_patch = grid_num_axis * grid_num_axis
                        ids = [i for i in range(max_num_patch)]
                        random.shuffle(ids)
                        num_patch = int(max_num_patch * (1 - drop_rate))
                        if num_patch !=0:
                            ids = np.array(ids[:num_patch])
                            rows, cols = ids // grid_num_axis, ids % grid_num_axis
                            for r, c in zip(rows, cols):
                                add_noise_mask[:, :, r * args.patch:(r + 1) * args.patch,
                                c * args.patch:(c + 1) * args.patch] = 0
                        if args.PRM:
                            delta = delta * add_noise_mask
                        output = model(X + delta)
                        loss = criterion(output, y)
                        grad = torch.autograd.grad(loss, delta)[0].detach()
                        delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
                        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
                    delta = delta.detach()
                    model.train()
                    if len(handle_list)!=0:
                        for handle in handle_list:
                            handle.remove()
                    return delta
                delta = pgd_attack()
                X_adv = X + delta
                output = model(X_adv)
                loss = criterion(output, y)
            elif args.method == 'TRADES':
                X = X.cuda()
                y = y.cuda()
                epsilon = epsilon_base.cuda()
                beta = args.beta
                batch_size = len(X)
                delta = torch.zeros_like(X).cuda()

                if args.delta_init == 'random':
                    delta = 0.001 * torch.randn(X.shape).cuda()
                    delta.data = clamp(delta, lower_limit - X, upper_limit - X)
                criterion_kl = nn.KLDivLoss(size_average=False)
                model.eval()

                delta.requires_grad = True
                for _ in range(args.attack_iters):
                    add_noise_mask = torch.ones_like(X)
                    grid_num_axis = int(args.resize / args.patch)
                    max_num_patch = grid_num_axis * grid_num_axis
                    ids = [i for i in range(max_num_patch)]
                    random.shuffle(ids)
                    num_patch = int(max_num_patch * (1 - drop_rate))
                    if num_patch != 0:
                        ids = np.array(ids[:num_patch])
                        rows, cols = ids // grid_num_axis, ids % grid_num_axis
                        for r, c in zip(rows, cols):
                            add_noise_mask[:, :, r * args.patch:(r + 1) * args.patch,
                            c * args.patch:(c + 1) * args.patch] = 0
                    if args.PRM:
                        delta = delta * add_noise_mask
                    loss_kl = criterion_kl(F.log_softmax(model(X+delta), dim=1),
                                           F.softmax(model(X), dim=1))
                    grad = torch.autograd.grad(loss_kl, [delta])[0]

                    delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
                    delta.data = clamp(delta, lower_limit - X, upper_limit - X)
                if len(handle_list) != 0:
                    for handle in handle_list:
                        handle.remove()
                model.train()
                x_adv = Variable(X+delta, requires_grad=False)
                output = logits = model(X)
                loss_natural = F.cross_entropy(logits, y)
                loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                                F.softmax(model(X), dim=1))
                loss = loss_natural + beta * loss_robust
            elif args.method == 'MART':
                X = X.cuda()
                y = y.cuda()
                beta = args.beta
                kl = nn.KLDivLoss(reduction='none')
                model.eval()
                batch_size = len(X)
                epsilon = epsilon_base.cuda()
                delta = torch.zeros_like(X).cuda()
                if args.delta_init == 'random':
                    delta = 0.001 * torch.randn(X.shape).cuda()
                    delta.data = clamp(delta, lower_limit - X, upper_limit - X)
                delta.requires_grad = True
                for _ in range(args.attack_iters):
                    add_noise_mask = torch.ones_like(X)
                    grid_num_axis = int(args.resize / args.patch)
                    max_num_patch = grid_num_axis * grid_num_axis
                    ids = [i for i in range(max_num_patch)]
                    random.shuffle(ids)
                    num_patch = int(max_num_patch * (1 - drop_rate))
                    if num_patch != 0:
                        ids = np.array(ids[:num_patch])
                        rows, cols = ids // grid_num_axis, ids % grid_num_axis
                        for r, c in zip(rows, cols):
                            add_noise_mask[:, :, r * args.patch:(r + 1) * args.patch,
                            c * args.patch:(c + 1) * args.patch] = 0
                    if args.PRM:
                        delta = delta * add_noise_mask
                    output = model(X + delta)
                    loss = F.cross_entropy(output, y)
                    grad = torch.autograd.grad(loss, delta)[0].detach()
                    delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
                    delta.data = clamp(delta, lower_limit - X, upper_limit - X)
                delta = delta.detach()
                if len(handle_list) != 0:
                    for handle in handle_list:
                        handle.remove()
                model.train()
                x_adv = Variable(X+delta,requires_grad=False)
                logits = model(X)
                logits_adv = model(x_adv)
                adv_probs = F.softmax(logits_adv, dim=1)
                tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
                new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
                loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)
                nat_probs = F.softmax(logits, dim=1)
                true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()
                loss_robust = (1.0 / batch_size) * torch.sum(
                    torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
                loss = loss_adv + float(beta) * loss_robust
            else:
                raise ValueError(args.method)
            if hasattr(model.module, 'extra_loss'):
                _ = model(X)
                nat_extras = model.module.extra_outputs
                _ = model(X + delta)
                adv_extras = model.module.extra_outputs
                extra_loss = model.module.extra_loss(adv_extras, nat_extras, y)
                loss = (1 - model.module.alpha) * loss + model.module.alpha * extra_loss
            opt.zero_grad()
            (loss / args.accum_steps).backward()
            if args.method == 'AT':
                acc = (output.max(1)[1] == y.max(1)[1]).float().mean()
            else:
                acc = (output.max(1)[1] == y).float().mean()
            return loss, acc,y

        for step, (X, y) in enumerate(train_loader):
            batch_size = args.batch_size // args.accum_steps
            epoch_now = epoch - 1 + (step + 1) / len(train_loader)
            for t in range(args.accum_steps):
                X_ = X[t * batch_size:(t + 1) * batch_size].cuda()  # .permute(0, 3, 1, 2)
                y_ = y[t * batch_size:(t + 1) * batch_size].cuda()  # .max(dim=-1).indices
                if len(X_) == 0:
                    break
                loss, acc,y = train_step(X,y,epoch_now,mixup_fn)
                train_loss += loss.item() * y_.size(0)
                train_acc += acc.item() * y_.size(0)
                train_n += y_.size(0)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            opt.zero_grad()
            if hasattr(model.module, 'update_anchors'):
                model.module.update_anchors()

            if (step + 1) % args.log_interval == 0 or step + 1 == steps_per_epoch:
                logger.info('Training epoch {} step {}/{}, lr {:.4f} loss {:.4f} acc {:.4f}'.format(
                    epoch, step + 1, len(train_loader),
                    opt.param_groups[0]['lr'],
                           train_loss / train_n, train_acc / train_n
                ))
            lr = lr_schedule(epoch_now)
            opt.param_groups[0].update(lr=lr)
        path = os.path.join(args.out_dir, 'checkpoint_{}'.format(epoch))
        if args.test:
            with open(os.path.join(args.out_dir, 'test_PGD20.txt'),'a') as new:
                args.eval_iters = 20
                args.eval_restarts = 1
                pgd_loss, pgd_acc = evaluate_pgd(args, model, test_loader)
                logger.info('test_PGD20 : loss {:.4f} acc {:.4f}'.format(pgd_loss, pgd_acc))
                new.write('{:.4f}   {:.4f}\n'.format(pgd_loss, pgd_acc))
            with open(os.path.join(args.out_dir, 'test_acc.txt'), 'a') as new:
                meter_test = evaluate_natural(args, model, test_loader, verbose=False)
                new.write('{}\n'.format(meter_test))
        if epoch == args.epochs:
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'opt': opt.state_dict()}, path)
            logger.info('Checkpoint saved to {}'.format(path))


if os.path.exists(os.path.join(args.out_dir, 'checkpoint_{}'.format(args.epochs))):
    checkpoint = torch.load(os.path.join(args.out_dir, 'checkpoint_{}'.format(args.epochs)))
    model.load_state_dict(checkpoint['state_dict'])
    logger.info('Checkpoint loaded from {}'.format(os.path.join(args.out_dir, 'checkpoint_{}'.format(args.epochs))))

if not os.path.exists(os.path.join(args.out_dir, 'checkpoint_{}'.format(args.epochs))):
    if hasattr(model.module, 'make_internal_anchor_loader'):
        model.module.make_internal_anchor_loader(train_dataset)
        model.module.update_anchors()
    train_adv(args, model, train_loader, test_loader, logger)

args.eval_iters = 20
args.eval_restarts = 1
attack_iters = args.eval_iters # 50
restarts = args.eval_restarts # 10

logger.info(args.out_dir)
print(args.out_dir)

if hasattr(model.module, 'make_internal_anchor_loader'):
    model.module.make_internal_anchor_loader(test_dataset)
    model.module.update_anchors()
nb_params = sum([param.view(-1).size(0) for param in model.parameters()]) / 1e6
logger.info('Number of Parameters: {:.1f}M'.format(nb_params))
evaluate_natural(args, model, test_loader, verbose=False)

if not exist_attack_result(prev_logfiles, 'cw20'):
    cw_loss, cw_acc = evaluate_CW(args, model, test_loader)
    logger.info('cw20 : loss {:.4f} acc {:.4f}'.format(cw_loss, cw_acc))

if not exist_attack_result(prev_logfiles, 'PGD20'):
    pgd_loss, pgd_acc = evaluate_pgd(args, model, test_loader)
    logger.info('PGD20 : loss {:.4f} acc {:.4f}'.format(pgd_loss, pgd_acc))

if not exist_attack_result(prev_logfiles, 'PGD100'):
    args.eval_iters = 100
    pgd_loss, pgd_acc = evaluate_pgd(args, model, test_loader)
    logger.info('PGD100 : loss {:.4f} acc {:.4f}'.format(pgd_loss, pgd_acc))
    args.eval_iters = 20

if not os.path.exists(os.path.join(args.out_dir, 'result_'+'_autoattack.txt')):
    at_path = os.path.join(args.out_dir, 'result_'+'_autoattack.txt')
    evaluate_aa(args, model,at_path, args.AA_batch)
