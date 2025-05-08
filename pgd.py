from matplotlib import pyplot as plt
from utils import *
import torch.nn.functional as F
import numpy as np
from reliability_diagram import reliability_diagrams
from pytorch_grad_cam import AblationCAM, GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, lower_limit, upper_limit, opt=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta

def attack_cw(model, X, y, epsilon, alpha, attack_iters, restarts, lower_limit, upper_limit, opt=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = CW_loss(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = CW_loss(model(X+delta), y, reduction=False).detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta

def evaluate_pgd(args, model, test_loader, eval_steps=None, save_extras=False, anchors_knolwedge=False, conf_model_name=None):
    if save_extras:
        i = model.module.robust_blocks_ids[-1] if hasattr(model.module, 'robust_blocks_ids') else 9
        layers = [model.module.backbone.blocks[i][1].norm3] if hasattr(model.module, 'backbone') else [model.module.blocks[i].norm2]
        def reshape_transform(tensor):
            if tensor.size(1) % 2 != 0:
                side = int(np.sqrt(tensor.size(1) - 1))
                result = tensor[:, 1:, :].reshape(tensor.size(0), side, side, tensor.size(2))
            else:
                side = int(np.sqrt(tensor.size(1)))
                result = tensor.reshape(tensor.size(0), side, side, tensor.size(2))
            result = result.transpose(2, 3).transpose(1, 2)
            return result
        cam_input = GradCAM(model=model, target_layers=layers, reshape_transform=reshape_transform)
        if hasattr(model.module, 'update_anchors'):
            layers = [model.module.backbone.blocks[i][1].rectifier.qk]
            tmp1 = GradCAM(model=model, target_layers=layers, reshape_transform=reshape_transform)
            layers = [model.module.backbone.blocks[i][1].rectifier.qk2]
            tmp2 = GradCAM(model=model, target_layers=layers, reshape_transform=reshape_transform)
            cam_anchors = [tmp1, tmp2]
        else:
            cam_anchors = None
    attack_iters = args.eval_iters # 50
    restarts = args.eval_restarts # 10
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    print('Evaluating with PGD {} steps and {} restarts'.format(attack_iters, restarts))
    if args.dataset=="cifar":
        mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
        std = torch.tensor(cifar10_std).view(3,1,1).cuda()
    if args.dataset=="imagenette" or args.dataset=="imagenet" :
        mu = torch.tensor(imagenet_mean).view(3,1,1).cuda()
        std = torch.tensor(imagenet_std).view(3,1,1).cuda()
    upper_limit = ((1 - mu)/ std)
    lower_limit = ((0 - mu)/ std)
    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std
    conf_dict = {}
    for step, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        if hasattr(model.module, 'update_anchors') and anchors_knolwedge:
            model.module.update_anchors()
        if save_extras and step < 25:
            if hasattr(model.module, 'process_and_save_extras'):
                output = model(X)
                model.module.process_and_save_extras(X, y, path=os.path.join(save_extras, 'nat'))
            nat_cam = np.transpose(cam_input(input_tensor=X[0].unsqueeze(0)), (1, 2, 0))
            if cam_anchors is not None:
                nat_pred = output[0].max(0)[1]
                anchors_nat_cam = np.transpose(cam_anchors[0](input_tensor=X[0].unsqueeze(0)), (1, 2, 0))
                nat_anchors = model.module.anchors
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, lower_limit, upper_limit)
        if hasattr(model.module, 'update_anchors') and not anchors_knolwedge:
            model.module.update_anchors()
        # with torch.no_grad():
        output = model(X + pgd_delta)
        if save_extras and step < 25:
            if hasattr(model.module, 'process_and_save_extras'):
                model.module.process_and_save_extras(X + pgd_delta, y, path=os.path.join(save_extras, 'adv'))
            adv_cam = np.transpose(cam_input(input_tensor=(X + pgd_delta)[0].unsqueeze(0)), (1, 2, 0))
            if cam_anchors is not None:
                anchors_adv_cam = np.transpose(cam_anchors[1](input_tensor=(X + pgd_delta)[0].unsqueeze(0)), (1, 2, 0))
            std_cifar, mean_cifar = torch.tensor([0.2471, 0.2435, 0.2616]), torch.tensor([0.4914, 0.4822, 0.4465])
            nat = (X[0].cpu().permute(1, 2, 0).numpy() * std_cifar.numpy() + mean_cifar.numpy()).clip(0, 1)
            adv = ((X + pgd_delta)[0].cpu().permute(1, 2, 0).numpy() * std_cifar.numpy() + mean_cifar.numpy()).clip(0, 1)
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(nat)
            ax[0].imshow(nat_cam, cmap='jet', alpha=0.3)
            ax[0].set_title('Natural')
            ax[0].axis('off'), ax[0].set_xticks([]), ax[0].set_yticks([])
            ax[1].imshow(adv)
            ax[1].imshow(adv_cam, cmap='jet', alpha=0.3)
            ax[1].set_title('Adversarial')
            ax[1].axis('off'), ax[1].set_xticks([]), ax[1].set_yticks([])
            fig.suptitle(f"GradCAM")
            plt.savefig(os.path.join(save_extras, f'{step}_GradCAM.png'))
            plt.close()
            if cam_anchors is not None:
                fig, ax = plt.subplots(2, len(output[0])+2, figsize=(len(output[0])+2*3, 6))
                ax[0, 0].imshow(nat)
                ax[0, 0].imshow(nat_cam, cmap='jet', alpha=0.3)
                ax[0, 0].set_title('Natural')
                ax[0, 0].axis('off'), ax[0, 0].set_xticks([]), ax[0, 0].set_yticks([])
                ax[1, 0].imshow(adv)
                ax[1, 0].imshow(adv_cam, cmap='jet', alpha=0.3)
                ax[1, 0].set_title('Adversarial')
                ax[1, 0].axis('off'), ax[1, 0].set_xticks([]), ax[1, 0].set_yticks([])
                ax[0, 1].axis('off'), ax[0, 1].set_xticks([]), ax[0, 1].set_yticks([])
                ax[1, 1].axis('off'), ax[1, 1].set_xticks([]), ax[1, 1].set_yticks([])
                label, adv_pred = y[0], output[0].max(0)[1]
                nat = (nat_anchors.cpu().permute(0, 2, 3, 1).numpy() * std_cifar.numpy() + mean_cifar.numpy()).clip(0, 1)
                adv = (model.module.anchors.cpu().permute(0, 2, 3, 1).numpy() * std_cifar.numpy() + mean_cifar.numpy()).clip(0, 1)
                for i in range(len(nat)):
                    gt_txt = '(True label)' if i == label else ''
                    nat_pred_txt, adv_pred_txt = '-Predicted-' if i == nat_pred else '', '-Predicted-' if i == adv_pred else ''
                    ax[0, i+2].imshow(nat[i])
                    ax[0, i+2].imshow(anchors_nat_cam[..., i], cmap='jet', alpha=0.3)
                    ax[0, i+2].set_title(f'{nat_pred_txt}{gt_txt}')
                    ax[0, i+2].axis('off'), ax[0, i+2].set_xticks([]), ax[0, i+2].set_yticks([])
                    ax[1, i+2].imshow(adv[i])
                    ax[1, i+2].imshow(anchors_adv_cam[..., i], cmap='jet', alpha=0.3)
                    ax[1, i+2].set_title(f'{adv_pred_txt}{gt_txt}')
                    ax[1, i+2].axis('off'), ax[1, i+2].set_xticks([]), ax[1, i+2].set_yticks([])
                fig.suptitle(f"GradCAM")
                plt.savefig(os.path.join(save_extras, f'{step}_anchorsGradCAM.png'))
                plt.close()
                os.remove(os.path.join(save_extras, f'{step}_GradCAM.png'))
            if step == 24:
                return
        loss = F.cross_entropy(output, y)
        pgd_loss += loss.item() * y.size(0)
        pgd_acc += (output.max(1)[1] == y).sum().item()
        n += y.size(0)
        if step + 1 == eval_steps:
            break
        if (step + 1) % 10 == 0 or step + 1 == len(test_loader):
            print('{}/{}'.format(step+1, len(test_loader)), 
                pgd_loss/n, pgd_acc/n)
        if conf_model_name is not None:
            torch.set_printoptions(profile="full")
            adv_logit = model(X + pgd_delta)
            adv_conf, adv_pred = adv_logit.max(dim=1)
            nat_logit = model(X)
            nat_conf, nat_pred = nat_logit.max(dim=1)
            adv_conf, adv_pred = adv_conf.detach().cpu().numpy(), adv_pred.detach().cpu().numpy()
            nat_conf, nat_pred = nat_conf.detach().cpu().numpy(), nat_pred.detach().cpu().numpy()
            if conf_dict == {}:
                conf_dict = {'natural': {"true_labels": y.detach().cpu().numpy(), "pred_labels": nat_pred, "confidences": nat_conf},
                             'adversarial': {"true_labels": y.detach().cpu().numpy(), "pred_labels": adv_pred, "confidences": adv_conf}}
            else:
                conf_dict['natural']["true_labels"] = np.concatenate((conf_dict['natural']["true_labels"], y.detach().cpu().numpy()))
                conf_dict['natural']["pred_labels"] = np.concatenate((conf_dict['natural']["pred_labels"], nat_pred))
                conf_dict['natural']["confidences"] = np.concatenate((conf_dict['natural']["confidences"], nat_conf))
                conf_dict['adversarial']["true_labels"] = np.concatenate((conf_dict['adversarial']["true_labels"], y.detach().cpu().numpy()))
                conf_dict['adversarial']["pred_labels"] = np.concatenate((conf_dict['adversarial']["pred_labels"], adv_pred))
                conf_dict['adversarial']["confidences"] = np.concatenate((conf_dict['adversarial']["confidences"], adv_conf))
            torch.set_printoptions(profile="default")
    if conf_model_name is not None:
        conf_dict['void'] = conf_dict['natural']
        conf_dict['void2'] = conf_dict['natural']
        fig = reliability_diagrams(conf_dict, return_fig=True, num_cols=2, num_bins=20, draw_bin_importance=True)
        plt.savefig(f'reliability_diagram_PGD10_{conf_model_name}.png')
    return pgd_loss/n, pgd_acc/n

def evaluate_CW(args, model, test_loader, eval_steps=None, anchors_knolwedge=False):
    attack_iters = args.eval_iters # 50
    restarts = args.eval_restarts # 10
    cw_loss = 0
    cw_acc = 0
    n = 0
    model.eval()
    print('Evaluating with CW {} steps and {} restarts'.format(attack_iters, restarts))
    if args.dataset=="cifar":
        mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
        std = torch.tensor(cifar10_std).view(3,1,1).cuda()
    if args.dataset=="imagenette" or args.dataset=="imagenet":
        mu = torch.tensor(imagenet_mean).view(3,1,1).cuda()
        std = torch.tensor(imagenet_std).view(3,1,1).cuda()
    upper_limit = ((1 - mu)/ std)
    lower_limit = ((0 - mu)/ std)
    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std
    for step, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        if hasattr(model.module, 'update_anchors') and anchors_knolwedge:
            model.module.update_anchors()
        pgd_delta = attack_cw(model, X, y, epsilon, alpha, attack_iters, restarts, lower_limit, upper_limit)
        if hasattr(model.module, 'update_anchors') and not anchors_knolwedge:
            model.module.update_anchors()
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = CW_loss(output, y)
            cw_loss += loss.item() * y.size(0)
            cw_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
        if step + 1 == eval_steps:
            break
        if (step + 1) % 10 == 0 or step + 1 == len(test_loader):
            print('{}/{}'.format(step+1, len(test_loader)),
                cw_loss/n, cw_acc/n)
    return cw_loss/n, cw_acc/n


def CW_loss(x, y, reduction=True, num_cls=10, threshold=10,):
    batch_size = x.shape[0]
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    logit_mc = x_sorted[:, -2] * ind + x_sorted[:, -1] * (1. - ind)
    logit_gt = x[np.arange(batch_size), y]
    loss_value_ori = -(logit_gt - logit_mc)
    loss_value = torch.maximum(loss_value_ori, torch.tensor(-threshold).cuda())
    if reduction:
        return loss_value.mean()
    else:
        return loss_value


def get_xadv(x, v, d, lb=0., ub=1.):
    if isinstance(d, int):
        d = torch.tensor(d).repeat(len(x)).cuda()
    out = x + d.view(len(x), 1, 1, 1) * v
    out = torch.clamp(out, lb, ub)
    return out


def search_succ(model, x, y, queries, mask):
    queries[mask] += 1
    logits = model(x[mask])
    return logits.argmax(1) != y[mask]


def binary_search(model, x, y, d_t, x_final, sgn_t, queries, sgn, valid_mask, tol=1e-3, lb=0., ub=1.):
    sgn_norm = torch.norm(sgn.view(len(x), -1), 2, 1)
    sgn_unit = sgn / sgn_norm.view(len(x), 1, 1, 1)

    d_start = torch.zeros_like(y).float().cuda()
    d_end = d_t.clone()

    initial_succ_mask = search_succ(model, get_xadv(x, sgn_unit, d_t, lb, ub), y, queries, valid_mask)
    to_search_ind = valid_mask.nonzero().flatten()[initial_succ_mask]
    d_end[to_search_ind] = torch.min(d_t, sgn_norm)[to_search_ind]

    while len(to_search_ind) > 0:
        d_mid = (d_start + d_end) / 2.0
        search_succ_mask = search_succ(model, get_xadv(x, sgn_unit, d_mid, lb, ub), y, queries, to_search_ind)
        d_end[to_search_ind[search_succ_mask]] = d_mid[to_search_ind[search_succ_mask]]
        d_start[to_search_ind[~search_succ_mask]] = d_mid[to_search_ind[~search_succ_mask]]
        to_search_ind = to_search_ind[((d_end - d_start)[to_search_ind] > tol)]

    to_update_ind = (d_end < d_t).nonzero().flatten()
    if len(to_update_ind) > 0:
        d_t[to_update_ind] = d_end[to_update_ind]
        x_final[to_update_ind] = get_xadv(x, sgn_unit, d_end, lb, ub)[to_update_ind]
        sgn_t[to_update_ind] = sgn[to_update_ind]


def attack_rays(model, X, y, epsilon, query_limit, lower_limit, upper_limit):
    shape = list(X.shape)
    dim = np.prod(shape[1:])

    queries = torch.zeros_like(y).cuda()
    sgn_t = torch.sign(torch.ones(shape)).cuda()
    d_t = torch.ones_like(y).float().fill_(float("Inf")).cuda()
    working_ind = (d_t > epsilon).nonzero().flatten()

    stop_queries = queries.clone()
    dist = d_t.clone()
    x_final = get_xadv(X, sgn_t, d_t, lb=lower_limit, ub=upper_limit)

    block_level = 0
    block_ind = 0
    for i in range(query_limit):
        block_num = 2 ** block_level
        block_size = int(np.ceil(dim / block_num))
        start, end = block_ind * block_size, min(dim, (block_ind + 1) * block_size)

        valid_mask = (queries < query_limit)
        attempt = sgn_t.clone().view(shape[0], dim)
        attempt[valid_mask.nonzero().flatten(), start:end] *= -1.
        attempt = attempt.view(shape)

        binary_search(model, X, y, d_t, x_final, sgn_t, queries, attempt, valid_mask, lb=lower_limit, ub=upper_limit)

        block_ind += 1
        if block_ind == 2 ** block_level or end == dim:
            block_level += 1
            block_ind = 0

        dist = torch.norm((x_final - X).view(shape[0], -1), np.inf, 1)
        stop_queries[working_ind] = queries[working_ind]
        working_ind = (dist > epsilon).nonzero().flatten()
        if torch.sum(queries >= query_limit) == shape[0]:
            break
    return x_final

def evaluate_RayS(args, model, test_loader, eval_steps=None, anchors_knowledge=False):
    attack_iters = args.eval_iters # 50
    restarts = args.eval_restarts # 10
    query_limit = args.eval_queries # 5000
    rays_loss = 0
    rays_acc = 0
    n = 0
    model.eval()
    print('Evaluating with RayS {} steps and {} restarts'.format(attack_iters, restarts))
    if args.dataset=="cifar":
        mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
        std = torch.tensor(cifar10_std).view(3,1,1).cuda()
    if args.dataset=="imagenette" or args.dataset=="imagenet":
        mu = torch.tensor(imagenet_mean).view(3,1,1).cuda()
        std = torch.tensor(imagenet_std).view(3,1,1).cuda()
    upper_limit = ((1 - mu)/ std)
    lower_limit = ((0 - mu)/ std)
    epsilon = (args.epsilon / 255.) / std
    for step, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        if hasattr(model.module, 'update_anchors') and anchors_knowledge:
            model.module.update_anchors()
        x_adv = attack_rays(model, X, y, epsilon, query_limit, lower_limit, upper_limit)
        if hasattr(model.module, 'update_anchors') and not anchors_knowledge:
            model.module.update_anchors()
        with torch.no_grad():
            output = model(x_adv)
            loss = F.cross_entropy(output, y)
            rays_loss += loss.item() * y.size(0)
            rays_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
        if step + 1 == eval_steps:
            break
        if (step + 1) % 10 == 0 or step + 1 == len(test_loader):
            print('{}/{}'.format(step+1, len(test_loader)),
                rays_loss/n, rays_acc/n)
    return rays_loss/n, rays_acc/n


def evaluate_transfer(args, model, test_loader, transfer_model, eval_steps=None, anchors_knolwedge=False):
    attack_iters = args.eval_iters  # 50
    restarts = args.eval_restarts  # 10
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    print('Evaluating with PGD {} steps and {} restarts'.format(attack_iters, restarts))
    if args.dataset == "cifar":
        mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
        std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()
    if args.dataset == "imagenette" or args.dataset == "imagenet":
        mu = torch.tensor(imagenet_mean).view(3, 1, 1).cuda()
        std = torch.tensor(imagenet_std).view(3, 1, 1).cuda()
    upper_limit = ((1 - mu) / std)
    lower_limit = ((0 - mu) / std)
    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std
    for step, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        if anchors_knolwedge:
            if hasattr(model.module, 'update_anchors'): model.module.update_anchors()
            if hasattr(transfer_model.module, 'update_anchors'): transfer_model.module.update_anchors()
        pgd_delta = attack_pgd(transfer_model, X, y, epsilon, alpha, attack_iters, restarts, lower_limit, upper_limit)
        if not anchors_knolwedge:
            if hasattr(model.module, 'update_anchors'): model.module.update_anchors()
            if hasattr(transfer_model.module, 'update_anchors'): transfer_model.module.update_anchors()
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
        if step + 1 == eval_steps:
            break
        if (step + 1) % 10 == 0 or step + 1 == len(test_loader):
            print('{}/{}'.format(step + 1, len(test_loader)),
                  pgd_loss / n, pgd_acc / n)
    return pgd_loss / n, pgd_acc / n