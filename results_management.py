import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
from reliability_diagram import reliability_diagrams


def find_all_models():
    """
    Find all the .log and .txt files in the current directory and its subdirectories
    Ignore the files in the .git and old directories
    If a directory starts with "trades_architecture_" or "trades_vanilla_", it is the root of a model, and the model name is the name of the directory minus the prefix and the suffix "_TRADES_warmup"
    If the prefix is "trades_architecture_", the model name should be added the suffix "_ARD_PRM"
    For each model name, use it as the key of a dictionary, and the value is a list of the paths of the .log and .txt files of the model
    :return:
    """
    models = {}
    for root, dirs, files in os.walk('When-Adversarial-Training-Meets-Vision-Transformers'):
        if '.git' in root or 'OLD' in root:
            continue
        if root.startswith('./trades_architecture_') or root.startswith('./trades_vanilla_'):
            model_name = root.split('/')[1].replace('_TRADES_warmup', '').replace('trades_vanilla_', '').replace('trades_architecture_', '')
            if root.startswith('./trades_architecture_'):
                model_name += '_ARD_PRM'

            if model_name not in models:
                models[model_name] = []
            for file in files:
                if file.endswith('.log') or file.endswith('.txt'):
                    models[model_name].append(os.path.join(root, file))
    return models


def parse_log_file(file_path, results):
    """
    Parse the .log file and fill a dictionary with the information
    """
    txt_to_key = {'Evaluation test_loss': 'Clean', 'cw20 :': 'C&W20', 'PGD20 :': 'PGD20', 'PGD100 :': 'PGD100',
                  'RayS1000 :': 'RayS1000', 'cw20_withKnowledge :': 'C&W20 w/ knowledge', 'PGD20_withKnowledge :': 'PGD20 w/ knowledge',
                  'PGD100_withKnowledge :': 'PGD100 w/ knowledge', 'RayS1000_withKnowledge :': 'RayS1000 w/ knowledge',
                  'Transfer PGD deit_tiny_patch16_224:': 'PGD20 transfer DeiT-tiny', 'Transfer PGD robustquote6_tiny_patch16_224:': 'PGD20 transfer robustquote6-tiny',
                  'PGD20 (eps=2/255) :': 'PGD20 (eps=2/255)', 'PGD20 (eps=4/255) :': 'PGD20 (eps=4/255)', 'PGD20 (eps=16/255) :': 'PGD20 (eps=16/255)',
                  'PGD20 (eps=32/255) :': 'PGD20 (eps=32/255)', 'PGD20 (eps=64/255) :': 'PGD20 (eps=64/255)', 'PGD20 (eps=128/255) :': 'PGD20 (eps=128/255)'}
    with open(file_path, 'r') as file:
        lines = file.readlines()
    for line in lines:
        for key in txt_to_key:
            if key in line:
                results[txt_to_key[key]] = float(line.split('=')[-1]) if key == 'Evaluation test_loss' else float(line.split(' ')[-1])
    return results


def parse_txt_file(file_path, results):
    """
    Parse the .txt file and fill a dictionary with the information
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    key = 'AutoAttack'
    if 'withKnowledge' in file_path:
        key += ' w/ knowledge'
    elif 'Square' in file_path:
        key = 'Square'
    elif 'APGD-t' in file_path:
        key = 'APGD-t'
        study_confidence_in_APGD_t(lines)

    for line in lines:
        if 'robust accuracy: ' in line:
            results[key] = float(line.split(' ')[-1].replace('%', ''))/100
    return results


def parse_files(models):
    """
    Parse the .log and .txt files of the models and fill a dictionary with the information
    """
    results = {}
    for model in models:
        results[model] = {}
        for file in models[model]:
            if file.endswith('.log'):
                results[model] = parse_log_file(file, results[model])
            elif file.endswith('.txt'):
                results[model] = parse_txt_file(file, results[model])
    return results


def beautify_results(results):
    """
    Beautify the results dictionary
    """
    attacks_order = ['Clean', 'PGD20', 'PGD100', 'C&W20', 'AutoAttack', 'APGD-t', 'Square', 'RayS1000', 'PGD20 transfer DeiT-tiny',
                     'PGD20 transfer robustquote6-tiny', 'PGD20 (eps=2/255)', 'PGD20 (eps=4/255)', 'PGD20 (eps=16/255)',
                     'PGD20 (eps=32/255)', 'PGD20 (eps=64/255)', 'PGD20 (eps=128/255)', 'PGD20 w/ knowledge',
                     'PGD100 w/ knowledge', 'C&W20 w/ knowledge', 'AutoAttack w/ knowledge', 'RayS1000 w/ knowledge']
    models_order = ['cifar_deit_tiny_patch16_224', 'deit_confLoss_tiny_patch16_224', 'cifar_deit_tiny_patch16_224_ARD_PRM', 'cifar_fsr_tiny_patch16_224',
                    'cifar_sacnet_tiny_patch16_224', 'cifar_dh_at_tiny_patch16_224', 'cifar_robustquote3_tiny_patch16_224',
                    'cifar_robustquote6_tiny_patch16_224', 'cifar_robustquote9_unFreeze_tiny_patch16_224',
                    'cifar_robustquote9_alpha0_tiny_patch16_224', 'cifar_robustquote9_alpha10_tiny_patch16_224', 'cifar_robustquote9_alpha75_tiny_patch16_224',
                    'cifar_robustquote9_alpha50_tiny_patch16_224', 'cifar_robustquote9_tiny_patch16_224', 'cifar_robustquote9_confLoss_tiny_patch16_224',
                    'cifar_robustquote9_alpha90_tiny_patch16_224', 'cifar_TESTrobustquote9_tiny_patch16_224', 'cifar_robustquote9_tau01_tiny_patch16_224', 'cifar_robustquote9_tau05_tiny_patch16_224', 'cifar_robustquote9_tau09_tiny_patch16_224', 'cifar_robustquote9_tau1_tiny_patch16_224', 'cifar_robustquote11_tiny_patch16_224',
                    'cifar_robustquote39_tiny_patch16_224', 'cifar_robustquote69_tiny_patch16_224', 'cifar_robustquote910_tiny_patch16_224',
                    'cifar_robustquote911_tiny_patch16_224', 'cifar_deit_base_patch16_224', 'cifar_deit_base_patch16_224_ARD_PRM',
                    'cifar_fsr_base_patch16_224', 'cifar_sacnet_base_patch16_224', 'cifar_dh_at_base_patch16_224', 'cifar_robustquote9_base_patch16_224',
                    'imagenette_deit_tiny_patch16_224_ARD_PRM', 'imagenette_fsr_tiny_patch16_224',
                    'imagenette_sacnet_tiny_patch16_224', 'imagenette_dh_at_tiny_patch16_224', 'imagenette_deit_tiny_patch16_224',
                    'imagenette_robustquote9_tiny_patch16_224', 'imagenette_deit_base_patch16_224_ARD_PRM', 'imagenette_fsr_base_patch16_224',
                    'imagenette_sacnet_base_patch16_224', 'imagenette_dh_at_base_patch16_224', 'imagenette_deit_base_patch16_224',
                    'imagenette_robustquote9_base_patch16_224']
    # all models are maybe not present in the results
    # all attacks are maybe not present for each model
    # models and attacks missing are not filled
    results_beautified = {}
    for model in models_order:
        if model in results:
            results_beautified[model] = {}
            for attack in attacks_order:
                if attack in results[model]:
                    results_beautified[model][attack] = results[model][attack]*100
    return results_beautified

def prepare_for_easy_PivotTables(results):
    """
    Prepare the results dictionary for easy use with PivotTables
    """
    results_for_pivot = []
    for model in results:
        for attack in results[model]:
            results_for_pivot.append({'Model': model, 'Attack': attack, 'Accuracy': results[model][attack]})
    return results_for_pivot


def study_confidence_in_APGD_t(lines):
    def str2num(string):
        # remove all characters that are not digits, dots or commas
        string = ''.join([i for i in string if i.isdigit() or i == '.' or i == ','])
        return [float(i) if '.' in i else int(i) for i in string.split(',') if i != '']

    fill_tensor_flag, label_flag, nat_flag, nat_conf_flag, adv_flag, adv_conf_flag = False, False, False, False, False, False
    labels, nat_labels, adv_labels, nat_conf, adv_conf = None, None, None, None, None
    for line in lines:
        if 'tensor([' in line:
            # find the three floats in the line
            tmp = torch.tensor(str2num(line))
            fill_tensor_flag = True
            if 'gt' in line: label_flag = True
            elif 'conf_nat' in line: nat_conf_flag = True
            elif 'conf_adv' in line: adv_conf_flag = True
            elif 'nat' in line: nat_flag = True
            elif 'adv' in line: adv_flag = True
        elif fill_tensor_flag:
            tmp = torch.cat((tmp, torch.tensor(str2num(line))), 0)
        if 'device=' in line and fill_tensor_flag:
            tmp = tmp[:-1] # remove the last element which is the device
            if label_flag: labels, label_flag = tmp if labels is None else torch.cat((labels, tmp), 0), False
            elif nat_flag: nat_labels, nat_flag = tmp if nat_labels is None else torch.cat((nat_labels, tmp), 0), False
            elif adv_flag: adv_labels, adv_flag = tmp if adv_labels is None else torch.cat((adv_labels, tmp), 0), False
            elif nat_conf_flag: nat_conf, nat_conf_flag = tmp if nat_conf is None else torch.cat((nat_conf, tmp), 0), False
            elif adv_conf_flag: adv_conf, adv_conf_flag = tmp if adv_conf is None else torch.cat((adv_conf, tmp), 0), False
            fill_tensor_flag = False
    data = {'void': {"true_labels": labels.numpy(), "pred_labels": nat_labels.numpy(), "confidences": nat_conf.numpy()},
            'void2': {"true_labels": labels.numpy(), "pred_labels": nat_labels.numpy(), "confidences": nat_conf.numpy()},
            'natural': {"true_labels": labels.numpy(), "pred_labels": nat_labels.numpy(), "confidences": nat_conf.numpy()},
            'adversarial': {"true_labels": labels.numpy(), "pred_labels": adv_labels.numpy(), "confidences": adv_conf.numpy()},
            }
    fig = reliability_diagrams(data, return_fig=True, num_cols=2, num_bins=20, draw_bin_importance=True)
    plt.savefig('reliability_diagram_APGD-t_robustquote9_tiny_patch16_224.png')


if __name__ == '__main__':
    models = find_all_models()
    results = parse_files(models)
    results = beautify_results(results)
    results_for_pivot = prepare_for_easy_PivotTables(results)
    df = pd.DataFrame(results_for_pivot)
    df.to_csv('results.csv')