import argparse
import csv
import os
import re

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

import dataio
import model

# 
transforms = dataio.transforms
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

ABLATION_VARIANTS = [
    'no_cross_attn', 'no_dual', 'no_mask', 'no_aspp', 'no_pos', 'no_gate', 'full'
]
MODEL_NAMES = [
    'ASPP_dual', 'MultiResnet34', 'MultiASPP', 'ASPP_cATT', 'ASPP',
    'resnet50', 'resnet34', 'inception_v3'
]


def _ckpt_stem(ckpt_path):
    stem = os.path.splitext(os.path.basename(ckpt_path))[0]
    if stem.endswith('_best'):
        stem = stem[:-5]
    # 
    stem = re.sub(r'_ep\d+$', '', stem)
    # 
    return stem


def infer_ablation_from_ckpt(ckpt_path):
    stem = _ckpt_stem(ckpt_path)
    for variant in ABLATION_VARIANTS:
        if stem.endswith('_' + variant):
            return variant
    return None


def infer_model_from_ckpt(ckpt_path, flg='H'):
    stem = _ckpt_stem(ckpt_path)
    ablation = infer_ablation_from_ckpt(ckpt_path)
    if ablation is not None:
        stem = stem[: -(len(ablation) + 1)]
    if flg and stem.startswith(flg):
        stem = stem[len(flg):]
    for name in MODEL_NAMES:
        if stem == name:
            return name
    return None


def resolve_model_setting(cfgs, ckpt_arg=None, ablation_arg=None, model_arg=None):
    model_name = model_arg or cfgs['model_Classifier']
    ablation = ablation_arg or cfgs.get('ablation_variant', 'full')
    flg = cfgs['dataset']['testis']['flg']
    if ckpt_arg:
        ckpt_path = ckpt_arg
        inferred_model = infer_model_from_ckpt(ckpt_path, flg=flg)
        if model_arg is None and inferred_model is not None:
            model_name = inferred_model
        elif model_arg is None and inferred_model is None:
            print('Warning: cannot infer model from ckpt name, use cfgs: %s' % model_name)
        inferred_ablation = infer_ablation_from_ckpt(ckpt_path)
        if ablation_arg is None and inferred_ablation is not None:
            ablation = inferred_ablation
        elif ablation_arg is None and inferred_ablation is None and model_name == 'ASPP_dual':
            print('Warning: cannot infer ablation from ckpt name, use cfgs: %s' % ablation)
    else:
        prefix = './checkpoint/' + flg + model_name
        if model_name == 'ASPP_dual':
            prefix += '_' + ablation
        best_path = prefix + '_best.pth'
        last_path = prefix + '.pth'
        if os.path.isfile(best_path):
            ckpt_path = best_path
        elif os.path.isfile(last_path):
            ckpt_path = last_path
        else:
            raise FileNotFoundError('checkpoint not found: %s or %s' % (best_path, last_path))
    return model_name, ablation, ckpt_path


def load_checkpoint(net, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    if any(k.startswith('module.') for k in state.keys()):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    net.load_state_dict(state, strict=True)
    print('Loaded checkpoint:', ckpt_path)


def forward_net(net, images, mask, model_name, data=None, device=None):
    if model_name == 'ASPP_dual':
        if data is None or 'img_CLR' not in data or 'msk_CLR' not in data:
            raise KeyError(
                'ASPP_dual requires paired img_CLR and msk_CLR tensors; '
                'duplicating the G-modality input is not allowed'
            )
        input_a = [images, mask]
        input_b = [data['img_CLR'].to(device), data['msk_CLR'].to(device)]
        return net(input_a, input_b)
    out = net(images, mask)
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def main():
    parser = argparse.ArgumentParser(description='Evaluate checkpoint on outside/test set')
    parser.add_argument('--cfg', default='./cfgs.yaml', help='config yaml path')
    parser.add_argument('--ckpt', default=None, help='checkpoint path; default uses *_best.pth from cfgs')
    parser.add_argument(
        '--model', default=None,
        help='classifier name, e.g. resnet50/resnet34/ASPP_dual; if omitted, inferred from --ckpt'
    )
    parser.add_argument(
        '--ablation', default=None,
        help='ASPP_dual ablation variant; if omitted, inferred from --ckpt name, else from cfgs'
    )
    parser.add_argument(
        '--split', default='outside', choices=['outside', 'test'],
        help='outside=Outsiade_val.lst; test=TrainClassyfy_dir20.lst'
    )
    parser.add_argument('--lst', default=None, help='list file; only used when --split outside')
    parser.add_argument('--out', default=None, help='output csv path')
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    with open(args.cfg, 'r', encoding='utf-8') as f:
        cfgs = yaml.safe_load(f)

    model_name, ablation, ckpt_path = resolve_model_setting(
        cfgs, ckpt_arg=args.ckpt, ablation_arg=args.ablation, model_arg=args.model
    )
    print('Split: %s | Model: %s | ablation: %s | ckpt: %s' % (
        args.split, model_name, ablation, ckpt_path
    ))

    val_trans = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.Crop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if args.split == 'outside':
        list_file = args.lst if args.lst else './Outsiade_val.lst'
        dataset = dataio.testis(
            root=cfgs['dataset'],
            flag='outside',
            VH=cfgs['dataset']['testis']['flg'],
            transform=val_trans,
            list_file=list_file,
        )
        split_tag = 'outside_val'
    else:
        dataset = dataio.testis(
            root=cfgs['dataset'],
            flag='test',
            VH=cfgs['dataset']['testis']['flg'],
            transform=val_trans,
        )
        split_tag = 'test'
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    net = model.encode(
        name=model_name,
        num_classes=2,
        ablation_variant=ablation,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    load_checkpoint(net, ckpt_path, device)
    net.eval()

    rows = []
    labels_all = []
    probs_pos = []
    preds_all = []

    with torch.no_grad():
        for data in dataloader:
            images = data['images'].to(device)
            mask = data['masks'].to(device)
            labels = data['labels']
            prediction = forward_net(
                net, images, mask, model_name, data=data, device=device
            )
            prob = prediction.softmax(dim=1).cpu().numpy()
            pred = np.argmax(prob, axis=1)

            bs = labels.shape[0] if hasattr(labels, 'shape') else len(labels)
            for i in range(bs):
                label_i = int(labels[i].item() if torch.is_tensor(labels[i]) else labels[i])
                pred_i = int(pred[i])
                p0 = float(prob[i, 0])
                p1 = float(prob[i, 1])
                ck = data['ckName'][i]
                if 'im_path' in data:
                    im_path = data['im_path'][i]
                else:
                    im_path = dataset.im_list[len(rows)]
                rows.append({
                    'image_path': im_path,
                    'ckName': ck,
                    'label': label_i,
                    'pred': pred_i,
                    'prob_0': p0,
                    'prob_1': p1,
                })
                labels_all.append(label_i)
                probs_pos.append(p1)
                preds_all.append(pred_i)

    labels_all = np.asarray(labels_all)
    probs_pos = np.asarray(probs_pos)
    preds_all = np.asarray(preds_all)

    acc = accuracy_score(labels_all, preds_all)
    try:
        auc_score = roc_auc_score(labels_all, probs_pos)
    except ValueError:
        auc_score = float('nan')
    fpr, tpr, _ = roc_curve(labels_all, probs_pos)
    auc_trap = auc(fpr, tpr) if len(np.unique(labels_all)) > 1 else float('nan')
    cm = confusion_matrix(labels_all, preds_all, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    spec = tn / (tn + fp) if (tn + fp) > 0 else float('nan')

    if args.out is None:
        ckpt_stem = os.path.splitext(os.path.basename(ckpt_path))[0]
        out_dir = './results'
        os.makedirs(out_dir, exist_ok=True)
        args.out = os.path.join(out_dir, '%s_%s.csv' % (ckpt_stem, split_tag))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    with open(args.out, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(
            f, fieldnames=['image_path', 'ckName', 'label', 'pred', 'prob_0', 'prob_1']
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_path = os.path.splitext(args.out)[0] + '_summary.csv'
    with open(summary_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['ckpt', 'n', 'acc', 'auc', 'sensitivity', 'specificity', 'tp', 'tn', 'fp', 'fn'])
        writer.writerow([
            ckpt_path, len(rows), '%.6f' % acc, '%.6f' % auc_score,
            '%.6f' % sens, '%.6f' % spec, tp, tn, fp, fn
        ])

    print('Samples: %d' % len(rows))
    print('ACC: %.4f | AUC: %.4f (trapz=%.4f)' % (acc, auc_score, auc_trap))
    print('Sensitivity: %.4f | Specificity: %.4f' % (sens, spec))
    print('Confusion [tn fp; fn tp]:')
    print(cm)
    print('Per-sample CSV:', args.out)
    print('Summary CSV:', summary_path)
    print('ROC tip: use columns label + prob_1')


if __name__ == '__main__':
    main()
# 
