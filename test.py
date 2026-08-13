import torch
import yaml
import dataio
import model
import time
import numpy as np
from datetime import datetime
import torch.nn as nn
import os
import csv
import cv2
os.environ['CUDA_VISBLE_DEVICES'] = '3,4,5,6'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# from utils_metrics import plot_training_history
transforms = dataio.transforms
def save_history(path, history):
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 写入表头
        writer.writerow(history.keys())
        # 写入数据
        writer.writerows(zip(*history.values()))

if __name__ == '__main__':
    # load configures
    file_id = open('./cfgs.yaml')
    cfgs = yaml.safe_load(file_id)
    file_id.close()

    trans = transforms.Compose([
        transforms.CenterCrop((530,530)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.449], std=[.678])
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset_val = dataio.testis(root=cfgs['dataset'], flag='test', transform=trans)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    # model
    net = model.encode(name=cfgs['model_name'], num_classes=2)
    if cfgs['Val']:
        net.load_state_dict(torch.load('./checkpoint/' + cfgs['dataset']['testis']['flg']+cfgs['model_name']+'_best.pth'))

    # # multi_GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using GPUs: {cfgs['iGPU']}")
        net = nn.DataParallel(net, device_ids=cfgs['iGPU'])
    running_loss = 0.0
    best_acc = 0.0
    history = {'val_acc': []}
    gt = []
    pred = []
    net.eval()
    for i, data in enumerate(dataloader_val):
        gt.append(data['labels'])
        images1_val = data['images'].to(device)
        mask1_val = data['masks'].to(device)
        labels_val = data['labels'].to(device)
        images2_val = data['img_CLR'].to(device)
        mask2_val = data['msk_CLR'].to(device)
        input1_val = [images1_val, mask1_val]
        input2_val = [images2_val, mask2_val]

        # images = torch.cat((images, images,images), 1)
        with torch.no_grad():
            prediction =  net(input1_val, input2_val)  #, _
        prob = prediction.softmax(dim=1).cpu().detach().numpy()
        pred.append(np.argmax(prob, 1))
        if gt[i] != pred[i]:
            im = cv2.imread(dataset_val.im_list[i])
            mk = cv2.imread(dataset_val.mask_list[i])
            mk = cv2.resize(mk, (im.shape[1], im.shape[0]),  interpolation=cv2.INTER_CUBIC)
            fuse = np.hstack([im, mk])
            cv2.imwrite('./log/ASPP_dual/_%s_idx=%d_gt=%d_pred=%d.png' % (data['ckName'], i, gt[i], pred[i]), fuse)

    gt_arr, pred_arr = torch.cat(gt).numpy(), np.concatenate(pred)
    nTP = ((gt_arr - pred_arr) == 0).sum()#  284
    nFN = ((gt_arr - pred_arr) == 1).sum()
    nFP = ((gt_arr - pred_arr) == -1).sum()
    acc = nTP / gt_arr.size
    format_str = '%s: step [%d], acc = %.3f'
    print('Val:', format_str % (datetime.now(), i,  acc))
    history['val_acc'].append(acc)
    running_loss = 0.0
    save_history('./checkpoint/' + cfgs['dataset']['testis']['pth']+cfgs['model_name']+'_Test_log.csv', history)
    print('Best ACC:', best_acc)
