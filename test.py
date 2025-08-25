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
import pandas as pd

transforms = dataio.transforms
def save_history(path, history):
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(history.keys())
        writer.writerows(zip(*history.values()))

if __name__ == '__main__':
    # load configures
    file_id = open('./cfgs.yaml')
    cfgs = yaml.safe_load(file_id)
    file_id.close()

    trans = transforms.Compose([
        transforms.Crop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.449], std=[.678])
                ])

    dataset_val = dataio.testis(root=cfgs['dataset'], flag='test', transform=trans)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=1)

    # model checkpoint\testicular_dataASPP_best_C.pth
    net = model.encode(name=cfgs['model_name'], num_classes=2).train()
    net.load_state_dict(torch.load('./checkpoint/' + 'run1/testicular_dataASPP_dual_best.pth'))
    # loss
    criterion = torch.nn.CrossEntropyLoss()

    # multi_GPU
    device = torch.device("cuda"if torch.cuda.is_available() else "cpu")
    net.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using GPUs: {cfgs['iGPU']}")
        net = nn.DataParallel(net, device_ids=cfgs['iGPU'])
    running_loss = 0.0
    best_acc = 0.0
    all_results = []

    for i, data in enumerate(dataloader_val):
        images1 = data['images'].to(device)
        mask1 = data['masks'].to(device)
        labels = data['labels'].to(device)
        images2 = data['img_CLR'].to(device)
        mask2 = data['msk_CLR'].to(device)
        input1 = [images1, mask1]
        input2 = [images2, mask2]
        with torch.no_grad():
            prediction,_ =  net(input1, input2)
        prob = prediction.softmax(dim=1).cpu().detach().numpy()
        loss = criterion(prediction, labels)
        running_loss += loss.item()
        epoch_results = pd.DataFrame({
            'name': data['ckName'],
            'pred': np.argmax(prob, 1),
            'gt': data['labels'],
        })
        all_results.append(epoch_results)
    print('Best ACC:', best_acc)
    final_results = pd.concat(all_results, ignore_index=True)
    final_results.to_csv('all_epochs_resultsASPP.csv', index=False)
