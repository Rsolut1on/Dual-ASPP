import torch
import yaml
import dataio
import model
import time
import numpy as np
import tools
from datetime import datetime
import torch.nn as nn
import os
import csv
import pickle
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
    checkpoint_dir = tools.get_next_run_folder('./checkpoint')
    os.makedirs(checkpoint_dir, exist_ok=True)  # 创建文件夹
    print(f"Runing at: {checkpoint_dir}")

    trans = transforms.Compose([
        transforms.CenterCrop((530,530)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.321], std=[.192])
    ])

    dataset = dataio.testis(root=cfgs['dataset'], flag='train', transform=trans)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfgs['batch_size'], shuffle=True, num_workers=16)

    dataset_val = dataio.testis(root=cfgs['dataset'], flag='test', transform=trans)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=cfgs['batch_size'], shuffle=False, num_workers=16)

    # model
    net = model.encode(name=cfgs['model_name'], num_classes=2)
    if cfgs['fine_tunning']:
        net.load_state_dict(torch.load('./checkpoint/' + cfgs['dataset']['testis']['flg']+cfgs['model_name']+'.pth'))
    # loss
    criterion = torch.nn.CrossEntropyLoss()
    # optimal
    if cfgs['method'] == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), weight_decay=cfgs['weight_decay'])
    elif cfgs['method'] == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=cfgs['lr'], momentum=cfgs['momentum'],
                                    weight_decay=cfgs['weight_decay'])
    else:
        raise Exception('unknown optimizer name!')

    # # multi_GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using GPUs: {cfgs['iGPU']}")
        net = nn.DataParallel(net, device_ids=cfgs['iGPU'])
    running_loss = 0.0
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    for epoch in range(cfgs['max_iter']):  # loop over the dataset multiple times
        model.learning_rate_decay(optimizer, epoch, decay_rate=cfgs['decay_rate'], decay_steps=cfgs['decay_steps'])
        gt = []
        pred = []
        net.train()
        for i, data in enumerate(dataloader):
            start_time = time.time()
            gt.append(data['labels'])
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            images1 = data['images'].to(device)
            mask1 = data['masks'].to(device)
            labels = data['labels'].to(device)

            images2 = data['img_CLR'].to(device)
            mask2 = data['msk_CLR'].to(device)
            # images = torch.cat((images, images, mask), 1)
            input1 = [images1, mask1]
            input2 = [images2, mask2]
            prediction =  net(input1, input2)  #, _
            prob = prediction.softmax(dim=1).cpu().detach().numpy()
            pred.append(np.argmax(prob, 1))
            loss = criterion(prediction, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            duration = time.time() - start_time
            
            print_epoch = 5
            running_loss += loss.item()
            if i % print_epoch == 0:
                gt_arr, pred_arr = torch.cat(gt).numpy(), np.concatenate(pred)
                nTP = ((gt_arr - pred_arr) == 0).sum()#  284
                nFN = ((gt_arr - pred_arr) == 1).sum()
                nFP = ((gt_arr - pred_arr) == -1).sum()
                acc = nTP / gt_arr.size
                print(nTP, acc)
                if best_acc < acc:
                    torch.save(net.state_dict(), checkpoint_dir + cfgs['dataset']['testis']['pth']+cfgs['model_name']+'_best.pth')
                    best_acc = acc
                sec_per_batch = float(duration)
                format_str = '%s: step [%d, %5d], loss = %.3f (%.3f sec/batch)'
                print('Train:',format_str % (datetime.now(), epoch, i, running_loss / print_epoch, sec_per_batch))
                history['train_loss'].append(running_loss / print_epoch)
                history['train_acc'].append(acc)
                running_loss = 0.0
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
            loss = criterion(prediction.softmax(dim=1), labels_val)
            running_loss += loss.item()
        gt_arr, pred_arr = torch.cat(gt).numpy(), np.concatenate(pred)
        nTP = ((gt_arr - pred_arr) == 0).sum()#  284
        nFN = ((gt_arr - pred_arr) == 1).sum()
        nFP = ((gt_arr - pred_arr) == -1).sum()
        acc = nTP / gt_arr.size
        format_str = '%s: step [%d, %5d], loss = %.3f, acc = %.3f'
        print('Val:', format_str % (datetime.now(), epoch, i, running_loss / i, acc))
        history['val_loss'].append(running_loss / i)
        history['val_acc'].append(acc)
        running_loss = 0.0
        save_history(checkpoint_dir + cfgs['dataset']['testis']['pth']+cfgs['model_name']+'_log.csv', history)
    torch.save(net.state_dict(), checkpoint_dir + cfgs['dataset']['testis']['pth']+cfgs['model_name']+'.pth')
    print('Saved at: ', checkpoint_dir + cfgs['dataset']['testis']['pth']+cfgs['model_name']+'.pth')
    print('Best ACC:', best_acc)

    # plot_training_history(history)
