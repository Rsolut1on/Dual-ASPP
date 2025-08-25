import torch
import yaml
import os
os.environ['CUDA_VISBLE_DEVICES'] = '3,4,5,6'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import dataio
import model
import backbone
import cv2
import numpy as np
import time
from datetime import datetime
from matplotlib import pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import Subset

transforms = dataio.transforms

class Solver(object):
    def __init__(self, num_class=1, HW=[256, 256]):
        self.NumClass = num_class
        H, W = HW[0], HW[1]
        self.hori_translation = torch.zeros([1, self.NumClass, W, W])
        for i in range(W - 1):
            self.hori_translation[:, :, i, i + 1] = torch.tensor(1.0)
        self.verti_translation = torch.zeros([1, self.NumClass, H, H])
        for j in range(H - 1):
            self.verti_translation[:, :, j, j + 1] = torch.tensor(1.0)
        self.hori_translation = self.hori_translation.float()
        self.verti_translation = self.verti_translation.float()

    def get_hv(self):
        return self.hori_translation, self.verti_translation

    def create_exp_directory(self, exp_id):
        if not os.path.exists('models/' + str(exp_id)):
            os.makedirs('models/' + str(exp_id))

        csv = 'results_' + str(exp_id) + '.csv'
        with open(os.path.join(self.args.save, csv), 'w') as f:
            f.write('epoch, dice, Jac, clDice \n')

    def get_density(self, pos_cnt, bins=50):
        ### only used for Retouch in this code
        val_in_bin_ = [[], [], []]
        density_ = [[], [], []]
        bin_wide_ = []

        ### check
        for n in range(3):
            density = []
            val_in_bin = []
            c1 = [i for i in pos_cnt[n] if i != 0]
            c1_t = torch.tensor(c1)
            bin_wide = (c1_t.max() + 50) / bins
            bin_wide_.append(bin_wide)

            edges = torch.arange(bins + 1).float() * bin_wide
            for i in range(bins):
                val = [c1[j] for j in range(len(c1)) if ((c1[j] >= edges[i]) & (c1[j] < edges[i + 1]))]
                # print(val)
                val_in_bin.append(val)
                inds = (c1_t >= edges[i]) & (c1_t < edges[i + 1])  # & valid
                num_in_bin = inds.sum().item()
                # print(num_in_bin)
                density.append(num_in_bin)

            denominator = torch.tensor(density).sum()
            # print(val_in_bin)

            #### get density ####
            density = torch.tensor(density) / denominator
            density_[n] = density
            val_in_bin_[n] = val_in_bin
        print(density_)

        return density_, val_in_bin_, bin_wide_

def save_val_res(model_name, hori_translation, verti_translation, images, output, mask, names, k, prediction=None):
    if model_name == 'UNet' or model_name == 'SegNet' or model_name == 'UNet_2Plus':
        prediction = torch.sigmoid(prediction).cpu().detach().numpy()
    if model_name == 'DCnet':
        batch, channel, H, W = images.shape
        hori_show = hori_translation.repeat(batch, 1, 1, 1).cuda()
        verti_show = verti_translation.repeat(batch, 1, 1, 1).cuda()
        output_test = F.sigmoid(output)
        class_pred = output_test.view([batch, -1, 8, H, W])  # (B, C, 8, H, W)
        pred = torch.where(class_pred > 0.5, 1, 0)
        try:
            prediction, _ = backbone.Bilateral_voting(pred.float(), hori_show, verti_show)
            prediction = prediction.cpu().detach().numpy()
        except:
            print('Erro shape:', pred.shape, hori_show.shape)
            prediction = mask.cpu().detach().numpy()
    tmk = mask.cpu().detach().numpy()
    tim = images.cpu().detach().numpy()
    temp = 0
    if not os.path.exists(f'./val_check_img_K{k}/' ):
        os.makedirs(f'./val_check_img_K{k}/')
    for pre, mk, im in zip(prediction, tmk, tim):
        fuse = np.hstack([im[0], mk[0], pre[0]])
        plt.imshow(fuse, cmap='gray')
        plt.savefig(f'./val_check_img_K{k}/'+ names[temp])
        temp += 1

def train_seg_KF(train_loader, val_loader, k):
    # model
    model_name = cfgs['model_name']  # UNet UNet_2Plus SegNet
    net = model.encode(name=model_name, num_classes=NumClass).train()

    # DC loss init 
    hv = Solver(NumClass, (224, 224))
    hori_translation, verti_translation = hv.get_hv()

    # criterion = backbone.Cross_Entropy()
    criterion = backbone.connect_loss(hori_translation, verti_translation, num_class=1)
    # optimal
    # optimizer = torch.optim.SGD(net.parameters(), lr=cfgs['lr'], momentum=cfgs['momentum'],
    #                             weight_decay=cfgs['weight_decay'])
    optimizer = torch.optim.Adam(net.parameters(), lr=cfgs['lr'], weight_decay=cfgs['weight_decay'])

    # # multi_GPU
    device = torch.device("cuda:" + str(cfgs['iGPU']) if torch.cuda.is_available() else "cpu")
    net = nn.DataParallel(net)
    net.to(device)

    running_loss = 0.0
    for epoch in range(cfgs['max_iter']):  # loop over the dataset multiple times
        model.learning_rate_decay(optimizer, epoch, decay_rate=cfgs['decay_rate'], decay_steps=cfgs['decay_steps'])
        for i, data in enumerate(train_loader):
            start_time = time.time()

            # forward + backward + optimize
            images = data['images'].to(device)
            mask = data['masks'].to(device)
            labels = data['labels'].to(device)

            # one channel 2 three channel
            images = torch.cat((images, images, images), 1)

            optimizer.zero_grad()
            # CD net
            if model_name == 'DCnet':
                output, aux_out = net(images)
                loss_main = criterion(output, mask)
                loss_aux = criterion(aux_out, mask)
                loss = loss_main + 0.3 * loss_aux
            else:
                prediction = net(images)
                loss = criterion(prediction, mask)
            loss.backward()
            optimizer.step()

            # print statistics
            duration = time.time() - start_time
            print_epoch = 10
            running_loss += loss.item()
            if i % print_epoch == 0:
                sec_per_batch = float(duration)
                format_str = '%s: step [%d, %5d], loss = %.3f (%.3f sec/batch)'
                print(format_str % (datetime.now(), epoch, i, running_loss / print_epoch, sec_per_batch))
                running_loss = 0.0

        net.eval()
        with torch.no_grad():
            val_loss = 0.0
            for i, data in enumerate(val_loader):
                images = data['images'].to(device)
                mask = data['masks'].to(device)

                images = torch.cat((images, images, images), 1)
                if model_name == 'DCnet':
                    output, aux_out = net(images)
                    loss_main = criterion(output, mask)
                    loss_aux = criterion(aux_out, mask)
                    loss = loss_main + 0.3 * loss_aux
                else:
                    prediction = net(images)
                    loss = criterion(prediction, mask)
                val_loss += loss.item()
                if epoch == cfgs['max_iter'] - 1:
                    save_val_res(model_name, hori_translation, verti_translation, images, output, mask, data['ckName'], k)
            validation_loss = val_loss / len(val_loader)
            print(f"Validation Loss: {validation_loss}")
        
        if epoch % 16 == 0 and epoch != 0:
            if not os.path.exists(f'./checkpoint_K{k}/' ):
                os.makedirs(f'./checkpoint_K{k}/')
            torch.save(net.state_dict(),f'./checkpoint_K{k}/' 
                       + cfgs['model_name'] + f'ep{epoch}_{validation_loss:.3f}.pth')
            torch.save(net.state_dict(), f'./checkpoint_K{k}/' + cfgs['model_name'] + f'_{validation_loss:.3f}.pth')

if __name__ == '__main__':
    NumClass = 1
    # load configures
    file_id = open('./cfgs.yaml')
    cfgs = yaml.safe_load(file_id)
    file_id.close()

    trans = transforms.Compose([
        transforms.Crop(size=224, seg_mode=True),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    dataset = dataio.testis(root=cfgs['dataset'], flag='seg_train', transform=trans)
    kf = KFold(n_splits=5, shuffle=True, random_state=42) # 5-fold
    indices = list(range(len(dataset)))
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"Fold {fold + 1}:")
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=cfgs['batch_size'], shuffle=True, num_workers=4)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=cfgs['batch_size'], shuffle=True, num_workers=4)

        train_seg_KF(train_loader, val_loader, fold + 1)
                