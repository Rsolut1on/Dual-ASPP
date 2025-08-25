import glob
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import functional as F


def readTuple(path, filename):
    pfile = open(os.path.join(path, filename))
    print('Load data from ', os.path.join(path, filename))
    filenames = pfile.readlines()
    pfile.close()

    filenames = [f.strip() for f in filenames]
    filenames = [c.split(' ') for c in filenames]
    image_path = [os.path.join(path, c[0]) for c in filenames]
    mask_path = [os.path.join(path, c[1]) for c in filenames]
    labels_path = [int(c[2]) for c in filenames]
    return image_path, mask_path, labels_path


class testis(Dataset):
    def __init__(self, root, VH='V', flag='train', transform=None, montage=False):
        assert VH == 'V' or VH == 'H'
        if flag == 'train':
            self.im_list, self.mask_list, self.gt_list = readTuple(root['testis']['pth'], 'TrainClassyfy_dir80.lst')
        elif flag == 'test':
            self.im_list, self.mask_list, self.gt_list = readTuple(root['testis']['pth'], 'TrainClassyfy_dir20.lst')
        elif flag == 'Val':
            self.im_list, self.mask_list, self.gt_list = readTuple(root['testis']['pth'], 'Outsiade_val.lst')
        elif flag == 'seg_train':
            VH = 'Seg'
            self.im_list, self.mask_list, self.gt_list = readTuple(root['testis']['pth'], 'TrainTuple_dir80.lst')
        elif flag == 'seg_test':
            VH = 'Seg'
            self.im_list, self.mask_list, self.gt_list = readTuple(root['testis']['pth'], 'TestTuple_dir20.lst')
        else:
            raise Exception('flag must be train or test')
        if flag == 'seg_test':
            self.ckname = [i.split('/')[-1][:-4] for i in self.im_list]
        else:
            self.ckname = [i.split('/')[-1][:-4] for i in self.im_list]
        # self.ckname = [i.split('/')[-2] for i in self.im_list]
        self.imname = [i.split('/')[-1] for i in self.im_list]
        self.length = self.im_list.__len__()
        self.transform = transform
        self.flag = flag
        self.montage = montage

    def __len__(self):
        return self.length
    
    def __getitem__(self, item):
        image = Image.open(self.im_list[item])
        mask = Image.open(self.mask_list[item])

        item_CLR = np.random.randint(len(self.im_list)) # 对比学习 对比数据录入   /home/hedehuai/code/testicular/visualization_img
        img_CLR = Image.open(self.im_list[item].replace("G.", "C."))
        msk_CLR = Image.open(self.mask_list[item].replace("G_", "C_"))
        image = image.convert('L')
        mask = mask.convert('L')
        img_CLR = img_CLR.convert('L')
        msk_CLR = msk_CLR.convert('L')
        if not image.size == mask.size:
            mask = F.resize(mask, [image.size[1], image.size[0]], Image.NEAREST)
        if not img_CLR.size == msk_CLR.size:
            msk_CLR = F.resize(msk_CLR, [img_CLR.size[1], img_CLR.size[0]], Image.NEAREST)

        label = self.gt_list[item]
        label_CLR = self.gt_list[item_CLR]
        label_CLR = 1 if label==label_CLR else 0
        sample = {'images': image, 'img_CLR': img_CLR, 'msk_CLR': msk_CLR, 'masks': mask, 'labels': label, 'labels_CLR': label_CLR, 'ckName': self.ckname[item]}
        if self.transform:
            sample = self.transform(sample)
        if self.montage:
            sample = self.montage_f(sample)
        return sample
    
    def montage_f(self, sample):
        img_CLR = sample['img_CLR']
        mask = sample['masks'] > 0
        if len(mask.shape) == 2:  # 如果掩码是 [H, W]
            mask = mask.unsqueeze(0)  # 增加一个维度，变为 [1, H, W]
            mask = mask.expand(img_CLR.shape)  # 扩展为 [C, H, W]
        img_CLR[mask] = sample['images'][mask]
        sample['img_CLR'] = img_CLR
        return sample

def GroupTestis(root, flag='S', trans=None):
    Him_list, Hmask_list, Hgt_list = readTuple(root['testis']['pth'], 'HTestTuple.lst')
    Vim_list, Vmask_list, Vgt_list = readTuple(root['testis']['pth'], 'VTestTuple.lst')

    Hckname = [i.split('/')[-2] for i in Him_list]
    Vckname = [i.split('/')[-2] for i in Vim_list]
    SetVname = set(Vckname)
    SetHname = set(Hckname)
    InS = SetHname.intersection(SetVname)
    InU = SetHname.union(SetVname)
    if flag == 'H':
        S = SetHname
    elif flag == 'V':
        S = SetVname
    elif flag == 'I':
        S = InS
    elif flag == 'U':
        S = InU
    else:
        raise Exception('flag must be HVIU')
    data = {}
    for ck in S:
        data.update({ck: {'H': {'images': [], 'masks': [], 'label': []},
                          'V': {'images': [], 'masks': [], 'label': []}
                          }})
        for him, hmk, hg in zip(Him_list, Hmask_list, Hgt_list):
            if him.split('/')[-2] == ck:
                temp = trans({'images': Image.open(him), 'masks': Image.open(hmk)})
                data[ck]['H']['images'].append(temp['images'])
                data[ck]['H']['masks'].append(temp['masks'])
                data[ck]['H']['label'].append(hg)
        for vim, vmk, vg in zip(Vim_list, Vmask_list, Vgt_list):
            if vim.split('/')[-2] == ck:
                temp = trans({'images': Image.open(vim), 'masks': Image.open(vmk)})
                data[ck]['V']['images'].append(temp['images'])
                data[ck]['V']['masks'].append(temp['masks'])
                data[ck]['V']['label'].append(vg)
    return data
