import os
import torch
import scipy.io as sio
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import pandas as pd

from tqdm import tqdm
import torch.nn.functional as F
import pdb
from torchvision import transforms
from PIL import Image
import time
# from seg_losses import *
from matrics import *
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def write_img(self, out, data_o, gt, fn, **kwargs):
    out = torch.sigmoid(out)
    data_o = data_o.data
    outc = out.data.cpu()
    gt_c = gt.data.cpu()
    out_arr = np.array(outc, np.float16)
    gt_arr = np.array(gt_c, np.bool)
    print(torch.max(outc))
    print(self.save_dir)
    # save_dir =  fn[:-2].replace('p/Classification/', 'p/lesionsSeg_test/'+ self.save_dir[2:])# lesions
    # save_dir =  fn[:-6].replace('Code/Lung_CT_segmentation/','data/LungCT_np/lungSeg_test/'+ self.save_dir[2:]) # lungseg cell
    save_dir = fn[:-6].replace('_np/data/', '_np/lungSeg_test/' + self.save_dir[2:])  # lungseg cell
    # img /home/hedh/Data/LungCT_np/lung_label_annoatationNet/
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # name = fn[-2:] # lesions
    name = fn[-6:-4:]  # lung
    sio.savemat(save_dir + name + '.mat', {'lung_prob': out_arr})
    # tensor2img = transforms.ToPILImage()
    # out_img = tensor2img(out)
    # out_img.save('seg_show/' + name + '.png')

#     save_dir =  fn[:-2].replace('p/data/', 'p/lungSeg_gt/')
# #img /home/hedh/Data/LungCT_np/lung_label_annoatationNet/
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     name = fn[-2:]
#     sio.savemat(save_dir + name + '.mat', {'lung_gt':gt_arr})


def show_roc(labels, preds, name):

    fpr, tpr,_ = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='%s (area = %.2f)' % (name, roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.xlim([.0, 1.0])
    plt.ylim([.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Charateristic')
    plt.legend(loc='lower right')
    plt.show()

def plot_training_history(history):
        # 
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.show()
        # 