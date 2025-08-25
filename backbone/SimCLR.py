import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from .net import V1Filters, ASPP_Module # .net
import backbone
import numpy as np
import torch.nn.functional as F

PI = 3.141592653

class ASPP(nn.Module):
    def __init__(self, num_classes=2):
        super(ASPP, self).__init__()
        # self.conv1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        #                            nn.BatchNorm2d(64),
        #                            nn.ReLU(inplace=True))
        # image process
        self.conv1 = V1Filters(out_channel=64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.blocka_1 = ASPP_Module(in_channel=64, out_channel=128, stride=1)
        self.blocka_2 = ASPP_Module(in_channel=128, out_channel=256, stride=2)
        self.blocka_3 = ASPP_Module(in_channel=256, out_channel=256, stride=2)
        self.blocka_4 = ASPP_Module(in_channel=256, out_channel=256, stride=2)

        # mask process
        downsample1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
        )
        downsample2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2, padding=0),
            nn.BatchNorm2d(256),
        )
        # self.blocka_4 = backbone.BasicBlock(inplanes=512, planes=512, stride=2, downsample=downsample2)
        self.blockb_1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3), nn.ReLU(True))
        # self.blockb_2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(True))
        # self.blockb_3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), nn.ReLU(True))
        self.blockb_2 = backbone.BasicBlock(inplanes=64, planes=128, stride=1, downsample=downsample1)
        self.blockb_3 = backbone.BasicBlock(inplanes=128, planes=256, stride=2, downsample=downsample2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, *img_msk):
        x, mask = img_msk[0], img_msk[1]

        # stage1
        x = self.conv1(x)
        x = self.maxpool(x)
        # x = x * torch.nn.functional.adaptive_max_pool2d(mask, (x.shape[2], x.shape[3]))
        mask = self.blockb_1(mask) 
        mask = self.maxpool(mask)# b, 64,56,56
        x = x * mask + x

        # stage2
        x = self.blocka_1(x)# ->128
        # x = x * torch.nn.functional.adaptive_max_pool2d(mask, (x.shape[2], x.shape[3]))
        mask = self.blockb_2(mask)
        x = x * mask + x

        # stage3
        x = self.blocka_2(x) #->256
        # x = x * torch.nn.functional.adaptive_max_pool2d(mask, (x.shape[2], x.shape[3]))
        mask = self.blockb_3(mask)
        x = x * mask + x

        # stage4
        x = self.blocka_3(x)
        x = self.blocka_4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x2 = x
        x = self.fc(x)
        return x, x2

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = self._get_correlated_mask(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def _get_correlated_mask(self, batch_size):
        mask = torch.ones((2 * batch_size, 2 * batch_size), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask
    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)
        sim = torch.mm(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        positives = torch.cat([sim_i_j, sim_j_i], dim=0)

        negatives = sim[self.mask].view(N, -1)
        logits = torch.cat((positives.unsqueeze(1), negatives), dim=1)
        labels = torch.zeros(N).long().to(z.device)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss



class SimCLR(nn.Module):
    def __init__(
        self,
        net,
        image_size,
        channels = 3,
        hidden_layer = -2,
        project_hidden = True,
        project_dim = 128,
        augment_both = True,
        use_nt_xent_loss = False,
        augment_fn = None,
        temperature = 0.1
    ):
        super().__init__()
        self.net = NetWrapper(net, project_dim, layer = hidden_layer)
        self.augment = default(augment_fn, get_default_aug(image_size, channels))
        self.augment_both = augment_both
        self.temperature = temperature

        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)

        # send a mock image tensor to instantiate parameters
        self.forward(torch.randn(1, channels, image_size, image_size))

    def forward(self, x):
        b, c, h, w, device = *x.shape, x.device
        transform_fn = self.augment if self.augment_both else noop
		# 把原图使用不同数据增强和ViT提取成两个不同的图像特征(正样本对queries、keys)
        queries, _ = self.net(transform_fn(x))  
        keys, _    = self.net(self.augment(x))

        queries, keys = map(flatten, (queries, keys))
        # 计算loss
        loss = nt_xent_loss(queries, keys, temperature = self.temperature)
        return loss

# 定义投影头
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Linear(in_dim, 2048)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2048, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class SimCLR_ASPP(nn.Module):
    def __init__(self,num_channels):
        super().__init__()
        self.aspp = ASPP(num_channels)
        # self.aspp = encoder
        self.projection_head = ProjectionHead(in_dim=2048, out_dim=128)

    def forward(self, *img_msk):
        _, queries = self.aspp(img_msk[0], img_msk[1]) # 64,256
        out = self.projection_head(queries)
        return out


# 分类模型定义
class SimCLRClassifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super(SimCLRClassifier, self).__init__()
        self.encoder = encoder
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        with torch.no_grad():
            h = self.encoder(x)
        logits = self.fc(h)
        return logits