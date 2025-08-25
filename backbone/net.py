import torch
import torch.nn as nn
import backbone
import numpy as np
import math
import torch.nn.functional as F

import backbone.DCnet
import backbone.DCnet.resnet

# import cv2
PI = 3.141592653

class CrossAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=4):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.query_dim = feature_dim // num_heads
        self.key_dim = feature_dim // num_heads
        self.value_dim = feature_dim // num_heads

        self.query_projection = nn.Linear(feature_dim, feature_dim)
        self.key_projection = nn.Linear(feature_dim, feature_dim)
        self.value_projection = nn.Linear(feature_dim, feature_dim)
        self.out_projection = nn.Linear(feature_dim, feature_dim)

    def forward(self, query_features, key_value_features):
        batch_size = query_features.size(0)

        # Project the query, key, and value
        query = self.query_projection(query_features).view(batch_size, -1, self.num_heads, self.query_dim).transpose(1, 2)
        key = self.key_projection(key_value_features).view(batch_size, -1, self.num_heads, self.key_dim).transpose(1, 2)
        value = self.value_projection(key_value_features).view(batch_size, -1, self.num_heads, self.value_dim).transpose(1, 2)

        # Calculate the attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.query_dim)
        attention = F.softmax(scores, dim=-1)

        # Apply the attention to the values
        context = torch.matmul(attention, value).transpose(1, 2).contiguous().view(batch_size, -1, self.feature_dim)

        # Project the output
        output = self.out_projection(context)
        return output


class V1Filters(nn.Module):
    def __init__(self, out_channel=64, num_dir=8, scale=(1, 2, 4, 6), phi=(0, -PI/2), k=(2, 4, 8)):
        super(V1Filters, self).__init__()
        self.weight = []
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for j in range(num_dir):
            theta = j / num_dir * PI
            for i, sigma in enumerate(scale):
                for p in phi:
                    self.weight.append(torch.tensor(self.kernel(theta, sigma, p), dtype=torch.float, device=device))
        # for s in scale:
        #     for ik in k:
        #         self.weight.append(torch.tensor(self.kernel_off(s, ik), dtype=torch.float, device=device))
        #         self.weight.append(torch.tensor(self.kernel_on(s, ik), dtype=torch.float, device=device))
        # self.down = nn.Sequential(nn.Conv2d(1, out_channel, kernel_size=1, stride=2), nn.BatchNorm2d(out_channel))
        # self.relu = nn.ReLU(inplace=True)

        self.num_trainable_weight = out_channel - len(self.weight)
        if self.num_trainable_weight != 0:
            self.conv = nn.Conv2d(1, self.num_trainable_weight, kernel_size=7, stride=2, padding=3)

    def get_length(self, sigma, gamma):
        r = int(sigma * 2.5 / gamma)
        d = r * 2 + 1
        return r, d

    def kernel_off(self, sigma, k):
        weight = self.kernel_gauss(sigma, sigma*k) - self.kernel_gauss(sigma*k, sigma*k)
        return np.expand_dims(np.expand_dims(weight, axis=0), axis=0)

    def kernel_on(self, sigma, k):
        weight = self.kernel_gauss(sigma*k, sigma * k) - self.kernel_gauss(sigma, sigma * k)
        return np.expand_dims(np.expand_dims(weight, axis=0), axis=0)

    def kernel_gauss(self, sigma, sample_sigma):
        r = int(sample_sigma * 2.5)
        d = r * 2 + 1
        x, y = np.meshgrid(np.linspace(-r, r, num=d), np.linspace(-r, r, num=d))
        y = -y

        a = 1 / np.sqrt(2 * PI) / sigma
        weight = a * np.exp(-(1 / (2 * sigma * sigma)) * ((x * x) + (y * y)))
        return weight

    def kernel(self, theta, sigma, phi):
        slratio = 0.56
        gamma = 0.5
        lambda_ = sigma / slratio

        r, d = self.get_length(sigma, gamma)
        x, y = np.meshgrid(np.linspace(-r, r, num=d), np.linspace(-r, r, num=d))
        y = -y
        xp = x * np.cos(theta) + y * np.sin(theta)
        yp = -x * np.sin(theta) + y * np.cos(theta)

        gamma2 = gamma * gamma
        b = 1 / np.sqrt(2 * sigma * sigma)
        a = b / PI
        f = 2 * PI / lambda_

        weight = a*np.exp(-b*(xp*xp + gamma2*(yp*yp))) * np.cos(f*xp + phi)
        return np.expand_dims(np.expand_dims(weight, axis=0), axis=0)

    def forward(self, x):
        # short_cut = self.down(x)
        device = x.device # 获取输入张量的设备
        # 确保所有权重张量与输入张量在同一设备上
        weight = [w.to(device) for w in self.weight]
        
        out = []
        for w in weight:
            pad = int((w.shape[2]-1) / 2)
            out.append(nn.functional.conv2d(x, w, bias=None, stride=2, padding=pad, dilation=1, groups=1))

        if self.num_trainable_weight != 0:
            out.append(self.conv(x))
            x = torch.cat(out, dim=1)
        else:
            x = torch.cat(out, dim=1)
        return x


class ASPP_Module(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(ASPP_Module, self).__init__()
        self.atrous1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding=0),
                                 nn.BatchNorm2d(out_channel))
        self.atrous2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, dilation=1),
                                 nn.BatchNorm2d(out_channel))
        self.atrous3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=3, dilation=3),
                                 nn.BatchNorm2d(out_channel))
        self.atrous4 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=6, dilation=6),
                                 nn.BatchNorm2d(out_channel))
        self.atrous5 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=9, dilation=9),
            nn.BatchNorm2d(out_channel))

        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(out_channel * 5, out_channel, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU())

    def forward(self, x):
        x1 = self.atrous1(x)
        x2 = self.atrous2(x)
        x3 = self.atrous3(x)
        x4 = self.atrous4(x)
        x5 = self.atrous5(x)

        net = self.conv_1x1(torch.cat([x1, x2, x3, x4, x5], dim=1))
        return net

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


class ASPP_cATT(nn.Module):
    def __init__(self, num_classes=2):
        super(ASPP_cATT, self).__init__()
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
        self.c_att = CrossAttention(28*28)

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

        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.bn = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(.5)

        self.fc1 = nn.Linear(512*4, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, *img_msk):
        x, mask = img_msk[0], img_msk[1]

        # stage1
        x = self.conv1(x)
        x = self.maxpool(x)
        # x = x * torch.nn.functional.adaptive_max_pool2d(mask, (x.shape[2], x.shape[3]))
        mask = self.blockb_1(mask)
        mask = self.maxpool(mask)
        # x = x * mask + x

        # stage2
        x = self.blocka_1(x)# ->128
        # x = x * torch.nn.functional.adaptive_max_pool2d(mask, (x.shape[2], x.shape[3]))
        mask = self.blockb_2(mask)
        # x = x * mask + x

        # stage3
        x = self.blocka_2(x) #->256
        # x = x * torch.nn.functional.adaptive_max_pool2d(mask, (x.shape[2], x.shape[3]))
        mask = self.blockb_3(mask) # 256,28,28
        # x = x * mask + x
        x = torch.flatten(x, 2)
        mask = torch.flatten(mask, 2)
        x_att = self.c_att(x, mask) + x # b, 256, 784
        m_att = self.c_att(mask, x) + mask
        
        x = torch.cat((x_att, m_att), 1)
        x = self.bn(x) # b, 256,784*2
        x = x.view(-1, 512, 28, 28)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.relu(self.fc2(x))
        x2 = x
        x = self.fc3(x)
        return x, x2

class ASPP_SimCLR(nn.Module):
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

class MultiResnet34(nn.Module):
    def __init__(self, size=(300, 224, 148), num_classes=2):
        super(MultiResnet34, self).__init__()
        planes = 8
        self.size = size

        self.scale1 = backbone.resnet.resnet34(pretrained=True)
        self.scale1.fc = nn.Linear(512 * 1, planes)

        self.scale2 = backbone.resnet.resnet34(pretrained=True)
        self.scale2.fc = nn.Linear(512 * 1, planes)

        self.scale3 = backbone.resnet.resnet34(pretrained=True)
        self.scale3.fc = nn.Linear(512 * 1, planes)
        self.fc = nn.Linear(planes * 3, 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, *input):
        # scale1
        im_scale1 = torch.nn.functional.interpolate(input[0], size=self.size[0], mode='bilinear', align_corners=True)
        mk_scale1 = torch.nn.functional.interpolate(input[1], size=self.size[0], mode='bilinear', align_corners=True)
        s1 = self.scale1(im_scale1, mk_scale1)

        im_scale2 = torch.nn.functional.interpolate(input[0], size=self.size[1], mode='bilinear', align_corners=True)
        mk_scale2 = torch.nn.functional.interpolate(input[1], size=self.size[1], mode='bilinear', align_corners=True)
        s2 = self.scale1(im_scale2, mk_scale2)

        im_scale3 = torch.nn.functional.interpolate(input[0], size=self.size[2], mode='bilinear', align_corners=True)
        mk_scale3 = torch.nn.functional.interpolate(input[1], size=self.size[2], mode='bilinear', align_corners=True)
        s3 = self.scale1(im_scale3, mk_scale3)

        return self.fc(self.relu(torch.cat([s1[0], s2[0], s3[0]], dim=1))), []


class MultiASPP(nn.Module):
    def __init__(self, size=(300, 224, 148), num_classes=2):
        super(MultiASPP, self).__init__()
        planes = 8
        self.size = size

        self.scale1 = ASPP(num_classes=planes)
        self.scale2 = ASPP(num_classes=planes)
        self.scale3 = ASPP(num_classes=planes)
        self.fc = nn.Linear(planes * 3, 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, *input):
        # scale1
        im_scale1 = torch.nn.functional.interpolate(input[0], size=self.size[0], mode='bilinear', align_corners=True)
        mk_scale1 = torch.nn.functional.interpolate(input[1], size=self.size[0], mode='bilinear', align_corners=True)
        s1 = self.scale1(im_scale1, mk_scale1)

        im_scale2 = torch.nn.functional.interpolate(input[0], size=self.size[1], mode='bilinear', align_corners=True)
        mk_scale2 = torch.nn.functional.interpolate(input[1], size=self.size[1], mode='bilinear', align_corners=True)
        s2 = self.scale1(im_scale2, mk_scale2)

        im_scale3 = torch.nn.functional.interpolate(input[0], size=self.size[2], mode='bilinear', align_corners=True)
        mk_scale3 = torch.nn.functional.interpolate(input[1], size=self.size[2], mode='bilinear', align_corners=True)
        s3 = self.scale1(im_scale3, mk_scale3)

        return self.fc(self.relu(torch.cat((s1[0], s2[0], s3[0]), dim=1))), self.relu(torch.cat((s1[0], s2[0], s3[0]), dim=1))

