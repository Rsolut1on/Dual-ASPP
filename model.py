import torch
import backbone
import torchvision
nn = torch.nn


def encode(name='resnet50', num_classes=2, encoder=None):
    if name == 'resnet50':
        model = backbone.resnet50(pretrained=True)
        model.fc = nn.Linear(512 * 4, num_classes)
    elif name == 'resnet34':
        model = backbone.resnet34(pretrained=True)
        # model.conv1 = backbone.V1Filters(out_channel=64)
        model.fc = nn.Linear(512 * 1, num_classes)
    elif name == 'inception_v3':
        model = backbone.inception_v3(pretrained=True, aux_logits=False)
        model.fc = nn.Linear(2048, num_classes)
    elif name == 'ASPP':
        model = backbone.ASPP(num_classes=2)
    elif name == 'ASPP_cATT':
        model = backbone.ASPP_cATT(num_classes=2)
    elif name == 'MultiASPP':
        model = backbone.MultiASPP(num_classes=2)
    elif name == 'MultiResnet34':
        model = backbone.MultiResnet34(size=(300, 224, 112), num_classes=2)
    elif name == 'UNet':
        model = backbone.UNet(1, 1)
    elif name == 'UNet_2Plus':
        model = backbone.UNet_Nested(1, 1)
    elif name == 'SegNet':
        model = backbone.SegNet(1, 1)
    elif name == 'DCnet':
        model = backbone.DconnNet(num_classes)
    elif name == 'ASPP_dual':
        model = backbone.EnhancedCrossModalFusion(num_classes, dim=256)
    else:
        raise Exception('unknown model name, please check in resnet34, resnet50')
    return model


def learning_rate_decay(optimizer, epoch, decay_rate=0.1, decay_steps=10):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * (decay_rate ** (epoch // decay_steps))


class loss(nn.Module):
    def __init__(self):
        super(loss, self).__init__()

    def forward(self, pred, labels):
        eps = 1e-6
        prob = pred.sigmoid().view(-1).clamp(eps, 1.0-eps)
        prob_pos = prob[labels == 1]
        prob_neg = prob[labels == 0]

        cross_entropy = (-prob_pos.log()).mean() + \
                        (-(1.0 - prob_neg).log()).mean()
        return cross_entropy / 2


def loss_fun(name='CRL', batch_size=8):
    if name == 'CRL':
        loss = backbone.ContrastiveLoss()
    elif name == 'SimCRL':
        loss = backbone.NTXentLoss(batch_size, temperature=0.5)
    else:
        raise Exception('unknown loss name, please check in CRL')
    return loss