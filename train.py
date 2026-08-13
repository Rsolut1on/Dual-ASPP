import torch
import yaml
import dataio
import model
import time
import numpy as np
from datetime import datetime
from utils_metrics import plot_training_history
KMP_DUPLICATE_LIB_OK = True
transforms = dataio.transforms

if __name__ == '__main__':
    # load configures
    file_id = open('./cfgs.yaml')
    cfgs = yaml.safe_load(file_id)
    file_id.close()

    # 
    train_trans = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.Crop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_trans = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.Crop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = dataio.testis(root=cfgs['dataset'], flag='train', VH=cfgs['dataset']['testis']['flg'], transform=train_trans)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfgs['batch_size'], shuffle=True, num_workers=4, drop_last=True
    )

    dataset_val = dataio.testis(root=cfgs['dataset'], flag='test', VH=cfgs['dataset']['testis']['flg'], transform=val_trans)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=1)

    net = model.encode(
        name=cfgs['model_Classifier'],
        num_classes=2,
        ablation_variant=cfgs.get('ablation_variant', 'full'),
    ).train()
    if cfgs['fine_tunning']:
        net.load_state_dict(
            torch.load('./checkpoint/' + cfgs['dataset']['testis']['flg'] + cfgs['model_Classifier'] + '.pth')
        )
    criterion = torch.nn.CrossEntropyLoss()
    if cfgs['method'] == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=cfgs['lr'], weight_decay=cfgs['weight_decay'])
    elif cfgs['method'] == 'SGD':
        optimizer = torch.optim.SGD(
            net.parameters(), lr=cfgs['lr'], momentum=cfgs['momentum'], weight_decay=cfgs['weight_decay']
        )
    else:
        raise Exception('unknown optimizer name!')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using GPUs: {cfgs['iGPU']}")
        net = torch.nn.DataParallel(net, device_ids=cfgs['iGPU'])

    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    ckpt_prefix = './checkpoint/' + cfgs['dataset']['testis']['flg'] + cfgs['model_Classifier']
    if cfgs['model_Classifier'] == 'ASPP_dual':
        ckpt_prefix += '_' + cfgs.get('ablation_variant', 'full')

    def forward_net(images, mask, data=None):
        if cfgs['model_Classifier'] == 'ASPP_dual':
            input_a = [images, mask]
            if data is None or 'img_CLR' not in data or 'msk_CLR' not in data:
                raise KeyError(
                    'ASPP_dual requires paired img_CLR and msk_CLR tensors; '
                    'duplicating the G-modality input is not allowed'
                )
            input_b = [data['img_CLR'].to(device), data['msk_CLR'].to(device)]
            return net(input_a, input_b)
        out = net(images, mask)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

    for epoch in range(cfgs['max_iter']):
        net.train()
        model.learning_rate_decay(optimizer, epoch, decay_rate=cfgs['decay_rate'], decay_steps=cfgs['decay_steps'])
        gt = []
        pred = []
        running_loss = 0.0
        n_train = 0
        for i, data in enumerate(dataloader):
            start_time = time.time()
            gt.append(data['labels'])
            optimizer.zero_grad()
            images = data['images'].to(device)
            mask = data['masks'].to(device)
            labels = data['labels'].to(device)

            prediction = forward_net(images, mask, data)
            loss = criterion(prediction, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            n_train += labels.size(0)
            prob = prediction.softmax(dim=1).cpu().detach().numpy()
            pred.append(np.argmax(prob, 1))

            if i % 22 == 0:
                duration = time.time() - start_time
                format_str = '%s: step [%d, %5d], loss = %.3f (%.3f sec/batch)'
                print(format_str % (datetime.now(), epoch, i, loss.item(), float(duration)))

        gt_arr, pred_arr = torch.cat(gt).numpy(), np.concatenate(pred)
        train_acc = ((gt_arr - pred_arr) == 0).sum() / gt_arr.size
        train_loss = running_loss / max(n_train, 1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        gt = []
        pred = []
        net.eval()
        running_loss = 0.0
        n_val = 0
        for i, data in enumerate(dataloader_val):
            gt.append(data['labels'])
            images = data['images'].to(device)
            mask = data['masks'].to(device)
            labels = data['labels'].to(device)
            with torch.no_grad():
                prediction = forward_net(images, mask, data)
            prob = prediction.softmax(dim=1).cpu().detach().numpy()
            pred.append(np.argmax(prob, 1))
            loss = criterion(prediction, labels)
            running_loss += loss.item()
            n_val += 1

        gt_arr, pred_arr = torch.cat(gt).numpy(), np.concatenate(pred)
        val_acc = ((gt_arr - pred_arr) == 0).sum() / gt_arr.size
        val_loss = running_loss / max(n_val, 1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(
            'Epoch %d | train_loss=%.3f train_acc=%.3f | val_loss=%.3f val_acc=%.3f | lr=%.6f'
            % (epoch, train_loss, train_acc, val_loss, val_acc, optimizer.param_groups[0]['lr'])
        )

        if val_acc > best_acc:
            torch.save(net.state_dict(), ckpt_prefix + '_best.pth')
            best_acc = val_acc
            print('Saved best checkpoint, val_acc=%.3f' % best_acc)
    # 

    torch.save(net.state_dict(), ckpt_prefix + '.pth')
    print('Saved at: ', ckpt_prefix + '.pth')
    print('Best ACC:', best_acc)

    plot_training_history(history)
