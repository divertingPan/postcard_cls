import time
import os
import math
import argparse
import json
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from torchvision.models import resnet50
from torch.optim import Adam, AdamW, SGD
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
from tensorboard import notebook
from dataset import MyDataset
from senet.se_resnet import se_resnet50
from repvgg import create_RepVGG_B1g2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--EPOCH", default=200, type=int, help="train epochs")
    parser.add_argument("--BATCHSIZE", default=200, type=int, help="batch size")
    parser.add_argument("--LR", default=0.001, type=float, help="learning rate")
    parser.add_argument("--LOGSTEP", default=100, type=int, help="steps between printing logs")
    parser.add_argument("--NUMWORKER", default=18, type=int, help="dataloader num_worker")
    parser.add_argument("--NETWORK", default='senet', type=str, help="network structure",
                        choices=['resnet', 'senet', 'repvgg'])
    opt = parser.parse_args()
    print(json.dumps(vars(opt)))

    DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    log_dir = './log'
    timestamp = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime(time.time()))
    notebook.start("--logdir {}".format(log_dir))
    writer = SummaryWriter(os.path.join(log_dir, timestamp))

    df = pd.read_csv('./dataset/label.csv')
    weight_list = df['train_cls'].values
    weight_list = 1 / weight_list
    weight_list = torch.tensor(weight_list).float()

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize(256),
        transforms.RandomAffine(10, shear=0.1, fillcolor=(255, 255, 255)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        # transforms.RandomCrop(224, fill=(255, 255, 255)),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    # train_dataset = ImageFolder('./dataset/test_folder', transform=transform)

    df = pd.read_csv('./dataset/train.csv')
    df = df.sample(frac=1.0)
    image_path_list = df['path'].values
    label_list = df['label'].values
    train_size = len(df)

    # train_size = 20000
    # image_path_list = image_path_list[:train_size]
    # label_list = label_list[:train_size]

    train_dataset = MyDataset(image_path_list, label_list, train=True, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=opt.BATCHSIZE, shuffle=True, drop_last=True,
                              num_workers=opt.NUMWORKER,
                              pin_memory=True
                              )

    df = pd.read_csv('./dataset/val.csv')
    df = df.sample(frac=1.0)
    image_path_list = df['path'].values
    label_list = df['label'].values
    val_size = len(df)

    # val_size = 5000
    # image_path_list = image_path_list[:val_size]
    # label_list = label_list[:val_size]

    val_dataset = MyDataset(image_path_list, label_list, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=opt.BATCHSIZE, shuffle=False,
                            num_workers=opt.NUMWORKER,
                            pin_memory=True
                            )

    if opt.NETWORK == 'resnet':
        net = resnet50(pretrained=True)
        fc_features = net.fc.in_features
        net.fc = torch.nn.Linear(fc_features, 230)
    elif opt.NETWORK == 'senet':
        net = se_resnet50(num_classes=230)
    elif opt.NETWORK == 'repvgg':
        net = create_RepVGG_B1g2(num_classes=230)
    else:
        net = resnet50(num_classes=230)

    net = net.to(DEVICE)

    # freezing conv features
    # for k, v in net.named_parameters():
    #    if k not in ['fc.weight', 'fc.bias']:
    #        v.requires_grad = False

    loss_func = CrossEntropyLoss(weight=weight_list).to(DEVICE)
    optimizer = Adam(net.parameters(), lr=opt.LR, weight_decay=0.0001)
    # optimizer = AdamW(net.parameters())
    # optimizer = SGD(net.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.1)

    best_acc = 0
    total_train_step = train_size // opt.BATCHSIZE
    total_val_step = math.ceil(val_size / opt.BATCHSIZE)
    for epoch in range(opt.EPOCH):
        correct = 0
        mean_loss = 0
        net.train()
        for step, (train_img, train_label) in enumerate(train_loader):
            train_img, train_label = train_img.to(DEVICE), train_label.to(DEVICE)
            predict = net(train_img)
            loss = loss_func(predict, train_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct += predict.max(1)[1].eq(train_label).sum().item()
            mean_loss += loss.item()

            if (step + 1) % opt.LOGSTEP == 0:
                acc = correct / (opt.BATCHSIZE * (step + 1))
                loss = mean_loss / (step + 1)
                print('Epoch: {:d}  Step: {:d} / {:d} | acc: {:.4f} | LR: {:.6f}'.format(
                    epoch, step + 1, total_train_step, acc, optimizer.param_groups[0]['lr']))

        acc = correct / (opt.BATCHSIZE * (step + 1))
        loss = mean_loss / (step + 1)
        # writer.add_scalar("train/loss", loss, step + 1 + epoch * total_train_step)
        # writer.add_scalar("train/acc", acc, step + 1 + epoch * total_train_step)
        # writer.add_images("train/img", train_img, epoch)
        writer.add_scalar("train/loss", loss, epoch)
        writer.add_scalar("train/acc", acc, epoch)

        # scheduler.step()

        net.eval()
        correct = 0
        mean_loss = 0
        print('===== val =====')
        with torch.no_grad():
            for step, (val_img, val_label) in enumerate(val_loader):
                val_img, val_label = val_img.to(DEVICE), val_label.to(DEVICE)
                predict = net(val_img)
                loss = loss_func(predict, val_label)

                correct += predict.max(1)[1].eq(val_label).sum().item()
                mean_loss += loss.item()

            acc = correct / (opt.BATCHSIZE * (step + 1))
            loss = mean_loss / (step + 1)
            # writer.add_scalar("val/loss", loss, step + 1 + epoch * total_val_step)
            # writer.add_scalar("val/acc", acc, step + 1 + epoch * total_val_step)
            # writer.add_images("val/img", val_img, epoch)
            writer.add_scalar("val/loss", loss, epoch)
            writer.add_scalar("val/acc", acc, epoch)
            print('Epoch: {:d}  Step: {:d} / {:d} | acc: {:.4f} | LR: {:.6f}'.format(
                epoch, step + 1, total_val_step, acc, optimizer.param_groups[0]['lr']))

        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(), './log/{}/{}.pth'.format(timestamp, timestamp))
            print('Best acc: {:.4f}, model saved!'.format(acc))
        print('===== epoch =====')

    writer.close()
