import os
import time
import os
import random
import datetime
import numpy as np
from math import sqrt
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch import optim
import torch.autograd
from torch.utils.data import DataLoader
import torch.nn.functional as F
working_path = os.path.abspath('.')
import time
from skimage import io
from utils.utils import evaluate_model
from utils.utils import intersectionAndUnion, AverageMeter
###################### Data and Model ########################
from dataset import CustomImageDataset
#from datasets import WHU_CD as RS
#DATA_NAME = 'WHU_CD'
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import CustomImageDataset
# 这里修改成需要使用的网络名称
from models.model import MyFCNet as Net # MyFCNet, MyResNet, MyViTModel
###################### Data and Model ########################
NET_NAME = 'MyFCNet'
train_img_dir = '/data/wfy/Halli_Galli/Res_Images'
train_label_dir = '/data/wfy/Halli_Galli/labels.txt'
val_img_dir = '/data/wfy/Halli_Galli/val_Images'
val_label_dir = '/data/wfy/Halli_Galli/val_labels.txt'
######################## Parameters ########################
args = {
    'train_batch_size': 16,
    'val_batch_size': 16,
    'lr': 0.1,
    'epochs': 50,
    'gpu': True,
    'dev_id': 0,
    'multi_gpu': None,  #"0,1,2,3",
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'print_freq': 100,
    'predict_step': 5,
    'log_dir': os.path.join(working_path, 'logs', NET_NAME),
    }
###################### Data and Model ######################
writer = SummaryWriter(args['log_dir'])


def main():
    net = Net()
    if args['multi_gpu']:
        net = torch.nn.DataParallel(net, [int(id) for id in args['multi_gpu'].split(',')])
    print(torch.cuda.is_available())
    net.to(device=torch.device('cuda', int(args['dev_id'])))

    train_dataset = CustomImageDataset(img_dir=train_img_dir, label_file=train_label_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=args['train_batch_size'], shuffle=True)
    val_dataset = CustomImageDataset(img_dir=val_img_dir, label_file=val_label_dir)
    val_dataloader = DataLoader(val_dataset, batch_size=args['val_batch_size'], shuffle=False)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.1,
                          weight_decay=args['weight_decay'], momentum=args['momentum'], nesterov=True)

    train(train_dataloader, net, optimizer, val_dataloader)
    print('Training finished.')
    # predict(net, args)


def train(train_loader, net, optimizer, val_loader):
    bestF = 0.0
    bestacc = 0.0
    bestloss = 1.0
    bestaccT = 0.0

    curr_epoch = 0
    begin_time = time.time()
    all_iters = float(len(train_loader) * args['epochs'])
    while True:
        torch.cuda.empty_cache()
        net.train()
        start = time.time()
        acc_meter = AverageMeter()
        precision_meter = AverageMeter()
        recall_meter = AverageMeter()
        f1_meter = AverageMeter()
        train_loss = AverageMeter()
        curr_iter = curr_epoch * len(train_loader)

        for i, data in enumerate(train_loader):
            running_iter = curr_iter + i + 1
            adjust_lr(optimizer, running_iter, all_iters, args)
            imgs,labels = data
            if args['gpu']:
                imgs = imgs.to(torch.device('cuda', int(args['dev_id']))).float()
                labels = labels.to(torch.device('cuda', int(args['dev_id']))).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = net(imgs)#这个outputs应该为一个batch_size*1的值
            loss = F.binary_cross_entropy_with_logits(outputs, labels)
            loss.backward()
            optimizer.step()

            labels = labels.cpu().detach().numpy()
            outputs = outputs.cpu().detach()
            preds = F.sigmoid(outputs).numpy()
            acc_curr_meter = AverageMeter()
            precision_curr_meter = AverageMeter()
            recall_curr_meter = AverageMeter()
            f1_curr_meter = AverageMeter()
            accuracy, precision, recall, f1 = evaluate_model(preds, labels)
            acc_curr_meter.update(accuracy)
            precision_curr_meter.update(precision)
            recall_curr_meter.update(recall)
            f1_curr_meter.update(f1)
            acc_meter.update(acc_curr_meter.avg)
            precision_meter.update(precision_curr_meter.avg)
            recall_meter.update(recall_curr_meter.avg)
            f1_meter.update(f1_curr_meter.avg)

            train_loss.update(loss.cpu().detach().numpy())
            curr_time = time.time() - start

            if (i + 1) % args['print_freq'] == 0:
                print('[epoch %d] [iter %d / %d %.1fs] [lr %f] [train loss %.4f acc %.2f]' % (
                    curr_epoch, i + 1, len(train_loader), curr_time, optimizer.param_groups[0]['lr'],
                    train_loss.val, acc_meter.val * 100))
                writer.add_scalar('train loss', train_loss.val, running_iter)
                loss_rec = train_loss.val
                writer.add_scalar('train accuracy', acc_meter.val, running_iter)
                writer.add_scalar('train precision', precision_meter.val, running_iter)
                writer.add_scalar('train recall', recall_meter.val, running_iter)
                writer.add_scalar('train f1', f1_meter.val, running_iter)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], running_iter)

        val_F, val_acc, val_loss = validate(val_loader, net, curr_epoch)
        if val_F > bestF:
            bestF = val_F
            bestacc = val_acc
            torch.save(net.state_dict(), os.path.join(NET_NAME + '_e%d_OA%.2f_F%.2f.pth' % (
            curr_epoch, val_acc * 100, val_F * 100)))
        if acc_meter.avg > bestaccT: bestaccT = acc_meter.avg
        print('[epoch %d/%d %.1fs] Best rec: Train %.2f, Val %.2f, F1 score: %.2f ' \
              % (curr_epoch, args['epochs'], time.time() - begin_time, bestaccT * 100, bestacc * 100, bestF * 100,
              ))
        curr_epoch += 1
        if curr_epoch >= args['epochs']:
            return


def validate(val_loader, net, curr_epoch):
    # the following code is written assuming that batch size is 1
    net.eval()
    torch.cuda.empty_cache()
    start = time.time()

    val_loss = AverageMeter()
    F1_meter = AverageMeter()
    Acc_meter = AverageMeter()

    for vi, data in enumerate(val_loader):
        imgs, labels = data

        if args['gpu']:
            imgs = imgs.to(torch.device('cuda', int(args['dev_id']))).float()
            labels = labels.to(torch.device('cuda', int(args['dev_id']))).float().unsqueeze(1)

        with torch.no_grad():
            outputs = net(imgs)
            loss = F.binary_cross_entropy_with_logits(outputs, labels)
        val_loss.update(loss.cpu().detach().numpy())

        outputs = outputs.cpu().detach()
        labels = labels.cpu().detach().numpy()
        preds = F.sigmoid(outputs).numpy()
        accuracy, precision, recall, f1 = evaluate_model(preds, labels)
        F1_meter.update(f1)
        Acc_meter.update(accuracy)

    curr_time = time.time() - start
    print('%.1fs Val loss %.2f Acc %.2f F %.2f' % (
    curr_time, val_loss.average(), Acc_meter.average() * 100, F1_meter.average() * 100))

    writer.add_scalar('val_loss', val_loss.average(), curr_epoch)
    writer.add_scalar('val_Accuracy', Acc_meter.average(), curr_epoch)

    return F1_meter.avg, Acc_meter.avg, val_loss.avg


def adjust_lr(optimizer, curr_iter, all_iter, args):
    scale_running_lr = ((1. - float(curr_iter) / all_iter) ** 3.0)
    running_lr = args['lr'] * scale_running_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = running_lr


if __name__ == '__main__':
    main()

# tensorboard --logdir=logs --port=6006
