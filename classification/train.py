from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm

from Measure import AverageMeter
from loader import TxtDataset
from model import Resnet50

from datetime import datetime


def train(model, loader, criterion, optimizer, epoch):
    losses = AverageMeter()
    correct = AverageMeter()
    model.train()
    pbar = tqdm(loader,ncols=250)
    for i, (label, path, image) in enumerate(loader, start=1):
        optimizer.zero_grad()
        outputs = model(image.cuda())
        loss = criterion(outputs, label.cuda())

        _, preds = torch.max(outputs, 1)
        loss.backward()
        optimizer.step()
        correct.update(torch.sum(preds == label.cuda()).item())
        losses.update(loss)
        pbar.set_description(f'[Epoch {epoch}] '
                             f'train_loss : {losses.val:.4f}({losses.avg:.4f}) '
                             f'train_acc : {correct.val / loader.batch_size:.4f}({correct.sum / (i * loader.batch_size):.4f})')
        pbar.update()
    pbar.close()
    writer.add_scalar('loss/train_loss', losses.avg, epoch)
    writer.add_scalar('accuracy/train_acc', correct.sum/loader.dataset.__len__(), epoch)


def valid(model, loader, criterion, epoch):
    losses = AverageMeter()
    correct = AverageMeter()
    model.eval()
    pbar = tqdm(loader, ncols=250)
    with torch.no_grad():
        for i, (label, path, image) in enumerate(loader,start=1):
            outputs = model(image.cuda())
            loss = criterion(outputs, label.cuda())

            _, preds = torch.max(outputs, 1)
            losses.update(loss)
            pbar.update()
            correct.update(torch.sum(preds == label.cuda()).item())
            pbar.set_description(f'[Epoch {epoch}] '
                                 f'valid_loss : {losses.val:.4f}({losses.avg:.4f}) '
                                 f'valid_acc : {correct.val / loader.batch_size:.4f}({correct.sum / (i * loader.batch_size):.4f})')

        pbar.close()
        writer.add_scalar('loss/valid_loss', losses.avg, epoch)
        writer.add_scalar('accuracy/valid_acc', correct.sum / loader.dataset.__len__(), epoch)



def main():
    lr = 1e-3
    weight_decay = 1e-4

    root = '/nfs_shared/food-101/images'
    labels = {i.strip().lower(): n for n, i in enumerate(open('/nfs_shared/food-101/meta/labels.txt', 'r').readlines())}
    train_loader = DataLoader(TxtDataset('/nfs_shared/food-101/meta/train.txt', labels, root), batch_size=64,
                              num_workers=4, shuffle=True)
    valid_loader = DataLoader(TxtDataset('/nfs_shared/food-101/meta/test.txt', labels, root), batch_size=64,
                              num_workers=4, shuffle=True)

    global writer

    writer = SummaryWriter(f'/hdd/ms/food_run/{datetime.now().strftime("%Y%m%d/%H%M%S")}')

    model = Resnet50().cuda()
    writer.add_graph(model, torch.rand((2, 3, 224, 224)).cuda())

    model = nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss()
    # print([k for k,_ in model.named_parameters()])
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for ep in range(1, 50, 1):
        train(model, train_loader, criterion, optimizer, ep)
        valid(model, valid_loader, criterion, ep)


if __name__ == '__main__':
    main()
