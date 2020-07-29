from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms as trn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch

from tensorboardX import SummaryWriter
from datetime import datetime
from tqdm import tqdm
import logging
import argparse

from classification.autoaugment import ImageNetPolicy
from classification.Measure import AverageMeter
from classification.loader import TXTDataset
from classification.model import Resnet50, Resnet152, Mobilenet_v2, Densenet_121


def train(model, loader, criterion, optimizer, epoch):
    losses = AverageMeter()
    correct = AverageMeter()

    model.train()
    pbar = tqdm(loader, ncols=250)
    for i, (label, path, image) in enumerate(loader, start=1):
        optimizer.zero_grad()
        outputs = model(image.cuda())
        loss = criterion(outputs, label.cuda())
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        correct.update(torch.sum(preds == label.cuda()).item())
        losses.update(loss)

        pbar.set_description(f'[Epoch {epoch}] '
                             f'train_loss : {losses.val:.4f}({losses.avg:.4f}) '
                             f'train_prec : {correct.val / loader.batch_size:.4f}({correct.sum / (i * loader.batch_size):.4f})')
        pbar.update()
    pbar.close()
    logger.info(f'[EPOCH {epoch}] train_loss : {losses.avg:.4f}, '
                f'train_prec : {correct.sum / loader.dataset.__len__():.4f}')

    writer.add_scalar('loss/train_loss', losses.avg, epoch)
    writer.add_scalar('precision/train_precision', correct.sum / loader.dataset.__len__(), epoch)

    return losses, correct.sum / loader.dataset.__len__()


def valid(model, loader, criterion, epoch):
    losses = AverageMeter()
    correct = AverageMeter()
    model.eval()
    pbar = tqdm(loader, ncols=250)
    with torch.no_grad():
        for i, (label, path, image) in enumerate(loader, start=1):
            outputs = model(image.cuda())
            loss = criterion(outputs, label.cuda())

            _, preds = torch.max(outputs, 1)
            losses.update(loss)
            pbar.update()
            correct.update(torch.sum(preds == label.cuda()).item())
            pbar.set_description(f'[Epoch {epoch}] '
                                 f'valid_loss : {losses.val:.4f}({losses.avg:.4f}) '
                                 f'valid_prec : {correct.val / loader.batch_size:.4f}({correct.sum / (i * loader.batch_size):.4f})')

        pbar.close()
        logger.info(f'[EPOCH {epoch}] valid_loss : {losses.avg:.4f}, '
                    f'valid_prec : {correct.sum / loader.dataset.__len__():.4f}')
        writer.add_scalar('loss/valid_loss', losses.avg, epoch)
        writer.add_scalar('precision/valid_precision', correct.sum / loader.dataset.__len__(), epoch)

    return losses, correct.sum / loader.dataset.__len__()


def init_logger(save_dir):
    c_time = datetime.now().strftime("%Y%m%d/%H%M%S")
    log_dir = f'/{save_dir}/{c_time}'
    log_txt = f'/{save_dir}/{c_time}/log.txt'
    writer = SummaryWriter(log_dir)

    logger = logging.getLogger(c_time)
    logger.setLevel(logging.INFO)
    logger = logging.getLogger(c_time)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
    h_file = logging.FileHandler(filename=log_txt, mode='a')
    h_file.setFormatter(fmt)
    h_file.setLevel(logging.INFO)
    logger.addHandler(h_file)
    logger.info(f'Log directory ... {log_txt}')
    return log_dir


def main():
    parser = argparse.ArgumentParser(description="Train for Food Classification.")
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-wd', '--weight_decay', type=float, default=5e-4)
    parser.add_argument('-batch', '--batch_size', type=int, default=64)
    parser.add_argument('-save', '--save_dir', type=str, default=f'/hdd/')

    args = parser.parse_args()

    global writer
    global logger
    log_dir = init_logger(args.save_dir)

    transform = {'train': trn.Compose([trn.RandomRotation(30),
                                       trn.RandomResizedCrop(224),
                                       trn.RandomHorizontalFlip(),
                                       # ImageNetPolicy(),
                                       trn.ToTensor(),
                                       trn.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])]),
                 'valid': trn.Compose([trn.Resize(256),
                                       trn.CenterCrop(224),
                                       trn.ToTensor(),
                                       trn.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])])}

    root = '/nfs_shared/food/images'
    classes = '/nfs_shared/food/meta/classes.txt'
    train_loader = DataLoader(TXTDataset('/nfs_shared/food/meta/train.txt', classes, root, transform=transform['train']),
                              batch_size=args.batch_size, num_workers=4, shuffle=True)
    valid_loader = DataLoader(TXTDataset('/nfs_shared/food/meta/test.txt', classes, root, transform=transform['valid']),
                              batch_size=args.batch_size, num_workers=4, shuffle=True)

    model = Resnet152(freeze=False).cuda()
    writer.add_graph(model, torch.rand((2, 3, 224, 224)).cuda())
    model = nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss()

    # print([k for k, p in model.named_parameters() if p.requires_grad])

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.learning_rate,
                           weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 80], gamma=0.1)

    best_acc = 0.0
    for ep in range(1, 100, 1):
        _, _ = train(model, train_loader, criterion, optimizer, ep)
        loss, acc = valid(model, valid_loader, criterion, ep)
        scheduler.step()
        if acc > best_acc:
            best_acc = acc
            torch.save({'model_state_dict': model.module.state_dict(),
                        'optim_state_dict': optimizer.state_dict(),
                        'epoch': ep, 'accuracy': acc, 'loss': loss.avg},
                       f'{log_dir}/state_{ep}_acc_{acc:.2f}_loss_{loss.avg:.2f}.pt')


if __name__ == '__main__':
    main()
