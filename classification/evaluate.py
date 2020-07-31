from classification.loader import TXTDataset
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from torchvision.transforms import transforms as trn
from classification.model import Resnet50, Resnet152, Mobilenet_v2, Densenet_121
import json
from tqdm import tqdm
import warnings
from classification.Measure import AverageMeter

warnings.filterwarnings("ignore")
if __name__ == '__main__':
    ckpt = torch.load('/hdd/ms/food_run/records/food/mobilenet/state_95_acc_0.79_loss_0.75.pt')
    print(ckpt['accuracy'], ckpt['loss'])

    root = '/nfs_shared/food/images'
    classes = '/nfs_shared/food/meta/classes.txt'
    transform = trn.Compose([trn.Resize(256),
                             trn.CenterCrop(224),
                             trn.ToTensor(),
                             trn.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])])

    loader = DataLoader(TXTDataset('/nfs_shared/food/meta/test.txt', classes, root, transform=transform),
                        batch_size=128, num_workers=4, shuffle=False)

    label_enc = loader.dataset.get_labels()

    model = Mobilenet_v2().cuda()
    model.load_state_dict(ckpt['model_state_dict'])
    model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax()

    result = dict()
    result['model'] = 'Mobilenet_v2'
    result['ckpt'] = '/hdd/ms/food_run/records/food/mobilenet/state_95_acc_0.79_loss_0.75.pt'
    result['precision'] = ckpt['accuracy']
    result['top_1'] = .0
    result['top_5'] = .0
    result['loss'] = ckpt['loss'].item()
    result['label'] = label_enc
    result['result'] = []
    pbar = tqdm(loader, ncols=150)
    model.eval()
    correct_top5 = AverageMeter()
    correct_top1 = AverageMeter()
    with torch.no_grad():
        for i, (label, path, image) in enumerate(loader, start=1):
            outputs = model(image.cuda())
            outputs = softmax(outputs)
            loss = criterion(outputs, label.cuda())

            prob, indice = torch.topk(outputs.cpu(), k=5)
            correct_top1.update(torch.sum(indice[:, 0:1] == label.reshape(-1, 1)).item())
            correct_top5.update(torch.sum(indice == label.reshape(-1, 1)).item())
            pbar.set_description(
                f'top_1 : {correct_top1.val / loader.batch_size:.4f}({correct_top1.sum / (i * loader.batch_size):.4f}), '
                f'top_5 : {correct_top5.val / loader.batch_size:.4f}({correct_top5.sum / (i * loader.batch_size):.4f})')

            labels = [[label_enc[i] for i in l] for l in indice]
            result['result'].extend([{'image': i[0],
                                      'gt_index': int(i[1]),
                                      'gt': label_enc[i[1]],
                                      'predict': i[4],
                                      'probability': [float(j) for j in i[2]],
                                      'predict_index': [int(j) for j in i[3]]
                                      } for i in zip(path, label.numpy(), prob.numpy(), indice.numpy(), labels)])

            pbar.update()
        result['top_1'] = correct_top1.sum / loader.dataset.__len__()
        result['top_5'] = correct_top5.sum / loader.dataset.__len__()

    json.dump(result, open('result.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
