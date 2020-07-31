from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.transforms import transforms as trn
import os
from autoaugment import ImageNetPolicy


class TXTDataset(Dataset):
    def __init__(self, txt, labels, root, transform=None):
        self.root = root
        self.labels = {i.strip(): n for n, i in enumerate(open(labels, 'r').readlines())}
        self.l = self.open(txt)
        self.loader = default_loader

        self.transform = trn.Compose([
            trn.Resize((224, 224)),
            trn.ToTensor(),
            trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if transform is not None:
            self.transform = transform

        self.target_transform = lambda labels, x: labels[x]

    def __getitem__(self, idx):
        label, im = self.l[idx]
        path = os.path.join(self.root, label, im)
        image = self.transform(self.loader(path))
        target = self.target_transform(self.labels, label)

        return target, path, image

    def open(self, txt):
        with open(txt, 'r') as f:
            l = [(os.path.dirname(i), os.path.basename(i.strip())) for i in f.readlines()]

        return l

    def get_labels(self):
        return list(self.labels.keys())


    def __len__(self):
        return len(self.l)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    # nf=open('/nfs_shared/food-101/meta/test2.txt', 'w')
    # with open('/nfs_shared/food-101/meta/test.txt', 'r') as f:
    #     l = [i.strip() + '.jpg\n' for i in f.readlines()]
    #     nf.writelines(l)
    # exit()

    dt = TXTDataset('/nfs_shared/food-101/meta/train2.txt', '/nfs_shared/food-101/meta/classes.txt',
                    '/nfs_shared/food-101/images')
    dt2 = TXTDataset('/nfs_shared/kfood/meta/test.txt', '/nfs_shared/kfood/meta/classes.txt',
                    '/nfs_shared/kfood/images')

    dt3 = TXTDataset('/nfs_shared/food/meta/test.txt', '/nfs_shared/food/meta/classes.txt',
                     '/nfs_shared/food/images')

    for l, p, i in DataLoader(dt3,batch_size=128,num_workers=4):
        print(l, p)
