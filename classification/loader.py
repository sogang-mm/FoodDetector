from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.transforms import transforms as trn
import os


class TxtDataset(Dataset):
    def __init__(self, txt, labels, root, transform=None):
        self.root = root
        self.labels = labels
        self.l = self.open(txt)
        self.loader = default_loader

        self.transform = trn.Compose([
            trn.Resize((224, 224)),
            trn.ToTensor(),
            trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if transform is not None:
            self.transform = transform

    def __getitem__(self, idx):
        label, im = self.l[idx]
        path = os.path.join(self.root, label, im)
        image = self.transform(self.loader(path))

        return self.labels[label.lower().replace('_',' ')], path, image

    def open(self, txt):
        with open(txt, 'r') as f:
            l = [(i.strip() + '.jpg').split('/') for i in f.readlines()]

        return l
    def __len__(self):
        return len(self.l)


if __name__ == '__main__':
    labels = {i.strip().lower(): n for n, i in enumerate(open('/nfs_shared/food-101/meta/labels.txt', 'r').readlines())}
    print(labels)
    dt = TxtDataset('/nfs_shared/food-101/meta/train.txt', labels, '/nfs_shared/food-101/images')
    for l, p, i in dt:
        print(l, p)
