import os
import glob
from torch.utils.data import ConcatDataset
from torchvision.datasets import ImageFolder


def kfood_split(root):
    train = open('/nfs_shared/kfood/meta/train.txt', 'w')
    test = open('/nfs_shared/kfood/meta/test.txt', 'w')
    superset = os.listdir(root)
    total=0
    n_train=0
    n_test=0
    for s in superset:
        for c in os.listdir(os.path.join(root, s)):
            l = os.listdir(os.path.join(root, s, c))
            cnt = len(l)
            total+=cnt
            t_cnt = int(cnt * 0.75)
            n_train+=t_cnt
            n_test+=cnt-t_cnt
            print(s, c, cnt, t_cnt,cnt-t_cnt,total,n_train,n_test)
            for n, f in enumerate(os.listdir(os.path.join(root, s, c))):
                if n < t_cnt:
                    train.write(f'{s}/{c}/{f}\n')
                else:
                    test.write(f'{s}/{c}/{f}\n')


if __name__ == '__main__':
    # kfood('/nfs_shared/kfood')
    root = '/nfs_shared/kfood/images'
    kfood_split(root)
    print(os.path.dirname(root),os.path.basename(root))
