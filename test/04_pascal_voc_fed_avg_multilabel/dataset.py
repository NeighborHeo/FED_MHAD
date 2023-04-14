# %%
import argparse
import numpy as np
import pathlib
from typing import Dict, Any, Tuple
import torch
from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import unittest
import os
import inspect

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

transformations_train = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.RandomChoice([
                                    transforms.ColorJitter(brightness=(0.80, 1.20)),
                                    transforms.RandomGrayscale(p = 0.25)
                                    ]),
                                transforms.RandomHorizontalFlip(p = 0.25),
                                transforms.RandomRotation(25),
                                transforms.ToTensor(),
                            ])

transformations_valid = transforms.Compose([transforms.ToPILImage(),
                                        transforms.CenterCrop(224), 
                                        transforms.ToTensor(), 
                                        transforms.Normalize(mean = mean, std = std),
                                        ])


class mydataset(torch.utils.data.Dataset):
    def __init__(self, imgs, labels, train=False, verbose=False, transforms=None):
        self.img = imgs
        self.gt = labels
        self.train = train
        self.verbose = verbose
        self.aug = False
        self.transforms = transforms
        return
    def __len__(self):
        return len(self.img)
    def __getitem__(self, idx):
        img = self.img[idx]
        gt = self.gt[idx]
        # print(img.shape) # 3, 224, 224
        if self.transforms:
            img = self.transforms(img)
        idx = torch.tensor(0)
        # return img, gt, idx
        return img, gt
    def get_labels(self):
        return self.gt
    
class PascalVocPartition:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.priv_data = {}
        # self.path = pathlib.Path(args.datapath).joinpath('PASCAL_VOC_2012', f'N_clients_{args.N_parties}_alpha_{args.alpha:.1f}')
        self.path = pathlib.Path.home().joinpath('.data', 'PASCAL_VOC_2012', f'N_clients_{args.N_parties}_alpha_{args.alpha:.1f}')
    
    def load_partition(self, i: int):
        path = pathlib.Path.home().joinpath('.data', 'PASCAL_VOC_2012')
        if i == -1:
            party_img = np.load(path.joinpath('PASCAL_VOC_train_224_Img.npy'))
            party_label = np.load(path.joinpath('PASCAL_VOC_train_224_Label.npy'))
            party_img, party_label = self.filter_images_by_label_type(self.args.task, party_img, party_label)
            train_dataset = mydataset(party_img, party_label)
            
            test_imgs = np.load(path.joinpath('PASCAL_VOC_val_224_Img.npy'))
            test_labels = np.load(path.joinpath('PASCAL_VOC_val_224_Label.npy'))
            test_imgs, test_labels = self.filter_images_by_label_type(self.args.task, test_imgs, test_labels)
            test_partition = mydataset(test_imgs, test_labels)
        else :
            party_img = np.load(self.path.joinpath(f'Party_{i}_X_data.npy'))
            party_label = np.load(self.path.joinpath(f'Party_{i}_y_data.npy'))
            party_img, party_label = self.filter_images_by_label_type(self.args.task, party_img, party_label)
            train_dataset = mydataset(party_img, party_label)
            
            test_imgs = np.load(path.joinpath('PASCAL_VOC_val_224_Img.npy'))
            test_labels = np.load(path.joinpath('PASCAL_VOC_val_224_Label.npy'))
            test_imgs, test_labels = self.filter_images_by_label_type(self.args.task, test_imgs, test_labels)
            n_test = int(test_imgs.shape[0] / self.args.N_parties)
            test_dataset = mydataset(test_imgs, test_labels)
            test_partition = torch.utils.data.Subset(test_dataset, range(i * n_test, (i + 1) * n_test))

        print(f"client {i}_size of train partition: ", len(train_dataset), "images / ", "test partition: ", len(test_partition), "images")
        return train_dataset, test_partition
    
    def filter_images_by_label_type(self, task: str, imgs: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        print(f"filtering images by label type: {task}")
        if task == 'singlelabel':
            sum_labels = np.sum(labels, axis=1)
            index = np.where(sum_labels == 1)
            labels = labels[index]
            labels = np.argmax(labels, axis=1)
            imgs = imgs[index]
        elif task == 'multilabel_only':
            sum_labels = np.sum(labels, axis=1)
            index = np.where(sum_labels > 1)
            labels = labels[index]
            imgs = imgs[index]
        elif task == 'multilabel':
            pass
        return imgs, labels

class Test_PascalVocPartition(unittest.TestCase):
    def test_load_partition(self):
        args = argparse.Namespace()
        args.datapath = '~/.data'
        args.N_parties = 5
        args.alpha = 1.0
        args.task = 'multilabel'
        args.batch_size = 16
        print(f"{os.path.basename(__file__)}:{inspect.currentframe().f_lineno}")
        pascal = PascalVocPartition(args)
        train_dataset, test_parition = pascal.load_partition(0)
        train_dataset, test_parition = pascal.load_partition(1)
        train_dataset, test_parition = pascal.load_partition(2)
        train_dataset, test_parition = pascal.load_partition(3)
        train_dataset, test_parition = pascal.load_partition(4)
        train_dataset, test_parition = pascal.load_partition(-1)
        # self.assertEqual(len(train_dataset), 1000)
        # self.assertEqual(len(test_parition), 100)
        valLoader = DataLoader(test_parition, batch_size=args.batch_size)
        img, label = next(iter(valLoader))
        print(label.shape)
        print(label)
    


if __name__ == '__main__':
    unittest.main()
# %%
