import glob
import random
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transform
import torchvision.transforms as standard_transforms


class ImageDataset(Dataset):
    def __init__(self, root_path, transform=None, train=True, patch=False,flip=False):
        self.root_path = root_path

        self.transform = transform


        # if train:
        #     self.img_dir = os.path.join(self.root_path,'train/img')
        #     self.gt_dir = os.path.join(self.root_path,'train/ground_truth')
        # else:
        #     self.img_dir = os.path.join(self.root_path,'test/img')
        #     self.gt_dir = os.path.join(self.root_path,'test/ground_truth')
        # self.img_paths=[]
        # for img_path in glob.glob(os.path.join(self.img_dir, '*.tif')):
        #     self.img_paths.append(img_path)
        self.img_paths=root_path

        self.train = train
        self.patch = patch
        self.flip = flip
        self.nSamples = len(self.img_paths)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        img_path = self.img_paths[index] # select img
        # gt_path = img_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_') # create gt path
        gt_path = img_path.replace('images', 'ground_truth')
        gt_count = int(gt_path.split('_')[4][1:])

        img = cv2.imread(img_path,1)
        gt = cv2.imread(gt_path,1)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        gt = Image.fromarray(cv2.cvtColor(gt, cv2.COLOR_BGR2RGB))
        if self.transform is not None:
            img = self.transform(img) # tensor c,h,w
            gt = transform.ToTensor()(gt)
        if self.train and self.patch:
            img, gt = random_crop(img, gt)

        img = torch.Tensor(img)
        gt = torch.Tensor(gt)
        target = {}

        target['gt']=gt
        target['count']=gt_count


        return img,gt, gt_count

from sklearn.model_selection import KFold,StratifiedKFold
def loading_data(data_root):
    # the pre-proccssing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.387, 0.387, 0.387],
                                    std=[0.278, 0.278, 0.278]),
    ])
    img_paths=[]
    img_dir1 = os.path.join(data_root, 'BBBC005_v1_images')
    path_set = [img_dir1]
    for path in path_set:
        for img_path in glob.glob(os.path.join(path, '*.tif')):
            img_paths.append(img_path)

    kf = KFold(n_splits=5)

    print(kf)
    img_train=[]
    img_test=[]
    Train_idx_set=[]
    Test_idx_set =[]
    # 做split时只需传入数据，不需要传入标签
    for train_index, test_index in kf.split(img_paths):
        Train_idx_set.append(train_index)
        Test_idx_set.append(test_index)
    print("TRAIN:", Train_idx_set[0], "TEST:", Test_idx_set[0])
    for i in Train_idx_set[0]:
            img_train.append(img_paths[i])
    for j in Test_idx_set[0]:
            img_test.append(img_paths[j])





    # create the training dataset
    train_set = ImageDataset(img_train, train=True, transform=transform, patch=False, flip=False)
    # create the validation dataset
    val_set = ImageDataset(img_test, train=False, transform=transform)

    return train_set, val_set


# random crop augumentation
def random_crop(img,dmap, num_patch=4):
    half_h = 128
    half_w = 128
    img_size1 = img.shape[0]
    result_img = np.zeros([num_patch, img.shape[0], half_h, half_w])
    result_dmap = np.zeros([num_patch, dmap.shape[0], half_h, half_w])
    # crop num_patch for each image
    for i in range(num_patch):
        start_h = random.randint(0, img.size(1) - half_h)
        start_w = random.randint(0, img.size(2) - half_w)
        end_h = start_h + half_h
        end_w = start_w + half_w
        # copy the cropped rect
        result_img[i] = img[:, start_h:end_h, start_w:end_w]
        result_dmap[i] = dmap[:, start_h:end_h, start_w:end_w]
    return result_img, result_dmap

if __name__ == '__main__':
    import itertools
    import time
    import datetime
    import sys
    import matplotlib.pyplot as plt

    import torchvision.transforms as transforms
    from torchvision.utils import save_image
    import torch

    from torch.utils.data import DataLoader
    from torchvision import datasets
    from torch.autograd import Variable
    from util.misc import *

    dataset_name =r'D:/cc/pix2pix/dataset/BBBC005'
    # Configure dataloaders
    train_set, val_set = loading_data(dataset_name)
    sampler_train = torch.utils.data.RandomSampler(train_set)
    sampler_val = torch.utils.data.SequentialSampler(val_set)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, 1, drop_last=True)

    data_loader_train = DataLoader(train_set, batch_sampler=batch_sampler_train,
                                   collate_fn=collate_fn_bc05, num_workers=0)

    data_loader_val = DataLoader(val_set, 1, sampler=sampler_val,
                                    drop_last=False, collate_fn=collate_fn_bc05, num_workers=0)

    for image, label,count in data_loader_train:
        image = image[0]
        density = label[0]
        count = count[0]
        print(image.size())
        print(density.size())
        print(count)

        img = np.transpose(image.numpy().squeeze(), [1, 2, 0]) * 0.2 + 0.45
        density = np.transpose(density.numpy().squeeze(), [1, 2, 0]) * 0.2 + 0.45
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.subplot(1, 2, 2)
        plt.imshow(density, cmap='jet')
        plt.show()
        break
