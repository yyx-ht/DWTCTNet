import torch
import torch.utils.data as data
import cv2
import numpy as np
import os
import scipy.misc as misc
from torch.utils.data import DataLoader

def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask


def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask


def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image = np.rot90(image)
        mask = np.rot90(mask)

    return image, mask

def read_own_data(root_path, mode):
    images = []
    masks = []
    image_root = os.path.join(root_path, mode + '/images')
    gt_root = os.path.join(root_path, mode + '/labels')
    for image_name in os.listdir(image_root):
        image_path = os.path.join(image_root, image_name)
        images.append(image_path)
    for label_name in os.listdir(gt_root):
        label_path = os.path.join(gt_root, label_name)
        masks.append(label_path)


    return images, masks


def own_data_loader(img_path, mask_path,size):
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, 0)
    img = cv2.resize(img, (size, size))
    mask = cv2.resize(mask, (size, size))
    img, mask = randomHorizontalFlip(img, mask)
    img, mask = randomVerticleFlip(img, mask)
    img, mask = randomRotate90(img, mask)
    mask = np.expand_dims(mask, axis=2)
    img = np.array(img, np.float32)/255.0
    mask = np.array(mask, np.float32) / 255.0
    mask[mask >= 0.1] = 1
    mask[mask < 0.1] = 0

    img = np.array(img, np.float32).transpose(2, 0, 1)
    mask = np.array(mask, np.float32).transpose(2, 0, 1)


    return img, mask


def own_data_test_loader(img_path, mask_path,size):
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, 0)
    img = cv2.resize(img, (size, size))
    mask = cv2.resize(mask, (size, size))
    mask = np.expand_dims(mask, axis=2)
    img = np.array(img, np.float32) / 255.0
    mask = np.array(mask, np.float32) / 255.0
    mask[mask >= 0.1] = 1
    mask[mask < 0.1] = 0
    img = np.array(img, np.float32).transpose(2, 0, 1)
    mask = np.array(mask, np.float32).transpose(2, 0, 1)

    return img, mask


class ImageFolder(data.Dataset):

    def __init__(self, root_path, mode,size):
        self.root = root_path
        self.mode = mode
        self.size =size
        self.images, self.labels = read_own_data(self.root, self.mode)



    def __getitem__(self, index):
        if self.mode == 'test1':
            img, mask = own_data_test_loader(self.images[index], self.labels[index],self.size)
        else:
            img, mask = own_data_loader(self.images[index], self.labels[index],self.size)
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        return img, mask

    def __len__(self):
        assert len(self.images) == len(self.labels), 'The number of images must be equal to labels'
        return len(self.images)






