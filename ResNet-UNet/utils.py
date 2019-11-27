from PIL import Image
from torch.utils.data import Dataset
from skimage.transform import resize
from tqdm import tqdm
import skimage.io
import os
import pandas as pd
import numpy as np

def printMask(fileName, mask):
    fish = mask[:, :, 0]
    flower = mask[:, :, 1]
    gravel = mask[:, :, 2]
    sugar = mask[:, :, 3]

    skimage.io.imsave("{}_fish.jpg".format(fileName), fish)
    skimage.io.imsave("{}_flower.jpg".format(fileName), flower)
    skimage.io.imsave("{}_gravel.jpg".format(fileName), gravel)
    skimage.io.imsave("{}_sugar.jpg".format(fileName), sugar)


def printMask0(fileName, mask):
    fish = mask[0, :, :]
    flower = mask[1, :, :]
    gravel = mask[2, :, :]
    sugar = mask[3, :, :]

    skimage.io.imsave("{}_fish.jpg".format(fileName), fish)
    skimage.io.imsave("{}_flower.jpg".format(fileName), flower)
    skimage.io.imsave("{}_gravel.jpg".format(fileName), gravel)
    skimage.io.imsave("{}_sugar.jpg".format(fileName), sugar)


def saveMask():
    trainData = pd.read_csv("./data/train.csv")
    n = trainData.shape[0]
    num_n = n / 4
    split_n = int(4 * (num_n // 5) * 4)
    for i in tqdm(range(0, split_n, 4)):
        fileName = trainData.iloc[i, 0][:-9]

        img = Image.open("./data/train_images/{}.jpg".format(fileName))
        img = img.resize((480, 320), Image.ANTIALIAS)
        img.save("./data/modified_train_images/{}.jpg".format(fileName))

        # fish, flower, gravel, sugar
        for j in range(0, 4):
            maskData = trainData.iloc[i+j, 1]
            npMask = np.zeros(1400 * 2100).astype(np.uint8)
            if type(maskData) == type(""):
                masks = maskData.split(" ")
                for k in range(0, len(masks), 2):
                    start = int(masks[k]) - 1
                    length = int(masks[k+1])
                    for l in range(length):
                        npMask[start + l] = 255
            npMask = npMask.reshape((-1, 1400)).T
            npMask = Image.fromarray(npMask)
            npMask = npMask.resize((480, 320), Image.ANTIALIAS)
            npMask = np.array(npMask)
            if j == 0:
                fish = npMask
            if j == 1:
                flower = npMask
            if j == 2:
                gravel = npMask
            if j == 3:
                sugar = npMask
        resultMask = np.array([fish, flower, gravel, sugar])
        resultMask = resultMask.transpose(1, 2, 0)
        # printMask(fileName, resultMask)
        np.save("./data/train_mask/{}_mask.npy".format(fileName), resultMask)

    for i in tqdm(range(split_n, n, 4)):
        fileName = trainData.iloc[i, 0][:-9]

        img = Image.open("./data/train_images/{}.jpg".format(fileName))
        img = img.resize((480, 320), Image.ANTIALIAS)
        img.save("./data/modified_test_images/{}.jpg".format(fileName))

        # fish, flower, gravel, sugar
        for j in range(0, 4):
            maskData = trainData.iloc[i+j, 1]
            npMask = np.zeros(1400 * 2100).astype(np.uint8)
            if type(maskData) == type(""):
                masks = maskData.split(" ")
                for k in range(0, len(masks), 2):
                    start = int(masks[k]) - 1
                    length = int(masks[k+1])
                    for l in range(length):
                        npMask[start + l] = 255
            npMask = npMask.reshape((-1, 1400)).T
            npMask = Image.fromarray(npMask)
            npMask = npMask.resize((480, 320), Image.ANTIALIAS)
            npMask = np.array(npMask)
            if j == 0:
                fish = npMask
            if j == 1:
                flower = npMask
            if j == 2:
                gravel = npMask
            if j == 3:
                sugar = npMask
        resultMask = np.array([fish, flower, gravel, sugar])
        resultMask = resultMask.transpose(1, 2, 0)
        # printMask(fileName, resultMask)
        np.save("./data/test_mask/{}_mask.npy".format(fileName), resultMask)


# 配合DataLoader生成
class ImageDataset(Dataset):
    def __init__(self, imagePath, maskPath, transform=None, target_transform=None):
        imgs = self.__make_dataset(imagePath, maskPath)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __make_dataset(self, imagePath, maskPath):
        imgs = []
        for fileName in os.listdir(maskPath):
            img = os.path.join(imagePath, "{}.jpg".format(fileName[:-9]))
            mask = os.path.join(maskPath, fileName)
            imgName = fileName[:-9]
            imgs.append((img, mask, imgName))
        return imgs

    def __getitem__(self, index):
        img_path, mask_path, imgName = self.imgs[index]
        img = Image.open(img_path)
        mask = np.load(mask_path)
        labels = np.zeros(4)
        for i in range(4):
            if np.sum(mask[:, :, i]) > 0:
                labels[i] = 1
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return img.float(), mask.float(), imgName, labels

    def __len__(self):
        return len(self.imgs)

if __name__ == "__main__":
    saveMask()