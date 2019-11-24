from PIL import Image as loadImage
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

def getData():
    trainData = pd.read_csv("./train.csv")

    n = trainData.shape[0]
    # n = 42
    print(n)
    Fish = []
    Flower = []
    Gravel = []
    Sugar = []
    Image = []

    i = 0
    while i < n:
        if i % 10 == 0:
            print(i)
        filePath = trainData.iloc[i, 0]
        mask = trainData.iloc[i, 1]
        newMask = np.zeros((1400, 2100, 1))
        if pd.isnull(mask) == False:
            if i < n-1 and "_" not in trainData.iloc[i+1, 0]:
                mask = mask + trainData.iloc[i+1, 0]
                i += 1

            pixels = mask.split(" ")
            lastPixels = 0
            for j in range(1, len(pixels), 2):
                if len(pixels[j]) < lastPixels:
                    continue
                start = int(pixels[j-1])
                maskRange = int(pixels[j])
                for k in range(maskRange):
                    xIndex = (start + k - 1) % 1400
                    yIndex = (start + k - 1) // 1400
                    newMask[xIndex][yIndex][0] = 1
                lastPixels = len(pixels[j])

        # if "00dec6a.jpg_Sugar" in filePath:
        #     newImage = np.zeros((1400, 2100, 3))
        #     for x in range(1400):
        #         for y in range(2100):
        #             if newMask[x][y] == 1:
        #                 newImage[x][y][0], newImage[x][y][1], newImage[x][y][2] = 255, 255, 255
        #     cv2.imwrite("./{}.png".format(filePath), newImage)

        if "Fish" in filePath:
            fileName = filePath.split("_")[0]
            try:
                image = loadImage.open("./train_images/{}".format(fileName))
                # image = cv2.imread("./train_images/{}".format(fileName))
                Image.append(np.array(image))
                Fish.append(newMask)
            except:
                while i < n and ("_" not in trainData.iloc[i, 0] or fileName in trainData.iloc[i, 0]):
                    i += 1
                i -= 1

        elif "Flower" in filePath:
            Flower.append(newMask)
        elif "Gravel" in filePath:
            Gravel.append(newMask)
        elif "Sugar" in filePath:
            Sugar.append(newMask)

        i += 1

    # Image: 1400*2100*3 (0-255, uint8 注意不要溢出)
    # Fish, Flower, Gravel, Sugar: 1400*2100*1 (0为空, 1为mask)
    return np.array(Image), \
           np.array(Fish), \
           np.array(Flower), \
           np.array(Gravel), \
           np.array(Sugar)


# 配合DataLoader生成
class ImageDataset(Dataset):
    def __init__(self, Image, Mask, transform=None, target_transform=None):
        self.image = Image
        self.mask = Mask
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_x = self.image[index]
        img_y = self.mask[index]
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x.float(), img_y.float()

    def __len__(self):
        return len(self.mask)


if __name__ == "__main__":
    getData()