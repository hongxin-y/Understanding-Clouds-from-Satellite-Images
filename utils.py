import cv2
import pandas as pd
import numpy as np


def getData():
    trainData = pd.read_csv("./train.csv")

    n = trainData.shape[0]
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
        if pd.isnull(mask):
            i += 1
            continue
        if i < n-1 and "_" not in trainData.iloc[i+1, 0]:
            mask = mask + trainData.iloc[i+1, 0]
            i += 1

        newMask = np.zeros((1400, 2100))
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
                newMask[xIndex][yIndex] = 1
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
            image = cv2.imread("./train_images/{}".format(fileName))
            Image.append(image)
            Fish.append(newMask)
        if "Flower" in filePath:
            Flower.append(newMask)
        if "Gravel" in filePath:
            Gravel.append(newMask)
        if "Sugar" in filePath:
            Sugar.append(newMask)

        i += 1

    # Image: 1400*2100*3 (0-255, uint8 注意不要溢出)
    # Fish, Flower, Gravel, Sugar: 1400*2100 (0为空, 1为mask)
    return Image, Fish, Flower, Gravel, Sugar


if __name__ == "__main__":
    getData()