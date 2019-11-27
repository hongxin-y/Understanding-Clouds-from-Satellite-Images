import pandas as pd
import numpy as np
from PIL import Image

def label_accuracy(path):
    data = pd.read_csv(path)
    threshold = [0.60, 0.67, 0.63, 0.45]
    cnt = 0
    for i in range(data.shape[0]):
        for j in range(1, 5):
            if (data.iloc[i, j] < threshold[j-1] and data.iloc[i, j+4] == 0) or \
                    (data.iloc[i, j] >= threshold[j-1] and data.iloc[i, j+4] == 1):
                cnt += 1

    print("Accuracy: {}".format(cnt / (4*data.shape[0])))
    return cnt / (4*data.shape[0])


def get_Dice(output, target, mask_threshold, valid_output=True):
    img1 = np.array(Image.open(output))
    img2 = np.array(Image.open(target))
    
    img1_cnt = 0.001
    img2_cnt = 0.001
    img1_img2_cnt = 0.001

    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if valid_output and img1[i][j] > mask_threshold and img2[i][j] == 255:
                img1_img2_cnt += 1
                img1_cnt += 1
                img2_cnt += 1
            elif valid_output and img1[i][j] > mask_threshold:
                img1_cnt += 1
            elif img2[i][j] == 255:
                img2_cnt += 1

    Dice = 2*img1_img2_cnt / (img1_cnt + img2_cnt)
    return Dice



def calculate_dice(path):
    label_threshold = [0.60, 0.67, 0.63, 0.45]
    mask_threshold = [102, 102, 102, 102]
    sum_dice = 0

    label_data = pd.read_csv(path)
    for i in range(label_data.shape[0]):
        fileName = label_data.iloc[i, 0]

        if label_data.iloc[i, 1] > label_threshold[0]:
            dice_fish = get_Dice("./data/output/{}_output_fish.jpg".format(fileName[:-4]),
                                 "./data/output/{}_truth_fish.jpg".format(fileName[:-4]), mask_threshold[0], True)
        else:
            dice_fish = get_Dice("./data/output/{}_output_fish.jpg".format(fileName[:-4]),
                                 "./data/output/{}_truth_fish.jpg".format(fileName[:-4]), mask_threshold[0], False)

        if label_data.iloc[i, 2] > label_threshold[1]:
            dice_flower = get_Dice("./data/output/{}_output_flower.jpg".format(fileName[:-4]),
                                   "./data/output/{}_truth_flower.jpg".format(fileName[:-4]), mask_threshold[1], True)
        else:
            dice_flower = get_Dice("./data/output/{}_output_flower.jpg".format(fileName[:-4]),
                                   "./data/output/{}_truth_flower.jpg".format(fileName[:-4]), mask_threshold[1], False)
        if label_data.iloc[i, 3] > label_threshold[2]:
            dice_gravel = get_Dice("./data/output/{}_output_gravel.jpg".format(fileName[:-4]),
                                   "./data/output/{}_truth_gravel.jpg".format(fileName[:-4]), mask_threshold[2], True)
        else:
            dice_gravel = get_Dice("./data/output/{}_output_gravel.jpg".format(fileName[:-4]),
                                   "./data/output/{}_truth_gravel.jpg".format(fileName[:-4]), mask_threshold[2], False)

        if label_data.iloc[i, 4] > label_threshold[3]:
            dice_sugar = get_Dice("./data/output/{}_output_sugar.jpg".format(fileName[:-4]),
                                  "./data/output/{}_truth_sugar.jpg".format(fileName[:-4]), mask_threshold[3], True)
        else:
            dice_sugar = get_Dice("./data/output/{}_output_sugar.jpg".format(fileName[:-4]),
                                  "./data/output/{}_truth_sugar.jpg".format(fileName[:-4]), mask_threshold[3], False)

        print("{}: File: {}, Avg Dice: {}".format(i, fileName, (dice_fish + dice_flower + dice_gravel + dice_sugar) / 4))

        sum_dice += (dice_fish + dice_flower + dice_gravel + dice_sugar)

    ave_dice = sum_dice / (label_data.shape[0] * 4)

    print("Avg Dice: {}".format(ave_dice))
    return ave_dice



if __name__ == "__main__":
    # label_accuracy("./data/output.csv")
    calculate_dice("./data/output.csv")