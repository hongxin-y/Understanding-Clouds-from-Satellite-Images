import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from PIL import Image
from sklearn.metrics import accuracy_score, recall_score, f1_score, auc, roc_curve

def label_accuracy(path):
    data = pd.read_csv(path)
    threshold = [0.78, 0.69, 0.63, 0.42]
    cnt = 0
    for i in range(data.shape[0]):
        for j in range(1, 5):
            if (data.iloc[i, j] < threshold[j-1] and data.iloc[i, j+4] == 0) or \
                    (data.iloc[i, j] >= threshold[j-1] and data.iloc[i, j+4] == 1):
                cnt += 1

    print("Accuracy: {}".format(cnt / (4*data.shape[0])))
    return cnt / (4*data.shape[0])


def label_metrics(path):
    data = pd.read_csv(path)

    fish_thresold = 0.6
    flower_threshold = 0.56
    gravel_threshold = 0.62
    sugar_threshold = 0.4

    fish_pred = []
    fish_true = []
    flower_pred = []
    flower_true = []
    gravel_pred = []
    gravel_true = []
    sugar_pred = []
    sugar_true = []

    for i in range(data.shape[0]):
        if data.iloc[i, 1] < fish_thresold:
            fish_pred.append(0)
        else:
            fish_pred.append(1)
        fish_true.append(data.iloc[i, 5])

        if data.iloc[i, 2] < flower_threshold:
            flower_pred.append(0)
        else:
            flower_pred.append(1)
        flower_true.append(data.iloc[i, 6])

        if data.iloc[i, 3] < gravel_threshold:
            gravel_pred.append(0)
        else:
            gravel_pred.append(1)
        gravel_true.append(data.iloc[i, 7])

        if data.iloc[i, 4] < sugar_threshold:
            sugar_pred.append(0)
        else:
            sugar_pred.append(1)
        sugar_true.append(data.iloc[i, 8])


    print("Fish")
    fpr, tpr, _ = roc_curve(fish_true, fish_pred)
    print("AUC: {}".format(auc(fpr, tpr)))
    print("Accuracy: {}".format(accuracy_score(fish_true, fish_pred)))
    print("Recall: {}".format(recall_score(fish_true, fish_pred)))
    print("F1: {}".format(f1_score(fish_true, fish_pred)))

    print("Flower")
    fpr, tpr, _ = roc_curve(flower_true, flower_pred)
    print("AUC: {}".format(auc(fpr, tpr)))
    print("Accuracy: {}".format(accuracy_score(flower_true, flower_pred)))
    print("Recall: {}".format(recall_score(flower_true, flower_pred)))
    print("F1: {}".format(f1_score(flower_true, flower_pred)))

    print("Gravel")
    fpr, tpr, _ = roc_curve(gravel_true, gravel_pred)
    print("AUC: {}".format(auc(fpr, tpr)))
    print("Accuracy: {}".format(accuracy_score(gravel_true, gravel_pred)))
    print("Recall: {}".format(recall_score(gravel_true, gravel_pred)))
    print("F1: {}".format(f1_score(gravel_true, gravel_pred)))

    print("Sugar")
    fpr, tpr, _ = roc_curve(sugar_true, sugar_pred)
    print("AUC: {}".format(auc(fpr, tpr)))
    print("Accuracy: {}".format(accuracy_score(sugar_true, sugar_pred)))
    print("Recall: {}".format(recall_score(sugar_true, sugar_pred)))
    print("F1: {}".format(f1_score(sugar_true, sugar_pred)))


def draw_label_accuracy(path):
    data = pd.read_csv(path)
    fish = []
    flower = []
    gravel = []
    sugar = []


    thresholds = [i / 100 for i in range(10, 80, 1)]
    for threshold in tqdm(thresholds):
        cnt = 0
        for i in range(data.shape[0]):
            if (data.iloc[i, 1] < threshold and data.iloc[i, 1+4] == 0) or \
                    (data.iloc[i, 1] >= threshold and data.iloc[i, 1+4] == 1):
                cnt += 1

        fish.append(cnt / (data.shape[0]))


    for threshold in tqdm(thresholds):
        cnt = 0
        for i in range(data.shape[0]):
            if (data.iloc[i, 2] < threshold and data.iloc[i, 2 + 4] == 0) or \
                    (data.iloc[i, 2] >= threshold and data.iloc[i, 2 + 4] == 1):
                cnt += 1

        flower.append(cnt / (data.shape[0]))


    for threshold in tqdm(thresholds):
        cnt = 0
        for i in range(data.shape[0]):
            if (data.iloc[i, 3] < threshold and data.iloc[i, 3 + 4] == 0) or \
                    (data.iloc[i, 3] >= threshold and data.iloc[i, 3 + 4] == 1):
                cnt += 1

        gravel.append(cnt / (data.shape[0]))


    for threshold in tqdm(thresholds):
        cnt = 0
        for i in range(data.shape[0]):
            if (data.iloc[i, 4] < threshold and data.iloc[i, 4 + 4] == 0) or \
                    (data.iloc[i, 4] >= threshold and data.iloc[i, 4 + 4] == 1):
                cnt += 1

        sugar.append(cnt / (data.shape[0]))


    plt.subplot(2, 2, 1)
    plt.plot(thresholds, fish)
    plt.xlabel("Label Threshold")
    plt.ylabel("Classification Accuracy")
    plt.title("Fish")

    plt.subplot(2, 2, 2)
    plt.plot(thresholds, flower)
    plt.xlabel("Label Threshold")
    plt.ylabel("Classification Accuracy")
    plt.title("Flower")

    plt.subplot(2, 2, 3)
    plt.plot(thresholds, gravel)
    plt.xlabel("Label Threshold")
    plt.ylabel("Classification Accuracy")
    plt.title("Gravel")

    plt.subplot(2, 2, 4)
    plt.plot(thresholds, sugar)
    plt.xlabel("Label Threshold")
    plt.ylabel("Classification Accuracy")
    plt.title("Sugar")

    plt.subplots_adjust(wspace=0.5, hspace=0.6)
    plt.savefig("./Label_Thresholds_Accuracy.png")


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


def get_Dice0(output, target, mask_threshold, valid_output=True):
    img1 = np.array(Image.open(output))
    img2 = np.array(Image.open(target))

    img1_cnt = 0.001
    img2_cnt = 0.001
    img1_img2_cnt = 0.001

    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if valid_output and img1[i][j][0] > mask_threshold and img2[i][j][0] == 255:
                img1_img2_cnt += 1
                img1_cnt += 1
                img2_cnt += 1
            elif valid_output and img1[i][j][0] > mask_threshold:
                img1_cnt += 1
            elif img2[i][j][0] == 255:
                img2_cnt += 1

    Dice = 2 * img1_img2_cnt / (img1_cnt + img2_cnt)
    return Dice

def calculate_dice(path):
    # label_threshold = [0.78, 0.69, 0.63, 0.42]
    # mask_thresholds = [[60, 40, 50, 40]]
    label_threshold = [0.6, 0.56, 0.62, 0.4]
    mask_thresholds = [[80, 100, 30, 70]]
    for mask_threshold in mask_thresholds:
        sum_dice = 0
        sum_fish = 0
        sum_flower = 0
        sum_gravel = 0
        sum_sugar = 0

        label_data = pd.read_csv(path)
        for i in tqdm(range(label_data.shape[0])):
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

            # print("{}: File: {}, Avg Dice: {}".format(i, fileName, (dice_fish + dice_flower + dice_gravel + dice_sugar) / 4))

            sum_fish += dice_fish
            sum_flower += dice_flower
            sum_gravel += dice_gravel
            sum_sugar += dice_sugar
            sum_dice += (dice_fish + dice_flower + dice_gravel + dice_sugar)

        ave_fish = sum_fish / label_data.shape[0]
        ave_flower = sum_flower / label_data.shape[0]
        ave_gravel = sum_gravel / label_data.shape[0]
        ave_sugar = sum_sugar / label_data.shape[0]
        ave_dice = sum_dice / (label_data.shape[0] * 4)

        print("ave fish dice: {}".format(ave_fish))
        print("ave flower dice: {}".format(ave_flower))
        print("ave gravel dice: {}".format(ave_gravel))
        print("ave sugar dice: {}".format(ave_sugar))
        print("mask threshold:{}, Avg Dice: {}".format(mask_threshold, ave_dice))
    return


def find_threshold(path):
    # fish
    label_threshold = 0.78
    mask_thresholds = range(30, 220, 10)
    dice_sav =[]
    for mask_threshold in mask_thresholds:
        print(mask_threshold)

        sum_dice = 0

        label_data = pd.read_csv(path)
        for i in range(100):
            fileName = label_data.iloc[i, 0]

            if label_data.iloc[i, 1] > label_threshold:
                dice_fish = get_Dice("./data/output/{}_output_fish.jpg".format(fileName[:-4]),
                                     "./data/output/{}_truth_fish.jpg".format(fileName[:-4]), mask_threshold, True)
            else:
                dice_fish = get_Dice("./data/output/{}_output_fish.jpg".format(fileName[:-4]),
                                     "./data/output/{}_truth_fish.jpg".format(fileName[:-4]), mask_threshold, False)

            # print("{}: File: {}, Avg Dice: {}".format(i, fileName, (dice_fish + dice_flower + dice_gravel + dice_sugar) / 4))

            sum_dice += dice_fish

        ave_dice = sum_dice / (100)
        dice_sav.append(ave_dice)

    with open('0_78_fish_eval.plk', 'wb') as f:
        pickle.dump(dice_sav, f)

    plt.plot(mask_thresholds, dice_sav)
    plt.savefig("0_78_fish_eval.png")
    plt.clf()

    # flower
    label_threshold = 0.69
    mask_thresholds = range(30, 220, 10)
    dice_sav =[]
    for mask_threshold in mask_thresholds:
        print(mask_threshold)

        sum_dice = 0

        label_data = pd.read_csv(path)
        for i in range(100):
            fileName = label_data.iloc[i, 0]

            if label_data.iloc[i, 2] > label_threshold:
                dice_flower = get_Dice("./data/output/{}_output_flower.jpg".format(fileName[:-4]),
                                       "./data/output/{}_truth_flower.jpg".format(fileName[:-4]), mask_threshold, True)
            else:
                dice_flower = get_Dice("./data/output/{}_output_flower.jpg".format(fileName[:-4]),
                                       "./data/output/{}_truth_flower.jpg".format(fileName[:-4]), mask_threshold, False)

            # print("{}: File: {}, Avg Dice: {}".format(i, fileName, (dice_fish + dice_flower + dice_gravel + dice_sugar) / 4))

            sum_dice += dice_flower

        ave_dice = sum_dice / (100)
        dice_sav.append(ave_dice)

    with open('0.56_fish_flower.plk', 'wb') as f:
        pickle.dump(dice_sav, f)

    plt.plot(mask_thresholds, dice_sav)
    plt.savefig("0.56_fish_flower.png")
    plt.clf()

    # gravel
    label_threshold = 0.63
    mask_thresholds = range(30, 220, 10)
    dice_sav =[]
    for mask_threshold in mask_thresholds:
        print(mask_threshold)

        sum_dice = 0

        label_data = pd.read_csv(path)
        for i in range(100):
            fileName = label_data.iloc[i, 0]

            if label_data.iloc[i, 3] > label_threshold:
                dice_gravel = get_Dice("./data/output/{}_output_gravel.jpg".format(fileName[:-4]),
                                       "./data/output/{}_truth_gravel.jpg".format(fileName[:-4]), mask_threshold, True)
            else:
                dice_gravel = get_Dice("./data/output/{}_output_gravel.jpg".format(fileName[:-4]),
                                       "./data/output/{}_truth_gravel.jpg".format(fileName[:-4]), mask_threshold, False)

            # print("{}: File: {}, Avg Dice: {}".format(i, fileName, (dice_fish + dice_flower + dice_gravel + dice_sugar) / 4))

            sum_dice += dice_gravel

        ave_dice = sum_dice / (100)
        dice_sav.append(ave_dice)

    with open('0.55_gravel_eval.plk', 'wb') as f:
        pickle.dump(dice_sav, f)

    plt.plot(mask_thresholds, dice_sav)
    plt.savefig("0.55_gravel_eval.png")
    plt.clf()

    # sugar
    label_threshold = 0.42
    mask_thresholds = range(30, 220, 10)
    dice_sav =[]
    for mask_threshold in mask_thresholds:
        print(mask_threshold)

        sum_dice = 0

        label_data = pd.read_csv(path)
        for i in range(100):
            fileName = label_data.iloc[i, 0]

            if label_data.iloc[i, 4] > label_threshold:
                dice_sugar = get_Dice("./data/output/{}_output_sugar.jpg".format(fileName[:-4]),
                                      "./data/output/{}_truth_sugar.jpg".format(fileName[:-4]), mask_threshold, True)
            else:
                dice_sugar = get_Dice("./data/output/{}_output_sugar.jpg".format(fileName[:-4]),
                                      "./data/output/{}_truth_sugar.jpg".format(fileName[:-4]), mask_threshold, False)

            # print("{}: File: {}, Avg Dice: {}".format(i, fileName, (dice_fish + dice_flower + dice_gravel + dice_sugar) / 4))

            sum_dice += dice_sugar

        ave_dice = sum_dice / (100)
        dice_sav.append(ave_dice)

    with open('0.65_sugar_eval.plk', 'wb') as f:
        pickle.dump(dice_sav, f)

    plt.plot(mask_thresholds, dice_sav)
    plt.savefig("0.65_sugar_eval.png")
    plt.clf()


def draw_dice_threshold(fish, flower, gravel, sugar):
    with open(fish, 'rb') as f:
        fish_dice = pickle.load(f)
    with open(flower, 'rb') as f:
        flower_dice = pickle.load(f)
    with open(gravel, 'rb') as f:
        gravel_dice = pickle.load(f)
    with open(sugar, 'rb') as f:
        sugar_dice = pickle.load(f)

    mask_thresholds = [i/255 for i in range(30, 220, 10)]
    print(np.where(fish_dice == np.max(fish_dice)))
    print(np.where(flower_dice == np.max(flower_dice)))
    print(np.where(gravel_dice == np.max(gravel_dice)))
    print(np.where(sugar_dice == np.max(sugar_dice)))
    plt.subplot(2, 2, 1)
    plt.plot(mask_thresholds, fish_dice)
    plt.xlabel("Mask Threshold")
    plt.ylabel("Average Dice")
    plt.title("Fish")

    plt.subplot(2, 2, 2)
    plt.plot(mask_thresholds, flower_dice)
    plt.xlabel("Mask Threshold")
    plt.ylabel("Average Dice")
    plt.title("Flower")

    plt.subplot(2, 2, 3)
    plt.plot(mask_thresholds, gravel_dice)
    plt.xlabel("Mask Threshold")
    plt.ylabel("Average Dice")
    plt.title("Gravel")

    plt.subplot(2, 2, 4)
    plt.plot(mask_thresholds, sugar_dice)
    plt.xlabel("Mask Threshold")
    plt.ylabel("Average Dice")
    plt.title("Sugar")

    plt.subplots_adjust(wspace=0.5, hspace=0.6)
    plt.savefig("./Mask_Thresholds_Dice.png")




def draw_output_masks(imgName, label_threshold, mask_threshold):
    output_fish = np.array(Image.open("./data/output/{}_output_fish.jpg".format(imgName)))
    truth_fish = np.array(Image.open("./data/output/{}_truth_fish.jpg".format(imgName)))
    output_flower = np.array(Image.open("./data/output/{}_output_flower.jpg".format(imgName)))
    truth_flower = np.array(Image.open("./data/output/{}_truth_flower.jpg".format(imgName)))
    output_gravel = np.array(Image.open("./data/output/{}_output_gravel.jpg".format(imgName)))
    truth_gravel = np.array(Image.open("./data/output/{}_truth_gravel.jpg".format(imgName)))
    output_sugar = np.array(Image.open("./data/output/{}_output_sugar.jpg".format(imgName)))
    truth_sugar = np.array(Image.open("./data/output/{}_truth_sugar.jpg".format(imgName)))

    final_output_fish = np.zeros((320, 480))
    if max(output_fish.reshape((-1))) >= label_threshold[0]*255:
        for i in range(320):
            for j in range(480):
                if output_fish[i][j] >= mask_threshold[0]:
                    final_output_fish[i][j] = 255

    final_output_flower = np.zeros((320, 480))
    if max(output_flower.reshape((-1))) >= label_threshold[1]*255:
        for i in range(320):
            for j in range(480):
                if output_flower[i][j] >= mask_threshold[1]:
                    final_output_flower[i][j] = 255

    final_output_gravel = np.zeros((320, 480))
    if max(output_gravel.reshape((-1))) >= label_threshold[2]*255:
        for i in range(320):
            for j in range(480):
                if output_gravel[i][j] >= mask_threshold[2]:
                    final_output_gravel[i][j] = 255

    final_output_sugar = np.zeros((320, 480))
    if max(output_sugar.reshape((-1))) >= label_threshold[3]*255:
        for i in range(320):
            for j in range(480):
                if output_sugar[i][j] >= mask_threshold[3]:
                    final_output_sugar[i][j] = 255

    Image.fromarray(final_output_fish).convert('RGB').save("./data/report_output/{}_output_fish.jpg".format(imgName))
    Image.fromarray(final_output_flower).convert('RGB').save("./data/report_output/{}_output_flower.jpg".format(imgName))
    Image.fromarray(final_output_gravel).convert('RGB').save("./data/report_output/{}_output_gravel.jpg".format(imgName))
    Image.fromarray(final_output_sugar).convert('RGB').save("./data/report_output/{}_output_sugar.jpg".format(imgName))

    Image.fromarray(truth_fish).convert('RGB').save("./data/report_output/{}_truth_fish.jpg".format(imgName))
    Image.fromarray(truth_flower).convert('RGB').save("./data/report_output/{}_truth_flower.jpg".format(imgName))
    Image.fromarray(truth_gravel).convert('RGB').save("./data/report_output/{}_truth_gravel.jpg".format(imgName))
    Image.fromarray(truth_sugar).convert('RGB').save("./data/report_output/{}_truth_sugar.jpg".format(imgName))


if __name__ == "__main__":
    # label_accuracy("./data/output.csv")
    # label_metrics("./data/output.csv")
    # draw_label_accuracy("./data/output.csv")
    calculate_dice("./data/output.csv")
    # find_threshold("./data/output.csv")
    # draw_dice_threshold("./pic_18_final/0_6_fish_eval.plk",
    #                     "./pic_18_final/0.56_fish_flower.plk",
    #                     "./pic_18_final/0.55_gravel_eval.plk",
    #                     "./pic_18_final/0.65_sugar_eval.plk")

    #
    # imgName = "d412278"
    # draw_output_masks(imgName,  [0.78, 0.69, 0.63, 0.42], [60, 40, 50, 40])
    # print(get_Dice0("./data/report_output/{}_output_flower.jpg".format(imgName),
    #          "./data/report_output/{}_truth_flower.jpg".format(imgName), 127, valid_output=True))