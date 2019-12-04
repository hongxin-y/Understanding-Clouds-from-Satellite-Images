import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import keras
import cv2
import skimage.io
from PIL import Image
from keras.applications.xception import Xception
from keras import backend as K
from keras import layers, optimizers
from keras.models import Sequential, Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K
from keras.models import load_model
import os

from utils import mask2rle, dice_coef, rle2mask

IMG_PATH = './train_images/'
LABELS = ['fish', 'flower', 'gravel', 'sugar']
IMG_LIST = os.listdir(IMG_PATH)

def read_data(file_path):
    # Read Label
    df = pd.read_csv(file_path)
    df['Image'] = df['Image_Label'].map(lambda x: x.split('.')[0])
    df['Label'] = df['Image_Label'].map(lambda x: x.split('_')[1])
    data_df = pd.DataFrame({'Image': df['Image'][::4]})
    data_df['rle_fish'] = df['EncodedPixels'][::4].values
    data_df['rle_flower'] = df['EncodedPixels'][1::4].values
    data_df['rle_gravel'] = df['EncodedPixels'][2::4].values
    data_df['rle_sugar'] = df['EncodedPixels'][3::4].values
    data_df.set_index('Image', inplace=True, drop=True)
    data_df.fillna('', inplace=True);
    data_df[['is_fish', 'is_flower', 'is_gravel', 'is_sugar']] = (data_df[['rle_fish', 'rle_flower', 'rle_gravel', 'rle_sugar']] != '').astype('int8')
    return data_df

class DataGenerator(keras.utils.Sequence):
    def __init__(self, id_list, batch_size = 8, width=512, height=352, shuffle=False, mode = "train", path=IMG_PATH, flips=False):
        self.id_list = id_list
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.shuffle = shuffle
        self.mode = mode
        self.path = path
        self.flips = flips
        self.on_epoch_end()
        
    def __len__(self):
        return (len(self.id_list) - 1) // self.batch_size + 1

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.id_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)
        return 
    
    def __data_generation(self, indexes):
        length = len(indexes)
        X = np.zeros((length, self.height, self.width, 3), dtype = np.float32) # 3 is # of channels, RGB
        y = np.zeros((length, 4), dtype = np.int8)
        for k in range(length):
            img = cv2.imread(self.path + self.id_list[indexes[k]] + '.jpg')
            img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)

            # data augmentation
            horizontal_flip = False
            vertical_flip = False
            if (self.flips):
                if np.random.uniform(0, 1) > 0.5: horizontal_flip = True
                if np.random.uniform(0, 1) > 0.5: vertical_flip = True
            if horizontal_flip: img = cv2.flip(img, 0)
            if vertical_flip: img = cv2.flip(img, 1)

            # normalize X and get y
            X[k] = img / 128. - 1.
            if self.mode == 'train' or self.mode == 'validate':
                y[k] = data_df.loc[self.id_list[indexes[k]], ['is_fish', 'is_flower', 'is_gravel', 'is_sugar']].values
        return X, y

    def __getitem__(self, k):
        batch_indexes = self.indexes[k*self.batch_size:(k+1)*self.batch_size]
        X, y = self.__data_generation(batch_indexes)
        if self.mode == 'train' or self.mode == 'validate':
            return X, y
        return X

def evaluation_class(model, test_df, threshold = 0.5, mode = "testing"):
    # prediction
    pred_gen = DataGenerator(test_df.index, mode='predict')
    predicitions = model.predict_generator(pred_gen, verbose=2)
    test_df.loc[:,'pred_fish'],  test_df.loc[:,'pred_flower'], test_df.loc[:,'pred_gravel'], test_df.loc[:,'pred_sugar'] = 0, 0, 0, 0
    test_df[['pred_fish', 'pred_flower', 'pred_gravel', 'pred_sugar']] = predicitions

    th = threshold
    log = "Evaluation of " + mode + " data:\n"
    for label in LABELS:
        log += label + ": "
        auc = roc_auc_score(test_df["is_" + label].values, test_df["pred_" + label].values)
        acc = accuracy_score(test_df["is_" + label].values, (test_df["pred_" + label].values > th).astype(int))
        f1 = f1_score(test_df["is_" + label].values, (test_df["pred_" + label].values > th).astype(int))
        log += "AUC = " + str(np.round(auc, 3)) + ", ACC = " + str(np.round(acc, 3)) + ", f1 = " + str(np.round(f1, 3)) + "\n"

    log += "Overall: "
    auc = roc_auc_score(test_df[['is_fish', 'is_flower', 'is_gravel', 'is_sugar']].values.reshape(-1),\
        test_df[['pred_fish', 'pred_flower', 'pred_gravel', 'pred_sugar']].values.reshape(-1))
    acc = accuracy_score(test_df[['is_fish', 'is_flower', 'is_gravel', 'is_sugar']].values.reshape(-1),\
        (test_df[['pred_fish', 'pred_flower', 'pred_gravel', 'pred_sugar']].values > th).astype(int).reshape((-1)))
    f1 = f1_score(test_df[['is_fish', 'is_flower', 'is_gravel', 'is_sugar']].values.reshape(-1),\
        (test_df[['pred_fish', 'pred_flower', 'pred_gravel', 'pred_sugar']].values > th).astype(int).reshape((-1)))
    log += "AUC = " + str(np.round(auc, 3)) + ", ACC = " + str(np.round(acc, 3)) + ", f1 = " + str(np.round(f1, 3)) + "\n"
    return log

def evaluation_segmentation(test_df, thresholds = [0.8,0.5,0.7,0.7], mode = "testing"):
    log = "Evaluation of " + mode + " data:\n"
    for k, label in enumerate(LABELS):
        test_df['score_'+ label] = test_df.apply(lambda x:dice_coef(x["rle_" + label],x["pred_rle_" + label],x["pred_vec_" + label],thresholds[k-1]), axis=1)
        dice = test_df['score_'+ label].mean()
        log += label + ": Kaggle Dice =" + str(np.around(dice, 3)) + "\n"
    dice = np.mean( test_df[['score_fish','score_flower','score_gravel','score_sugar']].values)
    log += "Overall : Kaggle Dice =" + str(np.around(dice, 3)) + "\n"
    return log

def generate_segmentation(cam, weights, test_df):
    test_df.loc[:,'pred_rle_fish'],  test_df.loc[:,'pred_rle_flower'], test_df.loc[:,'pred_rle_gravel'], test_df.loc[:,'pred_rle_sugar'] = "", "", "", ""
    test_df.loc[:,'pred_vec_fish'],  test_df.loc[:,'pred_vec_flower'], test_df.loc[:,'pred_vec_gravel'], test_df.loc[:,'pred_vec_sugar'] = 0, 0, 0, 0

    for i, idx in enumerate(test_df.index.values):
        img_path = IMG_PATH + idx + '.jpg'
        # calculate 4 masks
        for k, label in enumerate(LABELS):
            output, pred, _ = get_rle_probs(cam, weights, img_path, label_idx = k)
            test_df.loc[idx, "pred_rle_" + label] = mask2rle((output > 0.3).astype(int))
            test_df.loc[idx, "pred_vec_" + label] = pred
    return test_df

def get_rle_probs(cam, weights, image_file, label_idx = None):
    img = cv2.resize(cv2.imread(image_file), (512, 352))
    x = np.array(img)[None,:,:] / 128. - 1.#np.expand_dims(img, axis=0) / 128. - 1.
    global_pooling_output, pred_vec = cam.predict(x)
    global_pooling_output = np.squeeze(global_pooling_output)

    if label_idx == None: label_idx = np.argmax(pred_vec)

    channels_weights = weights[:, label_idx]
    output = np.dot(global_pooling_output.reshape((16 * 11, 2048)), channels_weights).reshape(11, 16)
    output = scipy.ndimage.zoom(output, (32, 32), order=1)

    # map pixels into [0,1]
    mx = np.round(np.max(output), 1)
    mn = np.round(np.min(output), 1)
    output = (output - mn) / (mx - mn)
    output = cv2.resize(output, (525, 350))

    return output, pred_vec[0, label_idx], label_idx

def save_segmentation(cam, weights, num, path, thresholds = [0.8,0.5,0.7,0.7]):
    th = thresholds
    for k in np.random.randint(0, len(IMG_LIST), num):
        while IMG_LIST[k].split(".")[0] not in data_df.index: k = np.random.randint(0, len(IMG_LIST))
        img = cv2.resize(cv2.imread(IMG_PATH + IMG_LIST[k]), (512, 352))
        mask_pred, probs, label_idx = get_rle_probs(cam, weights, IMG_PATH + IMG_LIST[k], label_idx = None)
        label = LABELS[label_idx]
        rle_true = data_df.loc[IMG_LIST[k].split('.')[0], "rle_" + label]
        rle_pred = mask2rle((mask_pred > th[label_idx]).astype(int))
        mask_true = rle2mask(rle_true)[::4,::4]

        # draw picture
        # plt.imshow(img, alpha=0.5)
        # plt.imshow(mask_true, alpha=0.5)
        # plt.imshow(mask_pred, cmap='jet', alpha=0.5)
        skimage.io.imsave(path + IMG_LIST[k] + "_pred_" + label + ".jpg", (mask_pred > th[label_idx]).astype("uint8")*100)
        skimage.io.imsave(path + IMG_LIST[k] + "_heat_" + label + ".jpg", (mask_pred*100).astype("uint8"))
        skimage.io.imsave(path + IMG_LIST[k] + "_true_" + label + ".jpg", mask_true*100)
        # skimage.io.imsave(path + IMG_LIST[k] + "_orig_" + label + ".jpg", mask_true)
        dice = dice_coef(rle_true, rle_pred, probs, th[label_idx])
        print("Dice = " + str(np.round(dice,3)))
        # plt.savefig(path + IMG_LIST[k] + "_" + label + ".jpg")

if __name__ == "__main__":
    data_df = read_data('train.csv')

    # split training and test data
    train_df, test_df = train_test_split(data_df, random_state=42, test_size=0.1)

    # split validation and training data
    train_df, validate_df = train_test_split(train_df, random_state=42, test_size=0.2)
    train_idx, validate_idx = train_df.index, validate_df.index
    train_gen = DataGenerator(train_idx, flips=True, shuffle=True)
    val_gen = DataGenerator(validate_idx, mode='validate')
    print("Data Generation Done")

    
    # Xception pre-train model
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(None, None, 3))

    # freeze non-batchnorm layers
    for layer in base_model.layers:
        if not isinstance(layer, layers.BatchNormalization): layer.trainable = False

    # add global average pooling layer
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(4, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=x)

    # compile and use BCE loss, lr = 0.001
    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=0.001), metrics=['accuracy'])
    print('Model Building Done')

    # train
    model.fit_generator(train_gen, epochs=2, verbose=2, validation_data=val_gen, steps_per_epoch = None)
    # unfroze the layers and train with lr = 0.0001
    for layer in model.layers: 
        layer.trainable = True
    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=0.0001), metrics=['accuracy'])
    model.fit_generator(train_gen, epochs=2, verbose=2, validation_data=val_gen, steps_per_epoch = None)
    print("Training Done")

    model.save('model.h5')
    
    # model = load_model("model.h5")
    # evaluation_class
    class_th = 0.5
    log = evaluation_class(model, train_df, threshold = class_th, mode = "training")
    log += evaluation_class(model, validate_df, threshold = class_th, mode = "validate")
    log += evaluation_class(model, test_df, threshold = class_th, mode = "testing")
    print(log)

    # generate sigmentation figure
    # a new model to generate segmentation figure
    weights = model.layers[-1].get_weights()[0]
    cam = Model(inputs=model.input, outputs=(model.layers[-3].output, model.layers[-1].output))
    train_df = generate_segmentation(cam, weights, train_df)
    validate_df = generate_segmentation(cam, weights, validate_df)
    test_df = generate_segmentation(cam, weights, test_df)

    # evaluation final result
    segmentation_ths = [0.8,0.5,0.7,0.7]
    log = evaluation_segmentation(train_df, thresholds = segmentation_ths, mode = "training")
    log += evaluation_segmentation(validate_df, thresholds = segmentation_ths, mode = "validate")
    log += evaluation_segmentation(test_df, thresholds = segmentation_ths, mode = "testing")
    print(log)

    # save figures
    save_segmentation(cam, weights, 25, "./", thresholds = segmentation_ths)