import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import keras
import cv2
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
from keras.callbacks import Callback
import os

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    K.cast(y_true_f, tf.float32)
    K.cast(y_pred_f, tf.float32)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Read Label
train = pd.read_csv('train.csv')[:1000]
train['Image'] = train['Image_Label'].map(lambda x: x.split('.')[0])
train['Label'] = train['Image_Label'].map(lambda x: x.split('_')[1])
train2 = pd.DataFrame({'Image':train['Image'][::4]})
train2['e1'] = train['EncodedPixels'][::4].values
train2['e2'] = train['EncodedPixels'][1::4].values
train2['e3'] = train['EncodedPixels'][2::4].values
train2['e4'] = train['EncodedPixels'][3::4].values
train2.set_index('Image',inplace=True,drop=True)
train2.fillna('',inplace=True); train2.head()
train2[['d1','d2','d3','d4']] = (train2[['e1','e2','e3','e4']]!='').astype('int8')
train2[['d1','d2','d3','d4']].head()


# Generate Data
class DataGenerator(keras.utils.Sequence):
    # USES GLOBAL VARIABLE TRAIN2 COLUMNS E1, E2, E3, E4
    # 'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=8, shuffle=False, width=512, height=352, scale=1/128., sub=1., mode='train',
                 path='./train_images/', flips=False):
        # 'Initialization'
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.path = path
        self.scale = scale
        self.sub = sub
        self.path = path
        self.width = width
        self.height = height
        self.mode = mode
        self.flips = flips
        self.on_epoch_end()

    def __len__(self):
        # 'Denotes the number of batches per epoch'
        ct = int(np.floor(len(self.list_IDs) / self.batch_size))
        if len(self.list_IDs) > ct * self.batch_size: ct += 1
        return int(ct)

    def __getitem__(self, index):
        # 'Generate one batch of data'
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__data_generation(indexes)
        if (self.mode == 'train') | (self.mode == 'validate'):
            return X, y
        else:
            return X

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        self.indexes = np.arange(int(len(self.list_IDs)))
        if self.shuffle: np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        # 'Generates data containing batch_size samples'
        # Initialization
        lnn = len(indexes)
        X = np.empty((lnn, self.height, self.width, 3), dtype=np.float32)
        y = np.zeros((lnn, 4), dtype=np.int8)

        # Generate data
        for k in range(lnn):
            img = cv2.imread(self.path + self.list_IDs[indexes[k]] + '.jpg')
            img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
            # AUGMENTATION FLIPS
            hflip = False
            vflip = False
            if (self.flips):
                if np.random.uniform(0, 1) > 0.5: hflip = True
                if np.random.uniform(0, 1) > 0.5: vflip = True
            if vflip: img = cv2.flip(img, 0)  # vertical
            if hflip: img = cv2.flip(img, 1)  # horizontal
            # NORMALIZE IMAGES
            X[k, ] = img * self.scale - self.sub
            # LABELS
            if (self.mode == 'train') | (self.mode == 'validate'):
                y[k, ] = train2.loc[self.list_IDs[indexes[k]], ['d1', 'd2', 'd3', 'd4']].values

        return X, y

class Metrics(Callback):
    def on_train_begin(self, log = {}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, log = {}):
        val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
        val_targ = self.model.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print (" — val_f1: %f — val_precision: %f — val_recall %f" %(_val_f1, _val_precision, _val_recall))


# USE KERAS XCEPTION MODEL
base_model = Xception(weights='imagenet', include_top=False, input_shape=(None, None, 3))


# FREEZE NON-BATCHNORM LAYERS IN BASE
for layer in base_model.layers:
    if not isinstance(layer, layers.BatchNormalization): layer.trainable = False


# BUILD MODEL NEW TOP
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(4, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=x)


# COMPILE MODEL
metrics = Metrics()
model.compile(loss='binary_crossentropy', optimizer = optimizers.Adam(lr=0.001), metrics=['accuracy'])


# SPLIT TRAIN AND VALIDATE
idxT, idxV = train_test_split(train2.index, random_state=42, test_size=0.2)
train_gen = DataGenerator(idxT, flips=True, shuffle=True)
val_gen = DataGenerator(idxV, mode='validate')
print("Data Generation done")


# TRAIN NEW MODEL TOP LR=0.001 (with bottom frozen)
h = model.fit_generator(train_gen, epochs = 2, verbose=2, validation_data = val_gen)
# TRAIN ENTIRE MODEL LR=0.0001 (with all unfrozen)
for layer in model.layers: layer.trainable = True
model.compile(loss='binary_crossentropy', optimizer = optimizers.Adam(lr=0.001), metrics=['accuracy'])
h = model.fit_generator(train_gen, epochs = 2, verbose=2, validation_data = val_gen)


# PREDICT HOLDOUT SET
train3 = train2.loc[train2.index.isin(idxV)].copy()
oof_gen = DataGenerator(train3.index.values, mode='predict')
oof = model.predict_generator(oof_gen, verbose=2)
for k in range(1,5): train3['o'+str(k)] = 0
train3[['o1','o2','o3','o4']] = oof


# COMPUTE ACCURACY AND ROC_AUC_SCORE
types = ['Fish','Flower','Gravel','Sugar']
for k in range(1,5):
    print(types[k-1],': ',end='')
    auc = np.round( roc_auc_score(train3['d'+str(k)].values,train3['o'+str(k)].values  ),3 )
    acc = np.round( accuracy_score(train3['d'+str(k)].values,(train3['o'+str(k)].values>0.5).astype(int) ),3 )
    print('AUC =',auc,end='')
    print(', ACC =',acc)
print('OVERALL: ',end='')
auc = np.round( roc_auc_score(train3[['d1','d2','d3','d4']].values.reshape((-1)),train3[['o1','o2','o3','o4']].values.reshape((-1)) ),3 )
acc = np.round( accuracy_score(train3[['d1','d2','d3','d4']].values.reshape((-1)),(train3[['o1','o2','o3','o4']].values>0.5).astype(int).reshape((-1)) ),3 )
f1 = np.round( f1_score(train3[['d1','d2','d3','d4']].values.reshape((-1)),(train3[['o1','o2','o3','o4']].values>0.5).astype(int).reshape((-1)) ),3 )
dice = np.round(dice_coef(train3[['d1','d2','d3','d4']].values.reshape((-1)),(train3[['o1','o2','o3','o4']].values>0.5).astype(int).reshape((-1)) ))
print('AUC =',auc, end='')
print(', ACC =',acc)
print(', dice = ', dice, end = '')
print('f1 = ', f1)

'''
# NEW MODEL FROM OLD TO EXTRACT ACTIVATION MAPS
all_layer_weights = model.layers[-1].get_weights()[0]
cam_model = Model(inputs=model.input, 
        outputs=(model.layers[-3].output, model.layers[-1].output)) 
# DISPLAY 25 RANDOM IMAGES
PATH = './train_images/'
IMGS = os.listdir(PATH)
for k in np.random.randint(0,5000,25):

    # LOAD IMAGE AND PREDICT CLASS ACTIVATION MAP
    img = cv2.resize( cv2.imread(PATH+IMGS[k]), (512, 352))
    x = np.expand_dims(img, axis=0)/128. -1.
    last_conv_output, pred_vec = cam_model.predict(x) 
    last_conv_output = np.squeeze(last_conv_output) 
    pred = np.argmax(pred_vec)
    layer_weights = all_layer_weights[:, pred] 
    final_output = np.dot(last_conv_output.reshape((16*11, 2048)), layer_weights).reshape(11,16) 
    final_output = scipy.ndimage.zoom(final_output, (32, 32), order=1) 
    # DISPLAY IMAGE WITH CLASS ACTIVATION MAPS
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    mx = np.round( np.max(final_output),1 )
    mn = np.round( np.min(final_output),1 )
    final_output = (final_output-mn)/(mx-mn)
    mask0 = (final_output>0.3).astype(int)
    contour0 = mask2contour(mask0,5)
    plt.imshow(img, alpha=0.5)
    plt.imshow(final_output, cmap='jet', alpha=0.5)
    plt.title('Found '+types[pred]+'  -  Pr = '+str(np.round(pred_vec[0,pred],3)) )

    # DISPLAY IMAGE WITH MASKS
    plt.subplot(1,2,2)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rle = train2.loc[IMGS[k].split('.')[0],'e'+str(pred+1)]
    mask = rle2mask2X(rle,shrink=(512,352))
    contour = mask2contour(mask,5)
    img[contour==1,:2] = 255
    img[contour0==1,2] = 255
    diff = np.ones((352,512,3),dtype=np.int)*255-img
    img=img.astype(int); img[mask0==1,:] += diff[mask0==1,:]//4
    plt.imshow( img )
    dice = np.round( dice_coef8(mask,mask0),3 )
    plt.title('Dice = '+str(dice)+'  -  '+IMGS[k]+'  -  '+types[pred])

    plt.show()
'''