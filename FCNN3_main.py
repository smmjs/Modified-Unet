# -*- coding: utf-8 -*-
"""
Created on Mon April  5 11:23:29 2021

@author: Briana Santo
"""
import os
import sys
import random
import warnings

import numpy as np
import pandas as pd
import glob

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.morphology import label
from skimage.transform import resize

from keras import backend
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.models import Model, load_model
from keras.preprocessing import image

import tensorflow as tf

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

os.environ["CUDA_VISIBLE_DEVICES"]="0"

#general params to be adjusted
num_epochs = 25
num_steps = 250
batch_sz = 64

#non-adjustable network params
size = 128
seed = 42 #for img, gt sheering in data aug

#data paths
cwd = os.getcwd()

fcnn_summary = 'fcnn3_sz'+str(size)+'_b'+str(batch_sz)
fcnn_summary = cwd + '/' + fcnn_summary

train_data = cwd + '/gt_preprocessing/train_data/'
test_data = cwd + '/gt_preprocessing/test_data/'

train_ims = glob.glob(train_data + 'train512/*.tif')
train_gt = glob.glob(train_data + 'train512_gt/*.tif')
test_ims = glob.glob(test_data + 'test512/*.tif')
test_gt = glob.glob(test_data + 'test512_gt/*.tif')

#print(test_gt.index(test_data+'test512_gt/TESTINGIMAGE_mask.tif'))
#print('\n\n\n\n\n\n\n\n\n\n')

#data info and confirm directory alignment
sample_im = plt.imread(train_ims[0])
#sample_gt = plt.imread(train_gt[0])
rows = 128
cols = 128
dims = 1
#print('size:',size)
#plt.subplot(211), plt.imshow(sample_im), plt.title('sample img')
#plt.subplot(212), plt.imshow(sample_gt), plt.title('sample mask')
#plt.show()

#create img train and test tensors
print('\n Retrieving training images and masks ... ')
x_train = np.zeros((len(train_ims), rows, cols, dims), dtype=np.uint8)
y_train = np.zeros((len(train_gt), rows, cols, 1), dtype=np.bool)

for i in range(len(train_ims)):
    im = imread(train_ims[i])
    #im = im[:,:,0]
    im = resize(im, (rows, cols, dims), mode='constant', preserve_range=True)
    gt = imread(train_gt[i])
    #gt = gt[:,:,0]
    gt = resize(gt, (rows, cols, dims), mode='constant', preserve_range=True)
    x_train[i] = im
    y_train[i] = gt

#print('x_train',x_train[0].shape)
#print('y_train',y_train.shape)
#plt.imshow(x_train[0].reshape(rows,cols))
#plt.show()

print('\n Retrieving test images and masks ... ')
x_test = np.zeros((len(train_ims), rows, cols, dims), dtype=np.uint8)
y_test = np.zeros((len(train_gt), rows, cols, 1), dtype=np.bool)

for i in range(len(test_ims)):
    im = imread(test_ims[i])
    #im = im[:,:,0]
    im = resize(im, (rows, cols, dims), mode='constant', preserve_range=True)
    gt = imread(test_gt[i])
    #gt = gt[:,:,0]
    gt = resize(gt, (rows, cols, dims), mode='constant', preserve_range=True)
    x_test[i] = im
    y_test[i] = gt

print('\n Done ! ')

#functions
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = backend.flatten(y_true)
    y_pred_f = backend.flatten(y_pred)
    intersection = backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (backend.sum(y_true_f) + backend.sum(y_pred_f) + smooth)

def mean_iou(y_true, y_pred):

    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        backend.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return backend.mean(backend.stack(prec), axis=0)

#create training data generators
image_datagen = image.ImageDataGenerator(shear_range=0.5, rotation_range=50, zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2, fill_mode='reflect')
mask_datagen = image.ImageDataGenerator(shear_range=0.5, rotation_range=50, zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2, fill_mode='reflect')

# Keep the same seed for image and mask generators so they fit together
image_datagen.fit(x_train[:int(x_train.shape[0]*0.9)], augment=True, seed=seed)
mask_datagen.fit(y_train[:int(y_train.shape[0]*0.9)], augment=True, seed=seed)

#complete training data augmentation
x=image_datagen.flow(x_train[:int(x_train.shape[0]*0.9)],batch_size=batch_sz,shuffle=True, seed=seed)
y=mask_datagen.flow(y_train[:int(y_train.shape[0]*0.9)],batch_size=batch_sz,shuffle=True, seed=seed)

#create test data generators
image_datagen_val = image.ImageDataGenerator()
mask_datagen_val = image.ImageDataGenerator()

image_datagen_val.fit(x_train[int(x_train.shape[0]*0.9):], augment=True, seed=seed)
mask_datagen_val.fit(y_train[int(y_train.shape[0]*0.9):], augment=True, seed=seed)

#complete test data augmentation
x_val=image_datagen_val.flow(x_train[int(x_train.shape[0]*0.9):],batch_size=batch_sz,shuffle=True, seed=seed)
y_val=mask_datagen_val.flow(y_train[int(y_train.shape[0]*0.9):],batch_size=batch_sz,shuffle=True, seed=seed)

#create random generators
train_generator = zip(x, y)
val_generator = zip(x_val, y_val)

#building Unet
inputs = Input((rows, cols, dims))
s = Lambda(lambda x: x / 255) (inputs)

c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (s)
c1 = Dropout(0.1) (c1)
c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)
c2 = Dropout(0.1) (c2)
c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p2)
c3 = Dropout(0.2) (c3)
c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)
c4 = Dropout(0.3) (c4)
c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c4)

u5 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (c4)
u5 = concatenate([u5, c3])
c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u5)
c5 = Dropout(0.2) (c5)
c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c2])
c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6)
c6 = Dropout(0.2) (c6)
c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c1], axis=3)
c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7)
c7 = Dropout(0.1) (c7)
c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c7)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
model.summary()

#training
earlystopper = EarlyStopping(patience=20, verbose=1)
checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)
results = model.fit_generator(train_generator, validation_data=val_generator, validation_steps=10, steps_per_epoch=num_steps, epochs=num_epochs, callbacks=[earlystopper, checkpointer])
hist_df = pd.DataFrame(results.history)
hist_df.to_csv(fcnn_summary)

#prediction and model testing
model = load_model('model-dsbowl2018-1.h5', custom_objects={'mean_iou': mean_iou})
preds_train = model.predict(x_train[:int(x_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(x_train[int(x_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(x_test, verbose=1)

#threshold predictions
preds_train_t = (preds_train > 0.6).astype(np.uint8)
preds_val_t = (preds_val > 0.6).astype(np.uint8)
preds_test_t = (preds_test > 0.6).astype(np.uint8)

#plotting loss
plt.plot(results.history['loss'], label='BCE loss (training data)')
plt.plot(results.history['val_loss'], label='BCE loss (validation data)')
plt.title('BCE Loss for Mitochondrial Segmentation')
plt.ylabel('Log loss value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
fig_path = fcnn_summary + '_loss.png'
plt.savefig(fig_path,pad_inches = 0.25)
plt.close()

#plotting ji / iou
plt.plot(results.history['mean_iou'], label='JI (training data)')
plt.plot(results.history['val_mean_iou'], label='JI (validation data)')
plt.title('Jacard Index for Mitochondrial Segmentation')
plt.ylabel('JI - Mean Intersection over Union')
plt.xlabel('No. epoch')
plt.legend(loc="lower right")
fig_path = fcnn_summary + '_iou.png'
plt.savefig(fig_path,pad_inches = 0.25)
plt.close()

#test masks
preds_test_upsampled = []
for i in range(len(preds_test)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]),
                                       (rows, cols),
                                       mode='constant', preserve_range=True))

#qualitative assessment of network performance
plt.subplot(311), plt.imshow(x_test[136].reshape(rows,cols)), plt.title('random image'), plt.axis('off')
plt.subplot(312), plt.imshow(y_test[116].reshape(rows,cols)), plt.title('random image groundtruth'), plt.axis('off')
plt.subplot(313), plt.imshow(preds_test_upsampled[136].reshape(rows,cols)), plt.title('network prediction'), plt.axis('off')
fig_path = fcnn_summary + '_sample_seg.png'
plt.savefig(fig_path,pad_inches = 0.25)
