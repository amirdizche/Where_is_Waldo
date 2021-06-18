# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 16:19:32 2021 # connected to PyCharm 2

@author: Amirhassan

Find Waldo using Faster R-CNN 
"""
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf


DF = pd.read_csv('annotations.csv') # read ground truth Waldo locations
DF_train = DF[0:25]
DF_val = DF[25:27].reset_index()
DF_test = DF[27:].reset_index()
#print(DF)

# define anchor boxes for training
# sample box height and weight from oroginal images
# half of the boxes include Waldo (True boxes)
# half do not include Waldo (Flase boxes)
def sample_anchor_boxes(DF, box_w, box_h):
  X = []
  Y = []
  
  dx = np.linspace(-(box_w//2), box_w//2, box_w//4)
  dy = np.linspace(-(box_h//2), box_h//2, box_h//4)
  offset = np.meshgrid(dx, dy)
  
  for i in range(len(DF)):
    img = cv2.imread('images/' + DF.filename[i])
    img_w = DF.width[i]
    img_h = DF.height[i]

    # ground truth bounding boxes
    xmin = DF.xmin[i]
    xmax = DF.xmax[i]
    ymin = DF.ymin[i]
    ymax = DF.ymax[i]

    x_GT = (xmax + xmin)//2
    y_GT = (ymax + ymin)//2

    # "true" anchor boxes
    cnt = 0
    for yshift in range(-box_h//4, box_h//4, 1):   # shift the anchor boxes around 
      for xshift in range(-box_w//4, box_w//4, 1): # to have many samples of Waldo
        x = x_GT + xshift
        y = y_GT + yshift
        xmin_roi = x - box_w//2
        xmax_roi = x + box_w//2
        ymin_roi = y - box_h//2
        ymax_roi = y + box_h//2

        if xmin_roi < 0 or xmax_roi >= img_w or ymin_roi < 0 or ymax_roi >= img_h:
          continue

        img_roi = img[ymin_roi:ymax_roi, xmin_roi:xmax_roi, :]
        
        ########################################################################
        ### FILL IN THE BLANK ###
        ''' For this part, I give you the answer.'''
        label = np.ones((box_h//4,box_w//4, 5))
        label[:,:, 1] = x_GT - x - offset[0]
        label[:,:, 2] = y_GT - y - offset[1]
        label[:,:, 3] = (xmax-xmin)/box_w
        label[:,:, 4] = (ymax-ymin)/box_h
        ########################################################################
        
        X.append(img_roi)
        Y.append(label)
        cnt += 1

    # "false" anchor boxes
    while True:
      x = np.random.randint(box_w//2,img_w-box_w//2-1)
      y = np.random.randint(box_h//2,img_h-box_h//2-1)

      xmin_roi = x - box_w//2
      xmax_roi = x + box_w//2
      ymin_roi = y - box_h//2
      ymax_roi = y + box_h//2

      if (xmin_roi > xmin and xmin_roi < xmax) or (ymin_roi > ymin and ymin_roi < ymax) or (xmax_roi > xmin and xmax_roi < xmax) or (ymax_roi > ymin and ymax_roi < ymax):
        continue

      img_roi = img[ymin_roi:ymax_roi, xmin_roi:xmax_roi, :]
            
      ##########################################################################
      label = '''FILL IN THE BLANK'''
      label = np.zeros((box_h//4,box_w//4, 5))
      label[:,:, 1] = x - offset[0]
      label[:,:, 2] = y - offset[1]
      label[:,:, 3] = (xmax-xmin)/box_w
      label[:,:, 4] = (ymax-ymin)/box_h
      ##########################################################################
      
      X.append(img_roi)
      Y.append(label)
      cnt -= 1

      if cnt <= 0:
        break
        
  return np.array(X), np.array(Y)

#Now that we have the anchor box sampling function defined, let's build a training dataset using it

# anchor boxes
anchor_boxes = [[20, 20]] # for simplicity, we will use only one fixed-size anchor box.

(rpn_x_train, rpn_y_train) = sample_anchor_boxes(DF_train, anchor_boxes[0][0], anchor_boxes[0][1])
(rpn_x_val, rpn_y_val) = sample_anchor_boxes(DF_val, anchor_boxes[0][0], anchor_boxes[0][1])

# Visualization of some of the samples.
plt.figure(figsize=(16,16))
for i in range(49):
  plt.subplot(7,7,i+1)
  id = np.random.randint(len(rpn_x_train))
  temp = cv2.cvtColor(rpn_x_train[id], cv2.COLOR_BGR2RGB)
  mask = cv2.cvtColor(rpn_y_train[id,:,:,0].astype('float32'), cv2.COLOR_GRAY2RGB)
  mask = cv2.resize(1-mask, (anchor_boxes[0][0],anchor_boxes[0][0]))
  temp = temp/255.0
  temp = temp - mask*0.7
  temp = np.clip(temp, 0, 1)
  plt.imshow(temp)
  plt.grid([])
  plt.xticks([])
  plt.yticks([])
  plt.title(rpn_y_train[id, 2, 2, 1:5])

# import VGG16 as backbone
  
#backbone = keras.applications.vgg16.VGG16(weights="imagenet",
#                                          include_top=False)

#backbone = tf.keras.applications.VGG16(weights="imagenet",include_top=False)
backbone = tf.keras.applications.VGG16(weights='vgg.h5',include_top=False, pooling='avg')

for layer in backbone.layers:
  layer.trainable = False
backbone_output = backbone.get_layer('block3_conv3').output
conv = keras.layers.Conv2D(512, kernel_size=(3,3), padding='same', activation='relu')(backbone_output)
box = keras.layers.Conv2D(5, kernel_size=(1,1), activation='sigmoid')(conv)
rpn_model = keras.models.Model(backbone.input, box)
rpn_model.summary()


from keras import backend as K
# defibe the RPN loss
def rpn_loss(y_true, y_pred):
  
  ## CLASS LOSS
  class_loss = K.binary_crossentropy(y_pred[:, :, :, 0], y_true[:, :, :, 0])
  
  ## REGRESSION LOSS
  x = y_true[:, :, :, 1:5] - y_pred[:, :, :, 1:5]
  x_abs = K.abs(x)
  
  x_bool = K.less_equal(x_abs, 1.0)
  x_bool = K.cast(x_bool, 'float32')
  
  p = y_true[:,:,:,0]
  p = K.expand_dims(p, axis=3)
  p = K.repeat_elements(p, 4, axis=3)
  
  regression_loss = K.sum( p * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5)))
  
  return class_loss + 0.0001*regression_loss

# define training parameters and train the model
adam = keras.optimizers.Adam(lr=0.00001)
rpn_model.compile(optimizer=adam, loss=rpn_loss)

es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

rpn_model.fit(rpn_x_train, rpn_y_train,
          epochs=500, batch_size = 128,
          callbacks=[es],
         validation_data=(rpn_x_val, rpn_y_val))



# Test the model

id = 30 # 27~30 (inclusive) are images left out for testing
img = cv2.imread('images/' + DF.filename[id])

pred = rpn_model.predict(np.expand_dims(img,axis=0))


thres = 0.5

mask = np.squeeze(pred[:,:,:,0] > thres).astype("float32")
mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
mask = cv2.resize(1-mask, (img.shape[1], img.shape[0]))

bboxes = []
for j in range(pred.shape[1]):
  for i in range(pred.shape[2]):
    if pred[0,j,i,0] > thres:
      x = i*4 + pred[0,j,i,1]
      y = j*4 + pred[0,j,i,2]
      w = anchor_boxes[0][0]*pred[0,j,i,3]
      h = anchor_boxes[0][0]*pred[0,j,i,4]
      
      xmin = (x - w/2).astype(int)
      ymin = (y - w/2).astype(int)
      xmax = (x + w/2).astype(int)
      ymax = (y + w/2).astype(int)
      
      bboxes.append([xmin, ymin, xmax, ymax])

print('%d proposals were found.' % len(bboxes))


thres = 0.5
plt.figure(figsize=(48,32))
temp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
temp = temp/255.0
temp = temp - mask*0.8
temp = np.clip(temp, 0, 1)

temp = cv2.rectangle(temp, (DF.xmin[id], DF.ymin[id]), (DF.xmax[id], DF.ymax[id]), (0,1,0), 2) # to compare with the ground truth

plt.imshow(temp)
plt.grid([])
plt.xticks([])
plt.yticks([])
plt.show()

