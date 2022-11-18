#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8
num_epoch = 1500
Batch_size=2
# In[1]:

import os
import numpy as np
import tensorflow as tf
#from tensorflow import keras
from tensorflow.python import keras
from tensorflow.keras import layers
import nibabel as nib
from scipy import ndimage
import medpy
import copy
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Concatenate
import numpy as np
from tensorflow.python.keras.models import Model, load_model
from tensorflow.keras.layers import BatchNormalization,Conv2D, Input, UpSampling2D, Activation, concatenate, MaxPooling2D, Conv2DTranspose, Reshape, Permute,Add
from tensorflow.keras.optimizers import SGD, Adam
#from scipy.misc import imresize, imsave, imread
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from medpy.metric import dc, precision, recall
from tensorflow.python.keras.metrics import Metric
import matplotlib.pyplot as plt
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import sys
lrate1=3e-3;
lrate=float(lrate1)

import re
from scipy import linalg
import scipy.ndimage as ndi
# from six.moves import range
import threading


os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
ind = '20220323'
run_count = '02_temp'
split_num = 'Splitx'

#pip install medpy


# In[3]:


# In[3]:


T1_Data=np.load('HippoCamp_Data_selected_256x256.npy')
T1_Label=np.load('HippoCamp_Label_selected_256x256X4.npy')
#Add_Data=np.load('HippoCamp_AdditionalData_selected_256x256.npy')


# In[2]:


# In[ ]:


'''
%matplotlib inline
import matplotlib.pyplot as plt
plt.imshow(x_train[0],cmap='gray')

plt.show()
'''


# In[3]:


import patchify
from patchify import patchify
Patch_width = 64
Patch_height = 64
step=64


# In[4]:


x1=T1_Data
print(x1.shape)
#import patchify
#from patchify import patchify
#Patch_width = 128
#Patch_height = 128
idx=0
img=x1[0]
patches_img=patchify(img,(Patch_width,Patch_height), step=step)
x=patches_img.shape[1]
y=patches_img.shape[0]
z=x1.shape[0]
dim=x*y*z
img_stack_sm = np.zeros((dim, Patch_width, Patch_height))

for i in range (x1.shape[0]):
    img=x1[i]
    patches_img=patchify(img,(Patch_width,Patch_height), step=step)
    
    #print(x,y,z, dim)
    
    
    for j in range(patches_img.shape[0]):
        for k in range (patches_img.shape[1]):
            New_patch_img=patches_img[j,k,:,:]
            #plt.imshow(New_patch_img,cmap='gray')
            #plt.show()
            img_stack_sm[idx, :, :] = New_patch_img
            idx=idx+1
            #print(idx)
            
Patch_trainx=img_stack_sm[0:len(img_stack_sm)]
print(Patch_trainx.shape)


# In[5]:


x1=T1_Label
print(x1.shape)
#import patchify
#from patchify import patchify
#Patch_width = 128
#Patch_height = 128
idx=0
img=x1[0]
patches_img=patchify(img,(Patch_width,Patch_height,4), step=step)
x=patches_img.shape[1]
y=patches_img.shape[0]
z=x1.shape[0]
dim=x*y*z
img_stack_sm = np.zeros((dim, Patch_width, Patch_height,4))

for i in range (x1.shape[0]):
    img=x1[i]
    patches_img=patchify(img,(Patch_width,Patch_height,4), step=step)
    
    #print(x,y,z, dim)
    
    
    for j in range(patches_img.shape[0]):
        for k in range (patches_img.shape[1]):
            New_patch_img=patches_img[j,k,:,:]
            #plt.imshow(New_patch_img,cmap='gray')
            #plt.show()
            img_stack_sm[idx, :, :] = New_patch_img
            idx=idx+1
            #print(idx)
            
Patch_trainy=img_stack_sm[0:len(img_stack_sm)] 
print(Patch_trainy.shape)


# In[6]:


import numpy
#get_ipython().run_line_magic('matplotlib', 'inline')
#import matplotlib.pyplot as plt
import cv2
K=0
width = Patch_width
height = Patch_height
#width = 256
#height = 256
Data_2D=Patch_trainx
Label_2D=Patch_trainy
img_stack_sm2 = np.zeros((len(Data_2D), width, height))
img_stack_sm3 = np.zeros((len(Label_2D), width, height))
img_stack_sm4 = np.zeros((len(Data_2D), width, height))
img_stack_sm5 = np.zeros((len(Label_2D), width, height))

for idx in range(len(Label_2D)):
    if (np.max(Label_2D[idx,:,:,0])!=0):
        KD2=Data_2D[idx,:,:]
        a=numpy.amin(KD2)
        b=numpy.amax(KD2)
        KD3=((KD2-a)/(b-a))*255
        #KD4=KD3[40:300,80:400]
        #print(KD3.shape)
        K2=KD3.astype('uint8')
        #K2=cv2.resize(KD4,(256,256), interpolation = cv2.INTER_AREA)
        #print(KD3)
        #plt.imshow(K2,cmap='gray')
        #plt.show()
        KL3=Label_2D[idx,:,:,0]
        #a=numpy.amin(KL3)
        #b=numpy.amax(KL3)
        #KL3=((KL3-a)/(b-a))*255
        #print(KL3.shape)
        #KL4=KL3[40:300,80:400]
        K3=KL3.astype('uint8')
        #print(KL5.shape)
        #K3=cv2.resize(KL5,(256,256), interpolation = cv2.INTER_AREA)
        #plt.imshow(K3[:,:,0],cmap='gray')
        #plt.show()
        #plt.imshow(K3[:,:,1],cmap='gray')
        #plt.show()
        img_stack_sm2[K,:,:]=K2
        img_stack_sm3[K,:,:]=K3
        K=K+1
    else:
        KD2=Data_2D[idx,:,:]
        a=numpy.amin(KD2)
        b=numpy.amax(KD2)
        KD3=((KD2-a)/(b-a))*255
        #KD4=KD3[40:300,80:400]
        #print(KD3.shape)
        K2=KD3.astype('uint8')
        #K2=cv2.resize(KD4,(256,256), interpolation = cv2.INTER_AREA)
        #print(KD3)
        #plt.imshow(K2,cmap='gray')
        #plt.show()
        KL3=Label_2D[idx,:,:,0]
        #a=numpy.amin(KL3)
        #b=numpy.amax(KL3)
        #KL3=((KL3-a)/(b-a))*255
        #print(KL3.shape)
        #KL4=KL3[40:300,80:400]
        K3=KL3.astype('uint8')
        #print(KL5.shape)
        #K3=cv2.resize(KL5,(256,256), interpolation = cv2.INTER_AREA)
        #plt.imshow(K3[:,:,0],cmap='gray')
        #plt.show()
        #plt.imshow(K3[:,:,1],cmap='gray')
        #plt.show()
        img_stack_sm4[K,:,:]=K2
        img_stack_sm5[K,:,:]=K3
       
        
        #print(K2,K3)
    #img = T1_test_x[idx, :, :]
    #img_sm = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    #img_stack_sm[idx, :, :] = img_sm
print(K)


# In[7]:


D_P=img_stack_sm2[0:K]
L_P=img_stack_sm3[0:K]
D_N=img_stack_sm4[0:K]
L_N=img_stack_sm5[0:K]

print(L_P.shape)
print(L_N.shape)


# In[8]:


Data=[]
Data.extend(D_P)
Data.extend(D_N)
Data=np.asarray(Data)
print(Data.shape)

Label=[]
Label.extend(L_P[:,:,:])
Label.extend(L_N[:,:,:])
#Label=np.concatenate((L_P, L_N), axis=0)
Label=np.asarray(Label)
print(Label.shape)

# In[9]:


Patch_trainx=Data
Patch_trainy=Label


# In[10]:


print(Patch_trainx.shape)
print(Patch_trainy.shape)



# In[3]:


# In[ ]:


#Patch_width=256
#Patch_height=256


# In[11]:


New_train_xx=np.reshape(Patch_trainx, (-1,Patch_width,Patch_height, 1))
New_train_yy=np.reshape(Patch_trainy, (-1,Patch_width,Patch_height, 1))
#New_valid_xx=np.reshape(Patch_validx_new, (-1,Patch_width,Patch_height, 1))
#New_valid_yy=np.reshape(Patch_validy_new, (-1,Patch_width,Patch_height, 2))
#New_test_xx=np.reshape(Patch_testx_new, (-1,Patch_width,Patch_height, 1))
#New_test_yy=np.reshape(Patch_testy_new, (-1,Patch_width,Patch_height, 2))

#New_train_xx1=np.reshape(Patch_trainx1, (-1,Patch_width,Patch_height, 1))
#New_train_yy1=np.reshape(Patch_trainy1, (-1,Patch_width,Patch_height, 2))

New_Data=New_train_xx/255
New_Label=New_train_yy/255
#valid_x=New_valid_xx/255
#valid_y=New_valid_yy/255
#test_x=New_test_xx/255
#test_y=New_test_yy/255

#train_x1=New_train_xx1/255
#train_y1=New_train_yy1/255
print(New_Data.shape)
print(New_Label.shape)
#print(valid_x.shape)
#print(valid_y.shape)
#print(test_x.shape)
#print(test_y.shape)


# In[4]:


from sklearn.model_selection import train_test_split

x_train1, x_test, y_train1, y_test = train_test_split(New_Data, New_Label, test_size=0.3, random_state=42,shuffle=True)
#x_train, x_valid, y_train, y_valid = train_test_split(x_train1, y_train1, test_size=0.3, random_state=42,shuffle=True)





# In[5]:


# In[59]:



def recall_m(y_true, y_pred):
        true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
        possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
        predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+tf.keras.backend.epsilon()))

def dice_coef(y_true, y_pred, smooth=1.):
    flatten_layer = tf.keras.layers.Flatten()  # instantiate the layer
    y_true_f = flatten_layer(y_true)
    y_pred_f = flatten_layer(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def tversky_loss(y_true, y_pred, alpha=0.15, beta=0.85, smooth=1e-10):
    """ Tversky loss function.
    Parameters
    ----------
    y_true : keras tensor
        tensor containing target mask.
    y_pred : keras tensor
        tensor containing predicted mask.
    alpha : float
        real value, weight of '0' class.
    beta : float
        real value, weight of '1' class.
    smooth : float
        small real value used for avoiding division by zero error.
    Returns
    -------
    keras tensor
        tensor containing tversky loss.
    """
    flatten_layer = tf.keras.layers.Flatten()  # instantiate the layer
    y_true_f = flatten_layer(y_true)
    y_pred_f = flatten_layer(y_pred)
    truepos = tf.keras.backend.sum(y_true * y_pred)
    fp_and_fn = alpha * tf.keras.backend.sum(y_pred * (1 - y_true)) + beta * tf.keras.backend.sum((1 - y_pred) * y_true)
    answer = 1 - (truepos + smooth) / ((truepos + smooth) + fp_and_fn)
    return answer

def wcce(y_true, y_pred):
        #Kweights = K.constant(weights)
        #if not K.is_tensor(y_pred): y_pred = K.constant(y_pred)
        #y_true = K.cast(y_true, y_pred.dtype)
        return K.binary_crossentropy(y_true, y_pred)  
        return wcce

def dice_coef(y_true, y_pred):
    
    #y_pred = K.round(y_pred)
    flatten_layer = tf.keras.layers.Flatten()  # instantiate the layer
    y_true_f = flatten_layer(y_true)
    y_pred_f = flatten_layer(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# In[6]:


'''
# In[51]:
import tensorflow
tensorflow.keras.backend.clear_session


lrate1=3e-4
lrate=float(lrate1)

def expend_as(tensor, rep):
  my_repeat = tf.keras.layers.Lambda(lambda x, repnum: tf.keras.backend.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep})(tensor)
  return my_repeat

intial="glorot_normal"

def create_model(learn=lrate):

    inputs = tf.keras.Input((64,64,1))


    layer = Conv2D(16, (3, 3),padding='same', use_bias=False,kernel_initializer=intial, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(inputs)
    layer = Activation('relu')(layer)


    for i in range(1):
      layer = Conv2D(32, (3, 3),padding='same', use_bias=False,kernel_initializer=intial, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(layer)
      layer = BatchNormalization()(layer)      
      layer1 = Activation('relu')(layer)
      layer=layer1

    pool1 = MaxPooling2D(pool_size=(2, 2))(layer1)
    layer=pool1
    for i in range(2):
      layer = Conv2D(32, (3, 3),padding='same', use_bias=False,kernel_initializer=intial, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(layer)
      layer = BatchNormalization()(layer)      
      layer2 = Activation('relu')(layer)
      layer=layer2



    pool2 = MaxPooling2D(pool_size=(2, 2))(layer2)

    layer=pool2
    for i in range(1):

      layer = Conv2D(64, (3, 3),padding='same', use_bias=False,kernel_initializer=intial, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(layer)
      layer = BatchNormalization()(layer)      
      layer3 = Activation('relu')(layer)
      layer=layer3

    pool3 = MaxPooling2D(pool_size=(2, 2))(layer3)
    layer=pool3

    for i in range(1):

      layer = Conv2D(64, (3, 3),padding='same', use_bias=False,kernel_initializer=intial, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(layer)
      layer = BatchNormalization()(layer)      
      layer4 = Activation('relu')(layer)
      layer=layer4

    pool4 = MaxPooling2D(pool_size=(2, 2))(layer4)

    layer=pool4
    for i in range(1):
      layer = Conv2D(128, (3, 3),padding='same', use_bias=False,kernel_initializer=intial, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(layer)
      layer = BatchNormalization()(layer)      
      layer = Activation('relu')(layer)



############################################-----------------ATTENTION---1
    shape = tf.keras.backend.int_shape(layer)
    gating =  Conv2D(shape[3]*2, (3, 3),padding='same', use_bias=False,kernel_initializer=intial, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(layer)
    gating = BatchNormalization()(gating)      
    gating = Activation('relu')(gating)
##########################################
    shape_x=tf.keras.backend.int_shape(layer4)
    shape_g=tf.keras.backend.int_shape(gating)
    theta_x = Conv2D(64, (2, 2), strides=(2, 2), padding='same',kernel_initializer=intial, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(layer4)
    shape_theta_x = tf.keras.backend.int_shape(theta_x)
   
    phi_g = Conv2D(64, (1, 1), padding='same',kernel_initializer=intial, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(gating)
    upsample_g = Conv2DTranspose(64, (3, 3),strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),padding='same')(phi_g)
    concat_xg = keras.layers.Add()([upsample_g, theta_x])
    act_xg = Activation('relu')(concat_xg)
    psi = Conv2D(1, (1, 1), padding='same',kernel_initializer=intial, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = tf.keras.backend.int_shape(sigmoid_xg)
    upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32
    upsample_psi = expend_as(upsample_psi, shape_x[3])
    y = tf.keras.layers.Multiply()([upsample_psi, layer4])
    result = Conv2D(shape_x[3], (1, 1), padding='same',kernel_initializer=intial, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(y)
    result_bn = BatchNormalization()(result)

##########################################

    up = concatenate([Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',activation="relu")(layer),result_bn], axis=3)
#############################################################-------------------



############################################-----------------ATTENTION-----2
    shape = tf.keras.backend.int_shape(up)
    gating =  Conv2D(shape[3]*2, (3, 3),padding='same', use_bias=False,kernel_initializer=intial, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(up)
    gating = BatchNormalization()(gating)      
    gating = Activation('relu')(gating)
##########################################
    shape_x=tf.keras.backend.int_shape(layer3)
    shape_g=tf.keras.backend.int_shape(gating)
    theta_x = Conv2D(64, (2, 2), strides=(2, 2), padding='same',kernel_initializer=intial, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(layer3)
    shape_theta_x = tf.keras.backend.int_shape(theta_x)
   
    phi_g = Conv2D(64, (1, 1), padding='same',kernel_initializer=intial, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(gating)
    upsample_g = Conv2DTranspose(64, (3, 3),strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),padding='same')(phi_g)
    concat_xg = keras.layers.Add()([upsample_g, theta_x])
    act_xg = Activation('relu')(concat_xg)
    psi = Conv2D(1, (1, 1), padding='same',kernel_initializer=intial, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = tf.keras.backend.int_shape(sigmoid_xg)
    upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32
    upsample_psi = expend_as(upsample_psi, shape_x[3])
    y = tf.keras.layers.Multiply()([upsample_psi, layer3])
    result = Conv2D(shape_x[3], (1, 1), padding='same',kernel_initializer=intial, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(y)
    result_bn = BatchNormalization()(result)

##########################################

    up = concatenate([Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',activation="relu")(up), result_bn], axis=3)
#############################################################-------------------



############################################-----------------ATTENTION-----3
    shape = tf.keras.backend.int_shape(up)
    gating =  Conv2D(shape[3]*2, (3, 3),padding='same', use_bias=False,kernel_initializer=intial, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(up)
    gating = BatchNormalization()(gating)      
    gating = Activation('relu')(gating)
##########################################
    shape_x=tf.keras.backend.int_shape(layer2)
    shape_g=tf.keras.backend.int_shape(gating)
    theta_x = Conv2D(64, (2, 2), strides=(2, 2), padding='same',kernel_initializer=intial, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(layer2)
    shape_theta_x = tf.keras.backend.int_shape(theta_x)
   
    phi_g = Conv2D(64, (1, 1), padding='same',kernel_initializer=intial, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(gating)
    upsample_g = Conv2DTranspose(64, (3, 3),strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),padding='same')(phi_g)
    concat_xg = keras.layers.Add()([upsample_g, theta_x])
    act_xg = Activation('relu')(concat_xg)
    psi = Conv2D(1, (1, 1), padding='same',kernel_initializer=intial, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = tf.keras.backend.int_shape(sigmoid_xg)
    upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32
    upsample_psi = expend_as(upsample_psi, shape_x[3])
    y = tf.keras.layers.Multiply()([upsample_psi, layer2])
    result = Conv2D(shape_x[3], (1, 1), padding='same',kernel_initializer=intial, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(y)
    result_bn = BatchNormalization()(result)

##########################################

    up = concatenate([Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',activation="relu")(up), result_bn], axis=3)
##########################################

############################################-----------------ATTENTION-----4
    shape = tf.keras.backend.int_shape(up)
    gating =  Conv2D(shape[3]*2, (3, 3),padding='same', use_bias=False,kernel_initializer=intial, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(up)
    gating = BatchNormalization()(gating)      
    gating = Activation('relu')(gating)
##########################################
    shape_x=tf.keras.backend.int_shape(layer1)
    shape_g=tf.keras.backend.int_shape(gating)
    theta_x = Conv2D(64, (2, 2), strides=(2, 2), padding='same',kernel_initializer=intial, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(layer1)
    shape_theta_x = tf.keras.backend.int_shape(theta_x)
   
    phi_g = Conv2D(64, (1, 1), padding='same',kernel_initializer=intial, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(gating)
    upsample_g = Conv2DTranspose(64, (3, 3),strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),padding='same')(phi_g)
    concat_xg = keras.layers.Add()([upsample_g, theta_x])
    act_xg = Activation('relu')(concat_xg)
    psi = Conv2D(1, (1, 1), padding='same',kernel_initializer=intial, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = tf.keras.backend.int_shape(sigmoid_xg)
    upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32
    upsample_psi = expend_as(upsample_psi, shape_x[3])
    y = tf.keras.layers.Multiply()([upsample_psi, layer1])
    result = Conv2D(shape_x[3], (1, 1), padding='same',kernel_initializer=intial, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(y)
    result_bn = BatchNormalization()(result)

##########################################

    up = concatenate([Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',activation="relu")(up), result_bn], axis=3)
##########################################

    layer = Conv2D(1, (1, 1), padding='same',kernel_initializer=intial)(up)

    out = Activation('sigmoid')(layer)

    model = tf.keras.Model(inputs, out)
   
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learn), loss = tversky_loss, metrics=[dice_coef,'accuracy'])

    return model

# In[ ]:
model = create_model()
model.summary()
'''


# In[7]:


import tensorflow
from tensorflow.keras.backend import clear_session
from tensorflow.keras.layers import Layer, Lambda
from tensorflow.python.keras.utils import conv_utils
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.backend import conv2d, int_shape, repeat_elements
#####################new layer#################################################
#from keras import backend as K
#from keras.layers import Layer
#from keras.utils import conv_utils
#from keras import initializers, regularizers, constraints
class ASPP(Layer):
    def __init__(self, rank,
                     filters,
                     kernel_size,
                     no_layers,
                     strides=1,
                     padding='same',
                     data_format="channels_last",
                     dilation_rate=1,
                     activation=None,
                     use_bias=True,
                     kernel_initializer='glorot_uniform',
                     bias_initializer='zeros',
                     kernel_regularizer=None,
                     bias_regularizer=None,
                     activity_regularizer=None,
                     kernel_constraint=None,
                     bias_constraint=None,
                     **kwargs):
            super(ASPP, self).__init__(**kwargs)
            self.rank = rank
            self.filters = filters
            self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
            self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
            self.no_layers=no_layers
            self.padding = conv_utils.normalize_padding(padding)
            #self.data_format = K.normalize_data_format(data_format)
            self.dilation_rate = dilation_rate
            self.use_bias = use_bias
            self.data_format=data_format,
            self.kernel_initializer = initializers.get(kernel_initializer)
            self.bias_initializer = initializers.get(bias_initializer)
            self.kernel_regularizer = regularizers.get(kernel_regularizer)
            self.bias_regularizer = regularizers.get(bias_regularizer)
            self.activity_regularizer = regularizers.get(activity_regularizer)
            self.kernel_constraint = constraints.get(kernel_constraint)
            self.bias_constraint = constraints.get(bias_constraint)
            #self.input_spec = InputSpec(ndim=self.rank + 2)
            
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'rank': self.rank,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'no_layers': self.no_layers,
            'padding': self.padding,
            'dilation_rate': self.dilation_rate,
            'use_bias': self.use_bias,
            'data_format': self.data_format,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'bias_regularizer': self.bias_regularizer,
            'activity_regularizer': self.activity_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'bias_constraint': self.bias_constraint,
            'kernel_regularizer': self.kernel_regularizer,
            #'padding': self.padding,
        })
        return config
    '''
    def build(self, input_shape):
            channel_axis = -1
            if input_shape[channel_axis] is None:
                raise ValueError('The channel dimension of the inputs '
                                 'should be defined. Found `None`.')
            input_dim = input_shape[channel_axis]
            kernel_shape = self.kernel_size + (input_dim, self.filters)

            self.kernel = self.add_weight(shape=kernel_shape,
                                          initializer=self.kernel_initializer,
                                          name='kernel',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
            if self.use_bias:
                self.bias = self.add_weight(shape=(self.filters,),
                                            initializer=self.bias_initializer,
                                            name='bias',
                                            regularizer=self.bias_regularizer,
                                            constraint=self.bias_constraint)
            else:
                self.bias = None
            self.built = True
            super(ASPP, self).build(input_shape)
    '''
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        input_dim = input_shape[-1]
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        self.kernel = self.add_weight(name='kernel',
                                      shape=kernel_shape,
                                      initializer='uniform',
                                      trainable=True)
        super(ASPP, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
            b41=inputs
            '''
            layer=Convolution2D(48, 1, 1, activation='relu', name='layer_1')(b41)
            out=layer
            '''

            b1 = conv2d(inputs,self.kernel,strides=self.strides,padding=self.padding,dilation_rate=conv_utils.normalize_tuple(self.dilation_rate[0], self.rank,'dilation_rate'))
            b2 = conv2d(inputs,self.kernel,strides=self.strides,padding=self.padding,dilation_rate=conv_utils.normalize_tuple(self.dilation_rate[1], self.rank,'dilation_rate'))
            b3 = conv2d(inputs,self.kernel,strides=self.strides,padding=self.padding,dilation_rate=conv_utils.normalize_tuple(self.dilation_rate[1], self.rank,'dilation_rate'))
            b4 = conv2d(inputs,self.kernel,strides=self.strides,padding=self.padding,dilation_rate=conv_utils.normalize_tuple(self.dilation_rate[1], self.rank,'dilation_rate'))
            out=Concatenate()([b1,b2,b3,b4])
            print('the shape of the output of the new layer is=', out.shape)
            return out
    '''
    def compute_output_shape(self, input_shape):
            return (input_shape[0],8,16,48*(self.no_layers+1))
    '''
    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        print('the out shape_1=', space)
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            print('#################the loop is ',i)
            print('the out shape_2=', new_dim)
            new_space.append(new_dim)
        return (input_shape[0],) + tuple(new_space) + (self.filters*self.no_layers,)


############################end of new layer################################

def expend_as(tensor, rep):
		my_repeat = Lambda(lambda x, repnum: repeat_elements(x, repnum, axis=3), arguments={'repnum': rep})(tensor)
		return my_repeat



    


# In[8]:


intial="he_normal"
def attention(learn = 3e-4):
    

    inputs = Input((64,64,1))
    


    layer = Conv2D(16, (7, 7),padding='same', use_bias=False,kernel_initializer=intial)(inputs)
    layer = Activation('relu')(layer)


    for i in range(2):
        layer = Conv2D(32, (7, 7),padding='same', use_bias=False,kernel_initializer=intial)(layer)
        layer = BatchNormalization()(layer)
        layer1 = Activation('relu')(layer)
        layer=layer1

    pool1 = MaxPooling2D(pool_size=(2, 2))(layer1)
    layer=pool1
    for i in range(2):
        layer = Conv2D(32, (5, 5),padding='same', use_bias=False,kernel_initializer=intial)(layer)
        layer = BatchNormalization()(layer) 
        layer2 = Activation('relu')(layer)
        layer=layer2

	 

    pool2 = MaxPooling2D(pool_size=(2, 2))(layer2)

    layer=pool2
    for i in range(4):

	    layer = Conv2D(64, (3, 3),padding='same', use_bias=False,kernel_initializer=intial)(layer)
	    layer = BatchNormalization()(layer)       
	    layer3 = Activation('relu')(layer)
	    layer=layer3

    pool3 = MaxPooling2D(pool_size=(2, 2))(layer3)
    layer=pool3

    for i in range(4):

	    layer = Conv2D(64, (3, 3),padding='same', use_bias=False,kernel_initializer=intial)(layer)
	    layer = BatchNormalization()(layer)       
	    layer4 = Activation('relu')(layer)
	    layer=layer4


    pool4 = MaxPooling2D(pool_size=(2, 2))(layer4)

    layer=pool4
    for i in range(8):
	    layer = Conv2D(128, (3, 3),padding='same', use_bias=False,kernel_initializer=intial)(layer)
	    layer = BatchNormalization()(layer)       
	    layer = Activation('relu')(layer)
 #############################feature extract#######################
    #b41 = Lambda(lambda layer: K.expand_dims(layer, 1))(layer)
    #b41 = Lambda(lambda layer: K.expand_dims(layer, 1))(b41)
#################################  ASPP   ###########################################
    b41_1=layer
    layer=Conv2D(48, 1, 1, activation='relu', name='layer_1')(layer)

    s=ASPP(rank=2,filters=1024,kernel_size=(3,3),no_layers=4,dilation_rate=[6,12,18,24])(b41_1)
    '''
    s_0=Conv2D(1024, (3, 3), dilation_rate=(6,6) ,activation='relu', name='fc6_1',padding='same',use_bias=False)
    s=s_0(b41_1)
    s_1=K.conv2d(b41_1,s_0.kernel,strides=s_0.strides,padding=s_0.padding,dilation_rate=(12,12))
    print('###########shape of the s_1=',s_1.shape)
    s_2=K.conv2d(b41_1,s_0.kernel,strides=s_0.strides,padding=s_0.padding,dilation_rate=(18,18))
    s_3=K.conv2d(b41_1,s_0.kernel,strides=s_0.strides,padding=s_0.padding,dilation_rate=(24,24))
    s=Concatenate()([s,s_1,s_2,s_3])
    '''
    '''
    layer=Convolution2D(48, 1, 1, activation='relu', name='layer_1')(layer)
    s=Concatenate()([s,layer])
    '''
    '''
    b41=layer
    layer=Convolution2D(48, 1, 1, activation='relu', name='layer_1')(layer)
    b1 = SeparableConv2D(1024, (3, 3), dilation_rate=(6,6) ,activation='relu', name='fc6_1',padding='same',use_bias=False)(b41)
    '''

    b1 = Lambda(lambda s: s[:,:,:,0:1024])(s)
    b1 = BatchNormalization()(b1)
    #b1 = Dropout(0.5)(b1)
    b1 = Conv2D(1024, 1, 1, activation='relu', name='fc7_1')(b1)
    b1 = BatchNormalization()(b1)
    #b1 = Dropout(0.5)(b1)
    b1 = Conv2D(48, 1, 1, activation='relu', name='fc8_voc12_1')(b1)
    # hole = 12
    b2= Lambda(lambda s: s[:,:,:,1024:2048])(s)
    #b2 = SeparableConv2D(1024, (3, 3), dilation_rate=(12,12) ,activation='relu', name='fc7_1_2',padding='same',use_bias=False)(b41)
    b2 = BatchNormalization()(b2)
    #b2 = Dropout(0.5)(b2)
    b2 = Conv2D(1024, 1, 1, activation='relu', name='fc7_2')(b2)
    b2 = BatchNormalization()(b2)
    #b2 = Dropout(0.5)(b2)
    b2 = Conv2D(48, 1, 1, activation='relu', name='fc8_voc12_2')(b2)

    # hole = 18
    b3 = Lambda(lambda s: s[:,:,:,2048:3072])(s)
    #b3 = SeparableConv2D(1024, (3, 3), dilation_rate=(18,18), activation='relu', name='fc7_1_1',padding='same',use_bias=False)(b41)
    b3 = BatchNormalization()(b3)
    #b3 = Dropout(0.5)(b3)
    b3 = Conv2D(1024, 1, 1, activation='relu', name='fc7_3')(b3)
    b3 = BatchNormalization()(b3)
    #b3 = Dropout(0.5)(b3)
    b3 = Conv2D(48, 1, 1, activation='relu', name='fc8_voc12_3')(b3)

    # hole = 24
    b4 = Lambda(lambda s: s[:,:,:,3072:4096])(s)
    #b4 = SeparableConv2D(1024, (3, 3), dilation_rate=(24,24), activation='relu', name='fc6_1_2',padding='same',use_bias=False)(b41)
    b4 = BatchNormalization()(b4)
    #b4 = Dropout(0.5)(b4)
    b4 = Conv2D(1024, 1, 1, activation='relu', name='fc7_4')(b4)
    b4 = BatchNormalization()(b4)
    #b4 = Dropout(0.5)(b4)
    b4 = Conv2D(48, 1, 1, activation='relu', name='fc8_voc12_4')(b4)

    s = Concatenate()([b1, b2, b3, b4,layer])
    layer=s

############################################-----------------ATTENTION---1
    shape = int_shape(layer)
    gating = Conv2D(shape[3]*2, (3, 3),padding='same', use_bias=False,kernel_initializer=intial)(layer)
    gating = BatchNormalization()(gating)       
    gating = Activation('relu')(gating)
##########################################
    shape_x=int_shape(layer4)
    shape_g=int_shape(gating)
    theta_x = Conv2D(64, (2, 2), strides=(2, 2), padding='same',kernel_initializer=intial)(layer4) 
    shape_theta_x = int_shape(theta_x)
    
    phi_g = Conv2D(64, (1, 1), padding='same',kernel_initializer=intial)(gating)
    upsample_g = Conv2DTranspose(64, (3, 3),strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),padding='same')(phi_g)
    concat_xg = tf.keras.layers.Add()([upsample_g, theta_x])
    act_xg = Activation('relu')(concat_xg)
    psi = Conv2D(1, (1, 1), padding='same',kernel_initializer=intial)(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = int_shape(sigmoid_xg)
    upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32
    upsample_psi = expend_as(upsample_psi, shape_x[3])
    y = tf.keras.layers.Multiply()([upsample_psi, layer4])
    result = Conv2D(shape_x[3], (1, 1), padding='same',kernel_initializer=intial)(y)
    result_bn = BatchNormalization()(result)

##########################################

    up = concatenate([Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',activation="relu")(layer),result_bn], axis=3)
#############################################################-------------------



############################################-----------------ATTENTION-----2
    shape = int_shape(up)
    gating =  Conv2D(shape[3]*2, (3, 3),padding='same', use_bias=False,kernel_initializer=intial)(up)
    gating = BatchNormalization()(gating)       
    gating = Activation('relu')(gating)
##########################################
    shape_x=int_shape(layer3)
    shape_g=int_shape(gating)
    theta_x = Conv2D(64, (2, 2), strides=(2, 2), padding='same',kernel_initializer=intial)(layer3) 
    shape_theta_x = int_shape(theta_x)
    
    phi_g = Conv2D(64, (1, 1), padding='same',kernel_initializer=intial)(gating)
    upsample_g = Conv2DTranspose(64, (3, 3),strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),padding='same')(phi_g)
    concat_xg = tf.keras.layers.Add()([upsample_g, theta_x])
    act_xg = Activation('relu')(concat_xg)
    psi = Conv2D(1, (1, 1), padding='same',kernel_initializer=intial)(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = int_shape(sigmoid_xg)
    upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32
    upsample_psi = expend_as(upsample_psi, shape_x[3])
    y = tf.keras.layers.Multiply()([upsample_psi, layer3])
    result = Conv2D(shape_x[3], (1, 1), padding='same',kernel_initializer=intial)(y)
    result_bn = BatchNormalization()(result)

##########################################

    up = concatenate([Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',activation="relu")(up), result_bn], axis=3)
#############################################################-------------------



############################################-----------------ATTENTION-----3
    shape = int_shape(up)
    gating =  Conv2D(shape[3]*2, (3, 3),padding='same', use_bias=False,kernel_initializer=intial)(up)
    gating = BatchNormalization()(gating)       
    gating = Activation('relu')(gating)
##########################################
    shape_x=int_shape(layer2)
    shape_g=int_shape(gating)
    theta_x = Conv2D(64, (2, 2), strides=(2, 2), padding='same',kernel_initializer=intial)(layer2) 
    shape_theta_x = int_shape(theta_x)
    
    phi_g = Conv2D(64, (1, 1), padding='same',kernel_initializer=intial)(gating)
    upsample_g = Conv2DTranspose(64, (3, 3),strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),padding='same')(phi_g)
    concat_xg = tf.keras.layers.Add()([upsample_g, theta_x])
    act_xg = Activation('relu')(concat_xg)
    psi = Conv2D(1, (1, 1), padding='same',kernel_initializer=intial)(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = int_shape(sigmoid_xg)
    upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32
    upsample_psi = expend_as(upsample_psi, shape_x[3])
    y = tf.keras.layers.Multiply()([upsample_psi, layer2])
    result = Conv2D(shape_x[3], (1, 1), padding='same',kernel_initializer=intial)(y)
    result_bn = BatchNormalization()(result)

##########################################

    up = concatenate([Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',activation="relu")(up), result_bn], axis=3)
##########################################

############################################-----------------ATTENTION-----4
    shape = int_shape(up)
    gating =  Conv2D(shape[3]*2, (3, 3),padding='same', use_bias=False,kernel_initializer=intial)(up)
    gating = BatchNormalization()(gating)       
    gating = Activation('relu')(gating)
##########################################
    shape_x=int_shape(layer1)
    shape_g=int_shape(gating)
    theta_x = Conv2D(64, (2, 2), strides=(2, 2), padding='same',kernel_initializer=intial)(layer1) 
    shape_theta_x = int_shape(theta_x)
    
    phi_g = Conv2D(64, (1, 1), padding='same',kernel_initializer=intial)(gating)
    upsample_g = Conv2DTranspose(64, (3, 3),strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),padding='same')(phi_g)
    concat_xg = tf.keras.layers.Add()([upsample_g, theta_x])
    act_xg = Activation('relu')(concat_xg)
    psi = Conv2D(1, (1, 1), padding='same',kernel_initializer=intial)(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = int_shape(sigmoid_xg)
    upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32
    upsample_psi = expend_as(upsample_psi, shape_x[3])
    y = tf.keras.layers.Multiply()([upsample_psi, layer1])
    result = Conv2D(shape_x[3], (1, 1), padding='same',kernel_initializer=intial)(y)
    result_bn = BatchNormalization()(result)

##########################################

    up = concatenate([Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',activation="relu")(up), result_bn], axis=3)

##########################################



    layer1 = Conv2D(3, (1, 1) ,padding='same',kernel_initializer=intial)(up)


    layer = Conv2D(1, (1, 1) ,padding='same',kernel_initializer=intial)(layer1)


    out = Activation('sigmoid')(layer)

    model = tf.keras.Model(inputs=inputs, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learn), loss = tversky_loss, metrics=[dice_coef,'accuracy'])
    
    
    
    return model

model = attention()
model.summary()


# In[9]:


from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger,EarlyStopping,ReduceLROnPlateau

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
mc = ModelCheckpoint('CrF1ASPP_Attention_DeepMIR.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
cv = CSVLogger('CrF1ASPP_Attention_DeepMIR.csv',append=True)
rlp = ReduceLROnPlateau(monitor='val_loss', factor=0.01,patience=50, min_lr=0.0001)


# In[10]:





# In[11]:


cvscores, dc,precision,recall,sensitivity,specificity,accuracy,iou,ICC = []
for cr_vali in range(0,5):
    x_train, x_valid, y_train, y_valid = train_test_split(x_train1, y_train1, test_size=0.3, random_state=42,shuffle=True)
    train_x=x_train
    train_y=y_train
    valx=x_valid
    val_gt=y_valid
    
    #New generator with rotation and shear where interpolation that comes with rotation and shear are thresholded in masks. 
    #This gives a binary mask rather than a mask with interpolated values. 
    seed=24
    #from keras.preprocessing.image import ImageDataGenerator

    img_data_gen_args = dict(rotation_range=90,                         
                         width_shift_range=0.3,
                         height_shift_range=0.3,
                         shear_range=0.5,
                         zoom_range=0.3,
                         horizontal_flip=True,
                         vertical_flip=True,
                         fill_mode='reflect')

    mask_data_gen_args = dict(rotation_range=90,
                         width_shift_range=0.3,
                         height_shift_range=0.3,
                         shear_range=0.5,
                         zoom_range=0.3,
                         horizontal_flip=True,
                         vertical_flip=True,
                         fill_mode='reflect',
                         preprocessing_function = lambda x: np.where(x>0, 1, 0).astype(x.dtype)) #Binarize the output again. 

    image_data_generator = ImageDataGenerator(**img_data_gen_args)
    image_data_generator.fit(x_train, augment=True, seed=seed)

    image_generator = image_data_generator.flow(x_train, seed=seed)
    valid_img_generator = image_data_generator.flow(x_valid, seed=seed)

    mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
    mask_data_generator.fit(y_train, augment=True, seed=seed)
    mask_generator = mask_data_generator.flow(y_train, seed=seed)
    valid_mask_generator = mask_data_generator.flow(y_valid, seed=seed)

    def my_image_mask_generator(image_generator, mask_generator):
        train_generator = zip(image_generator, mask_generator)
        for (img, mask) in train_generator:
            yield (img, mask)

    my_generator = my_image_mask_generator(image_generator, mask_generator)

    validation_datagen = my_image_mask_generator(valid_img_generator, valid_mask_generator)

    steps_per_epoch = 3*(len(x_train))//Batch_size

    results = model.fit(my_generator, validation_data=validation_datagen, steps_per_epoch=steps_per_epoch, validation_steps=steps_per_epoch, epochs=num_epoch, shuffle=True,callbacks=[es,mc, cv, rlp])

    model.save('F1_ASPP_AttentionDeepMIR_10_10_22_DA.h5')
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
    
    # For the checkpoint file with threshold 0.5
    from medpy.metric import dc, precision, recall
    #import tensorflow as tf
    #model.load_weights('F2_ASPP_AttentionDeepMIR_12_08_22_DA.h5')
    #model.load_weights('New_Ensemble_xception_Hippo_subsampled_2D_deeplab_crop_numberofepoch_'+str(Patch_width)+'_'+str(Patch_height)+'_'+str(step)+'_TVL_0.5_0.5_Normalized_'+str(num_epoch)+'Batch_size'+ str(Batch_size)+ '_2.h5')
    #model.load_weights('New_Ensemble_xception_Hippo_subsampled_2D_deeplab_crop_numberofepoch_'+str(Patch_width)+'_'+str(Patch_height)+'_'+str(step)+'_TVL_0.5_0.5_Normalized_'+str(num_epoch)+'Batch_size'+ str(Batch_size)+ '_3.h5')
    pred = model.predict(x_test, batch_size=2)
    print(pred.shape)
    threshold = 0.5
    print("---------------Threshold = " + str(threshold) + " --------------")
    pred[pred >= threshold] = 1
    pred[pred < threshold] = 0
    dc = dc(pred,y_test)
    precision = precision(pred,y_test)
    recall = recall(pred,y_test)

    print("Test dc is : " + str(dc))
    print("Test precision is : " + str(precision))
    print("Test recall is : " + str(recall))


    from sklearn.metrics import confusion_matrix, f1_score, jaccard_score

    y_test[y_test >= threshold] = 1
    y_test[y_test < threshold] = 0

    #pred_img[pred_img != lesion] = 0
    #pred_img = pred_img / lesion
    pred_linear = np.reshape(pred, (pred.shape[0] * pred.shape[1] * pred.shape[2]))

    #gt_img[gt_img != lesion] = 0
    #gt_img = gt_img / lesion
    gt_linear = np.reshape(y_test, (y_test.shape[0] * y_test.shape[1] * y_test.shape[2]))

    tn, fp, fn, tp = confusion_matrix(y_true = gt_linear, y_pred = pred_linear).ravel()
    sen = tp / (tp + fn)
    prec = tp / (tp + fp)
    specificity = tn / (tn + fp)
    accu = (tp + tn) / (tp + tn + fp + fn)

    #haus_score = -1
    #dc = f1_score(y_true = gt_linear, y_pred = pred_linear)
    iou = jaccard_score(y_true = gt_linear, y_pred = pred_linear)

    import pingouin as pg
    ICC= pg.corr(gt_linear, pred_linear)


    print("Test specificity is : " + str(specificity))
    print("accu is : " + str(accu))
    print("Iou is : " + str(iou))
    print("ICC is : " + str(ICC))
    dc = dc.append(dc * 100)
    precision = precision.append(precision * 100)
    recall = recall.append(recall * 100)
    sensitivity = sensitivity.append(sen * 100)
    specificity = specificity.append(specificity * 100)
    accuracy = accuracy.append(accu * 100)
    iou = iou.append(iou * 100)
    ICC = ICC.append(ICC)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
print("%.2f%% (+/- %.2f%%)" % (np.mean(dc), np.std(dc)))
print("%.2f%% (+/- %.2f%%)" % (np.mean(precision), np.std(precision)))
print("%.2f%% (+/- %.2f%%)" % (np.mean(recall), np.std(recall)))
print("%.2f%% (+/- %.2f%%)" % (np.mean(sensitivity), np.std(sensitivity)))
print("%.2f%% (+/- %.2f%%)" % (np.mean(specificity), np.std(specificity)))
print("%.2f%% (+/- %.2f%%)" % (np.mean(accuracy), np.std(accuracy)))
print("%.2f%% (+/- %.2f%%)" % (np.mean( iou), np.std( iou)))
print("%.2f%% (+/- %.2f%%)" % (np.mean(ICC), np.std(ICC)))


# In[12]:


#print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


# In[12]:





# In[ ]:




