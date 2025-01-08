#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


#fold = 1                 # fold Number
#test_D = 21-r            # Test data

import patchify
from patchify import patchify

Patch_width = 64
Patch_height = 64
step=32

num_epoch = 1500
Batch_size=64

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import nibabel as nib
from scipy import ndimage
import math
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K
import tensorflow.keras
from tensorflow.python.keras.callbacks import CSVLogger
import shutil
from tensorflow.python.keras.layers import Concatenate
import shutil
import numpy as np
import time

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, Add,BatchNormalization,Conv2D, Input, UpSampling2D, Activation, concatenate, MaxPooling2D, Conv2DTranspose, Reshape, Permute,Add
from tensorflow.python.keras.layers.core import Lambda, RepeatVector, Reshape
from tensorflow.python.keras.layers.convolutional import Conv2D, Conv2DTranspose
from tensorflow.python.keras.layers.pooling import MaxPooling3D, GlobalMaxPool3D, MaxPooling2D, MaxPool3D
from tensorflow.python.keras.layers.merge import concatenate, add
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras import regularizers
from tensorflow.python.keras.layers import UpSampling2D

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


#pip install medpy
import medpy
import copy

# In[2]:

from sklearn.model_selection import train_test_split
import numpy as np



# In[ ]:


def recall_m(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	recall = true_positives / (possible_positives + K.epsilon())
	return recall

def precision_m(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return precision

def f1_m(y_true, y_pred):
	precision = precision_m(y_true, y_pred)
	recall = recall_m(y_true, y_pred)
	return 2*((precision*recall)/(precision+recall+K.epsilon()))

def dice_coef(y_true, y_pred, smooth=1.):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def tversky_loss(y_true, y_pred, alpha=0.5, beta=0.5, smooth=1e-10):
	y_true = K.flatten(y_true)
	y_pred = K.flatten(y_pred)
	truepos = K.sum(y_true * y_pred)
	fp_and_fn = alpha * K.sum(y_pred * (1 - y_true)) + beta * K.sum((1 - y_pred) * y_true)
	answer = 1 - (truepos + smooth) / ((truepos + smooth) + fp_and_fn)
	return answer
	# Combined loss of weighted multi-class logistic loss and dice loss
def customized_loss(y_true, y_pred):
	return (0.1 * K.categorical_crossentropy(y_true, y_pred)) + (1 * tversky_loss(y_true, y_pred))


# In[13]:


def iou(y_true, y_pred, label: int):
	# extract the label values using the argmax operator then
	# calculate equality of the predictions and truths to the label
	y_true = K.cast(K.equal(K.argmax(y_true), label), K.floatx())
	y_pred = K.cast(K.equal(K.argmax(y_pred), label), K.floatx())
	# y_true = K.cast(y_true, K.floatx())
	# y_pred = K.cast(y_pred, K.floatx())

	# calculate the |intersection| (AND) of the labels
	intersection = K.sum(y_true * y_pred)

	# calculate the |union| (OR) of the labels
	union = K.sum(y_true) + K.sum(y_pred) - intersection


	# intersection = K.print_tensor(intersection, message = "intersection: ")
	# union = K.print_tensor(union, message = "union: ")

	# avoid divide by zero - if the union is zero, return 1
	# otherwise, return the intersection over union
	ret = K.switch(K.equal(union, 0), 1.0, intersection / union)
	# ret = K.print_tensor(ret, message = "iou: ")
	return ret

def mean_iou(y_true, y_pred):
	   

	# get number of labels to calculate IoU for
	num_labels = K.int_shape(y_pred)[-1]

	# initialize a variable to store total IoU in
	# total_iou = K.variable(value = 0, name = "total_iou")

	total_iou = 0

	# print("Num Labels:", num_labels)
	# iterate over labels to calculate IoU for
	for label in range(1, num_labels):
		# print("Label:", label)
		total_iou = total_iou + iou(y_true, y_pred, label)
		# divide total IoU by number of labels to get mean IoU
	return total_iou / num_labels


# In[14]:
# In[15]:


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

# In[16]:


# In[9]:


intial="he_normal"
def attention(learn = 3e-4):
	    

	inputs = Input((96,48,3))
	    


	layer = Conv2D(32, (7, 7),padding='same', use_bias=False,kernel_initializer=intial)(inputs)
	layer = Activation('relu')(layer)


	for i in range(2):
		layer = Conv2D(64, (7, 7),padding='same', use_bias=False,kernel_initializer=intial)(layer)
		layer = BatchNormalization()(layer)
		layer1 = Activation('relu')(layer)
		layer=layer1

	pool1 = MaxPooling2D(pool_size=(2, 2))(layer1)
	layer=pool1
	for i in range(2):
		layer = Conv2D(64, (5, 5),padding='same', use_bias=False,kernel_initializer=intial)(layer)
		layer = BatchNormalization()(layer) 
		layer2 = Activation('relu')(layer)
		layer=layer2

		 

	pool2 = MaxPooling2D(pool_size=(2, 2))(layer2)

	layer=pool2
	for i in range(4):

		layer = Conv2D(128, (3, 3),padding='same', use_bias=False,kernel_initializer=intial)(layer)
		layer = BatchNormalization()(layer)       
		layer3 = Activation('relu')(layer)
		layer=layer3

	pool3 = MaxPooling2D(pool_size=(2, 2))(layer3)
	layer=pool3

	for i in range(4):

		layer = Conv2D(256, (3, 3),padding='same', use_bias=False,kernel_initializer=intial)(layer)
		layer = BatchNormalization()(layer)       
		layer4 = Activation('relu')(layer)
		layer=layer4


	pool4 = MaxPooling2D(pool_size=(2, 2))(layer4)

	layer=pool4
	for i in range(8):
		layer = Conv2D(512, (3, 3),padding='same', use_bias=False,kernel_initializer=intial)(layer)
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


# In[17]:







name =["21-r", "22", "23-r", "24-r", "28-r", "29-r", "35", "36", "37-r", "38-1", "39",  "40", "42",  "45", "48"]
# In[2]:
for fold in range(1,16):
	test_D = name[fold-1]
	print(fold,test_D)
	Data1=np.load('/media/abn/Hard_Disk/Hippocampus_subfield_segmentation_30112023/Fold'+str(fold)+'/Data_Train_Fold'+str(fold)+'_3channel.npy')
	T1_Data=Data1
	T1_Data=np.asarray(T1_Data)
	print(T1_Data.shape)

	Val1=np.load('/media/abn/Hard_Disk/Hippocampus_subfield_segmentation_30112023/Fold'+str(fold)+'/Data_Validation_Fold'+str(fold)+'_3channel.npy')
	T1_Data_Val=Val1
	T1_Data_Val=np.asarray(T1_Data_Val)
	print(T1_Data_Val.shape)


	# In[3]:


	Label1=np.load('/media/abn/Hard_Disk/Hippocampus_subfield_segmentation_30112023/Fold'+str(fold)+'/Label_Train_Fold'+str(fold)+'.npy')
	T1_Label=Label1
	T1_Label=np.asarray(T1_Label)
	print(T1_Label.shape)

	L_Val1=np.load('/media/abn/Hard_Disk/Hippocampus_subfield_segmentation_30112023/Fold'+str(fold)+'/Label_Validation_Fold'+str(fold)+'.npy')
	T1_Label_Val=L_Val1
	T1_Label_Val=np.asarray(T1_Label_Val)
	print(T1_Label_Val.shape)


	# In[4]:


	print(T1_Data.shape)
	print(T1_Label.shape)


	# In[5]:


	#T1_Data=np.asarray(T1_Data,dtype=np.float32)
	#T1_Label=np.asarray(T1_Label,dtype=np.float32)


	# In[6]:


	print(T1_Data.shape)
	print(T1_Label.shape)





	# In[7]:


	New_train_xx=np.reshape(T1_Data, (-1,96,48, 3))
	New_train_yy=np.reshape(T1_Label, (-1,96,48, 1))
	New_valid_xx=np.reshape(T1_Data_Val, (-1,96,48, 3))
	New_valid_yy=np.reshape(T1_Label_Val, (-1,96,48, 1))

	New_Data=New_train_xx/255
	New_Label=New_train_yy
	valid_x=New_valid_xx/255
	valid_y=New_valid_yy/255
	#test_x=New_test_xx/255
	#test_y=New_test_yy/255

	#train_x1=New_train_xx1/255
	#train_y1=New_train_yy1/255
	print(New_Data.shape)
	print(New_Label.shape)
	print(valid_x.shape)
	print(valid_y.shape)
	#print(test_x.shape)
	#print(test_y.shape)


	# In[8]:


	# In[8]:


	#from sklearn.model_selection import train_test_split

	#x_train, x_valid, y_train, y_valid = train_test_split(New_Data, New_Label, test_size=0.33, random_state=42,shuffle=True)
	#x_train, x_valid, y_train, y_valid = train_test_split(x_train1, y_train1, test_size=0.33, random_state=42,shuffle=True)


	# In[9]:


	x_train = New_Data.astype('float32')
	x_valid = valid_x.astype('float32')
	y_train = New_Label.astype('float32')
	y_valid = valid_y.astype('float32')
	#x_test = x_test.astype('float32')
	#y_test = y_test.astype('float32')


	# In[10]:


	mean = np.mean(x_train)
	std = np.std(x_train)
	x_train  = x_train - mean
	x_train  = x_train  / std

	x_valid = x_valid - mean
	x_valid = x_valid / std

	#x_test = x_test - mean
	#x_test = x_test / std

	
	# In[19]:


	Test_Data_SWI = nib.load('/media/abn/Hard_Disk/Hippocampus_subfield_segmentation_30112023/Fold'+str(fold)+'/Test/Data/PMBM'+str(test_D)+'_Reslice_Reoriented_FluidRemoved_cropped.nii.gz')
	Test_Data1_SWI = Test_Data_SWI.get_fdata()
	Test_Data_T1 = nib.load('/media/abn/Hard_Disk/Hippocampus_subfield_segmentation_30112023/Fold'+str(fold)+'/Test/Data/PMBM'+str(test_D)+'_Rigid_reg_t1_to_SWI_cropped.nii.gz')
	Test_Data1_T1 = Test_Data_T1.get_fdata()
	Test_Data_T2 = nib.load('/media/abn/Hard_Disk/Hippocampus_subfield_segmentation_30112023/Fold'+str(fold)+'/Test/Data/PMBM'+str(test_D)+'_Rigid_reg_t2_to_SWI_cropped.nii.gz')
	Test_Data1_T2 = Test_Data_T2.get_fdata()
	Test_Label = nib.load('/media/abn/Hard_Disk/Hippocampus_subfield_segmentation_30112023/Fold'+str(fold)+'/Test/Label/PMBM'+str(test_D)+'_Reslice_Reoriented_Label_S0_cropped.nii.gz')
	Test_Label1 = Test_Label.get_fdata()


	# In[22]:


	import numpy
	Test_Data1_SWI=np.asarray(Test_Data1_SWI)
	Test_Data1_T1=np.asarray(Test_Data1_T1)
	Test_Data1_T2=np.asarray(Test_Data1_T2)
	Test_Label1=np.asarray(Test_Label1)
	width1 = 96
	height1 = 48
	img_stack_sm2 = np.zeros((len(Test_Data1_SWI), width1, height1,3))

	for idx in range(len(Test_Data1_SWI)):
		#if (np.max(Label_2D[idx])!=0):
		KD2=Test_Data1_SWI[idx,:,:]
		a=numpy.amin(KD2)
		b=numpy.amax(KD2)
		KD3=((KD2-a)/(b-a))*255
		KD5=KD3
		KD6=KD5.astype('uint8')
		K2=KD6
		img_stack_sm2[idx,:,:,0]=K2
		
		KD2=Test_Data1_T1[idx,:,:]
		a=numpy.amin(KD2)
		b=numpy.amax(KD2)
		KD3=((KD2-a)/(b-a))*255
		KD5=KD3
		KD6=KD5.astype('uint8')
		K2=KD6
		img_stack_sm2[idx,:,:,1]=K2
		
		KD2=Test_Data1_T2[idx,:,:]
		a=numpy.amin(KD2)
		b=numpy.amax(KD2)
		KD3=((KD2-a)/(b-a))*255
		KD5=KD3
		KD6=KD5.astype('uint8')
		K2=KD6
		img_stack_sm2[idx,:,:,2]=K2
        



	Test_Data1 = img_stack_sm2

	Test_Data2=np.reshape(Test_Data1, (-1,96,48, 3))
	Test_Label2=np.reshape(Test_Label1, (-1,96,48, 1))
	Test_Data3=Test_Data2/255

	print(Test_Data3.shape)
	print(Test_Label2.shape)
	x_test = Test_Data3
	y_test = Test_Label2
	x_test  = x_test - mean
	x_test  = x_test  / std


	# In[ ]:





	# In[23]:


	# For the checkpoint file with threshold 0.5
	from medpy.metric import dc, precision, recall
	from sklearn.metrics import jaccard_score
	import tensorflow as tf
	import scipy
	import numpy
	model.load_weights('/media/abn/Hard_Disk/Hippocampus_subfield_segmentation_30112023/Fold'+str(fold)+'/Hippocampus_So_Fold_'+str(fold)+'_DeepAIM_06_12_23_3channel.h5')
	#model.load_weights('New_Ensemble_xception_Hippo_subsampled_2D_deeplab_crop_numberofepoch_'+str(Patch_width)+'_'+str(Patch_height)+'_'+str(step)+'_TVL_0.5_0.5_Normalized_'+str(num_epoch)+'Batch_size'+ str(Batch_size)+ '_2.h5')
	#model.load_weights('New_Ensemble_xception_Hippo_subsampled_2D_deeplab_crop_numberofepoch_'+str(Patch_width)+'_'+str(Patch_height)+'_'+str(step)+'_TVL_0.5_0.5_Normalized_'+str(num_epoch)+'Batch_size'+ str(Batch_size)+ '_3.h5')
	pred = model.predict(x_test, batch_size=64)
	print(pred.shape)
	print(y_test.shape)
	threshold = 0.5
	print("---------------Threshold = " + str(threshold) + " --------------")
	pred[pred >= threshold] = 1
	pred[pred < threshold] = 0
	dc = dc(pred,y_test)
	precision = precision(pred,y_test)
	recall = recall(pred,y_test)
	JAC=scipy.spatial.distance.jaccard(numpy.ndarray.flatten(y_test), numpy.ndarray.flatten(pred))
	#JAC1=jaccard_score(numpy.ndarray.flatten(y_test), numpy.ndarray.flatten(pred))
	
	#print("Test Jaccard index is : " + str(JAC1))
	#print(y_test)
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

	print("Test dc is : " + str(dc))
	print("Test precision is : " + str(precision))
	print("Test recall is : " + str(recall))
	print("Test specificity is : " + str(specificity))
	print("Test Jaccard index is : " + str(JAC))
	print("accu is : " + str(accu))
	print("Iou is : " + str(iou))
	print("ICC is : " + str(ICC))




	# In[26]:


	F_removed = np.reshape(pred, (-1,96,48))


	# In[27]:


	final_img = nib.Nifti1Image(F_removed, Test_Data_SWI.affine)
	nib.save(final_img, '/media/abn/Hard_Disk/Hippocampus_subfield_segmentation_30112023/Fold'+str(fold)+'/Test/Result/DeepAIM_PMBM'+str(test_D)+'_Reslice_Reoriented_FluidRemoved_3channel.nii.gz')



