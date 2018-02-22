import numpy as np
import math
import scipy.io
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, adam
from keras.utils import np_utils
from keras.utils.visualize_util import plot
from keras.models import load_model
import time


m = scipy.io.loadmat('testing.mat')
n = np.array(m['test'])

Xdata = np.array(n[:,1])
Xdata=Xdata[0:]

Xdata1=np.zeros((Xdata.shape[0],1,40,170))

i=0

for x in Xdata:

    Xdata1[i,0,:,:]=x
    i=i+1

Xdata1 = Xdata1.astype('float32')
Xdata1 /= 255


m = scipy.io.loadmat('training_with_error.mat')
n = np.array(m['c'])

Xdata = np.array(n[:,1])
Xdata=Xdata[0:]

Y1= np.array(n[:,2])
Y1=Y1[0:]

Xdata1=np.zeros((Xdata.shape[0],1,40,170))
Ydata1=np.zeros((Xdata.shape[0],2544))

i=0

for x in Xdata:

     Xdata1[i,0,:,:]=x
     i=i+1

i=0

for x in Y1:

     Ydata1[i,:]=x[0,0:2544]
     i=i+1

Xdata1 = Xdata1.astype('float32')
Xdata1/=255

batch_size = 64
nb_classes = 2544
nb_epoch = 1
img_rows, img_cols = 40, 170

model = Sequential()

model.add(Convolution2D(64, 5, 5, border_mode='same', input_shape=(1,40,170)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Convolution2D(64, 5, 5, border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(128, 5, 5, border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Convolution2D(128, 5, 5, border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(256, 5, 5, border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Convolution2D(256, 5, 5, border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(512, 5, 5, border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Convolution2D(512, 5, 5, border_mode='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(2000))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(2000))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax52'))

#
# def softmax52(x):
#
#     list=[];
#
#     for i in np.arange(0,48):
#
#         Temp=x[:,0+i*53:53+i*53] # IAM 53, Rimes 54,
#         Temp = K.exp(Temp - K.max(Temp, axis=1, keepdims=True))
#         Temp = Temp / K.sum(Temp, axis=1, keepdims=True)
#         list.append(Temp)
#
#     return T.concatenate(list, axis=1)


print(model.summary())

sgd = SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd,  metrics=['recall','precision'])

datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.05,  height_shift_range=0.05, zoom_range=0.1)

model.fit_generator(datagen.flow(Xdata1, Ydata1 ,batch_size=batch_size),samples_per_epoch=Xdata1.shape[0], nb_epoch=nb_epoch,class_weight=[1-int(i%53==0)+0.001 for i in range(1,2545)])

 ###################################################################


m = scipy.io.loadmat('testing.mat')
n = np.array(m['test'])

Xdata = np.array(n[:,1])
Xdata=Xdata[0:]

Xdata1=np.zeros((Xdata.shape[0],1,40,170))

i=0

for x in Xdata:

    Xdata1[i,0,:,:]=x
    i=i+1

Xdata1 = Xdata1.astype('float32')
Xdata1 /= 255

start = time.time()

r=model.predict(Xdata1)

end = time.time()

print(end - start)

print(Xdata.shape[0]/(end-start))
