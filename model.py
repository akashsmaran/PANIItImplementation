import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import natsort
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import img_to_array
from sklearn import model_selection
import pickle
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import LeakyReLU
from pandas import ExcelWriter
from pandas import ExcelFile

"""
path="training/training"
im_file=os.listdir(path)
im_file=natsort.natsorted(im_file)
info=pd.read_csv("training/solution.csv")

data=[]
labels=[]
img_dim=(240,240,3)
for f in im_file:
    im=cv2.imread(path+'/'+f)
    im=cv2.resize(im,(img_dim[1],img_dim[0]))
    #cv2.imshow("img",im)
    #cv2.waitKey(0) and ord('q')
    image=img_to_array(im)
    data.append(image)
for i in range(5000):
    label=info.category[i]
    labels.append(label)

num_classes=len(np.unique(labels))
data=np.array(data,dtype='float32')/255.0
labels=np.array(labels)
labels2=labels

lb=LabelBinarizer()
labels=lb.fit_transform(labels)
x_train,x_test,y_train,y_test=model_selection.train_test_split(data,labels,test_size=0.25,random_state=42)

model=Sequential()
model.add(Conv2D(32,(3,3),padding="same",activation="linear",input_shape=(240,240,3)))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(0.25))

model.add(Conv2D(32,(3,3),padding="same",activation="linear"))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),padding="same",activation="linear"))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3),padding="same",activation="linear"))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128,(3,3),padding="same",activation="relu"))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024,activation="linear"))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(num_classes,activation="softmax"))

model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=[keras_metrics.precision()])
model.summary()

checkpoint=ModelCheckpoint(filepath='model_best.hdf5',save_best_only=True,verbose=1)
train=model.fit(x_train,y_train,batch_size=100,epochs=40,validation_data=(x_test,y_test),verbose=0, callbacks=[checkpoint])
"""
###########################################



path2="testing"
im_file=os.listdir(path2)
im_file=natsort.natsorted(im_file)

model=load_model("model_best1.hdf5")
lb=pickle.loads(open("lb.pickle","rb").read())

labels=[]
img_dim=(240,240,3)
for f in im_file:
    im=cv2.imread(path2+'/'+f)
    im=cv2.resize(im,(img_dim[1],img_dim[0]))
    im=im.astype("float32")/255.0
    im=img_to_array(im)
    im=np.expand_dims(im,axis=0)
    proba=model.predict(im)[0]
    idx=np.argmax(proba)
    label=lb.classes_[idx]
    labels.append(label)

ids = [i for i in range(40000)]
df = pd.DataFrame({'id':ids,
                   'category':labels})
 
writer = ExcelWriter('abc.xlsx')
df.to_excel(writer,'Sheet1',index=False)
writer.save()

