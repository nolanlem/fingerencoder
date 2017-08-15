#!/usr/bin/env python

from keras.layers import Input, Dense
from keras.models import Model, load_model 
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import os
import sys
import h5py

from PIL import Image 
import glob

import argparse

# my utility module for loading imgs
from util import utility

# set up argument parser
parser = argparse.ArgumentParser(description='autoenc arg parser')
parser.add_argument('-path','--path', action='store', dest='thedir', default='./data/')
parser.add_argument('-epochs','--epochs', action='store', dest='numberepochs', default=500)

results = parser.parse_args()
print 'the path is ', results.thedir

# DEFINE MODEL ARCHITECTURE
INPUT_SHAPE = (1000, 300, 3)
input_img = Input(shape=INPUT_SHAPE)  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same', name='encoder')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3,3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
decoder = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoder)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
print autoencoder.summary()


def getImages(sourcedir, crop=False):
    if os.path.isdir(sourcedir) == True: 
        print 'reading files from %r' %(sourcedir)
        imgdata = np.zeros((1,1000,300,3))
        basename = os.path.basename(glob.glob(sourcedir + '/*')[1])
        imgformat = os.path.splitext(basename)[1]
        theimgs = glob.glob(sourcedir + "*" + imgformat)
        numimgs = len(theimgs)
        print 'there are %r files and they are of %r type' %(numimgs, imgformat)
        for i,img in enumerate(theimgs):
            sys.stdout.write('Reading %r/%r---> %s \r' %(i,numimgs,img))
            sys.stdout.flush()
            sourceimg = mpimg.imread(img)
            sourceimg = np.array(sourceimg, dtype='float32')
            sourceimg = sourceimg/255.0

            sourceimg = np.reshape(sourceimg,(1,sourceimg.shape[0],sourceimg.shape[1],sourceimg.shape[2])) 
            #print sourceimg.shape
            imgdata = np.vstack((imgdata,sourceimg))
            #print imgdata.shape
        imgdata = imgdata[1:] # remove first entry of zeros
    else:
        print '%r not a dir' %(sourcedir)
        imgdata = 0 
    print '...finished \n \n'
    return imgdata 


# get the images loaded 
# thedir = '/Volumes/nolanHD/pro-arts-show/fingers/data/resized-fingers/'
#thedir = './data/resized-fingers/'
# thedir = './data/trial-fingers/' # small dir for testing 
thedir = results.thedir

# separate training and testing data
x_train = getImages(thedir)
print 'done'
test_end = int(len(x_train)*0.9)
x_test = x_train[test_end:]
x_train = x_train[:test_end]

from keras.callbacks import ModelCheckpoint

# train ae for N epochs 
numepoch = int(results.numberepochs)
modelfn = 'resized-fingers-ae-' + str(numepoch) + '-epoch' # model name
checkptname = modelfn + '.h5'
#checkptname_ = checkptname
modelname = './saved-models/' + checkptname
checkpointer = ModelCheckpoint(filepath=modelname, verbose=1, save_best_only=True, save_weights_only=False)

print 'training autoencoder for %r epochs' %(numepoch)
# TRAIN THE MODEL
autoencoder.fit(x_train, x_train, epochs=numepoch, 
                batch_size=50, 
                shuffle=True,
                validation_data=(x_test,x_test),
                callbacks=[checkpointer]
               ) 

autoencoder.save(modelname)

print 'done, now load .h5 model and run prediction module below...'

# MAKE PREDICTION  and RECONSTRUCT
encoded_imgs = autoencoder.predict(x_test) 

print encoded_imgs.shape
theimg = encoded_imgs[0]
#theimg = x_train[0]
print theimg.shape
plt.imshow(theimg)
ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.axis('off')
imgtitle = modelfn + '_rendered_finger.pdf'
#plt.savefig('./renders/rendered_finger.pdf')
plt.savefig(imgtitle)

plt.show()
plt.close()

print '!!!!!!!!!!!done!!!!!!!!!!!!'
