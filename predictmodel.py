from keras.models import Model, load_model 
from util import utility

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt

autoencoder = load_model('./saved-models/resized-fingers-ae-500-epoch.h5') 
x_test = utility.getfingerImages('./data/trial-fingers/')

encoded_imgs = autoencoder.predict(x_test) 

print encoded_imgs.shape

theimg = encoded_imgs[0] 

plt.imshow(encoded_imgs[0]) 
ax = plt.gca() 
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(True)
ax.axis('off')
imgtitle = './renders/myrenderedfinger.pdf' 

plt.savefig(imgtitle)

print 'done'





