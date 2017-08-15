import sys
import numpy as np
from PIL import Image 
import os 
import glob
import matplotlib.image as mpimg


def myfunc(s):
	print 'yo'

# function to extract jpgs from sourcedir to numpy array for training
def getImages(sourcedir):
	if os.path.isdir(sourcedir) == True: 
		#print os.path.isdir(sourcedir)
		print 'reading files from %r' %(sourcedir)
		imgdata = np.zeros((1,360*360))
		basename = os.path.basename(glob.glob(sourcedir + '/*')[1])
		imgformat = os.path.splitext(basename)[1]
		print 'files are of %r type' %(imgformat)
		for img in glob.glob(sourcedir + "*" + imgformat):
			sys.stdout.write('\r Reading ---> %s' %img)
			sys.stdout.flush()
			
			sourceimg = Image.open(img)
			width = sourceimg.size[0]
			height = sourceimg.size[1]
			sourceimg = sourceimg.crop((width-width, height-width-20,width,height-20))         
			sourceimg = np.asarray(sourceimg, dtype='float32')
	        #sourceimg = sourceimg.reshape((1,sourceimg.shape[0]*sourceimg.shape[1]))
			sourceimg = sourceimg/255.
			sourceimg = sourceimg.reshape((1,sourceimg.shape[0]*sourceimg.shape[1])) 
			imgdata = np.vstack((imgdata,sourceimg)) 
		imgdata = imgdata[1:] # remove first entry of zeros
	else:
		print '%r not a dir' %(sourcedir)
		imgdata = 0 
	print '...finished'
	return imgdata 

def getfingerImages(sourcedir, crop=False):
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


