import numpy as np
import pylab
from skimage import data, color, data, exposure, restoration, feature
from skimage.util.dtype import dtype_range
from skimage.util import img_as_ubyte
from skimage.morphology import disk
from skimage.filters import rank
from itertools import cycle
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk
from scipy.signal import convolve2d as conv2

class ImageFilter:

	img = []
	def __init__(self, image):
		self.img = image

	def display(self, pre_im, post_im, z):
		pylab.imshow(pre_im[z], cmap=pylab.cm.bone)
		# pylab.imshow(post_im[z], cmap=pylab.cm.bone)
		pylab.show()
      

	def histoEqualization(self, toShow=False, z = 0):
		equ_image = exposure.equalize_hist(self.img)
				
		if toShow == True:
			
			self.display(self.img, equ_image, z)
		return equ_image

	def edgeDetection(self, toShow=False, z = 0):
		return feature.canny(self.img)

	def deconvolution(self, toShow=False, z = 0):
		
		imgd = self.img
		psf = np.ones((10,10))/100
		for i in range(2):
			img_slice = conv2(self.img[i], psf, 'same')
			img_slice += 0.1 * img_slice.std() * np.random.standard_normal(img_slice.shape)
			deconvolved, _ = restoration.unsupervised_wiener(img_slice, psf)
			imgd[i] = deconvolved
		return imgd

	def truncationROIfinder(mask, masked_img_arr, ):
		#Taken from work with Heidi

		# Identify lesion
		non_zeros = np.where(mask!=0)
		start = non_zeros[0][0]
		end = non_zeros[0][-1]
		print "Lesion: "+ str(start) + "-" + str(end)

		#extract 3x3x3 ROI with max intensity
		max_roi = -float('inf')
		roi = []

		for z,y,x in zip(non_zeros[0], non_zeros[1], non_zeros[2]):
		    acc = []
		    valid = True
		    
		    for _z in range(z, min(end, z + 3)):
		        for _y in range(y, y + 3):
		            for _x in range(x, x + 3):
		                if mask[_z][_y][_x] == 0 or masked_img_arr[_z][_y][_x] > 300:
		                    valid = False
		                else:
		                    acc.append(masked_img_arr[_z][_y][_x])
		    
		    # Check if we found a new max intesity ROI
		    if valid and len(acc) >= 9 * (min(end - start, 3)) and len(acc) != 0:
		        avg = np.mean(np.asarray(acc))
		        if avg > max_roi:
		            max_roi = avg
		            roi = np.asarray(acc[:])

		if max_roi != -float('inf'):
		    max_roi = (max_roi - normalized) / normalized * 100
		else:
		    # If this failed somehow take 90th percentile and normalize
		    max_roi = (np.percentile(flat_img_arr, 90) - normalized) / normalized * 100

		return max_roi, roi

