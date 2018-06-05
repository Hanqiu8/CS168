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

