from __future__ import division
from __future__ import print_function

import random
import numpy as np
import cv2
import os
from Cropping import cropping, prepareImg
from PIL import Image


def preprocessingImg(img, imgSize, dataAugmentation=False):
	
	if img is None:
		img = np.zeros([imgSize[1], imgSize[0]])

	
	if dataAugmentation:
		stretch = (random.random() - 0.5) 
		wStretched = max(int(img.shape[1] * (1 + stretch)), 1) 
		img = cv2.resize(img, (wStretched, img.shape[0])) 
	
	#Smooth
	img = cv2.GaussianBlur(img,(5,5),0)
	
	#increase contrast
	pxmin = np.min(img)
	pxmax = np.max(img)
	imgContrast = (img - pxmin) / (pxmax - pxmin) * 255


	np.seterr(divide='ignore', invalid='ignore')

	#increase line width
	kernel = np.ones((3, 3), np.uint8)
	imgMorph = cv2.erode(imgContrast, kernel, iterations = 1)

	cv2.imwrite('out.png', imgMorph)
	img = cv2.imread('out.png', cv2.IMREAD_GRAYSCALE)
	img= Image.fromarray(img, 'L')
	#img.show()

	#Cropping
	img = prepareImg(cv2.imread('out.png'), 50)
	res = cropping(img, kernelSize=25, sigma=11, theta=7, minArea=100)
	for (j, w) in enumerate(res):
		(wordBox, wordImg) = w
		(x, y, w, h) = wordBox
		cv2.imwrite('out.png', wordImg) 
	
	img = cv2.imread('out.png', cv2.IMREAD_GRAYSCALE)

	img = np.array(img)
	
	#dhmiourgia ths eikonas sto megethos pou theloume
	(wt, ht) = imgSize
	(h, w) = img.shape
	fx = w / wt
	fy = h / ht
	f = max(fx, fy)
	newSize = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1)) 
	img = cv2.resize(img, newSize, interpolation=cv2.INTER_CUBIC)
	target = np.ones([ht, wt]) * 255
	target[0:newSize[1], 0:newSize[0]] = img
	
	img= Image.fromarray(img, 'L')
	#img.show()

	img = cv2.transpose(target)

	#normalization
	(mean, std) = cv2.meanStdDev(img)
	mean = mean[0][0]
	std = std[0][0]
	img = img - mean
	img = img / std if std>0 else img
	return img

