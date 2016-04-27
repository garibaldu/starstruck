
import pylab as pl
import numpy as np

import scipy.ndimage.interpolation as int

def affinetrans(im,trans):
	# trans[0] = scale, trans[1] = angle, trans[2] = xshift, trans[3] = yshift

	xdim,ydim = np.shape(im)
	xgrid = np.linspace(0,xdim-1,xdim) - xdim/2.
	ygrid = np.linspace(0,ydim-1,ydim) - ydim/2.
	ygrid, xgrid = np.meshgrid(ygrid,xgrid)

	xout = trans[0]*np.cos(trans[1])*xgrid - trans[0]*np.sin(trans[1])*ygrid + trans[2] + xdim/2.
	yout = trans[0]*np.sin(trans[1])*xgrid + trans[0]*np.cos(trans[1])*ygrid + trans[3] + ydim/2.

	xout, yout = xout.flatten(), yout.flatten()
	coords = np.vstack((xout,yout))

	newim = int.map_coordinates(im, coords, order=1, mode='constant')
	return np.reshape(newim,(xdim,ydim))

def test_affinetrans():

	im = pl.imread('pics/2a.png')

	# Identity -- should be 0
	newim = affinetrans(im,[1,0,0,0])
	print(np.sum(im-newim))

	#newim = affinetrans(im,0.5,np.pi/7,-10.3,8.7)
	newim = affinetrans(im,[2,0,0,0])
	pl.imshow(newim)
	newim2 = affinetrans(newim,[0.5,0,0,0])
	pl.figure()
	pl.imshow(newim2)
	#newim2 = affinetrans(newim,2,-np.pi/7,10.3,-8.7)
	print (np.sum(im-newim2))

	newim = affinetrans(im,[1.0,np.pi/3,0,0])
	#newim = affinetrans(im,[0.5,np.pi/3,-80.3,8.7])
	pl.figure(), pl.imshow(newim)

#test_affinetrans()


