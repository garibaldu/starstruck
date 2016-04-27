
import pylab as pl
import numpy as np
import scipy as sp

import affinetrans

def c2p(im):
        nx = np.shape(im)[0]
        ny = np.shape(im)[1]
        rg = np.logspace(0, np.log10(np.min((nx/2., ny/2.))),nx)
        thetag = np.linspace(0, 2*np.pi, ny+1)
        thetag = thetag[:-1]
        thetagrid, rgrid = np.meshgrid(thetag, rg)
        x = rgrid*np.cos(thetagrid) + nx/2. -1
        y = rgrid*np.sin(thetagrid) + ny/2. -1
        return sp.ndimage.map_coordinates(im, [y,x], order=3)

def highpass(h,w):
        eta1 = np.cos(np.pi*np.linspace(-0.5,0.5,h))
        eta = np.ones((1,np.shape(eta1)[0]))*eta1
        neta1 = np.cos(np.pi*np.linspace(-0.5,0.5,w))
        neta = np.ones((1,np.shape(neta1)[0]))*neta1
        X = np.dot(eta.T,neta)
        return (1.0-X)*(2.0-X)

pl.ion()
im = pl.imread('pics/2a.png')
xdim,ydim = np.shape(im)
fullsize = 500

I = np.zeros((fullsize,fullsize))
xstart = 150
ystart = 150
I[xstart:xdim+xstart,ystart:ydim+ystart] = im

J = affinetrans.affinetrans(I,[1,0,120,110])
K = affinetrans.affinetrans(I,[1,np.pi/4,0,0])
#K = affinetrans.affinetrans(I,[1,0,150,130])
#K = affinetrans.affinetrans(I,[1,0,10,20])

print "Absolute difference after translation, rotation"
print np.real(np.sum(np.abs(I-J))), np.real(np.sum(np.abs(I-K)))

pl.figure(), pl.imshow(I), pl.title("Original Image")
pl.figure(), pl.imshow(J), pl.title("Translated Image")
pl.figure(), pl.imshow(K), pl.title("Rotated Image")

fI = np.fft.fftshift(np.fft.fft2(I))
ffI = fI*np.conj(fI)
fJ = np.fft.fftshift(np.fft.fft2(J))
ffJ = fJ*np.conj(fJ)
fK = np.fft.fftshift(np.fft.fft2(K))
ffK = fK*np.conj(fK)

print "Fourier difference after translation, rotation"
print np.real(np.sum(np.abs(ffI-ffJ))), np.real(np.sum(np.abs(ffI-ffK)))

#pl.figure(), pl.imshow(np.log(np.abs(fJ)+1))
#pl.figure(), pl.imshow(np.log(np.abs(fJ)+1))
#pl.figure(), pl.imshow(np.log(np.abs(fI-fJ)+1))

h = highpass(fullsize,fullsize)
pI = c2p(h*np.abs(fI))
pJ = c2p(h*np.abs(fJ))
pK = c2p(h*np.abs(fK))

pl.figure(), pl.imshow(np.log(np.abs(pI)+1)), pl.title("Log Polar Original Image")
pl.figure(), pl.imshow(np.log(np.abs(pJ)+1)), pl.title("Log Polar Translated Image")
pl.figure(), pl.imshow(np.log(np.abs(pK)+1)), pl.title("Log Polar Rotated Image")

gI = np.fft.fftshift(np.fft.fft2(pI))
ggI = gI*np.conj(gI)
gJ = np.fft.fftshift(np.fft.fft2(pJ))
ggJ = gJ*np.conj(gJ)
gK = np.fft.fftshift(np.fft.fft2(pK))
ggK = gK*np.conj(gK)
print "Fourier-Mellin difference after translation, rotation"
print np.real(np.sum(np.abs(ggI-ggJ))), np.real(np.sum(np.abs(ggI-ggK)))
#pl.figure(), pl.imshow(np.log(np.abs(gI)+1))

pl.figure(), pl.imshow(np.log(np.abs(gI))), pl.title("Fourier-Mellin Original Image")
pl.figure(), pl.imshow(np.log(np.abs(gJ))), pl.title("Fourier-Mellin Translated Image")
pl.figure(), pl.imshow(np.log(np.abs(gK))), pl.title("Fourier-Mellin Rotated Image")
pl.show()
