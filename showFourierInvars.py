
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
h = highpass(fullsize,fullsize)

I = np.zeros((fullsize,fullsize))
xstart = 150
ystart = 150
I[xstart:xdim+xstart,ystart:ydim+ystart] = im
J = affinetrans.affinetrans(I,[1,0,120,110])
K = affinetrans.affinetrans(I,[1,np.pi/4,0,0])

print "Absolute difference after translation, rotation"
print np.real(np.sum(np.abs(I-J))), np.real(np.sum(np.abs(I-K)))

pl.figure(), 
pl.subplot(431) 
pl.imshow(I, cmap='gray'), pl.title("Original Image")
pl.subplot(432) 
pl.imshow(J, cmap='gray'), pl.title("Translated Image")
pl.subplot(433) 
pl.imshow(K, cmap='gray'), pl.title("Rotated Image")

fI = np.fft.fftshift(np.fft.fft2(I))
ffI = fI*np.conj(fI)
fJ = np.fft.fftshift(np.fft.fft2(J))
ffJ = fJ*np.conj(fJ)
fK = np.fft.fftshift(np.fft.fft2(K))
ffK = fK*np.conj(fK)

print "Fourier difference after translation, rotation"
print np.real(np.sum(np.abs(ffI-ffJ))), np.real(np.sum(np.abs(ffI-ffK)))



pI = c2p(h*np.abs(fI))
pJ = c2p(h*np.abs(fJ))
pK = c2p(h*np.abs(fK))

pl.subplot(434) 
pl.imshow(np.log(np.abs(pI)+1)), pl.title("Log Polar Original Image")
pl.subplot(435) 
pl.imshow(np.log(np.abs(pJ)+1)), pl.title("Log Polar Translated Image")
pl.subplot(436) 
pl.imshow(np.log(np.abs(pK)+1)), pl.title("Log Polar Rotated Image")

gI = np.fft.fftshift(np.fft.fft2(pI))
ggI = gI*np.conj(gI)
gJ = np.fft.fftshift(np.fft.fft2(pJ))
ggJ = gJ*np.conj(gJ)
gK = np.fft.fftshift(np.fft.fft2(pK))
ggK = gK*np.conj(gK)
print "Fourier-Mellin difference after translation, rotation"
print np.real(np.sum(np.abs(ggI-ggJ))), np.real(np.sum(np.abs(ggI-ggK)))

pl.subplot(437) 
pl.imshow(np.log(np.abs(gI))), pl.title("Fourier-Mellin Original Image")
pl.subplot(438) 
pl.imshow(np.log(np.abs(gJ))), pl.title("Fourier-Mellin Translated Image")
pl.subplot(439) 
pl.imshow(np.log(np.abs(gK))), pl.title("Fourier-Mellin Rotated Image")

# I want to convert these BACK to normal images now. This is in order to use intuitions about ...
# Other thought: better / worse to use wavelet transform that fft? Suspect so.
pl.subplot(4,3,10) 
pl.imshow(np.abs(np.fft.fft2(fI)), cmap='gray')
pl.subplot(4,3,11) 
pl.imshow(np.abs(np.fft.fft2(fJ)), cmap='gray')
pl.subplot(4,3,12) 
pl.imshow(np.abs(np.fft.fft2(fK)), cmap='gray')
pl.show()
pl.draw()
pl.imsave('out.png')

