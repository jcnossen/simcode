"""
Applies "Calibrating photon counts from a single image"
https://arxiv.org/abs/1611.05654
Multiple images can be passed to improve the estimate
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
from smlmtorch.util.progbar import progbar,pb_range

class GlobalCameraCalibration:
    def __init__(self, gain = 1, offset = 0, readnoise = 0):
        self.gain = gain
        self.offset = offset
        self.readnoise = readnoise

    def from_images(self, images):
        gain, offset = gain_offset_from_images(images, show_plot=False, ntiles=3)
        return GlobalCameraCalibration(gain, offset)


def mirror(img):
    # add mirrored borders to image
    h, w = img.shape
    return np.pad(img, ((0, h), (0, w)), mode = 'reflect') 


    
def mean_noise_energy(img, var_offset=0, kthreshold=1):
    # Compute the Fourier transform of the image
    f = np.fft.fft2(img) / np.sqrt(np.prod(img.shape))
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)**2

    # Compute the distance from the center
    h, w = img.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)

    # Compute the mean power of all the Fourier pixels with k > width/2
    mask = dist_from_center > w / 2 * kthreshold
    return np.mean(mag[mask]-var_offset)

def subdiv(img, nblocks=3):
    # shrink the image to make equal block size
    img = img[:img.shape[0]//nblocks*nblocks, :img.shape[1]//nblocks*nblocks]
    img = img.reshape((nblocks, img.shape[0]//nblocks, nblocks, img.shape[1]//nblocks))
    return img.transpose((0,2,1,3))


def gain_offset_from_images(images, show_plot=False, ntiles=3):
    if show_plot:    
        fig,ax=plt.subplots(2,2)
        ax[0,0].imshow(images[0]); ax[0,0].set_title('Raw image')

        m = mirror(images[0])
        ax[0,1].imshow(m); ax[0,1].set_title('Mirror')

    noise, mean_val = [], []

    for i in pb_range(len(images), desc='Step 1/2'):
        tiles = subdiv(images[i], ntiles)

        tiles = tiles.reshape((-1, tiles.shape[-2], tiles.shape[-1]))
        ex_tiles = [mirror(m) for m in tiles]
        noise.extend([mean_noise_energy(m) for m in ex_tiles])
        mean_val.extend([np.mean(m) for m in ex_tiles])

    # least square fit to fit offset and slope
    A = np.vstack([mean_val, np.ones(len(mean_val))]).T
    slope, var_offset = np.linalg.lstsq(A, noise, rcond=None)[0]

    offset = -var_offset / slope
    gains = []
    # compute on the full images
    for img in progbar(images, desc='Step 2/2'):
        m = mirror(images[0])
        ne = mean_noise_energy(m, var_offset)
        gains.append(ne/m.mean())
    gain = np.mean(gains)

    if show_plot:
        # plot the fit with 2 points
        x = np.linspace(0, max(mean_val), 100)
        ax[1,0].plot(mean_val, noise, '.')
        ax[1,0].plot(x, slope*x + var_offset)

        # print gain as text on the image
        ax[1,0].text(0.05, 0.9, f'gain = {gain:.2f}\noffset = {offset:.2f}', 
            transform=ax[1,0].transAxes, color='k', ha='left', va='bottom')

    return gain, offset


if __name__ == '__main__':
    H,W=600,600
    gain = 2
    offset = 50
    readnoise = 1
    images = np.zeros( (10,H,W),dtype=np.float32)
    # add some intensity varation over x and y
    images += np.linspace(0, 100, H)[:,None]
    images += np.linspace(0, 100, W)[None,:]
    # add a gradient from center
    images += np.sqrt((np.linspace(-100, 100, H)**2)[:,None] + 
                      (np.linspace(-100, 100, W)**2)[None,:])
    images = np.random.poisson(images/gain) * gain + offset + np.random.normal(0, 1, images.shape) * readnoise

    from smlmtorch import image_view

    tiles = subdiv(images[0], 3)
    fig,ax=plt.subplots(len(tiles),len(tiles[0]))
    for i in range(len(tiles)):
        for j in range(len(tiles[0])):
            ax[i,j].imshow(tiles[i,j])
            ax[i,j].axis('off')
    
    gain, offset = gain_offset_from_images(images, show_plot=True, ntiles=3)


    #for i in raneg()
    #ean_noise_energy