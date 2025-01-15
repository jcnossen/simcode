import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey
from scipy.signal import convolve
import torch
import time
import numba

@numba.njit(fastmath=True, parallel=True)
def draw_gaussians_nb(spots, w):
    """
    Render 2D Gaussians into an image using a small 5x5 kernel around each spot.
    spots: array of shape (N, 5):
           columns: [0]=x, [1]=y, [2]=sx, [3]=sy, [4]=photons
    w:     width (and height) of the output square image
    """
    image = np.zeros((w, w), dtype=np.float32)

    for i in numba.prange(spots.shape[0]):
        x = spots[i, 0]
        y = spots[i, 1]
        sx = spots[i, 2]
        sy = spots[i, 3]
        I  = spots[i, 4]

        ix = int(np.round(x))
        iy = int(np.round(y))

        for dy in range(-2, 3):
            for dx in range(-2, 3):
                xx = ix + dx
                yy = iy + dy

                # Skip if outside
                if (xx < 0) or (yy < 0) or (xx >= w) or (yy >= w):
                    continue

                # Compute the distance from (x,y)
                dx_f = (xx - x) / sx
                dy_f = (yy - y) / sy
                val = I * np.exp(-0.5*(dx_f*dx_f + dy_f*dy_f))
                image[yy, xx] += val

    return image

def _getfft(xy, photons, imgshape, zoom, device=None):
    """
    Creates an image of shape [w, w] by placing 2D Gaussians (5x5 patches) at 
    the (x, y) locations. Then returns the 2D FFT (shifted).
    """
    spots = np.zeros((len(xy), 5), dtype=np.float32)
    # Fill columns: (x, y, sx, sy, photons)
    spots[:, 0] = xy[:, 0] * zoom
    spots[:, 1] = xy[:, 1] * zoom
    # Hard-code the sigma to 0.5 px for this example
    spots[:, 2] = 0.5
    spots[:, 3] = 0.5
    spots[:, 4] = photons

    w = np.max(imgshape)*zoom
    w = int(w)

    # Render the Gaussians into the image via Numba
    img = draw_gaussians_nb(spots, w)

    # Multiply by a Tukey window (avoids FFT edge artifacts)
    wnd = tukey(w, 1/4).astype(np.float32)
    img *= wnd[:, None]
    img *= wnd[None, :]

    # Now compute the 2D FFT
    if device is not None:
        # For example, do the FFT on GPU using torch
        img_torch = torch.tensor(img, device=device, dtype=torch.float32)
        f_img = torch.fft.fftshift(torch.fft.fft2(img_torch)).cpu().numpy()
    else:
        # Or do the FFT in numpy
        f_img = np.fft.fftshift(np.fft.fft2(img))
    return f_img

def radialsum(sqimg):
    """
    radially-sums pixel values in a square image about its center
    """
    W = len(sqimg)
    Y, X = np.indices(sqimg.shape)
    R = np.sqrt((X - W//2)**2 + (Y - W//2)**2)
    R = R.astype(np.int32)
    return np.bincount(R.ravel(), sqimg.ravel())

def FRC(xy, photons, zoom, imgshape, pixelsize, display=True,
        smooth=0, mask=None, device=None):
    """
    Computes the Fourier Ring Correlation (FRC).
    If xy and photons are each Nx2 arrays, then they are treated as two separate
    localizations sets. Otherwise, a random 50/50 split (mask) is used.
    """
    if isinstance(xy, list) and len(xy) == 2:
        # xy[0], xy[1] and photons[0], photons[1] are separate sets
        pass
    else:
        # Random 50/50 split if no mask is provided
        if mask is None:
            mask = np.random.binomial(1, 0.5, len(xy)) == 1

        set1 = mask
        set2 = np.logical_not(set1)
        xy = [xy[set1], xy[set2]]
        photons = [photons[set1], photons[set2]]

    t0 = time.time()

    f1 = _getfft(xy[0], photons[0], imgshape, zoom, device=device)
    f2 = _getfft(xy[1], photons[1], imgshape, zoom, device=device)

    # Compute numerator and denominator of FRC
    x = np.real(f1 * np.conj(f2))  # cross-power spectrum
    frc_num = radialsum(x)
    frc_denom = np.sqrt(radialsum(np.abs(f1)**2) * radialsum(np.abs(f2)**2))
    frc = frc_num / frc_denom

    t1 = time.time()

    freq = np.fft.fftfreq(len(f1))
    frc = frc[:imgshape[0]*zoom//2]
    freq = freq[:imgshape[0]*zoom//2]

    # Optional smoothing
    if smooth > 0:
        frc = convolve(frc, np.ones(smooth)/smooth, mode='valid')
        freq = convolve(freq, np.ones(smooth)/smooth, mode='valid')

    # FRC resolution threshold (1/7 rule)
    below_thresh = np.where(frc < 1/7)[0]
    if len(below_thresh) > 0:
        frc_res_idx = below_thresh[0]
    else:
        frc_res_idx = 0  # fallback if no crossing found
    frc_res_freq = freq[frc_res_idx]

    frc_res = pixelsize / (zoom * frc_res_freq) if frc_res_freq != 0 else np.inf
    print(f"Elapsed time: {t1 - t0:.1f} s. FRC = {frc_res:.2f} nm")

    if display:
        plt.figure()
        plt.plot(freq * zoom / pixelsize, frc, label='FRC')
        plt.axhline(1/7, color='r', linestyle='--', label='1/7 Threshold')
        plt.title(
            f'FRC resolution: {frc_res:.2f} nm '
            f'({(1/(zoom*frc_res_freq)):.2f} px)'
        )
        plt.xlabel('Frequency [1/nm]')
        plt.legend()
        plt.show()

    return frc_res, frc, freq * zoom / pixelsize




if __name__ == "__main__":
    fn='C:/dev/smlmtorch/scripts/densities3/D1/results/sim_tubules_bg2.5_I500_psf_gauss1.3px/sfhd-ndi+smlm.hdf5'
    
    from smlmtorch import Dataset
    ds = Dataset.load(fn)
        
    #ds[mask].save('test1.hdf5')
    #ds[np.logical_not(mask)].save('test2.hdf5')

    ds=ds[ds.frame%6==0]
    
    res, curve, freq = FRC(ds.pos, ds.photons, 25
                      , ds.imgshape, 
                      pixelsize=108.3,
                      #mask=mask,
                      device=torch.device('cuda:0'))
    