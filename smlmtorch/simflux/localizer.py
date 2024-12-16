import smlmtorch.smlm.detection as detection
import os
import smlmtorch.util.caching as cache
import torch
from smlmtorch.util.multipart_tiff import tiff_read_file, tiff_get_movie_size, tiff_get_image_size, tiff_get_filenames
import numpy as np
from smlmtorch import Dataset
from smlmtorch.simflux.dataset import SFDataset
import smlmtorch.simflux.pattern_estimator as pe 
import matplotlib.pyplot as plt
import seaborn as sns
from smlmtorch.ui import image_view
from scipy.interpolate import InterpolatedUnivariateSpline
from smlmtorch.util.config_dict import config_dict
from smlmtorch import progbar

from fastpsf import Context, GaussianPSFMethods, Gauss3D_Calibration, CSplineMethods

class SMLMFitter:
    def __init__(self):
        self.cache_cfg = None
        
    def process(self, roi_iterator, imgshape, psf_model, fit_constants=None, pb_desc=None):
        summed_fits = []
        framenum = []
        roipos = []
        summed_crlb = []
        chisq = []
        ibg_fits = []
        ibg_fits_crlb = []
        
        numrois, rois = roi_iterator()
        pos = 0
        with progbar(total=numrois) as pb, torch.no_grad():
            if pb_desc is not None:
                pb.set_description(pb_desc)
                
            for rois_info, pixels, readnoise in rois:
                summed = pixels.sum(1)
                
                if readnoise is not None:
                    summed += readnoise
                    const_ = readnoise
                elif fit_constants is not None:
                    const_ = fit_constants[pos:pos+len(pixels)]
                else:
                    const_ = None
                          
                e = psf_model.Estimate(summed, constants=const_)[0]
                crlb_ = psf_model.CRLB(e, constants=const_)
                chisq_ = psf_model.ChiSquare(e, summed, constants=const_)
                
                summed_crlb.append(crlb_)
                chisq.append(chisq_)

                roipos.append( np.stack( (rois_info['y'], rois_info['x']), 1 ) )
                framenum.append(rois_info['id'])
                                    
                sh = pixels.shape # numspots, numpatterns, roisize, roisize
                pixels_rs = pixels.reshape((sh[0]*sh[1],sh[2],sh[3]))
                params = np.repeat(e, sh[1], axis=0)
                if const_ is not None:
                    const_ = np.repeat(const_, sh[1], axis=0)
                #params[:,self.psf_model.ParamIndex('I')]=1
                #params[:,self.psf_model.ParamIndex('bg')]=0
                r = psf_model.EstimateIntensityAndBackground(params, pixels_rs, constants=const_, cuda=True)
                ibg_ = r[0]
                ibg_crlb_ = r[1]
                
                #ibg_fits.append(ibg_)
                #ic = np.zeros((len(e)*sh[1],2))
                #ic [:,[0,1]] = ibg_
                #ic [:,[2,3]] = ibg_crlb_
                ibg_fits.append(ibg_.reshape((sh[0],sh[1],2))) # reshape into [spots,patterns,2]
                ibg_fits_crlb.append(ibg_crlb_.reshape((sh[0],sh[1],2))) # reshape into [spots,patterns,2]

                summed_fits.append(e)

                pos += len(pixels)
                pb.update(len(pixels))
            
        self.estimates = np.concatenate(summed_fits)
        self.chisq = np.concatenate(chisq)
        self.crlb = np.concatenate(summed_crlb)
        self.roipos = np.concatenate(roipos)
        self.framenum = np.concatenate(framenum)
        self.ibg_fits = np.concatenate(ibg_fits)
        self.ibg_fits_crlb = np.concatenate(ibg_fits_crlb)
        
        ds = SFDataset.fromEstimates(self.estimates, self.param_names, 
                                          roipos=self.roipos,
                                          crlb=self.crlb,
                                          framenum=self.framenum, imgshape=imgshape, 
                                          numPatterns=self.ibg_fits.shape[1], 
                                          chisq=self.chisq)
        ds.ibg[:] = np.concatenate(ibg_fits)
        ds.ibg_crlb[:] = np.concatenate(ibg_fits_crlb)
        
        ds.filterNaN()
        fits = self.estimates[ds.roi_id]
        limits = psf_model.ParamLimits()
        ds = ds[(fits > limits[0][None]).all(1) & 
                (fits < limits[1][None]).all(1)]
        
        return ds
    
    def save_stats(self, result_dir):
        ...

class SMLMFitter2D(SMLMFitter):
    def __init__(self, ctx: Context, psf_calib, roisize,  psf_sigma_binsize=1000):
        
        super().__init__()
        
        gm = GaussianPSFMethods(ctx)
        self.psf_constsigma = gm.CreatePSF_XYIBg(roisize, None, cuda=True)
        self.psf_sigmaxy = gm.CreatePSF_XYIBgSigmaXY(roisize, psf_calib, cuda=True)
        self.psf_fixedsigma = gm.CreatePSF_XYIBg(roisize, psf_calib, cuda=True)

        self.gauss_psf_calib = np.array(psf_calib)
        self.psf_sigma_binsize = psf_sigma_binsize

        self.param_names = self.psf_fixedsigma.param_names
        
        self.detection_template = self.psf_fixedsigma.ExpectedValue(
            [[roisize/2, roisize/2, 1, 0]])
        
        self.cache_cfg = [ psf_sigma_binsize ]

    @property
    def psf_model(self):
        return self.psf_fixedsigma

    def process(self, roi_iterator, imgshape):
        
        if self.psf_sigma_binsize is None:
            def estimate_with_fixed_sigma(pixels, readnoise, roi_ix):
                return self.psf_fixedsigma.Estimate(pixels)[0], None
            
            return super().process(roi_iterator, imgshape, pb_desc='Fitting with known PSF sigma', 
                            psf_model=self.psf_fixedsigma, fit_constants=None)
            
        framenum = []
        roipos = []
        fits = []
        numrois, rois = roi_iterator()
        with progbar(total=numrois) as pb, torch.no_grad():
            pb.set_description('Estimating PSF sigma')

            for rois_info, pixels, readnoise in rois:
                # ignore readnoise..
                summed = pixels.sum(1)
                e = self.psf_fixedsigma.Estimate(summed)[0]
                initial = np.zeros((len(e),6))
                initial[:,:4] = e
                initial[:,4:] = self.gauss_psf_calib[None]
                esxy = self.psf_sigmaxy.Estimate(summed, initial=initial)[0]
                framenum.append(rois_info['id'])
                roipos.append( np.stack( (rois_info['y'], rois_info['x']), 1 ) )
                fits.append(esxy)
                pb.update(len(pixels))
                
        framenum = np.concatenate(framenum)
        fits = np.concatenate(fits)
        numframes = np.max(framenum)+1
        
        nlocs = len(fits)
        ds = Dataset(nlocs, 2, imgshape=[0,0], haveSigma=True)
        ds.frame = np.maximum((framenum / self.psf_sigma_binsize - 0.5).astype(np.int32),0)
        ds = ds[np.isnan(fits).sum(1)==0]
        frames = ds.indicesPerFrame()
        if len(frames) > 2:
            self.medianSigma = np.array([np.median(fits[ds.roi_id[idx],4:],0) for idx in frames])
            print(f'number of PSF sigma bins: {len(self.medianSigma)}')
            self.sigma_t = (0.5+np.arange(len(frames))) * self.psf_sigma_binsize
            #self.medianSigma = [self.medianSigma[0], *self.medianSigma, self.medianSigma[-1]]
                
            self.sigma_t[0] = 0
            self.sigma_t[-1] = (len(frames)-1) * self.psf_sigma_binsize
            spl_x = InterpolatedUnivariateSpline(self.sigma_t, self.medianSigma[:,0], k=2)
            spl_y = InterpolatedUnivariateSpline(self.sigma_t, self.medianSigma[:,1], k=2)
            
            self.sigma = np.zeros((numframes,2))
            self.sigma[:,0] = spl_x(np.arange(numframes))
            self.sigma[:,1] = spl_y(np.arange(numframes))                
        else:
            self.sigma = np.zeros((numframes,2))
            self.medianSigma = np.median(fits[ds.roi_id,4:],0)
            self.sigma[:,0] = self.medianSigma[0]
            self.sigma[:,1] = self.medianSigma[1]
            self.sigma_t = None

            print(f'Median psf sigma: {self.medianSigma}')
        
            fig,ax=plt.subplots(2,1,sharex=True)
            ax[0].hist(fits[:,4], bins=100)
            ax[0].set_xlabel('Sigma X')
            ax[1].hist(fits[:,5], bins=100)
            ax[1].set_xlabel('Sigma Y')

        roi_sigma = self.sigma[framenum]

        def estimate_with_const_sigma(pixels, readnoise, roi_ix):
            #initial = initial_fits[roi_ix:roi_ix+len(pixels)]
            c = roi_sigma[roi_ix:roi_ix+len(pixels)]
                
            e = self.psf_constsigma.Estimate(pixels, constants=c)[0]
            return e, c

        ds = super().process(roi_iterator, imgshape, 
                    pb_desc='Fitting with interpolated PSF sigma', 
                    psf_model = self.psf_constsigma, 
                    fit_constants = roi_sigma)
        
        return ds
        
    def eval_psf(self, roi_ids):
        if self.psf_sigma_binsize is None:
            return self.psf_fixedsigma.ExpectedValue(self.estimates[roi_ids])
        
        return self.psf_constsigma.ExpectedValue(self.estimates[roi_ids], constants=self.sigma[roi_ids])

            

    def plotSigmaTimeSeries(self, result_dir, **figargs):
        if self.sigma_t is None:
            return
        plt.figure(**figargs)
        plt.plot(self.sigma_t, self.medianSigma[:,0],'o')
        plt.plot(self.sigma_t, self.medianSigma[:,1],'o')
        plt.plot(self.sigma[:,0], label='Sigma X')
        plt.plot(self.sigma[:,1], label='Sigma Y')
        plt.xlabel('Frames')
        plt.ylabel('Gaussian PSF Sigma [pixels]')
        plt.legend()
        plt.title(f'PSF Sigma vs time (using {self.psf_sigma_binsize} frames per bin)')
        plt.savefig(result_dir + "/sigma.png")
        plt.savefig(result_dir + "/sigma.svg")
    
    
    def save_stats(self, path, **figargs):
        if self.psf_sigma_binsize is not None:
            self.plotSigmaTimeSeries(path, **figargs)
            np.save(path + "/psf2D_sigma.npy", self.sigma)


class SMLMFitter3D(SMLMFitter):
    def __init__(self, ctx: Context, psf_calib: str, roisize: int, zrange, have_readnoise=False):
        super().__init__()

        self.psf_calib = psf_calib
        self.zrange = zrange
        self.roisize = roisize
        ext = os.path.splitext(psf_calib)[1]
        if ext == '.mat':
            template_model = CSplineMethods(ctx).CreatePSFFromFile(self.roisize, self.psf_calib)

            if have_readnoise:
                """ 
                Following Video-rate nanoscopy using sCMOS camera–specific single-molecule localization algorithms
                https://www.nature.com/articles/nmeth.2488 ,
                the readnoise can be modelled into the existing MLE by shifting the sample value 
                and expected value both by <readnoise var>/gain^2. 
                Here, the expected value is modified by passing a per-pixel background image to the fitter.
                """
                self.psf_model = CSplineMethods(ctx).CreatePSFFromFile(
                                            self.roisize, self.psf_calib, fitMode=CSplineMethods.BgImage)
            else:
                self.psf_model = template_model

            self.psf_model.SetLevMarParams(500, normalizeWeights=True)
            
        elif ext == '.yaml':
            calib = Gauss3D_Calibration.from_file(self.psf_calib)
            self.psf_model = GaussianPSFMethods(ctx).CreatePSF_XYZIBg(roisize, calib, cuda=True)
            self.gauss_psf_calib = calib
        else:
            raise ValueError(f'PSF model file has unknown extension: {psf_calib}')

        self.param_names = self.psf_model.param_names

        steps = 3
        x = np.array([[roisize/2, roisize/2, 0, 1, 0]]).repeat(steps,0)
        x[:,2] = np.linspace(self.psf_model.calib.zrange[0], self.psf_model.calib.zrange[1], steps)
        self.detection_template = template_model(x)
        
        plt.figure()
        plt.imshow(np.concatenate(self.detection_template,-1))
        
    def process(self, roi_iterator, imgshape):
        return super().process(roi_iterator, imgshape, psf_model=self.psf_model)

class SFLocalizer:
    def __init__(self, tif_path, *, pattern_frames, pixelsize, 
                 detection_threshold, psf_calib, roisize, device='cuda:0', 
                 zrange=None, gain=None, offset=None, readnoise=None, camera_calib=None, debug_mode=False, 
                 cache_name='cache',
                 maxframes=None, psf_sigma_binsize=None, result_dir='results'):

        self.fastpsf_ctx = None
        self.src_path = tif_path
        self.maxframes = maxframes
        
        self.pattern_frames = np.array(pattern_frames)
        
        if camera_calib is not None:
            self.camera_offset, self.camera_gain = np.load(camera_calib)
            assert gain is None and offset is None
        else:
            assert gain is not None and offset is not None
            if type(gain) == str:
                gain, offset = detection.load_gain_offset(gain, offset)
    
            self.camera_gain = gain
            self.camera_offset = offset
        
        self.pixelsize = pixelsize
        self.detection_threshold = detection_threshold
        self.psf_calib = psf_calib
        self.roisize = roisize
        self.zrange = zrange

        dir, fn = os.path.split(self.src_path)
        self.cache_dir = dir + f"/{cache_name}/" + os.path.splitext(fn)[0]+ "/"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.rois_path = self.cache_dir + "rois.npy"

        self.mod_fn = os.path.splitext(self.src_path)[0]+"-mod.pickle"
        self.numrois = None

        self.result_dir = dir + "/" + result_dir + "/" + os.path.splitext(fn)[0] + "/"
        os.makedirs(self.result_dir, exist_ok=True)
        
        self.device = torch.device(device)
        
        ofs_t = torch.tensor(self.camera_offset, device=self.device, dtype=torch.float32)
        gain_t = torch.tensor(self.camera_gain, device=self.device, dtype=torch.float32)
        
        self.camera_calib = lambda img: torch.clamp((img-ofs_t)*gain_t, min=1e-6)
        self.camera_readnoise = readnoise

        self.fastpsf_debug_mode = debug_mode
        self.psf_sigma_binsize = psf_sigma_binsize
        self.create_psf_model()
        
        #print("TIF filenames:", tiff_get_filenames(tif_path))
        
        self.sum_ds = None
        self.sf_ds = None
        
    def __del__(self):
        if self.fastpsf_ctx is not None:
            del self.fastpsf_ctx
            self.fastpsf_ctx = None

       
    def set_kxy(self, kxy):
        for i,pf in enumerate(self.pattern_frames):
            self.mod['k'][pf,:2] = kxy[i]

    def create_psf_model(self):
        torch.cuda.set_device(self.device)
        self.fastpsf_ctx = Context(debugMode = self.fastpsf_debug_mode)
        
        if type(self.psf_calib) == str:
            self.smlm_fitter = SMLMFitter3D(self.fastpsf_ctx, self.psf_calib, self.roisize, self.zrange, 
                                            self.camera_readnoise is not None)
        else:
            self.smlm_fitter = SMLMFitter2D(self.fastpsf_ctx, self.psf_calib, 
                                            self.roisize, self.psf_sigma_binsize)
        
        self.detection_template = self.smlm_fitter.detection_template

    @property
    def psf_model(self):
        return self.smlm_fitter.psf_model

    
    def detect_spots(self, ignore_cache=False, moving_window=False, alt_img_source=None, alt_rois_path=None,
                     batch_size=16*1024, max_filter_size=None, bg_filter_size=None):
        """
        moving_window: if false, processes frames in blocks of pattern_frames.size, never using the same frame twice
                        if true, process the frames with a moving window of size pattern_frames.size
                        
        alt_img_source: Function that returns (shape, numframes, image generator)
                        If None, reads from tif file
        """
            
        if alt_img_source is not None:
            self.imgshape, self.numframes, img_src = alt_img_source()
        else:
            self.imgshape, self.numframes = tiff_get_movie_size(self.src_path)
            img_src = None
            
        nframes = self.numframes
        if self.maxframes is not None:
            nframes = min(self.maxframes, nframes)
            
        # all parameters that influence detection
        cfg = config_dict(pf=self.pattern_frames.size, 
               gain=self.camera_gain, 
               ofs=self.camera_offset, 
               calib=self.psf_calib, 
               dt=self.detection_threshold, 
               bs=batch_size,
               tpl=np.array(self.detection_template),
               roisize=self.roisize, 
               mw=moving_window,
               max_filter_size=max_filter_size,
               bg_filter_size=bg_filter_size,
               nf=nframes)
        
        rois_path = self.rois_path
        if alt_rois_path is not None:
            rois_path = alt_rois_path

        if not cache.is_valid_cache(rois_path, cfg, self.src_path) or ignore_cache:
            detector = detection.SpotDetector(self.detection_threshold, self.imgshape, 
                                              torch.tensor(self.detection_template).to(self.device), 
                                              max_filter_size=cfg.max_filter_size, bg_filter_size=cfg.bg_filter_size, 
                                    spots_per_frame=100)
            
            #detector = torch.jit.script(detector)
            
            print(f"Processing {self.src_path} with {self.numframes} frames")

            if img_src is None:
                img_src = tiff_read_file(self.src_path, use_progbar=False, maxframes=nframes)
            
            def tif_moving_window(src, wndsize):
                buf = []
                for img in src:
                    buf.append(img)
                    if len(buf) < wndsize:
                        continue
                    
                    for wndimg in buf:
                        yield wndimg
                    buf.pop(0)
                
            def modify_framenum(info, rois):
                info['id'] *= self.pattern_frames.size
                return info, rois
            
            print(f'moving_window={moving_window}')
            if moving_window:
                img_src = tif_moving_window(img_src, self.pattern_frames.size)

            #from smlmtorch.multipart_tiff import tiff_read_file

            with torch.no_grad():
                numrois, numframes = detection.detect_spots_in_movie(detector, 
                                      self.camera_calib, img_src,
                                      sumframes = self.pattern_frames.size, 
                                      output_fn = rois_path,
                                      batch_size = batch_size,
                                      totalframes = nframes*self.pattern_frames.size if moving_window else self.numframes,
                                      callback = modify_framenum if not moving_window else None)
            
            cache.save_cache_cfg(rois_path, cfg, self.src_path, numrois)
        else:
            print(f'loading ROI cache: {rois_path}')
            _, numrois = cache.load_cache_cfg(rois_path)

        self.numrois = numrois
        print(f"Num ROIs: {self.numrois}")

        if self.numrois == 0:
            raise ValueError('No spots detected')
           
        self.moving_window = moving_window


            
    def fit_smlm(self, max_crlb_xy=20, max_photons=1e5, min_photons=300, max_chisq=4, ignore_cache=False) -> Dataset:
        def process():
            sum_ds = self.smlm_fitter.process(self.roi_iterator, self.imgshape)
            summed_fits = self.smlm_fitter.estimates
            
            self.smlm_fitter.save_stats(self.result_dir)
                
            print(f'Remaining localizations: {len(sum_ds)}/{len(summed_fits)}')
                        
            return sum_ds, summed_fits
        
        cache_cfg = dict(nr=self.numrois, fitter_cc=self.smlm_fitter.cache_cfg,
                         readnoise=self.camera_readnoise,
                         phf=[min_photons, max_photons])
        
        ds, summed_fits = cache.read(self.cache_dir + "summed_fits", cache_cfg, process, rebuild=ignore_cache)
        
        ds['pixelsize'] = self.pixelsize
        
        if max_chisq is not None:
            sel = ds[ds.chisq <= max_chisq * self.roisize*self.roisize]
            print(f'Chi-sq filtered: {len(sel)}/{len(ds)}')
            ds = sel
        
        if max_crlb_xy is not None:
            sel = (( np.max(ds.data.crlb.pos[:,:2],1) < max_crlb_xy / self.pixelsize ) )
            print(f"crlb filter: {np.sum(sel)}/{len(sel)}")
            ds = ds[sel]
        
        ds.save(self.result_dir + "smlm.hdf5")
        plt.imsave(self.result_dir + "render.png", ds.render(zoom=4))
        
        if len(ds)>100:
            plt.figure()
            plt.hist(ds.photons, bins=100, range=[0,np.percentile(ds.photons,99)])
            plt.xlabel('Photon counts')
            plt.savefig(self.result_dir + "intensities.svg")

        m_crlb = np.median(ds.crlb.pos,0) * self.pixelsize
        m_crlb_px = np.median(ds.crlb.pos,0) 
        print(f'Median intensity: {np.median(ds.photons):.0f}. Bg: {np.median(ds.bg):.2f} Median CRLB: X={m_crlb[0]:.1f} nm Y={m_crlb[1]:.1f} nm. ({m_crlb_px[0]:.2f}, {m_crlb_px[1]:.2f} px)')

        self.sum_ds = ds
        self.summed_fits = summed_fits#[self.sum_ds.roi_id]
        #assert (self.summed_fits[:,4] == self.sum_ds.data.estim.sigma[:,0]).all()
        return self.sum_ds
    
    def view_rois(self):
        info, rois = self.all_rois
        rois = rois[self.sum_ds.roi_id].sum(1)
        expval = self.smlm_fitter.eval_psf(self.sum_ds.roi_id)
        
        combined = np.concatenate([rois, expval], -1)
        image_view(combined, title='Spot images and fits')
        
    def fit_simflux(self, mp: pe.ModulationPattern, ds: SFDataset, 
                    iterations=100, lambda_=1, ignore_cache=False, 
                    normalizeWeights=False, distFromSMLM=0.5):
        
        #if type(self.psf_model) == CSplinePSF:
        #    raise ValueError('SIMFLUX fits only supported for 2D Gaussian PSF')

        assert mp.pattern_frames.size == ds.ibg.shape[1]
        mod_per_frame = mp.mod_at_frame(np.arange(ds.numFrames), self.pattern_frames.size)
        
        # map roi id to sum_ds indices
        id_map = np.ones(self.numrois,dtype=np.int32)*-1 
        id_map[ds.roi_id] = np.arange(len(ds))

        def process():
            npat = mp.pattern_frames.size
            
            torch.cuda.set_device(self.device)
            
            numrois, rois_iterable = self.roi_iterator()
            
            from fastpsf.simflux import SIMFLUX
            with SIMFLUX(self.fastpsf_ctx).CreateEstimator_Gauss2D(self.psf_calib, npat, self.roisize, npat) as sf_model :
                sf_model.SetLevMarParams(lambda_, iterations=iterations)
                fits = []
                crlb = []
                org_fits = []
                roi_ids = []
                start_ix = 0
                with progbar(total=len(ds)) as pb:
                    for rois_info, pixels, readnoise in rois_iterable:
                        nrois = len(pixels)
                        ix = id_map[start_ix : start_ix + len(pixels)]
                        # ix now maps ROI indices to sum_ds indices
                        rois_info = rois_info[ix>=0]
                        pixels = pixels[ix>=0]
                        ix = ix[ix>=0]
                        roi_ids.append(ix)
                        initial = self.summed_fits[ix, :sf_model.numparams]
                        initial[:, -1] /= len(mp.mod) # divide bg
                        org_fits.append(initial)
                        
                        mod_batch = mod_per_frame[ds.frame[ix]]
                        mod_batch['phase'] -= rois_info['y'][:,None] * mod_batch['k'][:,:,1]
                        mod_batch['phase'] -= rois_info['x'][:,None] * mod_batch['k'][:,:,0]
    
                        roi_mod_float = np.reshape(mod_batch.view(np.float32), (len(mod_batch), 6*len(mp.mod)))
                        #mod_batch = mod_batch.view(np.float32).reshape( (len(pixels),npat,6) )
                        fits_batch, diag, traces = sf_model.Estimate(pixels, initial=initial, constants=roi_mod_float)
                        fit_iterations = [len(tr) for tr in traces]
                        fits.append(fits_batch)
                        
                        crlb_ = sf_model.CRLB(fits_batch, constants=roi_mod_float)
                        crlb.append(crlb_)

                        pb.update(len(ix))
                        start_ix += nrois
                        
                fits = np.concatenate(fits)
                crlb = np.concatenate(crlb)
                org_fits = np.concatenate(org_fits)
                
                sf_ds = SFDataset.fromEstimates(fits, sf_model.param_names, 
                                                  roipos=ds.data.roipos,
                                                  crlb=crlb,
                                                  framenum=ds.frame, imgshape=self.imgshape, 
                                                  numPatterns=self.pattern_frames.size,
                                                  pixelsize=self.pixelsize)
                sf_ds.ibg[:] = ds.ibg
                sf_ds.roi_id[:] = np.concatenate(roi_ids)

                lp = np.concatenate([
                    sf_ds.local_pos.T, [sf_ds.photons], [sf_ds.background]],0).T
                
                limits = sf_model.ParamLimits()

                dist2D = np.sqrt(np.sum((org_fits[:,:2] - fits[:,:2])**2, 1))

                sf_ds = sf_ds[(lp > limits[0][None]).all(1) & 
                                (lp < limits[1][None]).all(1) &
                                (dist2D < distFromSMLM)]

                return sf_ds

        
        cache_cfg = dict(nr=self.numrois, 
                         it=iterations,
                         l=lambda_, 
                         mod=[mp.mod, mp.phase_offset_per_frame])
        
        sf_ds = cache.read(self.cache_dir + "sf_fits", cache_cfg, process, rebuild=ignore_cache)

        self.sf_ds = sf_ds
        self.sf_ds.save(self.result_dir + "sf.hdf5")
        return sf_ds

      
    def fit_ndi(self, mp: pe.ModulationPattern, ds: SFDataset, ndims=2):
        """
        Position fits using normally distributed intensities. 
        Normal distr. width is read from ds.ibg_crlb
        """
        
        ds = pe.ndi_fit_dataset(ds, mp, ndims, device=self.device)
        ds.save(self.result_dir + "ndi.hdf5")
        self.ndi_ds = ds

        return ds
        
      
    def estimate_angles(self, pitch_minmax_nm, **kwargs) -> pe.ModulationPattern:
        """
        Estimate angles and return them as a partially filled in ModulationPattern 
        """
        return pe.estimate_angles(pitch_minmax_nm, self.sum_ds, 
                        self.pattern_frames, 
                        result_dir = self.result_dir,
                        device = self.device,
                        moving_window = self.moving_window, 
                        **kwargs)

    def estimate_phases(self,mp:pe.ModulationPattern, **kwargs) -> pe.ModulationPattern:
        mp = pe.estimate_phases(self.sum_ds, 
                                mp,
                                fig_callback = lambda label: plt.savefig(self.result_dir + f"{label}.png"),
                                **kwargs
                                )
        return mp
      
    def estimate_kz(self, pitch_minmax_nm, frame_bins, kzsteps=100):
        kxy = torch.tensor( self.mod['k'][self.pattern_frames[:,0]][:,:2] )
        
        z_pitch_range = np.linspace(pitch_minmax_nm[0], pitch_minmax_nm[1],kzsteps)
        
        dev = self.device # torch.device('cpu')
        
        with torch.no_grad():
            for i, pf in enumerate(self.pattern_frames):
                kz = pe.estimate_kz( torch.from_numpy(self.sum_ds.pos).to(dev), 
                               torch.from_numpy(self.sum_ds.ibg[:,pf][:,:,0]).to(dev), kxy[i], z_pitch_range, 
                               torch.from_numpy(self.sum_ds.frame).to(dev),
                               frame_bins, 
                               fig_callback=lambda label: plt.savefig(self.result_dir + f"{label}-angle{i}.png"))
        
        return kz

    def drift_correct(self, **kwargs):
        self.drift = self.sum_ds.estimateDriftMinEntropy(
            coarseSigma=self.sum_ds.crlb.pos.mean(0)*4, 
            pixelsize=self.pixelsize, **kwargs)
        
        plt.savefig(self.result_dir+ "drift.png")

        self.sum_ds_undrift = self.sum_ds[:]
        self.sum_ds_undrift.applyDrift(self.drift)
        self.sum_ds_undrift.save(self.result_dir + "g2d_dme.hdf5")
        
        if self.sf_ds is not None:
            self.sf_ds_undrift  = self.sf_ds[:]
            self.sf_ds_undrift.applyDrift(self.drift)
            self.sf_ds_undrift.save(self.result_dir + "sf_dme.hdf5")

    @staticmethod    
    def extract_rois(image, roipos, roisize:int):
        r = np.arange(roisize)
        Y = roipos[:,0,None,None] + r[None,:,None]
        X = roipos[:,1,None,None] + r[None,None,:]
        return image[Y,X]
    
    def roi_iterator(self, maxrois=None):
        
        def iterator():
            for rois_info, pixels in detection.load_rois_iterator(self.rois_path, maxrois=maxrois):
                if self.camera_readnoise is not None:
                    roipos = np.zeros((len(pixels),2),dtype=np.int32)
                    roipos[:,0] = rois_info['y']
                    roipos[:,1] = rois_info['x']

                    readnoise_rois = self.extract_rois(self.camera_readnoise, roipos, self.roisize)
                else:
                    readnoise_rois = None
                
                yield rois_info, pixels, readnoise_rois
        
        return self.numrois, iterator()
    
    @property
    def all_rois(self):
        return detection.load_rois(self.rois_path)
    
        
