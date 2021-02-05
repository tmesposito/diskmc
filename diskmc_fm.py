#!/usr/bin/env python
#
# Disk forward modeling functions.

import numpy as np
from scipy.ndimage import gaussian_filter


def make_diskfm(mod_I, dataset, load_from_basis, basis_filename, save_basis,
                numbasis, annuli, OWA, numthreads):
    import pyklip.instruments.GPI as GPI
    from pyklip import fm
    from pyklip.fmlib import diskfm
    
    # Create object from diskfm.DiskFM class.
    print("\nInitializing DiskFM object...")
    modfm = diskfm.DiskFM(dataset.input.shape, numbasis, dataset, mod_I,
                          load_from_basis=load_from_basis, save_basis=save_basis,
                          annuli=annuli, subsections=subsections, OWA=OWA,
                          basis_filename=basis_fn,
                          numthreads=numthreads)
    
    
    return modfm


def do_fm_loci(dataset, new_model, c_list):
    
    fmsub_mod = do_loci_selfsub(ds_name='gpi/Reduced/%s' % ds, inp_mod=new_model, c_list=c_list,
                    pa_list=dataset.PAs_rad, fl=dataset.fl, d_ident=dataset.d_ident,
                    N_delta=dataset.N_delta, dr=dataset.dr, W=dataset.W, g=dataset.g,
                    N_a=dataset.N_a, IWA=dataset.IWA, OWA=dataset.OWA,
                    rad=None, star=dataset.star, skyPA=0., ctype=dataset.ctype,
                    plotNum=None, save=False, submod_out=True, subfunc_out=False,
                    s_ident='', silent=True)
    
    return fmsub_mod


def do_fm_pyklip(modfm, dataset, new_model):
    import pyklip.instruments.GPI as GPI
    from pyklip import fm
    from pyklip.fmlib import diskfm
    # Define KLIP parameters used for data.
    # Settings are for a9s1mv1_medcollapse.
    ann = 9
    subs = 1
    mvmt = 1
    minrot = None
    highpass = False
    sufx = '_%s_%s_%slk' % (mctype, s_ident, which)
    kl = 1
    
    # NOTE that a9s1mv1_medcollapse used subset of images: 70-99 inclusive.
    fl = [path_data + 'S20160228S%04d_spdc_distorcorr_phot_4p_hpNone_Jy_arcsec-2.fits' % ii for ii in range(70,114)]
    
    dataset = GPI.GPIData(fl, highpass=highpass, meas_satspot_flux=False)

    # Manually decreasing inner working angle to improve inner KLIP.
    dataset.IWA = 10 # [pix]
    dataset.OWA = 135
    # Manually set plate scale to best known value.
    dataset.lenslet_scale = 0.014166 # [arcsec/pix] best as of 6-2016
    numbasis = np.array([1, 2, 3, 10, 20, 50])
    maxnumbasis = 50
    
    
    star = np.array([140, 140]) #np.mean(dataset.centers, axis=0)
    collapse_spec = True
    
    # If desired, collapse the spec cube as sum of wavelength channels.
    if collapse_spec and dataset.prihdrs[0]['DISPERSR'] != 'WOLLASTON':
        input_collapsed = []
        ## Average all spec cubes along wavelength axis.
        # Sum each spec cube along wavelength axis to collapse channels.
        for fn in fl:
            # input_collapsed.append(numpy.nanmedian(fits.getdata(fn), axis=0))
            input_collapsed.append(np.sum(fits.getdata(fn), axis=0))
        input_collapsed = np.array(input_collapsed)
        dataset.input = input_collapsed
        
        # Average centers of all wavelength slices and store as new centers.
        centers_collapsed = []
        sl = 0
        while sl < dataset.centers.shape[0]:
            centers_collapsed.append(np.mean(dataset.centers[sl:sl+37], axis=0))
            sl += 37
        centers_collapsed = np.array(centers_collapsed)
        dataset.centers = centers_collapsed
        
        # Reduce dataset info from 37 slices to 1 slice.
        dataset.PAs = dataset.PAs[list(range(0,len(dataset.PAs),37))]
        dataset.filenums = dataset.filenums[list(range(0,len(dataset.filenums),37))]
        dataset.filenames = dataset.filenames[list(range(0,len(dataset.filenames),37))]
        dataset.wcs = dataset.wcs[list(range(0,len(dataset.wcs),37))]
        
        # Lie to pyklip about wavelengths.
        dataset.wvs = np.ones(input_collapsed.shape[0])
    
    # Create object from diskfm.DiskFM class.
    print("\nInitializing DiskFM object...")
    modfm = diskfm.DiskFM(dataset.input.shape, np.array(numbasis), dataset, mod_I,
                          load_from_basis=load_from_basis, save_basis=save_basis,
                          annuli=ann, subsections=subs, OWA=dataset.OWA,
                          basis_filename=basis_fn,
                          numthreads=numthreads)
    
# TEMP!!!
    modfm.maxnumbasis = maxnumbasis
    # modfm.numthreads = numthreads

    if mvmt is not None:
        fname = 'hd35841_pyklipfm_a%ds%dmv%d_hp%.1f_k%d-%d' % (ann, subs, mvmt, highpass, numbasis[0], numbasis[-1]) + sufx 
    elif minrot is not None:
        fname = 'hd35841_pyklipfm_a%ds%dmr%d_hp%.1f_k%d-%d' % (ann, subs, minrot, highpass, numbasis[0], numbasis[-1]) + sufx 
    
    
    if load_from_basis:
        # # Set model's aligned center property (do usual swap of y,x).
        # modfm.aligned_center = mod_cen_aligned[::-1]
        
        # Use loaded basis vectors to FM the original disk model (get images grouped by KL mode).
        fmsub_mod_imgs = modfm.fm_parallelized()
        
        # # Save the fm output FITS to disk.
        # modfm.save_fmout(dataset, fmsub_mod_imgs, path[:-1], fname, numbasis, '', False, None)
        
        # Take mean across the FM'd images for each KL mode.
        fmsub_mod = np.mean(fmsub_mod_imgs, axis=1)
        # Mask interior to the IWA (pyklip includes r=IWA pixels in first annulus).
        fmsub_mod[:, radii_data < dataset.IWA] = np.nan
        
        mod_I_fm = fmsub_mod[np.where(numbasis==kl)[0][0]]
        
    else:
# FIX ME!!! FM without saved bases is likely broken.
        # pyklip FM the model dataset (similar yet distinct function from pyklip.klip_dataset)
        # This writes the self-subtracted model and the klip'd data to disk but does not
        # output any arguments.
        if ann==1:
            padding = 0
        else:
            padding = 3
        
        print("KLIP FM without a saved basis set is NOT FUNCTIONAL! Will probably fail.")
        fmout = fm.klip_dataset(dataset, modfm, mode='ADI', outputdir=path_data, fileprefix=fname,
                annuli=ann, subsections=subs, OWA=dataset.OWA, N_pix_sector=None, movement=mvmt,
                minrot=minrot, numbasis=np.array(numbasis), maxnumbasis=maxnumbasis,
                numthreads=numthreads, calibrate_flux=False, aligned_center=star[::-1], #aligned_center=mod_cen_aligned[::-1]
                spectrum=None, highpass=highpass, save_klipped=False, padding=padding,
                mute_progression=False)
    
    
    
    # Update the model image in modfm object to a new model.
    modfm.update_disk(new_model)
    # # Load the KL basis info from log file instead of slowly recalculating.
#    modfm.load_basis_files(modfm.basis_filename)
    # FM the new disk model.
    fmsub_mod_imgs = modfm.fm_parallelized()
    
    # # Save the fm output FITS to disk.
    # modfm.save_fmout(dataset, fmsub_mod_imgs, path[:-1], fname, numbasis, '', False, None)
    
    # Take mean across the FM'd images for each KL mode.
    fmsub_mod = np.nanmean(fmsub_mod_imgs, axis=1)
    # # Mask interior to the IWA (pyklip includes r=IWA pixels in first annulus).
    # fmsub_mod[:, radii < dataset.IWA] = numpy.nan
    
    return fmsub_mod

