#!/usr/bin/env python

import os, sys, argparse, warnings
import subprocess 
from shutil import rmtree
import pdb
import time, datetime as dt
import glob
import pickle, hickle
import gzip
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import constants
from scipy.ndimage import zoom

import acor
from emcee import PTSampler, EnsembleSampler
#from emcee.utils import MPIPool
#from mpi4py import MPI

import pyklip.instruments.GPI as GPI
from pyklip import parallelized, klip, fm
from pyklip.fmlib import diskfm
from mcmc_tools import get_ann_stdmap, make_radii, get_radial_stokes
from mcfost.paramfiles import Paramfile

# !!!WARNING!!! Ignoring some annoying warnings about FITS files.
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

# Turn off interactive plotting.
plt.ioff() 


# |----- COMMAND LINE ARGUMENTS -----| #

# Call script as: python run_mcmc.py [-prec_I_back] [-partemp] s_ident d_ident ntemps \
# nwalkers nstep nburn [nthin] [mc_a]
parser = argparse.ArgumentParser(description='MCMC script for disk modeling with MCFOST')
# Parse flags for backwards precession and parallel tempering.
parser.add_argument('-partemp', action='store_false') # default is True
# Parse MCMC settings variables.
parser.add_argument('s_ident')
parser.add_argument('ntemps', type=int, default=10)
parser.add_argument('nwalkers', type=int, default=100) # MUST BE EVEN; 100 is recommended minimum
parser.add_argument('niter', type=int, default=500)
parser.add_argument('nburn', type=int, default=150) # 0 for no burn-in
parser.add_argument('nthin', type=int, nargs='?', default=1) # no thinning by default
parser.add_argument('mc_a', type=float, nargs='?', default=2.0) # 2.0 is emcee default
parser.add_argument('--init_samples_fn', type=str, nargs='?', default=None) # str filename for optional full_sampler log to use to initialize walkers

args = parser.parse_args()

# Turn namespace items into global variables.
globals().update(vars(args))


# |----- PATHS -----| #

ds = "hd35841_160228_H_spec"
# ds_aux = "hd35841_160318_H_pol"
root_path = os.path.expanduser('~/Research/')
log_path = root_path + 'data/gpi/Reduced/%s/logs/' % ds
gen_path = root_path + 'data/gpi/Reduced/%s/' % ds
model_path = root_path + 'data/gpi/Reduced/%s/models/morph_mcmc_%s/' % (ds, s_ident)

parfile = model_path + 'mcfost_param_35841_init_mmcmc_%s.para' % s_ident

# Make a subdirectory for the mcfost models if they will be saved.
if not os.path.isdir(model_path):
    os.makedirs(model_path)
if not os.path.isdir(model_path + "plots/"):
    os.makedirs(model_path + "plots/")
    # print "%s directory not found; created it and necessary subdirectories." % maindir


# |----- LOAD DATA -----| #
# Customize these paths to your data.

# Load radial Stokes Q data from FITS file.
data_Qr = fits.getdata(gen_path + '../hd35841_160318_H_pol/S20160318S0049_podc_radialstokesdc_sm2_perfwv_flex.01.01_smpol10_stpol2-11_mJy_arcsec-2.fits')[1]
# Load a mask that focuses on disk only (excludes outer noise areas).
mask = fits.getdata(gen_path + 'models/disk_mask-281x281_x140y140_tight.fits')
mask[mask==1] = np.nan


# |----- DATASET PARAMETERS -----| #

# skyPA = 74*np.pi/180. # [rad]
pa_list = []
pscale = 0.014166 # [arcsec/pix]
dist = 102.9 # [pc]

# Coordinates of the star in the data.
star = np.array([140, 140]) # [pix] (y,x)
# algo = 'loci'
# algo = 'pyklip'

# Define region of data-model residuals used for goodness-of-fit measurement.
# hw_y is the distance considered on either side of star along y-axis.
hw_y = 80
hw_x = 40
# r_fit defines the inner radial distance from the star considered.
r_fit = 14

# Wavelength at which models are computed, for unit conversion.
lam = 1.647 # [microns]
# Conversion factor from MCFOST's output W/m/Hz to Jy/arcsec^2.
nu = constants.c.value/(1e-6*lam) # [Hz]
conv_WtoJy = (1e3/pscale**2)*1.e26/nu # [(mJy arcsec^-2) / (W m^-2)]

# Option to spatially bin data and models by some factor.
bin_imgs = False
bin_factor = 2.


# |----- MCMC SETUP -----| #

# Flag to delete (False) or save (True) every MCFOST model after sampled.
write_model = False

# Set number of parallel threads to which emcee will assign walkers.
nthreads = 6
# Set number of threads that each MCFOST process will use.
omp_nthreads = 10
try:
    os.environ["OMP_NUM_THREADS"] = str(omp_nthreads) # force MCFOST to use omp_nthreads threads per process
    print "Set OMP_NUM_THREADS = %d for MCFOST (not persistent after MCMC)" % omp_nthreads
except:
    print "Unable to set OMP_NUM_THREADS = %d ; defaulting to the current shell value." % omp_nthreads


# |----- MCMC INITIALIZATION PARAMETERS -----| #

# For Gaussian initialization, set the mean value for each parameter.
inparams = dict(aexp=3.6, amin=27., debris_disk_vertical_profile_exponent=0.92, dust_mass=-6.8,
                dust_pop_0_mass_fraction=0.33, dust_pop_1_mass_fraction=0.33,
                dust_pop_2_mass_fraction=0.33, gamma_exp=1.0, inc=82.1,
                porosity=0.01, r_in=44.0, scale_height=4.0, surface_density_exp=0.7)

# For Gaussian initialization, set the sigma value for each parameter.
psigmas_lib = dict(aexp=1.0, amin=6.0, debris_disk_vertical_profile_exponent=0.4, dust_mass=0.4,
                dust_pop_0_mass_fraction=0.2, dust_pop_1_mass_fraction=0.2,
                dust_pop_2_mass_fraction=0.2, gamma_exp=0.5, inc=2.,
                porosity=0.3, r_in=10.0, scale_height=1.0, surface_density_exp=0.5)

# For a flat/uniform prior, set Min/Max value limits for parameters.
plims_lib = dict(aexp=(2.1, 6.5), amin=(1.1, 40.),
                debris_disk_vertical_profile_exponent=(0.11, 3.),
                dust_mass=(-8.7, -6.0), dust_pop_0_mass_fraction=(0.002, 1.),
                dust_pop_1_mass_fraction=(0.002, 1.), dust_pop_2_mass_fraction=(0.002, 1.),
                 gamma_exp=(-2.99, 3.0), inc=(76., 86.),
                porosity=(0.002, 0.95), r_in=(10.1, 53.),
                scale_height=(0.31, 15.), surface_density_exp=(-2.99, 3.0))


#ModObj

class ModObj:
    """
    Class that contains basic model info.
    """
    
    def __init__(self, pl, pkeys, parfile, gen_path, model_path, lam, s_ident):
        """
        Initialization code for ModObj.
        
        Inputs:
            
        
        """
        pl_dict = dict()
        
        self.pl = pl
        self.pkeys = pkeys
        self.pl_dict = pl_dict.update(zip(pkeys, pl))
        self.parfile = parfile
        self.gen_path = gen_path
        self.model_path = model_path
        self.lam = lam
        self.s_ident = s_ident


if __name__=='__main__':
    mc_main(ds, inparams, s_ident, nwalkers, niter, nburn, nthreads,
            data_Qr, mask, plot=False, save=False)


# Close MPI pool.
try:
    pool.close()
    print "\nMPI Pool closed"
except:
    print "\nNo MPI pools to close."

print "run_mcmc.py script finished\n"

# # Pause interactively before finishing script.
# pdb.set_trace()
