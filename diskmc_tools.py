#!/usr/bin/env python

import numpy as np
from scipy.ndimage import gaussian_filter


def make_radii(arr, cen):
    """
    Make array of radial distances from centerpoint cen in 2-D array arr.
    """
    # Make array of indices matching arr indices.
    grid = np.indices(arr.shape, dtype=float)
    # Shift indices so that origin is located at cen coordinates.
    grid[0] -= cen[0]
    grid[1] -= cen[1]
    
    return np.sqrt((grid[0])**2 + (grid[1])**2) # [pix]


def get_ann_stdmap(im, cen, radii, r_max=None, mask_edges=False):
    """
    Get standard deviation map from image im, measured in concentric annuli 
    (1 pixel wide) around cen. NaN's in im will be ignored (use for masking).
    
    Inputs:
        im: image from which to calculate standard deviation map.
        cen: list-like (y,x) coordinates for center of image im.
        radii: array of radial distances for pixels in im from cen.
        r_max: maximum radius to measure std dev.
        mask_edges: False or int N; mask out N pixels on all edges to mitigate
            edge effects biasing the standard devation at large radii.
    """
    
    if r_max==None:
        r_max = radii.max()
    
    if mask_edges:
        cen = np.array(im.shape)/2
        mask = np.ma.masked_invalid(gaussian_filter(im, mask_edges)).mask
        mask[cen[0]-mask_edges*5:cen[0]+mask_edges*5, cen[1]-mask_edges*5:cen[1]+mask_edges*5] = False
        im = np.ma.masked_array(im, mask=mask).filled(np.nan)
    
    stdmap = np.zeros(im.shape, dtype=float)
    for rr in np.arange(0, r_max, 1):
        # Central pixel often has std=0 b/c single element. Avoid this by giving
        # it same std dev as 2 pix-wide annulus from r=0 to 2.
        if rr==0:
            wr = np.nonzero((radii >= 0) & (radii < 2))
            stdmap[cen[0], cen[1]] = np.nanstd(im[wr])
        else:
            wr = np.nonzero((radii >= rr-0.5) & (radii < rr+0.5))
            #stdmap[wr] = np.std(im[wr])
            stdmap[wr] = np.nanstd(im[wr])
    
    return stdmap


def get_radial_stokes(Q, U, phi):
    """
    Take normal Stokes parameters Q and U and convert them to
    their radial counterparts Qr and Ur. Math from Schmid et al. 2006.
    Conversion matches that of GPItv and pipeline primitive.
    
    Inputs:
        Q: Stokes Q image.
        U: Stokes U image.
        phi: polar angle of a given position in Q or U image, calculated as
                np.arctan2(yy - star_y, xx - star_x)
    """
    
    Qr = Q*np.cos(2*phi) + U*np.sin(2*phi)
    Ur = -Q*np.sin(2*phi) + U*np.cos(2*phi)
    
    return Qr, Ur
