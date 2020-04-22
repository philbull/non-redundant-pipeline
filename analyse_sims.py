#!/usr/bin/env python3
"""
Analysis pipeline for simulated data; performs redundant calibration, coherent 
averaging, and power spectrum estimation.
"""
import numpy as np

import uvtools
import hera_cal as hc
from pyuvdata import UVCal, UVData

import utils
import time, copy


if __name__ == '__main__':
    # Analyse simulations
    
    input_data = "calibration/test_sim.uvh5"
    coherent_avg = True
    #red_grps, vecs, bl_lens, = uvd.get_redundancies()
    
    # (1) Perform redundant calibration
    cal = hc.redcal.redcal_run(input_data, 
                               filetype='uvh5', 
                               firstcal_ext='.first.calfits', 
                               omnical_ext='.omni.calfits', 
                               omnivis_ext='.omni_vis.uvh5', 
                               meta_ext='.redcal_meta.hdf5', 
                               iter0_prefix='', 
                               outdir=None, 
                               ant_metrics_file=None, 
                               clobber=True, 
                               nInt_to_load=None, 
                               pol_mode='1pol', 
                               bl_error_tol=1.0,
                               ex_ants=[], 
                               ant_z_thresh=4., 
                               max_rerun=10, 
                               solar_horizon=0.0, 
                               flag_nchan_low=0, 
                               flag_nchan_high=0, 
                               fc_conv_crit=1e-7, 
                               fc_maxiter=1000, 
                               oc_conv_crit=1e-12, 
                               oc_maxiter=3500, 
                               check_every=10, 
                               check_after=50, 
                               gain=0.4, 
                               add_to_history='', 
                               verbose=True, 
                               min_bl_cut=10.,
                               max_bl_cut=40.)
    
    # (2) Fix redundant cal. degeneracies
    # (this fixes the degens to the same values as the true/input gains)
    true_gains = None # FIXME: Load these! They will be in a calfits file
    new_gains = utils.fix_redcal_degeneracies(input_data, 
                                              cal['g_omnical'], 
                                              true_gains)
    
    # (3) Load data
    uvd_in = UVData()
    uvd_in.read_uvh5(input_data)
    
    # (3) Load calibration solutions and apply to data
    uvc = UVCal()
    uvc.read_calfits("calibration/test_sim.omni.calfits")
    uvd_cal = uvutils.uvcalibrate(uvd_in, uvc, inplace=False, prop_flags=True, 
                                  flag_missing=True)
    
    # (4) Perform coherent average (if requested)
    if coherent_avg:
        uvd_avg = utils.coherent_average_vis(uvd_cal, wgt_by_nsample=True, 
                                             inplace=False)
    
    # (5) Estimate power spectra
    # FIXME
    
    # Timing
    #tstart = time.time()
    #print("Run took %2.1f sec" % (time.time() - tstart))
    
