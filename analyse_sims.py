#!/usr/bin/env python3
"""
Analysis pipeline for simulated data; performs redundant calibration, coherent 
averaging, and power spectrum estimation.
"""
import numpy as np

import uvtools
import hera_cal as hc
import hera_pspec as hp
from pyuvdata import UVCal, UVData
import pyuvdata.utils as uvutils

import utils
import time, copy, sys, yaml

def default_cfg():
    """
    Set parameter defaults.
    """
    # General analysis parameters
    cfg_analysis = dict( coherent_avg=True )
    
    # Redcal parameters
    cfg_redcal = dict( firstcal_ext='.first.calfits', 
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
                       max_bl_cut=40. )

    #Pspec parameters
    cfg_pspec = dict( incoherent_ext='.inco.hdf5',
                      coherent_ext='.co.hdf5' )

    # Pspec run parameters
    cfg_pspec_run = dict( input_data_weight='identity',
                          norm='I',
                          spw_ranges=[(0, 120)],
                          rephase_to_dset=0,
                          taper='blackman-harris',
                          verbose=True,
                          overwrite=True,
                          bl_len_range=(0, 2.0e10),
                          bl_deg_range=(0., 180.),
                          exclude_auto_bls=False,
                          exclude_cross_bls=False,
                          exclude_permutations=True )
    
    # Combine into single dict
    cfg = {
            'redcal':       cfg_redcal,
            'analysis':     cfg_analysis,
            'pspec':        cfg_pspec,
            'pspec_run':    cfg_pspec_run,
          }
    return cfg


if __name__ == '__main__':
    # Analyse simulations
    
    # Get config file name
    if len(sys.argv) > 1:
        config_file = str(sys.argv[1])
    else:
        print("Usage: analyse_sims.py config_file")
        sys.exit(1)
    
    # Load config file
    cfg = utils.load_config(config_file, default_cfg())
            
    # Get input data filename
    input_data = cfg['analysis']['input_data']
    input_truegain = cfg['analysis']['input_truegain']
    #red_grps, vecs, bl_lens, = uvd.get_redundancies()


    # (1) Perform redundant calibration
    cal = hc.redcal.redcal_run(input_data, **cfg['redcal'])
    
    
    # (2) Load UVData
    uvd_in = UVData()
    uvd_in.read_uvh5(input_data)


    # (3) load true gains
    true_gains = utils.load_gain(input_truegain)

    # (4) Fix redundant cal. degeneracies and write new_gains in .calfits format
    # (this fixes the degens to the same values as the true/input gains)
    #true_gains = None # FIXME: Load these! They will be in a calfits file
    new_gains = utils.fix_redcal_degeneracies(input_data, 
                                              cal['g_omnical'], 
                                              true_gains)
    hc.redcal.write_cal('new_gains111.calfits',new_gains,
                        uvd_in.freq_array.flatten(),
                        np.unique(uvd_in.time_array))

    # (5) Load calibration solutions and apply to data
    uvc = UVCal()
    uvc.read_calfits("new_gains111.calfits")
    uvd_cal = uvutils.uvcalibrate(uvd_in, uvc, inplace=False, prop_flags=True, 
                                  flag_missing=True)
    
    # (4) Perform coherent average (if requested)
    #if coherent_avg:
    #    uvd_avg = utils.coherent_average_vis(uvd_cal, wgt_by_nsample=True, 
    #                                         inplace=False)


    # chenges few paramters structures for Pspec run
    spw = []
    spw.append(tuple(int(s) for s in 
                     cfg['pspec_run']['spw_ranges'].strip("()").split(",")))
    
    cfg['pspec_run']['spw_ranges'] = spw

    cfg['pspec_run']['bl_len_range'] = tuple(float(s) for s in cfg['pspec_run']['bl_len_range'].strip("()").split(","))

    cfg['pspec_run']['bl_deg_range'] = tuple(float(s) for s in cfg['pspec_run']['bl_deg_range'].strip("()").split(","))


    # (5) Estimate power spectra
    input_ext = utils.get_file_ext(input_data)
    psc_out_inco = input_ext+cfg['pspec']['incoherent_ext']
    psc_out_co = input_ext+cfg['pspec']['coherent_ext']
  
    pspecd = hp.pspecdata.pspec_run([uvd_cal,uvd_cal],
                                    psc_out_inco,**cfg['pspec_run'])

    ##if coherent_avg:
    ##    pspecd_avg = hp.pspecdata.pspec_run([uvd_avg,uvd_avg],
    ##                       psc_out_co,**cfg['pspec_run'])

    # FIXME
    
    # Timing
    tstart = time.time()
    print("Run took %2.1f sec" % (time.time() - tstart))
    
