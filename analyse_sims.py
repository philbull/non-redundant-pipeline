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
    cfg_analysis = dict( coherent_avg=True,
                         replace_outlier=False,
                         threshold=10.)
    
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

    # Pspec run parameters
    cfg_pspec_run = dict( input_data_weight='identity',
                          norm='I',
                          spw_ranges=[(0, 120)],
                          rephase_to_dset=0,
                          taper='blackman-harris',
                          verbose=True,
                          overwrite=True,
                          bl_len_range=(0, 20.),
                          bl_deg_range=(0., 5.),
                          interleave_times=False,
                          exclude_auto_bls=True,
                          exclude_cross_bls=False,
                          exclude_permutations=True,
                          store_window=False)
    
    # Combine into single dict
    cfg = {
            'redcal':       cfg_redcal,
            'analysis':     cfg_analysis,
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
    output_data = cfg['analysis']['output_data']
    input_truegain = cfg['analysis']['input_truegain']
    coherent_avg = cfg['analysis']['coherent_avg']
  
    input_ext = utils.remove_file_ext(input_data)
    psc_out_inco = input_ext+cfg['pspec']['incoherent_ext']
    psc_out_co = input_ext+cfg['pspec']['coherent_ext']
 

    # (1) Perform redundant calibration
    tstart = time.time()
    cal = hc.redcal.redcal_run(input_data, **cfg['redcal'])
    print("Red calibration run took %2.1f sec" % (time.time() - tstart))    

    #replace outlier
    if cfg['analysis']['replace_outlier']:
        threshold = cfg['analysis']['threshold']
        cal = utils.replace_gain_outlier(cal, threshold=threshold, inplace=True)
    
    #save the cal dict
    np.savez(input_ext+'_cal_dict.npz',**cal) 

    # (2) Load UVData
    uvd_in = UVData()
    uvd_in.read_uvh5(input_data)


    # (3) load true gains
    true_gains, _ = hc.io.load_cal(input_truegain)

    # (4) Fix redundant cal. degeneracies and write new_gains in .calfits format
    # (this fixes the degens to the same values as the true/input gains)
    new_gains = utils.fix_redcal_degeneracies(input_data, 
                                              cal['g_omnical'], 
                                              true_gains)
    hc.redcal.write_cal(input_ext+'_new.calfits', new_gains,
                        uvd_in.freq_array.flatten(),
                        np.unique(uvd_in.time_array))

    # (5) Load calibration solutions and apply to data
    uvc = UVCal()
    uvc.read_calfits(input_ext+'_new.calfits')
    uvd_cal = uvutils.uvcalibrate(uvd_in, uvc, inplace=False, prop_flags=True, 
                                  flag_missing=True)

    # Output calibrated data
    uvd_cal.write_uvh5(output_data, clobber=True)


    # (6) Perform coherent average (if requested)
    if coherent_avg:
        uvd_avg = utils.coherent_average_vis(uvd_cal, wgt_by_nsample=True, 
                                             inplace=False)
    
    # (7) Prepare input params for PSpec run (converts yaml lists to tuples) 
    spw = []
    spw.append(tuple(int(s) 
                for s in cfg['pspec_run']['spw_ranges'].strip("()").split(",")))
    cfg['pspec_run']['spw_ranges'] = spw
    
    for key in ['bl_len_range', 'bl_deg_range']:
        cfg['pspec_run'][key] = tuple(float(s) 
                          for s in cfg['pspec_run'][key].strip("()").split(","))

    # (8) Estimate power spectra
    tstart = time.time()
    pspecd = hp.pspecdata.pspec_run([uvd_cal,],
                                    psc_out_inco,
                                    **cfg['pspec_run'])

    if coherent_avg:
        pspecd_avg = hp.pspecdata.pspec_run([uvd_avg,],
                                            psc_out_co,
                                            **cfg['pspec_run'])

    print("Pspec run took %2.1f sec" % (time.time() - tstart))
    
