#!/usr/bin/env python3
"""
Generate simulations of a non-redundant array using hera_sim.
"""
from mpi4py import MPI
import numpy as np

import uvtools
import hera_cal as hc
import pyuvdata

from hera_sim.visibilities import VisCPU, conversions
from hera_sim.beams import PolyBeam

import utils
import time, copy, sys


def default_cfg():
    """
    Set parameter defaults.
    """
    # Simulation parameters
    cfg_sim = dict( xx,
                    beam_coeffs=[
                              2.35088101e-01, -4.20162599e-01,  2.99189140e-01, 
                             -1.54189057e-01,  3.38651457e-02,  3.46936067e-02, 
                             -4.98838130e-02,  3.23054464e-02, -7.56006552e-03, 
                             -7.24620596e-03,  7.99563166e-03, -2.78125602e-03,
                             -8.19945835e-04,  1.13791191e-03, -1.24301372e-04, 
                             -3.74808752e-04,  1.93997376e-04, -1.72012040e-05 
                               ],
                    a=1.,
                        )
    
    # Combine into single dict
    cfg = { 'simulation': cfg_sim }
    return cfg



if __name__ == '__main__':
    # Run simulations
    
    # Begin MPI
    comm = MPI.COMM_WORLD
    myid = comm.get_Rank()
    
    
    #gleamegc.dat'
    
    _cfg = ['']
    
    ants = utils.build_array()
    Nant = len(ants)
    ant_index = list(ants.keys())
    
    uvd = empty_uvdata(ntimes=20, nfreqs=40)

    # Create frequency array
    freq0 = 1e8
    ra_dec, flux = load_ptsrc_catalog(cfg['cat_name'], freq0=freq0, 
                                      freqs=np.unique(uvd.freq_array))

    # Best fit coeffcients for Chebyshev polynomials
    coeff = np.array()
    # From power-law fitting of the width of Fagnoni beam at 100 and 200 MHz
    spindex = -0.6975
    
    # Build list of beams by 
    beam_list = [ PolyBeam(beam_coeffs=coeff, 
                           spectral_index=spindex,
                           ref_freq=freq0) 
                  for i in range(Nant)]
    
    # Create VisCPU visibility simulator object (MPI-enabled)
    simulator = VisCPU(
        uvdata=uvd,
        beams=beam_list,
        beam_ids=ant_index,
        sky_freqs=freqs,
        point_source_pos=ra_dec,
        point_source_flux=flux,
        real_dtype=np.float64,
        complex_dtype=np.complex128,
        use_pixel_beams=False,
        bm_pix = 20,
        mpi_comm=comm
    )
        
    # Run simulation
    tstart = time.time()
    simulator.simulate()
    print("Run took %2.1f sec" % (time.time() - tstart))
    
    if myid != 0:
        # Wait for root worker to finish IO before quitting
        comm.Barrier()
        comm.Finalize() # FIXME
        sys.exit(0)
    
    # Write simulated data to file
    uvd = simulator.uvdata
    red_grps, vecs, bl_lens, = uvd.get_redundancies()
    uvd_n.write_uvh5("calibration/test_sim.uvh5", clobber=True)
    
    # Generate gain model
    gg = utils.generate_gains(nants, nfreqs, nmodes=8, seed=None)
    utils.save_simulated_gains(uvd, gains, outfile='test.calfits', overwrite=False)
    
    # Loop over all baselines and apply gain factor
    for bl in np.unique(uvd_g.baseline_array):
        
        # Calculate product of gain factors (time-indep. for now)
        ant1, ant2 = uvd_g.baseline_to_antnums(bl)
        gigj = true_gains[ant1] * true_gains[ant2].conj()
        
        # Get index in data array
        idxs = uvd_g.antpair2ind((ant1, ant2))
        uvd_g.data_array[idxs,0,:,0] *= np.atleast_2d(gigj.astype(uvd_g.data_array.dtype))
    uvd_g.write_uvh5("calibration/test_sim_g.uvh5", clobber=True)
    
    # Add noise
    uvd_n = utils.add_noise_from_autos(uvd_g, nsamp=1., seed=10, inplace=False)    
    uvd_n.write_uvh5("calibration/test_sim_n.uvh5", clobber=True)
    
    # Sync with other workers and finalise
    comm.Barrier()
    comm.Finalize()
    sys.exit(0)
