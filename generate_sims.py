#!/usr/bin/env python3
"""
Generate simulations of a slightly non-redundant array using hera_sim.
"""
from mpi4py import MPI
import numpy as np
from astropy.time import Time
from astropy.coordinates import EarthLocation

import uvtools
import hera_cal as hc
import pyuvdata
from pyuvdata import UVData

from hera_sim.visibilities import VisibilitySimulation, ModelData, VisCPU
from hera_sim import io
from hera_sim.beams import PolyBeam, PerturbedPolyBeam
from hera_sim import Simulator
from vis_cpu import conversions

from astropy.units import sday, rad
from astropy import units
from astropy.coordinates.angles import Latitude, Longitude
from pyradiosky import SkyModel

# Default setting for use_mpi (can be overridden by commandline arg)
USE_MPI_DEFAULT = True

try:
    import healpy
    import healvis
except:
    print("Unable to import healpy and/or healvis; diffuse mode unavailable")

try:
    import pygdsm
except:
    print("Unable to import pygdsm; vis_cpu diffuse mode unavailable")

import utils
import time, copy, sys


def default_cfg():
    """
    Set parameter defaults.
    """
    # Simulation specification
    cfg_spec = dict( nfreq=20,
                     start_freq=1.e8,
                     bandwidth=0.2e8,
                     start_time=2458902.33333,
                     integration_time=40.,
                     ntimes=40,
                     cat_name="gleamegc.dat",
                     apply_gains=True,
                     apply_noise=True,
                     ant_pert=False,
                     seed=None,
                     ant_pert_sigma=0.0,
                     hex_spec=(3,4), 
                     hex_ants_per_row=None, 
                     hex_ant_sep=14.6,
                     use_ptsrc=True )
                        
    # Diffuse model specification
    cfg_diffuse = dict( use_diffuse=False,
                        use_healvis=False,
                        nside=64,
                        obs_latitude=-30.7215277777,
                        obs_longitude = 21.4283055554,
                        obs_height = 1073,
                        beam_pol='XX',
                        diffuse_model='GSM',
                        eor_random_seed=42,
                        nprocs=1 )

    # EoR model specification (new style)
    cfg_eor = dict( eor_amp=1e-3, 
                    seed=None, 
                    min_delay=-np.inf, 
                    max_delay=np.inf )
    
    # Beam model parameters
    cfg_beam = dict( ref_freq=1.e8,
                     spectral_index=-0.6975,
                     seed=None,
                     perturb_scale=0.0,
                     mainlobe_scale_mean=1.0,
                     mainlobe_scale_sigma=0.0,
                     xstretch_mean=1.0,
                     xstretch_sigma=0.0,
                     ystretch_mean=1.0,
                     ystretch_sigma=0.0,
                     xystretch_same=True,
                     xystretch_dist=None,
                     rotation_dist='',
                     rotation_mean=0.0,
                     rotation_sigma=0.0,
                     mainlobe_width=0.3, 
                     nmodes=8,
                     beam_coeffs=[ 0.29778665, -0.44821433, 0.27338272, 
                                  -0.10030698, -0.01195859, 0.06063853, 
                                  -0.04593295,  0.0107879,  0.01390283, 
                                  -0.01881641, -0.00177106, 0.01265177, 
                                  -0.00568299, -0.00333975, 0.00452368,
                                   0.00151808, -0.00593812, 0.00351559
                                 ] )
    
    # Fluctuating gain model parameters
    cfg_gain = dict(nmodes=8, seed=None)
    
    # Noise parameters
    cfg_noise = dict(nsamp=1., seed=None, noise_file=None)

    # reflection parameters
    cfg_reflection = dict(amp=1.e-2, dly=800.)

    # xtalk parameters
    cfg_xtalk = dict(amp=1.e-2, dly=400.)
    
    # Combine into single dict
    cfg = { 'sim_beam':       cfg_beam,
            'sim_spec':       cfg_spec,
            'sim_diffuse':    cfg_diffuse,
            'sim_eor':        cfg_eor,
            'sim_noise':      cfg_noise,
            'sim_gain':       cfg_gain,
            'sim_reflection': cfg_reflection,
            'sim_xtalk':      cfg_xtalk,
           }
    return cfg


if __name__ == '__main__':
    # Run simulations (MPI-enabled)

    # Get config file name from args
    use_mpi = USE_MPI_DEFAULT
    if len(sys.argv) > 1:
        config_file = str(sys.argv[1])
        if len(sys.argv) > 2:
            use_mpi = bool(sys.argv[2])
            print("use_mpi:", use_mpi)
    else:
        print("Usage: analyse_sims.py config_file [use_mpi]")
        sys.exit(1)
    
    # Begin MPI
    if use_mpi:
        comm = MPI.COMM_WORLD
        myid = comm.Get_rank()
    else:
        comm = None
        myid = 0
    
    # Load config file
    cfg = utils.load_config(config_file, default_cfg())
    cfg_spec = cfg['sim_spec']
    cfg_diffuse = cfg['sim_diffuse']
    cfg_eor = cfg['sim_eor']
    cfg_out = cfg['sim_output']
    cfg_beam = cfg['sim_beam']
    cfg_gain = cfg['sim_gain']
    cfg_noise = cfg['sim_noise']
    cfg_reflection = cfg['sim_reflection']
    cfg_xtalk = cfg['sim_xtalk']
    
    # Construct array layout to simulate
    ants = utils.build_hex_array(hex_spec=cfg_spec['hex_spec'], 
                                     ants_per_row=cfg_spec['hex_ants_per_row'], 
                                     d=cfg_spec['hex_ant_sep'])

    # Construct array layout from UVData object
    #uvd = UVData()
    #uvd.read_uvh5('data_0_0_filtered.uvh5')
    #ants = utils.build_array_from_uvd(uvd, pick_data_ants=True)

    Nant = len(ants)
    ant_index = list(ants.keys())

    # Perturb antenna positions
    if cfg_spec['ant_pert']:
        np.random.seed(cfg_spec['seed'])
        for i in range(Nant):
            ants[i] = tuple(list(ants[i]) + 
                            (cfg_spec['ant_pert_sigma']*np.random.randn(3))) #3 for x,y,z
    
    # Build empty UVData object with correct dimensions
    uvd = utils.empty_uvdata(ants=ants, **cfg_spec)

    # Create frequency array
    freq0 = 100e6
    freqs = np.unique(uvd.freq_array)
    
    # Load point source catalogue
    if cfg_spec['use_ptsrc']:
        ra_dec, flux = utils.load_ptsrc_catalog(cfg_spec['cat_name'], 
                                                freq0=freq0, freqs=freqs, 
                                                usecols=(0,1,2,3), legacy=True)
  
    # Correct source locations so that vis_cpu uses the right frame
    obstime = Time(cfg_spec['start_time'], format="jd", scale="utc")

    location = EarthLocation.from_geodetic(lat=cfg_diffuse['obs_latitude'], lon=cfg_diffuse['obs_longitude'], 
                                       height=cfg_diffuse['obs_height'])
    ra_new, dec_new = conversions.equatorial_to_eci_coords(
                                    ra_dec[:, 0], 
                                    ra_dec[:, 1], 
                                    obstime, 
                                    location, 
                                    unit="rad", 
                                    frame="icrs")
    #ra_dec = np.column_stack((ra_new, dec_new))

    # Construct SkyModel object
    nsrc = ra_new.size
    sky_model = SkyModel(
                        ra=Longitude(ra_new, unit='rad'),
                        dec=Latitude(dec_new, unit='rad'),
                        stokes=np.array(
                            [
                                flux,                       # Stokes I
                                np.zeros((len(freqs), nsrc)), # Stokes Q = 0
                                np.zeros((len(freqs), nsrc)), # Stokes U = 0
                                np.zeros((len(freqs), nsrc)), # Stokes V = 0
                            ]
                        ),
                        name=np.array(["sources"] * nsrc),
                        spectral_type="full",
                        freq_array=freqs,
                    )
  
    # Build list of beams using best-fit coefficients for Chebyshev polynomials
    if cfg_beam['perturb']:

        mainlobe_scale_mean = cfg_beam['mainlobe_scale_mean']
        mainlobe_scale_sigma = cfg_beam['mainlobe_scale_sigma']
        xstretch_mean = cfg_beam['xstretch_mean']
        xstretch_sigma = cfg_beam['xstretch_sigma']
        ystretch_mean = cfg_beam['ystretch_mean']
        ystretch_sigma = cfg_beam['ystretch_sigma']
        rotation_mean = cfg_beam['rotation_mean']
        rotation_sigma = cfg_beam['rotation_sigma']

        np.random.seed(cfg_beam['seed'])

        # Perturb mainlobe
        mainlobe_scale = mainlobe_scale_sigma * np.random.randn(Nant) \
                       + mainlobe_scale_mean

        # Perturb xstretch and ystretch
        xstretch = np.full(Nant,xstretch_mean)
        ystretch = np.full(Nant,ystretch_mean)
        
        if cfg_beam['xystretch_dist'] == 'Gaussian':
            xstretch = xstretch_sigma * np.random.randn(Nant) + xstretch_mean
            if cfg_beam['xystretch_same']:
                ystretch = xstretch
            else:
                ystretch = ystretch_sigma * np.random.randn(Nant) + ystretch_mean
    
        if cfg_beam['xystretch_dist'] == 'Uniform': 
            xstretch = xstretch_mean + np.random.uniform(-2.*xstretch_sigma, 
                                                         2.*xstretch_sigma, 
                                                         Nant)
            if cfg_beam['xystretch_same']:
                ystretch = xstretch
            else:
                ystretch = ystretch_mean + np.random.uniform(-2.*ystretch_sigma, 
                                                             2.*ystretch_sigma, 
                                                             Nant)
        if cfg_beam['xystretch_dist'] == 'Outlier':
            xstretch[cfg_beam['outlier_ant_id']] = cfg_beam['outlier_xstretch']
            ystretch = xstretch

        # Perturb rotation
        rotation = np.zeros(Nant)
        if cfg_beam['rotation_dist'] == 'Gaussian': 
            rotation = rotation_sigma * np.random.randn(Nant) + rotation_mean
        elif cfg_beam['rotation_dist'] == 'Uniform':
            rotation = np.random.uniform(0.,360.,Nant)
        else:
            raise ValueError("rotation_dist '%s' not recognized" \
                             % cfg_beam['rotation_dist'])

        # Extract beam parameters needed by PerturbedPolyBeam only
        allowed_params = ['beam_coeffs', 'perturb_scale', 'mainlobe_width', 'transition_width',
                          'freq_perturb_coeffs', 'freq_perturb_scale', 'perturb_zeropoint', 
                          'spectral_index', 'ref_freq', 'polarized']
        beam_params = {key: cfg_beam[key] for key in cfg_beam if key in allowed_params}

        # Perturb sidelobe and other perturbation
        beam_list = [PerturbedPolyBeam(perturb_coeffs=np.random.randn(cfg_beam['nmodes']),
                                       mainlobe_scale= mainlobe_scale[i],
                                       xstretch=xstretch[i], 
                                       ystretch=ystretch[i],
                                       rotation=rotation[i],
                                       **beam_params) for i in range(Nant)]
    else:
        beam_list = [PolyBeam(cfg_beam['beam_coeffs'], spectral_index=cfg_beam['spectral_index'], ref_freq=cfg_beam['ref_freq']) for i in range(Nant)]

    
    # Use VisCPU to create point source sim, or load a template file with 
    # correct data structures instead
    if cfg_spec['use_ptsrc']:

        # Construct a data model
        data_model = ModelData(uvdata=uvd, 
                               sky_model=sky_model,
                               beams=beam_list)

        # Initialise VisCPU handler object
        viscpu = VisCPU(use_pixel_beams=False, precision=2, mpi_comm=comm)

        # Create a VisibilitySimulation object
        simulator = VisibilitySimulation(data_model=data_model, 
                                         simulator=viscpu)

        # Run the simulation
        tstart = time.time()
        simulator.simulate()
        if myid == 0:
            print("Simulation (point sources) took %2.1f sec" % (time.time() - tstart))
        
        # Write simulated data to file
        uvd = simulator.uvdata
        if myid == 0:
            if cfg_out['datafile_true'] != '':
                uvd.write_uvh5(cfg_out['datafile_true'], clobber=cfg_out['clobber'])

    # Simulate diffuse model using vis_cpu
    if cfg_diffuse['use_diffuse'] and not cfg_diffuse['use_healvis']:
        if myid == 0:
            print("Using GSM diffuse model (vis_cpu)")
        
        # FIXME: This can use a lot of memory in MPI mode, as there are Nprocs duplicates 
        # of the whole datacube!
        # Build SkyModel from GSM (pygdsm)
        gsm_sky = utils.gsm_sky_model(np.unique(uvd.freq_array), 
                                      resolution="hi", 
                                      nside=cfg_diffuse['nside'])

        # Construct a data model
        data_model = ModelData(uvdata=uvd, 
                               sky_model=gsm_sky,
                               beams=beam_list)

        # Initialise VisCPU handler object
        viscpu = VisCPU(use_pixel_beams=False, precision=2, mpi_comm=comm)

        # Create a VisibilitySimulation object
        simulator_diffuse = VisibilitySimulation(data_model=data_model, 
                                                 simulator=viscpu)

        # Run the simulation
        tstart = time.time()
        simulator_diffuse.simulate()
        if myid == 0:
            print("Simulation (diffuse) took %2.1f sec" % (time.time() - tstart))

        # Output diffuse-added data if requested
        if myid == 0:
            if cfg_out['datafile_post_diffuse'] != '':
                uvd.write_uvh5(cfg_out['datafile_post_diffuse'], 
                               clobber=cfg_out['clobber'],
                               fix_autos=True)

    # Close other MPI processes
    if myid != 0:
        # Wait for root worker to finish IO before ending all other worker procs
        print("Worker %d waiting to end" % myid)
        comm.Barrier()
        print("Worker %d finishing" % myid)
        sys.exit(0)

    # Simulate diffuse model using healvis (multi-threaded) (use https://github.com/hughbg/healvis.git)
    if cfg_diffuse['use_diffuse'] and cfg_diffuse['use_healvis']:
        
        # Create healvis baseline spec
        healvis_bls = []
        for i in range(len(ants)):
            for j in range(i, len(ants)):
                _bl = healvis.observatory.Baseline(ants[i], ants[j], i, j)
                healvis_bls.append(_bl)

        # Set times
        times = np.unique(uvd.time_array)
        Ntimes = times.size

        # Create Observatory object
        fov = 360. # deg
        obs = healvis.observatory.Observatory(cfg_diffuse['obs_latitude'], 
                                              cfg_diffuse['obs_longitude'], 
                                              cfg_diffuse['obs_height'],
                                              array=healvis_bls, 
                                              freqs=freqs)
        obs.set_pointings(times)
        obs.set_fov(fov)
        obs.set_beam(beam_list) # beam list
        
        # Create GSM sky model
        if cfg_diffuse["diffuse_model"] == "EOR":
            print("Using EOR diffuse model (healvis)")
            gsm = healvis.sky_model.construct_skymodel('flat_spec', freqs=freqs, Nside=cfg_diffuse['nside'], 
                                                    ref_chan=0, sigma=1e-3, seed=cfg_diffuse['eor_random_seed'])
        elif cfg_diffuse["diffuse_model"] == "GSM":
            print("Using GSM diffuse model (healvis)")
            gsm = healvis.sky_model.construct_skymodel('gsm', freqs=freqs,
                                                   ref_chan=0,
                                                   Nside=cfg_diffuse['nside'])
        else:
            raise ValueError("Invalid diffuse model: %s" % cfg_diffuse["diffuse_model"])

        gsm = healvis.sky_model.construct_skymodel('gsm', freqs=freqs, 
                                                   ref_chan=0,
                                                   Nside=cfg_diffuse['nside'])

        # Compute visibilities
        gsm_vis, _times, _bls = obs.make_visibilities(gsm,
                                              beam_pol=cfg_diffuse['beam_pol'], 
                                              Nprocs=cfg_diffuse['nprocs'])
        
        # Check that ordering of healvis output matches existing uvd object
        antpairs_hvs = [(healvis_bls[i].ant2, healvis_bls[i].ant1) for i in _bls]
        antpairs_uvd = [uvd.baseline_to_antnums(_b) for _b in uvd.baseline_array]
        
        # Check time ordering
        assert np.all(_times == uvd.time_array), \
                                     "healvis 'times' array does not match the " \
                                     "ordering of existing UVData.time_array"

        # Check baseline ordering (fix if wrong)
        try:
            assert antpairs_hvs == antpairs_uvd, \
                                     "healvis 'bls' array does not match the " \
                                     "ordering of existing UVData.baseline_array"

            # Add diffuse data to UVData object
            uvd.data_array[:,:,:,0] += gsm_vis

        except AssertionError:
            print("Healvis and UVData array ordering do not match; starting reordering.")

            # Get ordered list of baseline IDs for healvis baselines
            bls_hvs = [uvd.antnums_to_baseline(_bl[0], _bl[1]) for _bl in antpairs_hvs]
            
            # Loop over bl-times and assign values (can be slow)
            for i in range(len(_times)):
                idx = np.where((uvd.baseline_array == bls_hvs[i]) & (uvd.time_array == _times[i]))
                uvd.data_array[idx,:,:,0] += gsm_vis[i]
            print("Finished array reordering.")
        
        # Output diffuse-added data if requested
        if cfg_out['datafile_post_diffuse'] != '':
            uvd.write_uvh5(cfg_out['datafile_post_diffuse'], 
                           clobber=cfg_out['clobber'],
                           fix_autos=True)
    
    # Add new-style foreground and EoR simulation effects
    if cfg_eor["use_eor"]:
        # Construct new Simulator object and add flat-spectrum EoR to it
        sim = Simulator(data=simulator.uvdata)
        sim.add("noiselike_eor", 
                eor_amp=float(cfg_eor['eor_amp']), 
                seed=cfg_eor['seed'], 
                min_delay=cfg_eor['min_delay'], 
                max_delay=cfg_eor['max_delay'])
        
        if cfg_out['datafile_post_eor'] != '':
            sim.write(filename=cfg_out['datafile_post_eor'], 
                      clobber=cfg_out['clobber'],
                      fix_autos=True)

    # Add noise
    if cfg_spec['apply_noise']:
        uvd = utils.add_noise_from_autos(uvd, input_noise=cfg_noise['noise_file'], 
                                         nsamp=cfg_noise['nsamp'], 
                                         seed=cfg_noise['seed'], inplace=True)
        if cfg_out['datafile_post_noise'] != '':
            uvd.write_uvh5(cfg_out['datafile_post_noise'], 
                           clobber=cfg_out['clobber'],
                           fix_autos=True)
    
    # Add fluctuating gain model if requested
    if cfg_spec['apply_gains']:
        
        # Generate fluctuating gain model
        nfreqs = cfg_spec['nfreq']
        gg = utils.generate_gains(Nant, nfreqs, **cfg['sim_gain'])
        if cfg_out['gain_file'] != '':
            utils.save_simulated_gains(uvd, gg, 
                                       outfile=cfg_out['gain_file'], 
                                       overwrite=cfg_out['clobber'])
        
        # Loop over all baselines and apply gain factor
        for bl in np.unique(uvd.baseline_array):
            
            # Calculate product of gain factors (time-indep. for now)
            ant1, ant2 = uvd.baseline_to_antnums(bl)
            gigj = gg[ant1] * gg[ant2].conj()
            
            # Get index in data array
            idxs = uvd.antpair2ind((ant1, ant2))
            dtype = uvd.data_array.dtype
            uvd.data_array[idxs,0,:,0] *= np.atleast_2d(gigj.astype(dtype))
        
        # Output gain-multiplied data if requested
        uvd.write_uvh5(cfg_out['datafile_post_gains'], 
                       clobber=cfg_out['clobber'])

    # Add reflection gains
    if 'apply_reflection' in cfg_spec.keys() and cfg_spec['apply_reflection']:
        sim = Simulator(data=uvd)
        sim.add("reflection_gains", **cfg_reflection)
        if cfg_out['datafile_post_reflection'] != '':
            sim.write(cfg_out['datafile_post_reflection'], save_format="uvh5")

    # Add crosstalk
    if 'apply_xtalk' in cfg_spec.keys() and cfg_spec['apply_xtalk']:
        sim.add("cross_coupling_xtalk", **cfg_xtalk)
        if cfg_out['datafile_post_xtalk'] != '':
            sim.write(cfg_out['datafile_post_xtalk'], save_format="uvh5")

    
    # Sync with other workers and finalise
    if use_mpi:
        comm.Barrier()
        print("Worker %d finishing" % myid)
    sys.exit(0)
    
