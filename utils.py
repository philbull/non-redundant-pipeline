import numpy as np

import pyuvdata.utils as uvutils
import hera_pspec as hp
import hera_cal as hc

import copy, yaml


def add_noise_from_autos(uvd_in, nsamp=1, seed=None, inplace=False):
    """
    Add noise to a simulation, using the autos to estimate the noise rms.
    
    Parameters
    ----------
    uvd_in : UVData
        Input UVData object.
    
    nsamp : float, optional
        Rescale the generated noise according to some effective number of 
        samples (e.g. to emulate the effect of LST binning). Default: 1.
    
    seed : int, optional
        Random seed to set before generating random noise. Default: None.
    
    inplace : bool, optional
        Whether to add the noise directly to the input UVData object, or 
        operate on a copy. Default: False (operate on a copy).
    
    Returns
    -------
    uvd : UVData
        Output UVData object, now with noise included.
    """
    # Set random seed
    np.random.seed(seed)
    
    # Make a copy of the UVData object if needed
    if inplace:
        uvd = uvd_in
    else:
        uvd = copy.deepcopy(uvd_in)
    
    # Get channel width and integration time
    dnu = uvd.channel_width # in Hz
    dt = uvd.integration_time[0] # in sec
    
    # Get all autos
    v_auto = {}
    for ant in uvd.antenna_numbers:
        auto_idxs = uvd.antpair2ind(ant, ant)
        v_auto[ant] = uvd.data_array[auto_idxs]
    
    # Loop over baselines and add noise to each
    for bl in np.unique(uvd.baseline_array):
        
        # Get antenna numbers and indices in data array
        ant1, ant2 = uvd.baseline_to_antnums(bl)
        bl_idxs = uvd.antpair2ind(ant1, ant2)
        
        # Construct rms for each baseline pair, based on autos
        noise_shape = uvd.data_array[bl_idxs].shape
        std_ij = np.sqrt(v_auto[ant1] * v_auto[ant2].conj() / (nsamp * dt * dnu))
        n = 1.0 * np.random.normal(size=std_ij.size).reshape(std_ij.shape) \
          + 1.j * np.random.normal(size=std_ij.size).reshape(std_ij.shape)
        n *= std_ij / np.sqrt(2.) # to handle real and imaginary contributions
        
        # Add noise realisation
        uvd.data_array[bl_idxs,:,:,:] += n
    
    # Rescale nsamples_array by the assumed number of samples
    uvd.nsamples_array *= nsamp
    
    return uvd


def build_array():
    """
    Create a hexagonal array layout.
    """
    dist = 14.6
    ants = {}

    for i in range(0, 4):
        ants.update([(i, (-3.*dist/2 + i*14.6, 0., 0.))])   
    for i in range(4, 7):
        ants.update([(i, (-2.*dist/2 + (i-4)*14.6, -1.* np.sqrt(3) * dist/2, 0.))])   
    for i in range(7, 10):
        ants.update([(i, (-2.*dist/2 + (i-7)*14.6, +1.* np.sqrt(3) * dist/2, 0.))])
    return ants


def coherent_average_vis(uvd_in, wgt_by_nsample=True, bl_error_tol=1., 
                         inplace=False):
    """
    Coherently average together visibilities in redundant groups.
    
    Parameters
    ----------
    uvd_in : UVData
        Visibility data (should already be calibrated).
    
    wgt_by_nsample : bool, optional
        Whether to weight the average by the number of samples (nsamples) array. 
        If False, uses an unweighted average. Default: True.
    
    bl_error_tol : float, optional
        Tolerance in baseline length (in meters) to use when grouping baselines 
        into redundant groups. Default: 1.
    
    inplace : bool, optional
        Whether to do the averaging in-place, or on a new copy of the UVData 
        object.
    
    Returns
    -------
    uvd_avg : UVData
        UVData object containing averaged visibilities. The averages are 
        assigned to the first baseline in each redundant group (the other 
        baselines in the group are removed).
    """
    # Whether to work in-place or not
    if inplace:
        uvd = uvd_in
    else:
        uvd = copy.deepcopy(uvd_in)
    
    # Get antenna positions and polarizations
    antpos, ants = uvd.get_ENU_antpos()
    antposd = dict(zip(ants, antpos))
    pols = [uvutils.polnum2str(pol) for pol in uvd.polarization_array]

    # Get redundant groups
    reds = hc.redcal.get_pos_reds(antposd, bl_error_tol=bl_error_tol)

    # Eliminate baselines not in data
    antpairs = uvd.get_antpairs()
    reds = [[bl for bl in blg if bl in antpairs] for blg in reds]
    reds = [blg for blg in reds if len(blg) > 0]
    
    # Iterate over redundant groups and polarizations and perform average
    for pol in pols:
        for blg in reds:
            # Get data and weight arrays for this pol-blgroup
            d = np.asarray([uvd.get_data(bl + (pol,)) for bl in blg])
            f = np.asarray([(~uvd.get_flags(bl + (pol,))).astype(np.float) 
                            for bl in blg])
            n = np.asarray([uvd.get_nsamples(bl + (pol,)) for bl in blg])
            if wgt_by_nsample:
                w = f * n
            else:
                w = f
            
            # Take the weighted average
            wsum = np.sum(w, axis=0).clip(1e-10, np.inf)
            davg = np.sum(d * w, axis=0) / wsum
            navg = np.sum(n, axis=0)
            favg = np.isclose(wsum, 0.0)
            
            # Replace in UVData with first bl of blg
            bl_inds = uvd.antpair2ind(blg[0])
            polind = pols.index(pol)
            uvd.data_array[bl_inds, 0, :, polind] = davg
            uvd.flag_array[bl_inds, 0, :, polind] = favg
            uvd.nsample_array[bl_inds, 0, :, polind] = navg

    # Select out averaged bls
    bls = hp.utils.flatten([[blg[0] + (pol,) for pol in pols] for blg in reds])
    uvd.select(bls=bls)
    return uvd


def fix_redcal_degeneracies(data_file, red_gains, true_gains, outfile=None, 
                            overwrite=False):
    """
    Use the true (input) gains to fix the degeneracy directions in a set of 
    redundantly-calibrated gain solutions. This replaces the absolute 
    calibration that would normally be applied to a real dataset in order to 
    fix the degeneracies.
    
    Note that this step should only be using the true gains to fix the 
    degeneracies, and shouldn't add any more information beyond that.
    
    N.B. This is just a convenience function for calling the 
    remove_degen_gains() method of the redcal.RedundantCalibrator class. It 
    also assumes that only the 'ee' polarization will be used.
    
    Parameters
    ----------
    data_file : str
        Filename of the data file (uvh5 format) that is being calibrated. This 
        is only used to extract information about redundant baseline groups.
    
    red_gains : dict of array_like
        Dict containing 2D array of complex gain solutions for each antenna 
        (and polarization).
    
    true_gains : dict
        Dictionary of true (input) gains as a function of frequency. 
        Expected format: 
            key = antenna number (int)
            value = 1D numpy array of shape (Nfreqs,)
    
    outfile : str, optional
        If specified, save the updated gains to a calfits file. Default: None.
    
    overwrite : bool, optional
        If the output file already exists, whether to overwrite it. 
        Default: False.
    
    Returns
    -------
    new_gains : dict
        Dictionary with the same items as red_gains, but where the degeneracies 
        have been fixed in the gain solutions.
    
    uvc : UVCal, optional
        If outfile is specified, also returns a UVCal object containing the 
        updated gain solutions.
    """
    # Get ntimes from gain array belonging to first key in the dict
    ntimes = red_gains[list(red_gains.keys())[0]].shape[0]
    
    # Recreate true_gains dict in the format needed by remove_degen_gains
    # (expands 1D freq.-dep gain array into 2D array as fn. of freq. and time)
    true_gain_dict = {(ant, 'Jee'): np.outer(np.ones(ntimes), true_gains[ant]) 
                      for ant in true_gains.keys()}
    
    # Load data file and get redundancy information
    hd = hc.io.HERAData(data_file)
    reds = hc.redcal.get_reds(hd.antpos, pols=['ee',])
    
    # Create calibrator and fix degeneracies
    RedCal = hc.redcal.RedundantCalibrator(reds)
    new_gains = RedCal.remove_degen_gains(red_gains, 
                                          degen_gains=true_gain_dict, 
                                          mode='complex')
    
    # Save as file if requested
    if outfile is not None:
        uvc = hc.redcal.write_cal(outfile,
                                  new_gains,
                                  hd.freqs,
                                  hd.times,
                                  write_file=True,
                                  return_uvc=True,
                                  overwrite=overwrite)
        return new_gains, uvc
    else:
        # Just return the updated gain dict
        return new_gains


def generate_gains(nants, nfreqs, nmodes=8, seed=None):
    """
    Randomly generate fluctuating complex gains as a function of frequency (no 
    time dependence is included for now).
    
    The gains are constructed from a sum of sine waves, with unit Gaussian 
    random amplitudes that are then divided by mode number. This ensures that 
    highly-fluctuating spectral structure is suppressed relative to the smooth 
    structure.
    
    Parameters
    ----------
    nants : int
        Number of antennas to generate gains for.
        
    nfreqs : int
        Number of frequency channels.
        
    nmodes : int, optional
        Number of sine modes to include in the gain fluctuation model. 
        Increasing the number of modes will result in more rapidly-varying 
        spectral structure. Default: 8.
    
    seed : int, optional
        Random seed to set before generating random noise. Default: None.
    
    Returns
    -------
    gains : dict
        Dictionary of (1D) gain arrays for each antenna.
    """
    # Set random seed and create dimensionless frequency array
    np.random.seed(seed)
    nu = np.linspace(0., 1., nfreqs)
    
    gains = {}
    for ant in range(nants):
        
        # Generate random coeffs as a fn of frequency and sum sine waves
        p = 0
        coeffs = np.random.randn(nmodes) + 1.j * np.random.randn(nmodes)
        coeffs *= 1. / (1. + np.arange(nmodes))
        for n in range(coeffs.size):
            p += coeffs[n] * np.sin(np.pi * n * nu)
        p = (1. + 1.j)/np.sqrt(2.) + 0.1 * p # FIXME
        
        # Store in dict
        gains[ant] = p
    
    return gains


def save_simulated_gains(uvd, gains, outfile, overwrite=False):
    """
    Save simulated gains in a .calfits file.
    
    Parameters
    ----------
    uvd : UVData
        Data object.
    
    gains : dict
        Dictionary of (1D) gain arrays for each antenna.
        
    outfile : str
        Name of the output file.
    
    overwrite : bool, optional
        If the output file already exists, whether to overwrite it. 
        Default: False.
    """
    # Get unique time and frequency arrays
    times = np.unique(uvd.time_array)
    freqs = np.unique(uvd.freq_array)
    
    # Rename keys to use same format as redcal, and inflate in time direction
    gain_dict = {(ant, 'Jee'): np.outer(np.ones(ntimes), gains[ant]) 
                 for ant in gains.keys()}
    
    # Write to calfits file
    hc.redcal.write_cal(outfile,
                        gain_dict,
                        freqs,
                        times,
                        write_file=True,
                        return_uvc=False,
                        overwrite=overwrite)


def load_config(config_file, cfg_default):
    """
    Load a configuration file as a dict. Uses default values for parameters not 
    specified in the file.
    
    Parameters
    ----------
    config_file : str
        Filename of configuration file (YAML format).
        
    cfg_default : dict
        Dictionary containing default values.
    
    Returns
    -------
    cfg : dict
        Dictionary containing configuration.
    """
    # Open and load file
    with open(config_file) as f:
        cfg_in = yaml.load(f, Loader=yaml.FullLoader)
    
    # Overwrite defaults based on parameters in config file
    cfg = copy.deepcopy(cfg_default)
    for grp in cfg_in.keys():
        if isinstance(cfg[grp], dict):
            # Nested dict
            for key in cfg_in[grp].keys():
                cfg[grp][key] = cfg_in[grp][key]
        else:
            cfg[grp] = cfg_in[grp]
    return cfg
    