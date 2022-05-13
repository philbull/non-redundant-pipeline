# https://pyuvdata.readthedocs.io/en/latest/uvdata_parameters.html
import ephem, numpy as np
import sys
from pyuvdata import UVData

PI = 3.14159265359

def pol_str(pa):
  switcher = { 
    1: "pI",
    2: "pQ",
    3: "pU",
    4: "pV",
    -1: "RR",
    -2: "LL",
    -3: "RL",
    -4: "LR",
    -5: "XX",
    -6: "YY",
    -7: "XY",
    -8: "YX"
  } 

  return [ switcher[x] for x in pa ]
    
def julian_to_utc(j):
  DUBLIN_DATE = 2415020.00000
  d = ephem.Date(j-DUBLIN_DATE)
  return d

def print_uvdata(uvdata):
  
  print("Number of antennas with data present:", uvdata.Nants_data)
  print("Number of antennas in the array:", uvdata.Nants_telescope)
  print("Number of baselines:", uvdata.Nbls)
  print("Number of baseline-times:", uvdata.Nblts)
  print("Number of frequency channels:", uvdata.Nfreqs)
  print("Number of polarizations:", uvdata.Npols)
  print("Polarizations:", pol_str(uvdata.polarization_array))
  print("Number of spectral windows:", uvdata.Nspws)
  print("Number of times:", uvdata.Ntimes)
  print("ant_1_array:", uvdata.ant_1_array)
  print("ant_2_array:", uvdata.ant_2_array)
  ant_pairs = [ (uvdata.ant_1_array[i], uvdata.ant_2_array[i]) for i in range(len(uvdata.ant_1_array)) ]
  print("ant pairs:", uvdata.get_antpairs())
  print("Antenna names:", uvdata.antenna_names)
  print("Antenna numbers:", uvdata.antenna_numbers)
  print("Antenna positions:", uvdata.antenna_positions)
  print("Baseline array:", uvdata.baseline_array)
  bl_pairs = [ uvdata.baseline_to_antnums(bl) for bl in uvdata.baseline_array ]
  print("Baseline array (pairs):", bl_pairs)
  print("Channel width:", "{:.2f}".format(uvdata.channel_width), "[Hz]")
  print("Frequencies:")
  for i in range(uvdata.Nspws):
    print("  SPW", str(i)+":", "{:.2f}".format(uvdata.freq_array[i, 0]), "-", "{:.2f}".format(uvdata.freq_array[i, -1]), "[Hz]")
  print("Integration time:", [ "{:.2f}".format(x) for x in np.unique(uvdata.integration_time) ], "[s]")
  print("Nsamples in integration:",uvdata.nsample_array)
  if len(np.unique(uvdata.lst_array)) == 1:
    print("LST:", "{:.2f}".format(12*uvdata.lst_array[0]/PI))
  else: print("LSTS:", "{:.2f}".format(12*uvdata.lst_array[0]/PI), "-", "{:.2f}".format(12*uvdata.lst_array[-1]/PI))
  if len(np.unique(uvdata.time_array)) == 1:
    print("Time:", julian_to_utc(uvdata.time_array[0]), "[UTC]")
  else: print("Times:", julian_to_utc(uvdata.time_array[0]), "-", julian_to_utc(uvdata.time_array[-1]), "[UTC]", 
          uvdata.time_array[0], "-", uvdata.time_array[-1], "[JD]")
  print("Telescope name:", uvdata.telescope_name)
  degree_sign = u"\N{DEGREE SIGN}"
  print("Telescope location:", "Lat", "{:.2f}".format(uvdata.telescope_location_lat_lon_alt_degrees[0])+degree_sign, 
	"Lon", "{:.2f}".format(uvdata.telescope_location_lat_lon_alt_degrees[1])+degree_sign,
	"Alt", "{:.2f}".format(uvdata.telescope_location_lat_lon_alt_degrees[2])+"m")
  print("Vis units:", uvdata.vis_units)
  print("extra keywords:", uvdata.extra_keywords)

uvd = UVData()
uvd.read_uvh5(sys.argv[1], read_data=False) #, times=[2458902.4])
print_uvdata(uvd)
#uvd.write_uvfits("x.uvfits", force_phase=True, spoof_nonessential=True)
exit()
ants = uvd.antenna_positions

for i in range(len(ants)):
    for j in range(i+1, len(ants)):
        x = ants[i, 0]-ants[j, 0]
        y = ants[i, 1]-ants[j, 1]
        if abs(y) < 1: print(i, j, x, y, ants[i], ants[j])

index = np.where(uvd.baseline_array==223302)
print(index)
print(uvd.uvw_array[index])

print(uvd.get_ENU_antpos()[0][69]-uvd.get_ENU_antpos()[0][76])
