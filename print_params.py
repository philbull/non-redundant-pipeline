import yaml, sys


with open(sys.argv[1]) as f:
        cfg_in = yaml.load(f, Loader=yaml.FullLoader)

params = [ "which", "apply_gains", "start_time", "diffuse_model", "dummy_source", "az_za_corrections", "cat_name", "orig_yaml", "noise_file", "apply_noise", "use_diffuse", "nside", "ntimes", "nfreq", "load_points_sim" ]

def scan(d):
      for k, v in d.items():
        if isinstance(v, dict):
          v = scan(v)
        else:
          if k in params:
            print(k, "=", d[k])
      return d    

cfg_in = scan(cfg_in)
