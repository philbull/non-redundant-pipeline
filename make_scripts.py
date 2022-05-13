import numpy as np
import os, yaml

gen_to_analyse = {
    "generate_sims_4a_0.01.yaml":  "analyse_sims_eor10.yaml",
    "generate_sims_4a_0.02.yaml":  "analyse_sims_eor11.yaml",
    "generate_sims_4b_0.01.yaml":  "analyse_sims_eor12.yaml",
    "generate_sims_4b_0.02.yaml":  "analyse_sims_eor13.yaml",
    "generate_sims_4c_a.yaml":  "analyse_sims_eor14.yaml",
    "generate_sims_4c_b.yaml":  "analyse_sims_eor15.yaml",
    #"generate_sims_main001.yaml":  "analyse_sims_eor3.yaml",
    #"generate_sims_main002.yaml":  "analyse_sims_eor4.yaml",
    "generate_sims_outlier2_1.1.yaml":  "analyse_sims_eor8.yaml",
    "generate_sims_outlier7_1.1.yaml":  "analyse_sims_eor9.yaml",
    "generate_sims_side005.yaml":  "analyse_sims_eor1.yaml",
    "generate_sims_side02.yaml":  "analyse_sims_eor2.yaml",
    "generate_sims_unixystretch0.01.yaml":  "analyse_sims_eor7.yaml",
    "generate_sims_xystretch0.01.yaml":  "analyse_sims_eor5.yaml",
    "generate_sims_xystretch0.02.yaml":  "analyse_sims_eor6.yaml",
    "generate_sims.yaml":  "analyse_sims_eor.yaml"
}

def create_sim_script(runtime, mem, script_to_run, working_yaml, what, in_gen_yaml, in_analyse_yaml, script_name):

    script ="""
#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 8   # 8 cores
#$ -l h_rt={runtime}:00:00    
#$ -l h_vmem={mem}G     # per core
#$ -l gpu=1         # request 1 GPU per host
#$ -l owned

module load hdf5

source ~/.bashrc
conda activate sampler
date
python setup_yaml_files.py {what} {in_gen_yaml} {in_analyse_yaml} {working_yaml}
python {script} {working_yaml}
rm {working_yaml}
date
""".format(runtime=runtime, mem=mem, script=script_to_run, what=what, working_yaml=working_yaml, in_gen_yaml=in_gen_yaml, 
		in_analyse_yaml=in_analyse_yaml)

    with open(script_name, "w") as f:
        f.write(script)

def create_sampler_script(runtime, mem, file_root, niter, script_name):

    script ="""
#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 8   # 8 cores
#$ -l h_rt={runtime}:00:00    
#$ -l h_vmem={mem}G     # per core
#$ -l gpu=1         # request 1 GPU per host
#$ -l owned

module load hdf5

source ~/.bashrc
conda activate sampler
cd ../gain_sampler
date
python run_sampler.py {file_root} {niter}
date
""".format(runtime=runtime, mem=mem, file_root=file_root, niter=niter)

    with open(script_name, "w") as f:
        f.write(script)

def create_plots_script(runtime, mem, file_root, case, script_name):
    
    script ="""
#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 8   # 8 cores
#$ -l h_rt={runtime}:00:00    
#$ -l h_vmem={mem}G     # per core
#$ -l gpu=1         # request 1 GPU per host
#$ -l owned

module load hdf5

source ~/.bashrc
conda activate sampler
cd ../gain_sampler
date
sed s/SAMPLER_DIR/'{file_root}'/ paper_plots.ipynb > paper_plots_{case}.ipynb
papermill paper_plots_{case}.ipynb paper_plots_{case}_out.ipynb
rm paper_plots_{case}.ipynb 
date
""".format(runtime=runtime, mem=mem, file_root=file_root, case=case, script_name=script_name)

    with open(script_name, "w") as f:
        f.write(script)

def cl(case):
    if case == "": return "vanilla"
    else: return case

def cs(case):
    if case == "": return ""
    else: return "_"+case

with open("globals.yaml") as f:
    global_setup = yaml.safe_load(f)

open("Makefile", "w")


os.system("rm run_*_*")
all = ""
for gen in gen_to_analyse:
    case = gen[14:-5]

    create_sim_script(global_setup["sim_time_gen"], int(np.ceil(float(global_setup["sim_mem"])/8)), "generate_sims.py", "g_"+cl(case)+".yaml", 
		"generate", "yaml_files/"+gen, "", "run_"+cl(case)+"_gen")
    create_sim_script(global_setup["sim_time_analyse"], int(np.ceil(float(global_setup["sim_mem"])/8)), "analyse_sims.py", "a_"+cl(case)+".yaml", 
		"analyse", "yaml_files/"+gen, "yaml_files/"+gen_to_analyse[gen], "run_"+cl(case)+"_analyse")

    create_sampler_script(global_setup["sampler_time"], int(np.ceil(float(global_setup["sampler_mem"])/8)), 
		global_setup["output_root"]+"/"+global_setup["catalog"]+"/viscatBC"+cs(case), global_setup["niter"], "run_"+cl(case)+"_sampler")

    create_plots_script(global_setup["sampler_time"], int(np.ceil(float(global_setup["sampler_mem"])/8)), 
                (global_setup["output_root"]+"/"+global_setup["catalog"]+"/sampled_viscatBC").replace("/", "\\/")+cs(case), cl(case), "run_"+cl(case)+"_plots")
    

    file_root = global_setup["output_root"]+"/"+global_setup["catalog"]+"/viscatBC"+cs(case)
    with open("Makefile", "a") as f:
        f.write("# Plots\n")
        f.write(global_setup["output_root"]+"/"+global_setup["catalog"]+"/paper_plots_"+cl(case)+"_out.ipynb: "+file_root+"/sampled_viscatBC"+cs(case))
        f.write("\trun_"+cl(case)+"_plots")
        all += global_setup["output_root"]+"/"+global_setup["catalog"]+"/paper_plots_"+cl(case)+"_out.ipynb"+" "


        f.write("# Sampler\n")
        f.write(file_root+"/sampled_viscatBC"+cs(case)+": "+file_root+"_g_cal.uvh5\n")
        f.write("\tsh run_"+cl(case)+"_sampler\n\n")

        f.write("# Analyse\n")
        f.write(file_root+"_g_cal.uvh5: yaml_files/"+gen_to_analyse[gen]+" "+file_root+"_g.uvh5\n")
        f.write("\tsh run_"+cl(case)+"_analyse\n\n")

        f.write("# Generate\n")
        f.write(file_root+"_g.uvh5: yaml_files/"+gen+"\n")
        f.write("\tsh run_"+cl(case)+"_gen\n\n")

        
with open("Makefile", "a") as f:
    f.write("all: "+all)
 
