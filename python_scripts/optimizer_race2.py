import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import netcdf_read_write as nrw
import training_data_generation as tdg
import utils_optimizers as uopt
import optimize as opt
import healpy as hp
import utils_intensity_map as uim
import utils_healpy as uhp
import utils_deck_generation as idg

np_complex = np.vectorize(complex)
root_dir = "path/to_run/file"

list_dirs = ["run_file_name",
            ]

list_labels = list_dirs
"""
list_labels = ["cone, theta, quad",
               ]
"""

ndirs = len(list_dirs)

num_modes = 30
num_inputs = 16
random_seed = 12345
display_steradians = False

dataset = {}
dataset_params = {}
deck_gen_params = {}
facility_spec = {}
sys_params = {}
opt_params = {}
fitness_temporal = {}
fitness_overall = {}
opt_params = {}

for idir in range(ndirs):
    label = list_labels[idir]
    diag_dir = root_dir +"/"+ list_dirs[idir]
    
    sys_params[label] = tdg.define_system_params(diag_dir)
    dataset[label], dataset_params[label], deck_gen_params[label], facility_spec[label] = idg.load_data_dicts_from_file(sys_params[label])

    sys_params[label]["plot_file_type"] = ".png"
    opt_params[label] = uopt.define_optimizer_parameters(diag_dir, 0, 0, dataset_params[label], facility_spec[label], sys_params[label])
    num_sims = np.shape(dataset[label]["rms"])[0]

    num_evaluated = dataset[label]["num_evaluated"]
    total_outputs = np.shape(dataset[label]["rms"])[0]*np.shape(dataset[label]["rms"])[1]
    num_zero_vals = np.count_nonzero(dataset[label]["rms"]==0)
    ind = np.where(dataset[label]["rms"]==0)

    num_failed = (2 * num_evaluated) - (total_outputs - num_zero_vals)
    print(ind[0])
    dataset[label]["rms"][ind[0]] = 1000

    ind_bug = np.where(dataset[label]["avg_flux"][:,1] > 100.0)
    num_failed = len(ind_bug[0])
    
    ind_profile = 0

fig, ax = plt.subplots(figsize=(4,4), dpi=200)
    
for idir in range(ndirs):
    label = list_labels[idir]
    diag_dir = root_dir +"/"+ list_dirs[idir]
    num_sims = np.shape(dataset[label]["rms"])[0]
    
    inputs_mean = np.mean(dataset[label]["input_parameters"])
    input_standard_deviation = np.sqrt(np.var(dataset[label]["input_parameters"], axis=0))
    fitness = uopt.fitness_function(dataset[label], opt_params[label])
    number_of_timesteps = np.shape(dataset[label]["rms"][:,:])[1]
    rms_of_rms = np.sqrt(np.sum(dataset[label]["rms"][:,:]**2, axis=1) / float(number_of_timesteps))
    
    start_ind = 0
    x_markers = np.linspace(start_ind, num_sims, num_sims-start_ind)
    ax.semilogy(x_markers,np.minimum.accumulate(rms_of_rms[start_ind:]), label=label, linewidth=3)
    print(np.min(rms_of_rms)*100.0, np.argmin(rms_of_rms), label, )

ax.set_ylim([0.05, 1.0])
ax.set_xlim([0, 800])
ax.legend(fontsize=8)

fig.savefig("../plots/optimizer_race.png", dpi=300, bbox_inches='tight')
