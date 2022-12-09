import numpy as np
import utils_deck_generation as idg
import healpy_pointings as hpoint
import netcdf_read_write as nrw
import utils_intensity_map as uim
import os
import subprocess
import sys
from scipy.stats import qmc


def define_system_params(root_dir):
    sys_params = {}
    sys_params["num_processes"] = 10
    sys_params["num_ex_checkpoint"] = 1000
    sys_params["run_gen_deck"] = True
    sys_params["run_sims"] = True
    sys_params["run_compression"] = True
    sys_params["run_clean"] = True

    sys_params["root_dir"] = root_dir
    sys_params["sim_dir"] = "run_"
    sys_params["trainingdata_filename"] = "training_data_and_labels.nc"
    sys_params["ifriit_ouput_name"] = "p_in_z1z2_beam_all"
    sys_params["figure_location"] = "plots"
    sys_params["plot_file_type"] = ".pdf"

    return sys_params



def define_dataset_params(num_examples,
                          random_sampling=0,
                          random_seed=12345):
    dataset_params = {}
    # Number of samples, size of NN training set
    dataset_params["num_examples"] = num_examples
    dataset_params["random_seed"] = random_seed
    dataset_params["hemisphere_symmetric"] = True

    num_sim_params = 0
    # pointings
    dataset_params["surface_cover_radians"] = np.radians(45.0)
    num_sim_params += 2
    # defocus
    dataset_params["defocus_range"] = 20.0 # mm
    num_sim_params += 1
    #power
    dataset_params["min_power"] = 0.5 # fraction of full power
    num_sim_params += 1
    dataset_params["num_sim_params"] = num_sim_params

    dataset_params["imap_nside"] = 256

    dataset_params["run_type"] = "nif" #"test" #"nif"
    facility_spec = idg.import_nif_config()

    dataset_params["LMAX"] = 30
    dataset_params["num_coeff"] = int(((dataset_params["LMAX"] + 2) * (dataset_params["LMAX"] + 1))/2.0)
    # Assume symmetry
    dataset_params["num_output"] = int(facility_spec['num_cones']/2) * dataset_params["num_sim_params"]

    random_generator=np.random.default_rng(dataset_params["random_seed"])
    if random_sampling == 1:
        print("Random Sampling!")
        sample = random_generator.random((dataset_params["num_examples"], dataset_params["num_output"]))
    else:
        sampler = qmc.LatinHypercube(d=dataset_params["num_output"],
                                     strength=1, seed=random_generator, optimization="random-cd")
        sample = sampler.random(n=dataset_params["num_examples"])
    dataset_params["Y_train"] = sample.T

    return dataset_params, facility_spec



def generate_training_data(dataset_params, sys_params, facility_spec):
    dataset_params = idg.create_run_files(dataset_params, sys_params, facility_spec)
    Y_train = dataset_params["Y_train"]

    min_parallel = 0
    max_parallel = -1
    chkp_marker = 1.0
    run_location = sys_params["root_dir"] + "/" + sys_params["sim_dir"]
    X_train = np.zeros((dataset_params["num_coeff"] * 2, dataset_params["num_examples"]))
    avg_powers = np.zeros(dataset_params["num_examples"])
    filename_trainingdata = sys_params["root_dir"] + "/" + sys_params["trainingdata_filename"]
    if sys_params["run_sims"]:
        num_parallel_runs = int(dataset_params["num_examples"] / sys_params["num_processes"])
        if num_parallel_runs > 0:
            for ir in range(num_parallel_runs):
                min_parallel = ir * sys_params["num_processes"]
                max_parallel = (ir + 1) * sys_params["num_processes"] - 1
                X_train[:,min_parallel:max_parallel+1], avg_powers[min_parallel:max_parallel+1] = run_and_delete(min_parallel, max_parallel, dataset_params, sys_params, facility_spec['target_radius'])

                if sys_params["run_compression"]:
                    if ((max_parallel + 1) >= (chkp_marker * sys_params["num_ex_checkpoint"])):
                        print("Save training data checkpoint at run: " + str(max_parallel))
                        nrw.save_training_data(X_train[:,:max_parallel+1], Y_train[:,:max_parallel+1], avg_powers[:max_parallel+1], filename_trainingdata)
                        chkp_marker +=1

        if max_parallel != (dataset_params["num_examples"] - 1):
            min_parallel = max_parallel + 1
            max_parallel = dataset_params["num_examples"] - 1
            X_train[:,min_parallel:max_parallel+1], avg_powers[min_parallel:max_parallel+1] = run_and_delete(min_parallel, max_parallel, dataset_params, sys_params, facility_spec['target_radius'])

    if sys_params["run_compression"]:
        nrw.save_training_data(X_train, Y_train, avg_powers, filename_trainingdata)



def run_and_delete(min_parallel, max_parallel, dataset_params, sys_params, target_radius_microns):
    run_location = sys_params["root_dir"] + "/" + sys_params["sim_dir"]
    range_ind = max_parallel + 1 - min_parallel
    X_train = np.zeros((dataset_params["num_coeff"] * 2, range_ind))
    avg_powers = np.zeros(range_ind)
    for iex in range(min_parallel, max_parallel+1):
        idg.copy_ifriit_exc(run_location, iex)
    subprocess.check_call(["./bash_parallel_ifriit", run_location, str(min_parallel), str(max_parallel)])
    i = 0
    for iex in range(min_parallel, max_parallel+1):
        X_train[:,i], avg_powers[i] = nrw.retrieve_xtrain_and_delete(iex, dataset_params, sys_params, target_radius_microns)
        i += 1
    return X_train, avg_powers



def run_ifriit_input(num_examples, X_all, run_dir, LMAX, num_parallel, hemisphere_symmetric, run_clean):
    dataset_params, facility_spec = define_dataset_params(num_examples)
    dataset_params["hemisphere_symmetric"] = hemisphere_symmetric
    dataset_params["Y_train"] = X_all

    sys_params = define_system_params(run_dir)
    sys_params["num_processes"] = num_parallel
    sys_params["run_clean"] = run_clean # Create new run files

    dataset_params = idg.create_run_files(dataset_params, sys_params, facility_spec)
    generate_training_data(dataset_params, sys_params, facility_spec)

    X_all, Y_all, avg_powers_all = nrw.import_training_data_reversed(sys_params, LMAX)
    return Y_all, avg_powers_all



def main(argv):
    sys_params = define_system_params(argv[1])
    dataset_params, facility_spec = define_dataset_params(int(argv[2]), random_sampling=int(argv[4]), random_seed=int(argv[5]))
    dataset_params["hemisphere_symmetric"] = bool(int(argv[3]))
    generate_training_data(dataset_params, sys_params, facility_spec)

    return dataset_params, sys_params, facility_spec


if __name__ == "__main__":
    _, _, _ = main(sys.argv)
