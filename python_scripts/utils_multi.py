import numpy as np
import time
import re
import string
tiny = 1.0e-200

def critical_density(wavelength_l=351.0e-9):
    epi_0 = 8.85e-12
    mass_e = 9.11e-31
    charge_e = 1.6e-19
    c_s = 3.0e8

    omega_l = 2.0 * np.pi * c_s / wavelength_l

    n_crit = epi_0 * mass_e * omega_l**2 / charge_e**2

    return n_crit


def read_inputs(path, multi_data):

    with open(path) as input_file:
        for line in input_file:
            if re.search("nfuel", line, re.IGNORECASE):
                multi_input_line = line.split()[2]
                extract_number = multi_input_line.translate(str.maketrans('', '', string.punctuation))
                multi_data["fuel_boundary"] = int(extract_number)
            if re.search("wl", line, re.IGNORECASE):
                multi_input_line = line.split()[2]
                multi_input_line = re.sub(',', '', multi_input_line)
                multi_data["wavelength"] = float(multi_input_line) / 1.0e2

    return multi_data


def multi2ifriit_inputs(multi_data, itime, ind_interfaces):
    proton_mass = 1.6726e-27
    qe = 1.60e-19
    kb = 1.38e-23

    proton_mass_cgs = proton_mass * 1.0e3

    density = multi_data["rho"]
    charge_state = multi_data["charge_state"]
    atomic_mass = multi_data["atomic_mass"]

    ncells = len(multi_data["x"][0,:-1])
    ne_cgs = (density * charge_state) / (atomic_mass * proton_mass_cgs)
    ne = ne_cgs * 1.0e6
    cell_centred_radius = (multi_data["x"][:,:-1] + multi_data["x"][:,1:]) / 2.0 * 10000 # cm to microns
    radial_velocity_cc = (multi_data["v"][:,:-1] + multi_data["v"][:,1:]) / 2.0 / 100 # cm/s tp m/s

    te = multi_data['te'] * qe / kb # eV to K
    ti = multi_data['ti'] * qe / kb # eV to K

    ifriit_input_format = {}
    ifriit_input_format["xs"] = cell_centred_radius[itime] # microns
    ifriit_input_format["zs"] = np.array([0.0])
    ifriit_input_format["ne"] = np.reshape(ne[itime],(1,ncells))
    ifriit_input_format["te"] = np.reshape(te[itime],(1,ncells))
    ifriit_input_format["ti"] = np.reshape(ti[itime],(1,ncells))
    ifriit_input_format["vr"] = np.reshape(radial_velocity_cc[itime],(1,ncells))
    ifriit_input_format["vz"] = np.zeros((1,ncells))

    #print("Hardcoded for Foam and DT! Change inputs to vary which cells")
    print("Hardcoded for CH and DT! Change inputs to vary which cells")
    nmat = 4
    #                                     [ H,   D,   T,   C    ]
    ifriit_input_format["atomic_index"] = np.array([1.0, 2.0, 3.0, 12.011])
    ifriit_input_format["znuc"] =         np.array([1.0, 1.0, 1.0, 6.0])
    ifriit_input_format["frac"] = np.zeros((nmat,1,ncells))
    """
    # pure D2 layer
    ifriit_input_format["frac"][1,:,:ind_interfaces[0]] = 1.0
    # D2 wetted foam layer
    ifriit_input_format["frac"][1,:,ind_interfaces[0]:ind_interfaces[1]] = 0.90
    ifriit_input_format["frac"][0,:,ind_interfaces[0]:ind_interfaces[1]] = 0.05
    ifriit_input_format["frac"][3,:,ind_interfaces[0]:ind_interfaces[1]] = 0.05
    # Polystyrene
    ifriit_input_format["frac"][0,:,ind_interfaces[1]:] = 0.5
    ifriit_input_format["frac"][3,:,ind_interfaces[1]:] = 0.5
    """
    """
    # pure DT layer
    ifriit_input_format["frac"][1,:,:ind_interfaces[0]] = 0.5
    ifriit_input_format["frac"][2,:,:ind_interfaces[0]] = 0.5
    # DT wetted foam layer
    ifriit_input_format["frac"][1,:,:ind_interfaces[0]] = 0.45
    ifriit_input_format["frac"][2,:,:ind_interfaces[0]] = 0.45
    ifriit_input_format["frac"][0,:,:ind_interfaces[0]] = 0.05
    ifriit_input_format["frac"][3,:,:ind_interfaces[0]] = 0.05
    """
    # Plastic CH layer
    ifriit_input_format["frac"][0,:,ind_interfaces[0]:] = 0.5
    ifriit_input_format["frac"][3,:,ind_interfaces[0]:] = 0.5

    return ifriit_input_format, ncells, nmat


def multi_read_bin(filename, data_set):
    line_count = 0
    num_per_label = 0
    param_counter = 0
    ind_geo = 0
    old_label = ""
    max_num_times = 1000

    list_labels = []
    list_num_per_label = []
    ind_geo_count = {}
    print(filename)
    with open(filename) as file:
        for line in file:
            if line_count == 0:
                data_size = int(line)
            elif line_count<=data_size+1:
                label = str(line)
                if (label == old_label) or (old_label == ""):
                    num_per_label +=1
                else:
                    param_counter += 1
                    list_labels.append(old_label[:-4].strip())
                    list_num_per_label.append(num_per_label)
                    num_per_label = 1
                old_label = label
            if line_count==data_size+1:
                for ind in range(param_counter):
                    data_set[list_labels[ind]] = np.zeros((max_num_times, list_num_per_label[ind]))
                ind = 0
                num_per_label = 0
                previous_label_end = data_size
                ind_geo = 0
            if (line_count>data_size) and ((line_count<data_size*3+1) or (list_labels[0]=="TIME")):
                count_remainder = line_count - previous_label_end
                if count_remainder < list_num_per_label[ind]+1:
                    data_set[list_labels[ind]][ind_geo, num_per_label] = float(line)
                    num_per_label +=1
                else:
                    ind +=1
                    previous_label_end = line_count-1
                    num_per_label = 0
                    if ind == param_counter:
                      ind = 0
                      ind_geo +=1
                      data_set[list_labels[ind]][ind_geo, num_per_label] = float(line)
                    else:
                      data_set[list_labels[ind]][ind_geo, num_per_label] = float(line)
                      num_per_label +=1
                ind_geo_count[list_labels[ind]] = ind_geo
            if (line_count==data_size*3+1) and (list_labels[0]!="TIME"):
                list_labels = []
                list_num_per_label = []
                line_count=0
                num_per_label = 0
                param_counter = 0
                data_size = int(line)
                old_label = ""
            line_count += 1
    list_labels = list(data_set.keys())
    for ind in range(len(list_labels)):
        label = list_labels[ind]
        nt_size = ind_geo_count[list_labels[ind]]
        data_set[label] = data_set[label][:nt_size+1, :]
    return data_set


def multi_read_ascii(filename):

    start = time.time()

    file_data = []
    num_params = 0
    header_counter = 0
    data_counter = 0
    step_counter = 0
    ntsep = 0
    is_header = False
    is_data = False
    with open(filename) as file:
        for line in file:
            if "step" in line:
                header_counter = 0
                data_counter = 0
                is_header = True
                header_info = line.split()
                header_labels = [header_info[0][:-1], header_info[2][:-1]]
            if is_header:
                if header_counter == 0:
                    data_labels = line.split()
                if header_counter == 2:
                    data_labels = line.split()
                    num_params = len(data_labels)
                if header_counter == 4:
                    is_data = True
                    is_header = False
                header_counter += 1
            if is_data:
                x = np.array(line.split())
                y = x.astype(float)
                if len(y) < num_params:
                    is_data = False
                    num_radial_cells = data_counter
                    step_counter += 1
                data_counter +=1

    nstep = step_counter
    step_counter = 0
    data_counter = 0
    multi_data = {}
    for label in header_labels:
        multi_data[label] = np.zeros((nstep))
    for label in data_labels:
        if (label == "i") or (label == "x") or (label == "v"):
            multi_data[label] = np.zeros((nstep, num_radial_cells+1))
        else:
            multi_data[label] = np.zeros((nstep, num_radial_cells))
    #print(nstep, num_radial_cells)

    with open(filename) as file:
        for line in file:
            if "step" in line:
                header_counter = 0
                data_counter = 0
                is_header = True
                header_info = line.split()
                #print(header_info)
                multi_data[header_labels[0]][step_counter] = int(header_info[1])
                multi_data[header_labels[1]][step_counter] = float(header_info[3])
            if is_header:
                if header_counter == 4:
                    is_data = True
                    is_header = False
                header_counter += 1
            if is_data:
                x = np.array(line.split())
                y = x.astype(float)
                for ind in range(len(y)):
                    multi_data[data_labels[ind]][step_counter,data_counter] = y[ind]
                if len(y) < num_params:
                    is_data = False
                    step_counter += 1
                data_counter +=1

    end = time.time()
    print("Elapsed time: ",end - start)

    #print(multi_data.keys())

    return multi_data

def multi_printout(multi_data):
    print("Laser energy emitted = {:4.2f} kJ".format(multi_data["energy_laser_emitted"][-1] / 1000))
    print("Fusion energy gain (per emitted) = {:4.2f}".format(multi_data["energy_fusion_emitted"][-1] / multi_data["energy_laser_emitted"][-1]))
    print("Laser energy deposited = {:4.2f} kJ".format(multi_data["energy_laser_deposited"][-1] / 1000))
    print("Fusion energy gain (per deposited) = {:4.2f}".format(multi_data["energy_fusion_emitted"][-1] / multi_data["energy_laser_deposited"][-1]))
    print("Laser energy percentage absorbed = {:4.2f} %".format(multi_data["energy_laser_deposited"][-1] / multi_data["energy_laser_emitted"][-1] * 100))
    print("Fusion energy emitted = {:4.2f} kJ".format(multi_data["energy_fusion_emitted"][-1] / 1000))
    print("Initial mass = {:4.2f} mg".format(multi_data["CMI"][1,-1] * 1000)) # g to mg
    print("Initial mass DT = {:4.2f} mg (assuming pure DT fuel)".format(multi_data["CMI"][1,multi_data["fuel_boundary"]] * 1000)) # g to mg
    fusion_energy_per_mg_DT = 337.0e6 # joules/mg
    if multi_data["fuel_boundary"] != 0:
        multi_data["burn_fraction"] = multi_data["energy_fusion_emitted"][-1] / (multi_data["CMI"][1,multi_data["fuel_boundary"]] * 1000 * fusion_energy_per_mg_DT) * 100
    else:
        multi_data["burn_fraction"] = 0.0
    print("Percentage of DT burnt = {:4.2f} % (assuming pure DT fuel) \n".format(multi_data["burn_fraction"])) # g to mg

    print("Time critical surface 2/3 of initial gas region = {:4.2f} ns".format(multi_data["time"][multi_data["tind_r_two_thirds"]] * 1.0e9))
    print("IFAR at time critical surface 2/3 = {:4.2f} \n".format(multi_data["IFAR2"][multi_data["tind_r_two_thirds"]]))

    print("Time of max laser power = {:4.2f} ns".format(multi_data["time"][multi_data["ind_max_laser_power"]] * 1.0e9))
    print("Max laser power = {:4.2f} TW \n".format(multi_data["delta_laser"][multi_data["ind_max_laser_power"]] / 1.0e12))

    print("Time of max implosion velocity = {:4.2f} ns".format(multi_data["time"][multi_data["ind_max_vel"]] * 1.0e9))
    print("Peak implosion velocity = {:4.2f} km/s".format(np.abs(multi_data["implosion_velocity"][multi_data["ind_max_vel"]])))
    print("Min adiabat at max implosion velocity = {:4.2f}".format(multi_data["min_adiabat_imploding_DT"][multi_data["ind_max_vel"]]))
    print("Mean adiabat at max implosion velocity = {:4.2f}".format(multi_data["mass_averaged_adiabat_imploding_DT"][multi_data["ind_max_vel"]]))
    #print("IFAR at peak velocity = {:4.2f} \n".format(multi_data["IFAR2"][multi_data["ind_max_vel"]]))
    cell_mass_imploding = np.ma.masked_array(multi_data["cell_mass"], mask=multi_data["mask_imploding_cells"][multi_data["ind_max_vel"],:])
    print("Mass remaining at max implosion velocity = {:4.2f} mg".format(np.sum(cell_mass_imploding) * 1000)) # g to mg
    cell_mass_imploding_DT = np.ma.masked_array(multi_data["cell_mass"][:multi_data["fuel_boundary"]], mask=multi_data["mask_imploding_cells"][multi_data["ind_max_vel"],:multi_data["fuel_boundary"]])
    print("DT Mass remaining at max implosion velocity = {:4.2f} mg \n".format(np.sum(cell_mass_imploding_DT) * 1000)) # g to mg

    ind_min_hs_radius = np.argmin(multi_data["radius_outer_surf_hotspot"][multi_data["ind_max_vel"]:multi_data["tind_max_neut"]]) + multi_data["ind_max_vel"]
    print("Time of maximum convergence = {:4.2f} ns".format(multi_data["time"][ind_min_hs_radius] * 1.0e9))
    print("Areal density = {:4.2f} g/cm^-2".format(multi_data["RHORD"][ind_min_hs_radius, 0]))
    print("Convergence ratio = {:4.2f} \n".format(multi_data["convergence_ratio"][ind_min_hs_radius]))

    print("Time of minimum kinetic energy = {:4.2f} ns".format(multi_data["time"][multi_data["ind_min_kinetic_energy"]] * 1.0e9))
    print("Areal density = {:4.2f} g/cm^-2".format(multi_data["RHORD"][multi_data["ind_min_kinetic_energy"], 0]))
    print("Convergence ratio = {:4.2f} \n".format(multi_data["convergence_ratio"][multi_data["ind_min_kinetic_energy"]]))

    print("Time of maximum rhoR = {:4.2f} ns".format(multi_data["time"][multi_data["tind_max_rhor_DT"]] * 1.0e9))
    print("Max areal density = {:4.2f} g/cm^-2".format(multi_data["RHORD"][multi_data["tind_max_rhor_DT"], 0]))
    print("Convergence ratio = {:4.2f} \n".format(multi_data["convergence_ratio"][multi_data["tind_max_rhor_DT"]]))

    print("Time of maximum neutron flux = {:4.2f} ns".format(multi_data["time"][multi_data["tind_max_neut"]] * 1.0e9))
    print("Total neutrons released = {:4.2e}".format(multi_data["NEUTR"][-1, 0]))
    print("Areal density = {:4.2f} g/cm^-2".format(multi_data["RHORD"][multi_data["tind_max_neut"], 0]))
    print("Convergence ratio = {:4.2f} \n".format(multi_data["convergence_ratio"][multi_data["tind_max_neut"]]))

    print("Lawson criteria = {:4.2f} ".format(multi_data["lawson_criteria_chang2010"]))
    print("from Chang, P. et al. Phys. Rev. Lett. 104, 135002(2010).")


def multi_data_units(multi_data):
    multi_data["time"] = multi_data["TIME"][:,0]
    multi_data["ntimes"] = np.shape(multi_data["TIME"][:,0])[0]
    multi_data["delta_time"] = (multi_data["TIME"][1:,0] - multi_data["TIME"][:-1,0])

    multi_data["energy_laser_emitted"] = multi_data["LASER"][:,0] * 1.0e-7 #erg to joules
    multi_data["energy_fusion_emitted"] = multi_data["FUSIO"][:,0] * 1.0e-7 #erg to joules
    multi_data["energy_laser_deposited"] = multi_data["DELAS"][:,0] * 1.0e-7 #erg to joules

    multi_data["delta_fusion"] = (multi_data["energy_fusion_emitted"][1:]
                                  - multi_data["energy_fusion_emitted"][:-1]) / multi_data["delta_time"] #J to Watts
    multi_data["delta_laser"] = (multi_data["energy_laser_emitted"][1:]
                                 - multi_data["energy_laser_emitted"][:-1]) / multi_data["delta_time"] #J to Watts
    multi_data["delta_laser_dep"] = (multi_data["energy_laser_deposited"][1:]
                                     - multi_data["energy_laser_deposited"][:-1]) / multi_data["delta_time"] #J to Watts
    multi_data["delta_neutron_flux"] = (multi_data["NEUTR"][1:] - multi_data["NEUTR"][:-1])

    multi_data["implosion_velocity2"] = multi_data["VIMPL"][:,0] * 1.0e-5 # cm/s to um/ns
    multi_data["electron_number_density"] = multi_data["DENE"] * 1.0e6
    multi_data["cell_velocity"] = (multi_data["V"][:,:-1] + multi_data["V"][:,1:]) / 2.0 * 1.0e-5 # edge to cell centred and cm/s to um/ns

    multi_data["ind_max_laser_power"] = np.argmax(multi_data["delta_laser"])
    multi_data["tind_max_rhor_DT"] = np.argmax(multi_data["RHORD"])
    multi_data["tind_max_neut"] = np.argmax(multi_data["delta_neutron_flux"])

    multi_data["cell_mass"] = multi_data["CMI"][1,1:] - multi_data["CMI"][1,:-1]
    multi_data["power_laser_deposited_spatial"] = (multi_data["D"] * multi_data["cell_mass"] * 1.0e-7) #erg/s/g to Watts
    return multi_data


def multi_find_interfaces(multi_data):
    electron_number_density = multi_data["electron_number_density"]
    threshold = np.min(electron_number_density[0,:]) * 0.01
    ind_change_ne = np.where(abs(np.diff(electron_number_density[1,:]))>threshold)[0] + 1

    multi_data["ind_material_interfaces"] = np.where(np.diff(electron_number_density[1])>0.01*np.min(electron_number_density[1]))[0]
    multi_data["radius_material_interfaces"] = multi_data["X"][1,multi_data["ind_material_interfaces"]]
    return multi_data


def multi_mean_laser_dep_radius(multi_data):
    n_351 = critical_density()
    mean_deposition_radius = np.zeros(multi_data["ntimes"])
    mean_deposition_density = np.zeros(multi_data["ntimes"])
    mean_deposition_charge_state = np.zeros(multi_data["ntimes"])

    for tind in range(multi_data["ntimes"]):
        total_energy_deposited = np.sum(multi_data["power_laser_deposited_spatial"][tind,:])
        mean_deposition_radius[tind] = np.sum(multi_data["power_laser_deposited_spatial"][tind,:] * multi_data["XC"][tind,:]) / np.max([total_energy_deposited, tiny])
        mean_deposition_density[tind] = np.sum(multi_data["power_laser_deposited_spatial"][tind,:] * multi_data["electron_number_density"][tind,:] / multi_data["ZI"][tind,:]) / np.max([total_energy_deposited, tiny])
        mean_deposition_charge_state[tind] = np.sum(multi_data["power_laser_deposited_spatial"][tind,:] * multi_data["ZI"][tind,:]) / np.max([total_energy_deposited, tiny])

    intensity_14 = np.sum(multi_data["power_laser_deposited_spatial"][1:] / (4 * np.pi * multi_data["XC"][1:,]**2), axis=1) / 1.0e14

    multi_data["mean_deposition_radius"] = mean_deposition_radius
    multi_data["mean_deposition_density"] = mean_deposition_density[1:]
    multi_data["mean_deposition_charge_state"] = mean_deposition_charge_state

    surface_area = 4 * np.pi * np.max([multi_data["mean_deposition_radius"][1:], multi_data["radius_n_crit"]][1:], axis=0)**2
    pressure_estimate = 18.0 * (multi_data["mean_deposition_density"] / n_351)**(1./9.) * (intensity_14)**(7./9.)

    multi_data["surface_area"] = surface_area
    multi_data["intensity"] = intensity_14 * 1.0e14
    multi_data["pressure_estimate"] = pressure_estimate # Mbars

    return multi_data


def calc_rhor(multi_data):
    rhodr = multi_data["R"][1:,:] * (multi_data["X"][1:,1:] - multi_data["X"][1:,:-1])
    rhor = np.sum(rhodr, axis=1)
    multi_data["rhor_all"] = rhor
    multi_data["tind_max_rhor_all"] = np.argmax(rhor)
    return multi_data


def multi_critical_surface(multi_data, n_crit):
    ind_n_crit = np.zeros(multi_data["ntimes"], dtype='int64')
    itime_cut = multi_data["tind_max_rhor_DT"] #multi_data["ind_max_laser_power"]
    radius_n_crit = np.zeros(multi_data["ntimes"])
    for tind in range(multi_data["ntimes"]):
        ind_max_ne = np.argmax(multi_data["electron_number_density"][tind,:])
        ind_n_crit[tind] = np.argmin(np.abs(multi_data["electron_number_density"][tind,ind_max_ne:] - n_crit)) + ind_max_ne
        radius_n_crit[tind] = multi_data["XC"][tind,ind_n_crit[tind]]

    multi_data["radius_n_crit"] = radius_n_crit
    multi_data["ind_n_crit"] = ind_n_crit
    return multi_data


def lawson_criteria(multi_data):

    areal_density = multi_data["RHORD"][multi_data["tind_max_rhor_DT"]]
    yield_norm16 = multi_data["NEUTR"][-1, 0] / 1.0e16
    if multi_data["fuel_boundary"] == 0:
        multi_data["lawson_criteria_chang2010"] = 0.0
    else:
        mass_dt_stagnation = np.sum(multi_data["cell_mass"][:multi_data["fuel_boundary"]]) * 1000 # g to mg
        lawson_criteria_chang2010 = (areal_density)**(0.61) * (0.12 * yield_norm16 / mass_dt_stagnation)**(0.34)
        multi_data["lawson_criteria_chang2010"] = lawson_criteria_chang2010[0]
    return multi_data


def define_implosion_velocity(multi_data):
    multi_data["implosion_velocity"] = np.zeros(multi_data["ntimes"])
    multi_data["kinetic_energy_shell"] = np.zeros(multi_data["ntimes"])
    #multi_data["ind_imploding_cells"] = None*multi_data["ntimes"]

    multi_data["mask_imploding_cells"] = multi_data["cell_velocity"] > 0.
    cell_velocity_imploding = np.ma.masked_array(multi_data["cell_velocity"], mask=multi_data["mask_imploding_cells"])
    #print("imploding cell velocities", np.shape(cell_velocity_imploding))
    #print("mask", np.shape(mask_array))
    for tind in range(multi_data["ntimes"]-1):
        if multi_data["ind_ablation_front"][tind]==0:
            multi_data["kinetic_energy_shell"][tind] = 0.0
            multi_data["implosion_velocity"][tind] = 0.0
        else:
            cell_mass_imploding = np.ma.masked_array(multi_data["cell_mass"], mask=multi_data["mask_imploding_cells"][tind,:])
            multi_data["kinetic_energy_shell"][tind] = np.sum(0.5 * cell_mass_imploding * cell_velocity_imploding[tind]**2)
            multi_data["implosion_velocity"][tind] = np.sqrt(2.0 * multi_data["kinetic_energy_shell"][tind] / np.sum(cell_mass_imploding))

    multi_data["ind_max_vel"] = np.argmax(multi_data["implosion_velocity"][:multi_data["tind_max_neut"]]) #[:multi_data["tind_max_neut"]]
    multi_data["ind_min_kinetic_energy"] = np.argmin(multi_data["kinetic_energy_shell"][multi_data["ind_max_vel"]:]) + multi_data["ind_max_vel"]
    return multi_data


def find_sign_changes(test_list):
    "06/02/2025 taken from https://www.geeksforgeeks.org/python-program-to-get-indices-of-sign-change-in-a-list/ "
    res = []
    for idx in range(0, len(test_list) - 1):
        # checking for successive opposite index
        if test_list[idx] > 0 and test_list[idx + 1] < 0 or test_list[idx] < 0 and test_list[idx + 1] > 0:
            res.append(idx)
    return res


def define_capsule(multi_data):
    ind_inner_surf = np.zeros(multi_data["ntimes"], dtype='int64')
    ind_outer_surf = np.zeros(multi_data["ntimes"], dtype='int64')
    ind_outer_surf_DT = np.zeros(multi_data["ntimes"], dtype='int64')
    ind_ablation_front = np.zeros(multi_data["ntimes"], dtype='int64')
    radius_outer_surf = np.zeros(multi_data["ntimes"])
    radius_outer_surf_hotspot = np.zeros(multi_data["ntimes"])
    ind_outer_surf_hotspot = np.zeros(multi_data["ntimes"], dtype='int64')
    convergence_ratio = np.zeros(multi_data["ntimes"])
    convergence_ratio2 = np.zeros(multi_data["ntimes"])
    IFAR = np.zeros(multi_data["ntimes"])
    imax_density = np.zeros(multi_data["ntimes"], dtype='int64')

    for tind in range(multi_data["ntimes"]-1):
        density = multi_data["R"][tind,:]
        imax_density[tind] = np.argmax(density)
        max_density = np.max(density)
        temp_cutoff = multi_data["TI"][tind,imax_density[tind]]

        ind_outer_surf[tind] = np.argmin(np.abs(density[imax_density[tind]:] - max_density / 10)) + imax_density[tind]
        ind_outer_surf_DT[tind] = np.min([ind_outer_surf[tind], multi_data["fuel_boundary"]])
        radius_outer_surf[tind] = multi_data["XC"][tind,ind_outer_surf[tind]]

        if imax_density[tind] == 0:
            ind_inner_surf[tind] = 0
            ind_outer_surf_hotspot[tind] = 0
        else:
            ind_inner_surf[tind] = np.argmin(np.abs(density[:imax_density[tind]] - max_density / 10))
            ind_outer_surf_hotspot[tind] = np.argmin(np.abs(multi_data["TI"][tind,:imax_density[tind]] - multi_data["TI"][tind,0] / 6.0))
        ind_inner_surf[tind] = np.max([ind_inner_surf[tind], multi_data["ind_material_interfaces"][0]])
        radius_outer_surf_hotspot[tind] = multi_data["XC"][tind,ind_outer_surf_hotspot[tind]]

        IFAR[tind] = multi_data["XC"][tind,ind_outer_surf[tind]] / (multi_data["XC"][tind,ind_outer_surf[tind]] - multi_data["XC"][tind,ind_inner_surf[tind]])
        convergence_ratio[tind] = multi_data["XC"][1,ind_inner_surf[1]] / multi_data["XC"][tind,ind_outer_surf_hotspot[tind]]
        convergence_ratio2[tind] = multi_data["XC"][1,ind_inner_surf[1]] / multi_data["XC"][tind,ind_inner_surf[tind]]
        ind_n_crit = multi_data["ind_n_crit"][tind]
        """
        if (imax_density[tind] == 0) or (imax_density[tind] >= ind_n_crit):
            ind_ablation_front[tind] = len(density)-1
        else:
            ind_ablation_front[tind] = np.argmin(np.abs(multi_data["TE"][tind,imax_density[tind]:ind_n_crit] - multi_data["TE"][tind,ind_n_crit] / 6.0)) + imax_density[tind]
        """
        temp = find_sign_changes(multi_data["cell_velocity"][tind,imax_density[tind]:]) + imax_density[tind]
        if np.shape(temp) == (0,):
            ind_ablation_front[tind] = len(density)-1
        else:
            ind_ablation_front[tind] = temp[0]
        if np.sum(multi_data["cell_velocity"][tind,:ind_ablation_front[tind]])>0.0:
            ind_ablation_front[tind] = 0

    itime_cut = multi_data["tind_max_rhor_DT"]
    if itime_cut>0:
        multi_data["tind_r_two_thirds"] = np.argmin(np.abs(radius_outer_surf[:itime_cut] - multi_data["XC"][1,ind_inner_surf[1]]*2./3.))
    else:
        multi_data["tind_r_two_thirds"] = 0

    multi_data["ind_inner_surf"] = ind_inner_surf
    multi_data["ind_outer_surf"] = ind_outer_surf
    multi_data["ind_outer_surf_DT"] = ind_outer_surf_DT
    multi_data["radius_outer_surf_hotspot"] = radius_outer_surf_hotspot
    multi_data["ind_outer_surf_hotspot"] = ind_outer_surf_hotspot
    multi_data["ind_ablation_front"] = ind_ablation_front
    multi_data["convergence_ratio"] = convergence_ratio
    multi_data["convergence_ratio2"] = convergence_ratio2
    multi_data["IFAR2"] = IFAR
    multi_data["imax_density"] = imax_density
    return multi_data


def adiabat(multi_data):

    hbar = 1.0546e-34
    me = 9.1094e-31
    nucleon_mass = 1.6605e-27
    nucleon_number_DT = 2.505 # +0.005 is bodge to make fermi_density2pressure = 2.17
    fermi_density2pressure = (3.0 * np.pi **2)**(2.0/3.0) * hbar**2 / (5.0 * me) * (1. / (nucleon_number_DT * nucleon_mass) * 1000.)**(5./3.) / 1.0e11 # [all in SI, 1e3 converts density to SI and then 1e11 goes from pascal -> Mbar]
    print(fermi_density2pressure)
    total_pressure = multi_data["PT"] / (1e6*1e6) #from dyn/cm2 to Mbars
    density = multi_data["R"]
    fermi_pressure = fermi_density2pressure * density ** (5./3)
    multi_data["adiabat_ionised_DT"] = total_pressure / fermi_pressure

    """
    hbar = 1.05e-34
    me = 9.11e-31
    ne = multi_data["electron_number_density"]
    fermi_pressure2 = hbar**2 / (5.0 * me) * (3.0 * np.pi **2 * ne)**(2.0/3.0) * ne
    pressure_si = multi_data["PT"] / 10.0
    multi_data["adiabat_eoz"] = pressure_si / fermi_pressure2 #(multi_data["PI"] + multi_data["PE"]) / fermi_pressure #multi_data["PT"] / fermi_pressure#
    multi_data["adiabat_eoz"][0,:] = 0.0
    """

    shell_mass_averaged_adiabat_ionised_DT = np.zeros(multi_data["ntimes"])

    cell_mass_imploding_DT = np.ma.masked_array(multi_data["cell_mass"][:multi_data["fuel_boundary"]], mask=multi_data["mask_imploding_cells"][multi_data["ind_max_vel"],:multi_data["fuel_boundary"]])
    for ind in range(multi_data["ntimes"]-1):
        tind = ind + 1
        iis = multi_data["ind_inner_surf"][tind]
        ios = multi_data["ind_outer_surf_DT"][tind]

        cell_mass_imploding_DT = multi_data["cell_mass"][iis:ios]
        adiabat_imploding_DT = multi_data["adiabat_ionised_DT"][tind,iis:ios]

        shell_mass_averaged_adiabat_ionised_DT[tind] = np.sum(adiabat_imploding_DT * cell_mass_imploding_DT) / np.max([np.sum(cell_mass_imploding_DT), tiny])

    multi_data["mass_averaged_adiabat_imploding_DT"] = shell_mass_averaged_adiabat_ionised_DT
    multi_data["min_adiabat_imploding_DT"] = np.min(multi_data["adiabat_ionised_DT"], axis=1)

    return multi_data

