import numpy as np
import time
import re
import string

def critical_density(wavelength_l=351.0e-9):
    epi_0 = 8.85e-12
    mass_e = 9.11e-31
    charge_e = 1.6e-19
    c_s = 3.0e8
    
    omega_l = 2.0 * np.pi * c_s / wavelength_l
    
    n_crit = epi_0 * mass_e * omega_l**2 / charge_e**2
    
    print("Critical electron density for light wavelength {:.2f}nm is {:.2e}m^-3".format(wavelength_l*1.0e9,n_crit))
    
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

    print("Hardcoded for Foam and DT! Change inputs to vary which cells")
    #print("Hardcoded for CH and DT! Change inputs to vary which cells")
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
    print(nstep, num_radial_cells)

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
    
    print(multi_data.keys())
    
    return multi_data
