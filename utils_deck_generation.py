import os
import shutil
import numpy as np
import csv
import healpy_pointings as hpoint


def create_run_files(dataset_params, sys_params, run_data):

    num_output = dataset_params["num_output"]
    num_examples = dataset_params["num_examples"]

    coord_o = np.zeros(3)
    coord_o[2] = run_data['target_radius']
    run_data['pointings'] = np.zeros((run_data['nbeams'], 3))
    run_data["Port_centre_theta"] = np.zeros(run_data['nbeams'])
    run_data["Port_centre_phi"] = np.zeros(run_data['nbeams'])
    run_data["defocus"] = np.zeros(run_data['nbeams'])
    run_data["p0"] = np.zeros(run_data['nbeams'])
    run_data["fuse"] = [False] * run_data["nbeams"]
    run_data['beams_per_cone'] = [0] * run_data['num_cones']
    run_data['quad_from_each_cone'] = ('Q15T', 'Q13T', 'Q14T', 'Q11T', 'Q15B', 'Q16B', 'Q14B', 'Q13B')

    sim_params = np.zeros((num_output*2, num_examples))
    theta_pointings = np.zeros((run_data["nbeams"], num_examples))
    phi_pointings = np.zeros((run_data["nbeams"], num_examples))

    for iex in range(num_examples):
        if num_examples>1:
            ex_params =  dataset_params["Y_train"][:,iex]
        else:
            ex_params =  dataset_params["Y_train"]
        for icone in range(run_data['num_cones']):
            il = (icone*4) % num_output #4 = dataset_params["num_sim_params"]?
            iu = ((icone+1)*4-1) % num_output + 1
            cone_params = ex_params[il:iu]

            x = cone_params[0] * 2.0 - 1.0
            y = cone_params[1] * 2.0 - 1.0
            r, offset_phi = hpoint.square2disk(x, y)
            if icone > 3:
                if dataset_params["hemisphere_symmetric"]:
                    offset_phi = np.pi - offset_phi # Symmetric
                else:
                    offset_phi = (offset_phi + np.pi) % (2.0 * np.pi) # anti-symmetric
            offset_theta = r * dataset_params["surface_cover_radians"]

            cone_defocus = cone_params[2] * dataset_params["defocus_range"] # convert to mm
            cone_power = cone_params[3] * (1.0 - dataset_params["min_power"]) + dataset_params["min_power"]

            quad_name = run_data['quad_from_each_cone'][icone]
            quad_start_ind = run_data["Quad"].index(quad_name)
            quad_slice = slice(quad_start_ind, quad_start_ind+run_data['beams_per_quad'])

            cone_name = run_data['Cone'][quad_slice]
            cone_name = cone_name[0]

            run_data['beams_per_cone'][icone] = int(run_data["Cone"].count(cone_name)/2)
            sim_params[icone*4:(icone+1)*4,iex] = (offset_theta, offset_phi, cone_defocus, cone_power) #4 = dataset_params["num_sim_params"]?

            ind = run_data["Quad"].index(quad_name)
            cone_slice = slice(ind,ind+run_data['beams_per_cone'][icone],run_data['beams_per_quad'])
            quad_list_in_cone = run_data["Quad"][cone_slice]

            for quad_name in quad_list_in_cone:
                ind = run_data["Quad"].index(quad_name)
                quad_slice = slice(ind,ind+run_data['beams_per_quad'])
                beam_names = run_data['Beam'][quad_slice]

                run_data["Port_centre_theta"][quad_slice] = np.mean(run_data["Theta"][quad_slice])
                run_data["Port_centre_phi"][quad_slice] = np.mean(run_data["Phi"][quad_slice])
                port_theta = run_data["Port_centre_theta"][ind]
                port_phi = run_data["Port_centre_phi"][ind]

                rotation_matrix = np.matmul(np.matmul(hpoint.rot_mat(port_phi, "z"),
                                                      hpoint.rot_mat(port_theta, "y")),
                                  np.matmul(hpoint.rot_mat(offset_phi, "z"),
                                            hpoint.rot_mat(offset_theta, "y")))

                coord_n = np.matmul(rotation_matrix, coord_o)

                theta_pointings[quad_slice,iex] = np.arccos(coord_n[2] / run_data['target_radius'])
                phi_pointings[quad_slice,iex] = np.arctan2(coord_n[1], coord_n[0])

                run_data['pointings'][ind:ind+run_data['beams_per_quad'],:] = np.array(coord_n)
                run_data["defocus"][quad_slice] = cone_defocus
                run_data["p0"][quad_slice] = run_data['default_power'] * cone_power

        if sys_params["run_gen_deck"]:
            run_location = sys_params["root_dir"] + "/" + sys_params["sim_dir"] + str(iex)
            generate_input_deck(run_data, run_location)
            generate_input_pointing_and_pulses(run_data, run_location, dataset_params["run_type"])
    dataset_params["sim_params"] = sim_params
    dataset_params["theta_pointings"] = theta_pointings
    dataset_params["phi_pointings"] = phi_pointings
    return dataset_params



def import_nif_config():
    run_data = dict()

    run_data['nbeams'] = 192
    run_data['target_radius'] = 1100.0
    run_data['facility'] = "NIF"
    run_data['num_quads'] = 48
    run_data['num_cones'] = 8
    run_data['beams_per_quad'] = int(run_data['nbeams'] / run_data['num_quads'])
    run_data['default_power'] = 0.25 #TW per beam
    filename = "NIF_UpperBeams.txt"

    j = -1
    f=open(filename, "r")
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        if j==-1:
            key = row
            for i in range(len(row)):
                run_data[row[i]] = [None] * int(run_data['nbeams'])
        else:
            for i in range(len(row)):
                if i < 2:
                    run_data[key[i]][j] = row[i]
                elif i < 5:
                    run_data[key[i]][j] = float(row[i])
                else:
                    run_data[key[i]][j] = int(row[i])
        j=j+1
    f.close()
    filename = "NIF_LowerBeams.txt"
    f=open(filename, "r")
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        if j>96:
            for i in range(len(row)):
                if i < 2:
                    run_data[key[i]][j-1] = row[i]
                elif i < 5:
                    run_data[key[i]][j-1] = float(row[i])
                else:
                    run_data[key[i]][j-1] = int(row[i])
        j=j+1
    f.close()

    run_data["Theta"] = np.radians(run_data["Theta"])
    run_data["Phi"] = np.radians(run_data["Phi"])

    return run_data



def copy_ifriit_exc(run_location, iex):
    run_location = run_location + str(iex)
    shutil.copyfile("main", run_location+"/main")



def generate_input_deck(the_data, run_location):

    isExist = os.path.exists(run_location)

    if not isExist:
        os.makedirs(run_location)
        b='Created directory: '+run_location+"  "
        print("\r", b, end="")
    
    longStr = ("ifriit_inputs_base.txt")
    with open(longStr) as old_file:
        with open(run_location+"/ifriit_inputs.txt", "w") as new_file:
            for line in old_file:
                if "NBEAMS" in line:
                    new_file.write("    NBEAMS                      = " + str(the_data['nbeams']) + ",\n")
                elif "DIAGNOSE_INPUT_BEAMS_AND_EXIT_RADIUS_UM" in line:
                    new_file.write("    DIAGNOSE_INPUT_BEAMS_AND_EXIT_RADIUS_UM = " + str(the_data['target_radius']) + "d0,\n")
                else:
                    new_file.write(line)



def generate_input_pointing_and_pulses(dat, run_location, run_type):
    if (dat['facility'] == "NIF"):
        j = 0
        with open(run_location+'/ifriit_inputs.txt','a') as f:
            for beam in dat['Beam']:
                cone_name = dat["Cone"][dat["Beam"].index(beam)]
                if (cone_name == 23.5):
                    cpp="inner-23"
                elif (cone_name == 30):
                    cpp="inner-30"
                elif (cone_name == 44.5):
                    cpp="outer-44"                                       
                else:
                    cpp="outer-50"

                f.write('&BEAM\n')
                # if (cpp=="inner-23" or cpp=="inner-30"):
                #     f.write('    LAMBDA_NM           = '+str((1052.85+0.45)/3.)+',\n')   
                # else:
                f.write('    LAMBDA_NM           = {:.10f}d0,\n'.format(1052.85/3.))
                f.write('    FOC_UM              = {:.10f}d0,{:.10f}d0,{:.10f}d0,\n'.format(dat['pointings'][j][0],dat['pointings'][j][1],dat['pointings'][j][2]))
                if 't0' in dat.keys():
                    f.write('    POWER_PROFILE_FILE_TW_NS = "pulse_'+beam+'.txt"\n')
                    f.write('    T_0_NS              = {:.10f}d0,\n'.format(dat['t0']))
                else:
                    f.write('    P0_TW               = {:.10f}d0,\n'.format(dat['p0'][j]))
                if (run_type == "nif"):
                    f.write('    PREDEF_FACILITY     = "NIF"\n')
                    f.write('    PREDEF_BEAM         = "'+beam+'",\n')
                    f.write('    PREDEF_CPP          = "NIF-'+cpp+'",\n')
                    f.write('    CPP_ROTATION_MODE   = 1,\n')
                    #f.write('    CPP_ROTATION_DEG    = 45.0d0,\n')
                    f.write('    DEFOCUS_MM          = {:.10f}d0,\n'.format(dat['defocus'][j]))
                elif (run_type == "test"):
                    f.write('    THETA_DEG            = {:.10f}d0,\n'.format(np.degrees(dat['Port_centre_theta'][j])))
                    f.write('    PHI_DEG              = {:.10f}d0,\n'.format(np.degrees(dat['Port_centre_phi'][j])))
                    f.write('    FOCAL_M             = 10.0d0,\n')
                    f.write('    SG                  = 6,\n')
                    f.write('    LAW                  = 2,\n')
                    f.write('    RAD_1_UM            = 80.0d0,\n')
                    f.write('    RAD_2_UM            = 80.0d0,\n')
                if 'fuse' in dat.keys() and not dat['fuse'][j]:
                    f.write('    FUSE_QUADS          = .FALSE.,\n')
                else:
                    f.write('    FUSE_QUADS          = .TRUE.,\n')
                    f.write('    FUSE_BY_POINTINGS   = .TRUE.,\n')
                if 'xy-mispoint' in dat.keys():
                    f.write('    XY_MISPOINT_UM      = {:.10f}d0,{:.10f}d0,\n'.format(dat['xy-mispoint'][j][0],dat['xy-mispoint'][j][1]))
                f.write('/\n')
                f.write('\n')
                j = j + 1
            f.write('\n')
            f.write('! Last line must not be empty')
    else:
        print('Unknown facility',dat['facility'])
