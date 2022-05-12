import os
import shutil
import numpy as np


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

    j=0
    for beam in dat['Beam']:
        if 't0' in dat.keys():
            with open('pulse_'+beam+'.txt','w') as f:
                for i in range(len(dat['times'])):
                    f.write(str(dat['times'][i]) + ' ' + str(dat['powers'][j,i]) + '\n')
            j = j + 1