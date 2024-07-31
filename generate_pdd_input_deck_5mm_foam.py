import numpy as np
import utils_deck_generation as idg
import training_data_generation as tdg
import utils_beam_pointing as ubp
import matplotlib.pyplot as plt

dirname_pulse = "laser_pulse_data_pdd/revolver_wetted_foam/"
filename = "laser_pulse_nif_pdd_wetted_foam.csv"

num_examples = 1
dataset_params, facility_spec = tdg.define_dataset_params(num_examples)
deck_gen_params = idg.define_deck_generation_params(dataset_params, facility_spec)

facility_spec['target_radius'] = 2343.0
facility_spec["fuse"]=[False]*facility_spec["nbeams"]
facility_spec["defocus"] = np.zeros(facility_spec["nbeams"]) + 35.0
deck_gen_params['defocus'][iex,:] = 35.0 
deck_gen_params['t0'] = 10.0 
#facility_spec["default_power"] = 1.0

nbeams = facility_spec['nbeams']
beam_list = facility_spec["Beam"]
print(nbeams)

names,x,y,z = ubp.pointings_pdd(facility_spec)

pointings=dict()
iex = 0
tind = 0
deck_gen_params['p0'][iex,tind,:] = facility_spec["default_power"]

## pointing data
deck_gen_params['pointings'][iex,:,0] = x
deck_gen_params['pointings'][iex,:,1] = y
deck_gen_params['pointings'][iex,:,2] = z
deck_gen_params['port_centre_phi'] = facility_spec["Phi"]
deck_gen_params['port_centre_theta'] = facility_spec["Theta"] 

run_with_cbet = True 
base_input_txt_loc = ("ifriit_inputs_base.txt")
idg.generate_input_deck(dirname_pulse, base_input_txt_loc, run_with_cbet, facility_spec)

run_type = "nif"
idg.generate_input_pointing_and_pulses(iex, tind, facility_spec, deck_gen_params, dirname_pulse, run_type)

data = []
with open(dirname_pulse+filename) as input_file:
    for line in input_file:
        x = np.array(line.split())
        y = x.astype(float)
        data.append(y)

pulse_data = np.transpose(np.stack(data[:], axis=0))
times = pulse_data[0] * 1e9
total_power = pulse_data[1] / 1e12
print("times (ns)", times)
print("total power (TW)", total_power)
beam_powers = np.zeros((nbeams, len(total_power))) + total_power / nbeams

cone1_multiplier = 1.08
num_cones = int(facility_spec["num_cones"]/2)
cone_powers = np.zeros((num_cones)) + 1.0
cone_powers[0] = cone_powers[0] * cone1_multiplier
beams_per_cone = facility_spec["beams_per_cone"][:num_cones] * 2.0
print(cone_powers, beams_per_cone)
rebalance_term = np.sum(cone_powers * beams_per_cone) / np.sum(beams_per_cone)
cone_powers = cone_powers / rebalance_term

if 't0' in deck_gen_params.keys():
    j=0
    for ibeam in range(nbeams):
      beam = facility_spec['Beam'][ibeam]
      beam_powers[ibeam,:] = total_power / nbeams / rebalance_term
      if facility_spec['Cone'][ibeam] == 23.5:
          print("apply cone 1 multiplier, ", beam)
          beam_powers[ibeam,:] = beam_powers[ibeam,:] * cone1_multiplier
      with open(dirname_pulse+'pulse_'+beam+'.txt','w') as f:
          for i in range(len(total_power)):
              f.write(str(times[i]) + ' ' + str(beam_powers[ibeam,i]) + '\n')
      j = j + 1

total_power_check = np.sum(beam_powers, axis=0)
plt.figure()
plt.plot(times, total_power_check)
plt.plot(times, total_power)
plt.xlabel('Time (ns)')
plt.ylabel('Total Power (TW)')
plt.show()
