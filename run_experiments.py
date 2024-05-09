import os, sys
import os.path as o
import numpy as np
base_path =  o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "source"))
sys.path.append(base_path)

base = {'seed': str(0), 'n_seed': str(1), 'freq': str(100), 'n_sample': str(50)}
noise_sigma_list = np.linspace(0,1,10)


ode_name_list = ['LogisticODE', 'LinearOSC']

for ode_name in ode_name_list:
	if ode_name == 'LogisticODE':
		equation_numbers = [0]
	elif ode_name == 'LinearOSC':
		equation_numbers = [0,1] 	
	for equation_number in equation_numbers: 			
		for noise_sigma in noise_sigma_list:
			run_str = 'python -u ' + base_path+ '/run_simulation_vi.py --ode_name=' + ode_name + ' --x_id=' + str(equation_number) + ' --seed=' + base['seed'] + ' --noise_sigma=' + str(noise_sigma) + ' --n_seed=' + base['n_seed']  + ' --freq='+base['freq'] +' --n_sample=' + base['n_sample']
			os.system(run_str)
	
