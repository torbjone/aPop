import numpy as np
import os
from os.path import join
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg')
    at_stallo = True
else:
    at_stallo = False

username = os.getenv('USER')

if at_stallo:
    root_folder = join('/global', 'work', username, 'aPop')
else:
    root_folder = join('/home', username, 'work', 'aPop')

elec_distances = 10 * np.exp(np.linspace(0, np.log(1000), 30))
elec_x, elec_z = np.meshgrid(elec_distances, np.array([1000, 500, 0]))
elec_x = elec_x.flatten()
elec_z = elec_z.flatten()
elec_y = np.zeros(len(elec_z))
electrode_parameters = {
        'sigma': 0.3,
        'x': elec_x,
        'y': elec_y,
        'z': elec_z
}

# Time resolution of 2**-4 is almost identical to 2**-5
# root_folder = os.path.join(os.getenv('HOME'), 'work', 'aPop')
distributed_delta_params = {'input_type': 'distributed_delta',
                            'timeres_NEURON': 2**-4,
                            'cell_name': 'hay',
                            'timeres_python': 2**-4,
                            'cut_off': 200,
                            'end_t': 10000,
                            'syn_tau': 0.1,
                            'syn_weight': 1e-4,
                            'repeats': None,
                            'max_freq': 500,
                            'holding_potential': -80,
                            'conductance_type': 'generic',
                            'save_folder': 'shape_function',
                            'root_folder': root_folder,
                            'electrode_parameters': electrode_parameters,
                            }

vsd_params = {'input_type': 'distributed_delta',
                            'timeres_NEURON': 2**-4,
                            'cell_name': 'hay',
                            'timeres_python': 2**-4,
                            'cut_off': 100,
                            'end_t': 100,
                            'syn_tau': 0.1,
                            'syn_weight': 1e-3,
                            'repeats': None,
                            'max_freq': 500,
                            'holding_potential': -80,
                            'conductance_type': 'generic',
                            'save_folder': 'shape_function',
                            'root_folder': root_folder,
                            'electrode_parameters': electrode_parameters,
                            }



distributed_delta_classic_params = {'input_type': 'distributed_delta',
                            'timeres_NEURON': 2**-4,
                            'cell_name': 'hay',
                            'timeres_python': 2**-4,
                            'cut_off': 200,
                            'end_t': 10000,
                            'syn_tau': 0.1,
                            'syn_weight': 1e-4,
                            'repeats': None,
                            'max_freq': 500,
                            'conductance_type': None,
                            'holding_potential': None,
                            'save_folder': 'shape_function',
                            'root_folder': root_folder,
                            'electrode_parameters': electrode_parameters,
                            }

scale = 2
num_cells = 100 * scale ** 2
population_radius = 100. * scale
dr = 50.
population_sizes = np.arange(dr, population_radius + dr, dr)
layer_5_thickness = 200 # From Markram (2015): Thickness L1-L5: 1382 um. Hay cell height: 1169 um. 1382 - 1169 = 213 um
small_population_params = {'input_type': 'distributed_delta',
                           'name': 'small_population',
                           'timeres_NEURON': 2**-4,
                           'timeres_python': 2**-4,
                           'population_scale': scale,   # 10 means full population
                           'num_cells': num_cells,
                           'population_radius': population_radius,
                           'population_sizes': population_sizes,
                           'layer_5_thickness': layer_5_thickness,
                           'cut_off': 200,
                           'end_t': 1000,
                           'syn_tau': 0.1,
                           'syn_weight': 1e-4,
                           'repeats': None,
                           'max_freq': 500,
                           'holding_potential': -80,
                           'conductance_type': 'generic',
                           'save_folder': 'population',
                            'electrode_parameters': electrode_parameters,
                           }


asymmetry_params = {'input_type': 'distributed_asymmetry',
                            'timeres_NEURON': 2**-4,
                            'timeres_python': 2**-4,
                            'cut_off': 200,
                            'end_t': 10000,
                            'syn_tau': 0.1,
                            'syn_weight': 1e-4,
                            'repeats': None,
                            'max_freq': 500,
                            'holding_potential': -80,
                            'conductance_type': 'generic',
                            'asymmetry_fractions': [0.80],
                            'save_folder': 'asymmetry',
                            }