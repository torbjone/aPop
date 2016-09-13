from __future__ import division
import os
from os.path import join
from suppress_print import suppress_stdout_stderr
with suppress_stdout_stderr():
     import neuron
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg', warn=False)
    at_stallo = True
else:
    at_stallo = False
import numpy as np

# username = os.getenv('USER')

root_folder = '..'

elec_distances = 10 * np.exp(np.linspace(0, np.log(1000), 30))
lateral_elec_x, lateral_elec_z = np.meshgrid(elec_distances, np.array([1000, 500, 0]))
lateral_elec_x = lateral_elec_x.flatten()
lateral_elec_z = lateral_elec_z.flatten()
lateral_elec_y = np.zeros(len(lateral_elec_z))
lateral_electrode_parameters = {
        'sigma': 0.3,
        'x': lateral_elec_x,
        'y': lateral_elec_y,
        'z': lateral_elec_z
}

center_elec_z = np.arange(-400, 1501, 100)
center_elec_x = np.zeros(len(center_elec_z))
center_elec_y = np.zeros(len(center_elec_z))
center_electrode_parameters = {
        'sigma': 0.3,
        'x': center_elec_x,
        'y': center_elec_y,
        'z': center_elec_z
}

stick_lateral_electrode_parameters = lateral_electrode_parameters.copy()
stick_lateral_electrode_parameters['method'] = 'linesource'

stick_center_electrode_parameters = center_electrode_parameters.copy()
stick_center_electrode_parameters['method'] = 'linesource'

# Time resolution of 2**-4 is almost identical to 2**-5
dt = 2**-4
end_T = 2**13 - dt
cut_off = 2000

shape_function_params = {'name': 'shape_function',
                         'input_type': 'distributed_delta',
                         'timeres_NEURON': dt,
                         'cell_name': 'hay',
                         'timeres_python': dt,
                         'cut_off': cut_off,
                         'end_t': end_T,
                         'syn_tau': dt * 3,
                         'syn_weight': 1e-3,
                         'correlation': 0.0,
                         'max_freq': 500,
                         'holding_potential': -80,
                         'conductance_type': 'generic',
                         'save_folder': 'shape_function',
                         'root_folder': root_folder,
                         'lateral_electrode_parameters': lateral_electrode_parameters,
                         'center_electrode_parameters': center_electrode_parameters,
                         'num_synapses': 1000,
                         'input_firing_rate': 5,
                         'input_regions': ['homogeneous', 'distal_tuft', 'basal'],
                         'mus': [-0.5, 0.0, 2.0],
                         'distributions': ['uniform', 'linear_increase', 'linear_decrease'],
                         }

vsd_params = {'input_type': 'distributed_delta',
                            'timeres_NEURON': 2**-4,
                            'cell_name': 'hay',
                            'timeres_python': 2**-4,
                            'cut_off': cut_off,
                            'end_t': 100,
                            'syn_tau': 0.1,
                            'syn_weight': 1e-3,
                            'max_freq': 500,
                            'holding_potential': -80,
                            'conductance_type': 'generic',
                            'save_folder': 'shape_function',
                            'root_folder': root_folder,
                            'lateral_electrode_parameters': lateral_electrode_parameters,
                            }

distributed_delta_classic_params = {'input_type': 'distributed_delta',
                            'timeres_NEURON': dt,
                            'cell_name': 'hay',
                            'timeres_python': dt,
                            'cut_off': cut_off,
                            'end_t': end_T,
                            'syn_tau': 0.1,
                            'syn_weight': 1e-3,
                            'max_freq': 500,
                            'conductance_type': None,
                            'holding_potential': None,
                            'save_folder': 'shape_function',
                            'root_folder': root_folder,
                            'lateral_electrode_parameters': lateral_electrode_parameters,
                            'center_electrode_parameters': center_electrode_parameters,
                            'num_synapses': 1000,
                            }

scale = 10
num_cells = 100 * scale ** 2
population_radius = 100. * scale
dr = 50.
population_radii = np.arange(dr, population_radius + dr, dr)
layer_5_thickness = 200  # From Markram (2015): Thickness L1-L5: 1382 um. Hay cell height: 1169 um. 1382 - 1169 = 213 um
generic_population_params = {'input_type': 'distributed_delta',
                             'name': 'generic_population',
                             'timeres_NEURON': dt,
                             'cell_name': 'hay',
                             'timeres_python': dt,
                             'population_scale': scale,  # 10 means full population
                             'num_cells': num_cells,
                             'population_radius': population_radius,
                             'population_radii': population_radii,
                             'layer_5_thickness': layer_5_thickness,
                             'cut_off': cut_off,
                             'end_t': end_T,
                             'syn_tau': dt * 3,
                             'syn_weight': 1e-3,
                             'max_freq': 500,
                             'holding_potential': -80,
                             'conductance_type': 'generic',
                             'save_folder': 'population',
                             'lateral_electrode_parameters': lateral_electrode_parameters,
                             'center_electrode_parameters': center_electrode_parameters,
                             'root_folder': root_folder,
                             'num_synapses': 1000,
                             'input_firing_rate': 5,
                             'input_regions': ['basal'],#['homogeneous', 'distal_tuft', 'basal'],
                             'mus': [0.0],#[-0.5, 0.0, 2.0],
                             'distributions': ['linear_decrease'],#['uniform', 'linear_increase', 'linear_decrease'],
                             'correlations': [0.1],# [0.0, 0.01, 0.1, 1.0]
                             }

asymmetric_population_params = {'input_type': 'distributed_delta',
                             'name': 'asymetric_generic_population',
                             'timeres_NEURON': dt,
                             'cell_name': 'hay',
                             'timeres_python': dt,
                             'population_scale': scale,  # 10 means full population
                             'num_cells': num_cells,
                             'population_radius': population_radius,
                             'population_radii': population_radii,
                             'layer_5_thickness': layer_5_thickness,
                             'cut_off': cut_off,
                             'end_t': end_T,
                             'syn_tau': dt * 3,
                             'syn_weight': 1e-3,
                             'max_freq': 500,
                             'holding_potential': -80,
                             'conductance_type': 'generic',
                             'save_folder': 'asymmetric_population',
                             'lateral_electrode_parameters': lateral_electrode_parameters,
                             'center_electrode_parameters': center_electrode_parameters,
                             'root_folder': root_folder,
                             'num_synapses': 1000,
                             'input_firing_rate': 5,
                             'input_regions': ['dual0.80', 'dual1.00'],
                             'mus': [2.0],
                             'distributions': ['linear_increase'],#['uniform', 'linear_increase', 'linear_decrease'],
                             'correlations': [0., 0.1, 1.0]
                             }

classic_population_params = {'input_type': 'distributed_delta',
                             'name': 'classic_population',
                             'cell_name': 'hay',
                             'timeres_NEURON': dt,
                             'timeres_python': dt,
                             'population_scale': scale,  # 10 means full population
                             'num_cells': num_cells,
                             'population_radius': population_radius,
                             'population_radii': population_radii,
                             'layer_5_thickness': layer_5_thickness,
                             'cut_off': cut_off,
                             'end_t': end_T,
                             'syn_tau': dt * 3,
                             'syn_weight': 1e-3,
                             'max_freq': 500,
                             'holding_potential': -70,
                             'conductance_type': 'classic',
                             'save_folder': 'population',
                             'lateral_electrode_parameters': lateral_electrode_parameters,
                             'center_electrode_parameters': center_electrode_parameters,
                             'root_folder': root_folder,
                             'num_synapses': 1000,
                             'input_firing_rate': 5,
                             'input_regions': ['homogeneous', 'distal_tuft'],
                             'mus': None,
                             'holding_potentials': [-80, -70, -60],
                             'distributions': None,
                             'conductance_types': ['active', 'passive', 'Ih', 'Ih_frozen'],
                             'correlations': [0.0, 0.01, 0.1, 1.0]
                             }

stick_population_params = {'input_type': 'distributed_delta',
                             'name': 'stick_population',
                             'timeres_NEURON': dt,
                             'cell_name': 'infinite_neurite',
                             'timeres_python': dt,
                             'population_scale': scale,  # 10 means full population
                             'num_cells': num_cells,
                             'population_radius': population_radius,
                             'population_radii': population_radii,
                             'layer_5_thickness': 0,
                             'cut_off': cut_off,
                             'end_t': end_T * 1,
                             # 'split_sim': 'top',
                             'syn_tau': dt * 3,
                             'syn_weight': 1e-3,
                             'max_freq': 500,
                             'holding_potential': -80,
                             'g_w_bar_scaling': 10.,
                             'conductance_type': 'generic',
                             'save_folder': 'stick_population',
                             'lateral_electrode_parameters': stick_lateral_electrode_parameters,
                             'center_electrode_parameters': stick_center_electrode_parameters,
                             'root_folder': root_folder,
                             'num_synapses': 1000,
                             'input_firing_rate': 5,
                             'input_regions': ['top', 'homogeneous'],
                             'mus': [-0.5, 0.0, 2.0],
                             'distributions': ['uniform', 'increase'],
                             'correlations': [0.0, 0.01, 0.1, 1.0],
                             'correlation': 0.0,
                             }

asymmetry_params = {'input_type': 'distributed_asymmetry',
                            'timeres_NEURON': dt,
                            'timeres_python': dt,
                            'cut_off': cut_off,
                            'end_t': end_T,
                            'syn_tau': 0.1,
                            'syn_weight': 1e-3,
                            'max_freq': 500,
                            'holding_potential': -80,
                            'conductance_type': 'generic',
                            'asymmetry_fractions': [0.80],
                            'save_folder': 'asymmetry',
                            }


vmem_3D_population_params = {'input_type': 'distributed_delta',
                             'name': 'vmem_3D_population',
                             'timeres_NEURON': dt,
                             'cell_name': 'hay',
                             'timeres_python': dt,
                             'population_scale': scale,  # 10 means full population
                             'num_cells': num_cells,
                             'population_radius': population_radius,
                             'population_radii': population_radii,
                             'layer_5_thickness': layer_5_thickness,
                             'cut_off': cut_off,
                             'end_t': 100,
                             'syn_tau': dt * 3,
                             'syn_weight': 1e-3,
                             'max_freq': 500,
                             'holding_potential': -80,
                             'conductance_type': 'generic',
                             'save_folder': 'vmem_3D',
                             'lateral_electrode_parameters': lateral_electrode_parameters,
                             'center_electrode_parameters': center_electrode_parameters,
                             'root_folder': root_folder,
                             'num_synapses': 1000,
                             'input_firing_rate': 5,
                             'input_regions': ['homogeneous', 'distal_tuft', 'basal'],
                             'mus': [0.0],
                             'distributions': ['uniform', 'linear_increase'],
                             'correlations': [0., 0.1, 1.0]
                             }



if __name__ == '__main__':
    from NeuralSimulation import NeuralSimulation
    generic_population_params.update({'input_region': 'distal_tuft', 'mu': 0.0, 'correlation': 0.0, 'cell_number': 0})
    ns = NeuralSimulation(**generic_population_params)
