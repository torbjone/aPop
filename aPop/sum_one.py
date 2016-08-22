from Population import sum_one_population


from param_dicts import generic_population_params as param_dict
param_dict['distribution'] = 'linear_decrease'
param_dict['input_region'] = 'basal'
param_dict['mu'] = 0.0
param_dict['correlation'] = 0.10
num_cells = 2000
num_tsteps = int(round(param_dict['end_t']/param_dict['timeres_python'] + 1))

sum_one_population(param_dict, num_cells, num_tsteps)