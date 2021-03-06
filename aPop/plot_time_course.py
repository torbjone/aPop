import sys
import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from Population import initialize_population
from NeuralSimulation import NeuralSimulation
from tools import return_freq_and_psd_welch
from plotting_convention import mark_subplots, simplify_axes

root_folder = '..'

center_elec_z = np.arange(-200, 1201, 200)
center_elec_x = np.zeros(len(center_elec_z))
center_elec_y = np.zeros(len(center_elec_z))
center_elec_params = {
        'sigma': 0.3,
        'x': center_elec_x,
        'y': center_elec_y,
        'z': center_elec_z,
        'method': "linesource",
}

dt = 2**-3
param_dict = {'input_type': "distributed_delta",
              'name': 'stick_pop_example',
              'cell_number': 0,
              'cell_name': 'stick_cell',
              'dt': dt,
              'num_cells': 100,
              'population_radius': 20,
              'layer_5_thickness': 0,
              'cut_off': 0,
              'end_t': 200,
              'syn_tau': dt * 3,
              'syn_weight': 1e-3,
              'max_freq': 500,
              'holding_potential': -80,
              'g_w_bar_scaling': 20.,
              'conductance_type': 'generic',
              'save_folder': 'stick_time_course',
              'center_electrode_parameters': center_elec_params,
              'root_folder': root_folder,
              'num_synapses': 200,
              'input_firing_rate': 5,
              'input_regions': ['homogeneous', ],
              'mus': [None, 2.0],
              'distributions': ['increase'],
              'correlations': [0.0, 1.0],
             }


def sum_and_remove(param_dict, remove=False):
    print("summing LFP and removing single-cell LFP files. ")

    num_tsteps = int(round(param_dict['end_t']/param_dict['dt'] + 1))
    summed_center_sig = np.zeros((len(param_dict['center_electrode_parameters']['x']), num_tsteps))

    ns = None

    for cell_number in range(param_dict['num_cells']):
        param_dict.update({'cell_number': cell_number})
        ns = NeuralSimulation(**param_dict)
        sim_name = ns.sim_name
        try:
            center_lfp = 1000 * np.load(join(ns.sim_folder, 'center_sig_%s.npy' % sim_name))  # uV
        except IOError:
            print("Could not load ", cell_number, join(ns.sim_folder, 'center_sig_%s.npy' % sim_name))
        summed_center_sig += center_lfp

    np.save(join(ns.sim_folder, 'summed_center_signal_%s_%dum.npy' %
                 (ns.population_sim_name, param_dict['population_radius'])), summed_center_sig)

    if remove:
        param_dict.update({'cell_number': 0})
        ns = NeuralSimulation(**param_dict)
        sig_name = join(ns.sim_folder, 'center_sig_%s_*.npy' % ns.population_sim_name)
        #print("Remove files %s" % sig_name)
        os.system("rm %s" % sig_name)
    return True



def init_example_pop(param_dict):
    initialize_population(param_dict)

def simulate_populations(param_dict):
    print("Simulating example stick population. ")
    task = 1
    num_tasks = (len(param_dict['input_regions']) *
                 len(param_dict['distributions']) *
                 len(param_dict['mus']) *
                 len(param_dict['correlations'])*
                 param_dict['num_cells'])

    num_pops = (len(param_dict['input_regions']) *
                len(param_dict['distributions']) *
                len(param_dict['mus']) *
                len(param_dict['correlations']))
    pop_number = 0
    for input_region in param_dict['input_regions']:
        param_dict['input_region'] = input_region
        for distribution in param_dict['distributions']:
            param_dict['distribution'] = distribution
            for mu in param_dict['mus']:
                param_dict['mu'] = mu
                for correlation in param_dict['correlations']:
                    param_dict["correlation"] = correlation
                    for cell_idx in range(param_dict['num_cells']):
                        param_dict.update({'cell_number': cell_idx})
                        ns = NeuralSimulation(**param_dict)
                        ns.run_single_simulation()
                        if task % 50 == 0:
                            print("Simulating task {} of {}".format(task, num_tasks))

                        task += 1
                    pop_number += 1
                    print("Finished population {} / {}.".format(pop_number, num_pops))
                    sum_and_remove(param_dict, True)
    print("Done with all populations.")


def compare_populations(param_dict):

    fig = plt.figure(figsize=[6,  4])

    input_regions = ['homogeneous']
    fig.subplots_adjust(bottom=0.22, left=0.14, hspace=0.5, wspace=0.25)
    mu_clr_dict = {None: "k", 2.0: 'b'}
    dz = np.abs(np.diff(param_dict["center_electrode_parameters"]["z"])[0])
    normalize_factor = dz / 1
    tvec_name = "tvec_{}_{}.npy".format(param_dict["cell_name"],
                                        param_dict["input_type"])
    tvec = np.load(join(root_folder, param_dict["save_folder"],
            "simulations", tvec_name))
    shift_idx = np.argmin(np.abs(tvec - 150))
    name_dict = {None: "passive",
                 2.0: "passive + asym. restorative"}

    lines = []
    line_names = []

    for ir, input_region in enumerate(input_regions):
        param_dict['input_region'] = input_region
        for c, correlation in enumerate(param_dict['correlations']):
            param_dict["correlation"] = correlation

            ax_ = fig.add_subplot(2, 2, 1 + c*2, xlim=[0, 50], ylim=[-15, 10],
                                  title="c={}".format(correlation),
                                  ylabel="LFP ($\mu$V)")
            ax_shift = fig.add_subplot(2, 2, 2 + c*2, xlim=[150, 200], ylim=[-15, 10],
                                  title="c={}; shifted".format(correlation))

            if c == 1:
                ax_.set_xlabel("Time (ms)")
                ax_shift.set_xlabel("Time (ms)")
            # ax_.text(-30, 500, input_region, rotation=90, ha="center", va="center")
            for d, distribution in enumerate(param_dict['distributions']):
                param_dict['distribution'] = distribution
                for m, mu in enumerate(param_dict['mus']):
                    param_dict['mu'] = mu
                    ns = NeuralSimulation(**param_dict)
                    lfp = np.load(join(ns.sim_folder, 'summed_center_signal_%s_%dum.npy' %
                         (ns.population_sim_name, param_dict["population_radius"])))
                    for eidx, z in enumerate(param_dict["center_electrode_parameters"]["z"]):
                        if not 0 <= z < 200:
                            continue
                        ax_.axhline(z, color='gray', ls="--")
                        ax_shift.axhline(z, color='gray', ls="--")
                        y = (lfp[eidx, :]) * normalize_factor + z
                        y_shift = (lfp[eidx, :] - np.average(lfp[eidx, shift_idx:])) * normalize_factor + z
                        l, = ax_.plot(tvec, y, c=mu_clr_dict[mu])
                        ax_shift.plot(tvec, y_shift, c=mu_clr_dict[mu])
                    if ir == 0 and c == 0 and d == 0:
                        lines.append(l)
                        line_names.append(name_dict[mu])
    simplify_axes(fig.axes)
    mark_subplots(fig.axes)
    # fig.set_tight_layout(True)
    fig.legend(lines, line_names, frameon=False, ncol=2, loc=(0.2, 0.001))
    fig.savefig(join(root_folder, param_dict["save_folder"], "LFP_test_example_stick.pdf"))
    # plt.show()


if __name__ == '__main__':
    init_example_pop(param_dict)
    simulate_populations(param_dict)
    compare_populations(param_dict)
    # plot_LFP_PSDs(param_dict)