import sys
import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from aPop.Population import initialize_population
from aPop.NeuralSimulation import NeuralSimulation
from aPop.tools import return_freq_and_psd_welch
from aPop.plotting_convention import mark_subplots, simplify_axes

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
              'num_cells': 10,
              'population_radius': 20,
              'layer_5_thickness': 0,
              'cut_off': 100,
              'end_t': 500,
              'syn_tau': dt * 3,
              'syn_weight': 1e-3,
              'max_freq': 500,
              'holding_potential': -80,
              'g_w_bar_scaling': 10.,
              'conductance_type': 'generic',
              'save_folder': 'stick_pop_example',
              'center_electrode_parameters': center_elec_params,
              'root_folder': root_folder,
              'num_synapses': 100,
              'input_firing_rate': 5,
              'input_regions': ['top', 'homogeneous', ],
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


def plot_LFP_PSDs(param_dict):

    fig = plt.figure(figsize=[14, 8])

    fig.subplots_adjust(bottom=0.12, right=0.98, top=0.95)
    mu_clr_dict = {None: "k", 2.0: 'b'}

    tvec_name = "tvec_{}_{}.npy".format(param_dict["cell_name"],
                                        param_dict["input_type"])
    tvec = np.load(join(root_folder, param_dict["save_folder"],
            "simulations", tvec_name))
    dt = tvec[1] - tvec[0]
    num_tsteps = len(tvec)

    from matplotlib.mlab import window_hanning
    divide_into_welch = 2.
    welch_dict = {'Fs': 1000 / dt,
                  'NFFT': int(num_tsteps/divide_into_welch),
                  'noverlap': int(num_tsteps/divide_into_welch/2.),
                  'window': window_hanning,
                  'detrend': 'mean',
                  'scale_by_freq': True,
                  }

    name_dict = {None: "passive",
                 2.0: "passive + asym. restorative"}

    lines = []
    line_names = []
    num_cols = 4
    num_rows = 2

    soma_elec = np.argmin(np.abs(param_dict["center_electrode_parameters"]["z"] - 0))

    for ir, input_region in enumerate(param_dict['input_regions']):
        param_dict['input_region'] = input_region
        for c, correlation in enumerate(param_dict['correlations']):
            param_dict["correlation"] = correlation
            for d, distribution in enumerate(param_dict['distributions']):
                param_dict['distribution'] = distribution
                plot_number = ir * num_cols + c*2 + 1
                ax = fig.add_subplot(num_rows, num_cols, plot_number,
                                      title="{} input; c={}".format(input_region, correlation),
                                      xlabel="frequency (Hz)",
                                      ylabel="LFP-PSD ($\mu$V$^2$/Hz)")
                if c == 0:
                    ax.text(-0.55, 0.5, input_region + " input", fontsize=24,
                            transform=ax.transAxes, rotation=90, va='center')

                ax_mod = fig.add_subplot(num_rows, num_cols, plot_number + 1,
                                      title="{} input; c={}".format(input_region, correlation),
                                      xlabel="frequency (Hz)",
                                      ylabel="PSD modulation",
                                      ylim=[1e-2, 1e2])
                ax_mod.axhline(1, color='gray', ls="--")
                psd_dict = {}
                for m, mu in enumerate(param_dict['mus']):
                    param_dict['mu'] = mu
                    ns = NeuralSimulation(**param_dict)

                    lfp = np.load(join(ns.sim_folder, 'summed_center_signal_%s_%dum.npy' %
                         (ns.population_sim_name, param_dict["population_radius"])))[soma_elec, :]
                    freq, psd = return_freq_and_psd_welch(lfp, welch_dict)
                    psd_dict[mu] = psd
                    l, = ax.loglog(freq, psd[0], c=mu_clr_dict[mu], lw=2)

                    if ir == 0 and c == 0 and d == 0:
                        lines.append(l)
                        line_names.append(name_dict[mu])
                ax_mod.loglog(freq, psd_dict[2.0][0]/psd_dict[None][0],
                              c=mu_clr_dict[2.0], lw=2)
    simplify_axes(fig.axes)
    mark_subplots(fig.axes, ypos=1.11)
    fig.legend(lines, line_names, frameon=False, ncol=2, loc=(0.4, 0.01))
    fig.savefig(join(root_folder, param_dict["save_folder"], "LFP_PSDs.png"))


def compare_populations(param_dict):

    fig = plt.figure(figsize=[18.3, 9.3])
    fig.suptitle("Uniform synaptic input")
    fig.subplots_adjust(bottom=0.15)
    mu_clr_dict = {None: "k", 2.0: 'b'}
    dz = np.abs(np.diff(param_dict["center_electrode_parameters"]["z"])[0])
    normalize_factor = dz / 1
    tvec_name = "tvec_{}_{}.npy".format(param_dict["cell_name"],
                                        param_dict["input_type"])
    tvec = np.load(join(root_folder, param_dict["save_folder"],
            "simulations", tvec_name))
    shift_idx = np.argmin(np.abs(tvec - 200))
    name_dict = {None: "passive",
                 2.0: "passive + asym. restorative"}

    lines = []
    line_names = []

    for ir, input_region in enumerate(param_dict['input_regions']):
        param_dict['input_region'] = input_region
        for c, correlation in enumerate(param_dict['correlations']):
            param_dict["correlation"] = correlation

            ax_ = fig.add_subplot(2, 2, 1 + c, xlim=[0, 300], ylim=[-15, 10],
                                  title="c={}".format(correlation),
                                  xlabel="Time (ms)", ylabel="LFP ($\mu$V)")
            ax_shift = fig.add_subplot(2, 2, 3 + c, xlim=[200, 300], ylim=[-15, 10],
                                  title="c={}; Zoomed and shifted".format(correlation),
                                   xlabel="Time (ms)", ylabel="LFP ($\mu$V)")
            # ax_.text(-30, 500, input_region, rotation=90, ha="center", va="center")
            for d, distribution in enumerate(param_dict['distributions']):
                param_dict['distribution'] = distribution
                for m, mu in enumerate(param_dict['mus']):
                    param_dict['mu'] = mu
                    ns = NeuralSimulation(**param_dict)
                    lfp = np.load(join(ns.sim_folder, 'summed_center_signal_%s_%dum.npy' %
                         (ns.population_sim_name, param_dict["population_radius"] - 1)))
                    print(lfp.shape, np.max(np.abs(lfp)), param_dict["center_electrode_parameters"]["z"])
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
    fig.set_tight_layout(True)
    fig.legend(lines, line_names, frameon=False, ncol=2, loc=(0.4,0.01))
    fig.savefig(join(root_folder, param_dict["save_folder"], "LFP_test_example_stick_high_activity.png"))
    plt.show()

if __name__ == '__main__':
    # init_example_pop(param_dict)
    simulate_populations(param_dict)
    # compare_populations(param_dict)
    # plot_LFP_PSDs(param_dict)