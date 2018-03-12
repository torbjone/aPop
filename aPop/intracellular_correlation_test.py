import sys
import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from Population import initialize_population
from NeuralSimulation import NeuralSimulation
from tools import return_freq_and_psd_welch
from plotting_convention import mark_subplots, simplify_axes
import LFPy
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

np.random.seed(1234)

dt = 2**-5
param_dict = {'input_type': "distributed_delta",
              'name': 'stick_pop_example',
              'cell_number': 0,
              'cell_name': 'stick_cell',
              'dt': dt,
              'num_cells': 1,
              'population_radius': 1,
              'layer_5_thickness': 0,
              'cut_off': 100,
              'end_t': 4000,
              'syn_tau': dt * 3,
              'syn_weight': 1e-2,
              'max_freq': 500,
              'holding_potential': -80,
              'g_w_bar_scaling': 20.,
              'conductance_type': 'generic',
              'input_region': "bottom",
              'distribution': "uniform",
              'mu': 0.0,
              "correlation": 0.0,
              'save_folder': 'intracellular_freq_scaling',
              'center_electrode_parameters': center_elec_params,
              'root_folder': root_folder,
              'num_synapses': 100,
              'input_firing_rate': 10,
              'input_regions': ['top'],
              'mus': [None, 2.0],
              'distributions': ['homogeneous'],
              'correlations': [0.0, 1.0],
             }


def init_example_pop(param_dict):
    initialize_population(param_dict)

def simulate_populations(param_dict):
    print("Simulating example stick")
    num_synapses = param_dict['num_synapses']
    correlation = 0.5
    jitter = 100.
    np.random.seed(1234)
    ns = NeuralSimulation(**param_dict)
    synapse_params = {
        'e': 0.,                   # reversal potential (not used for ExpSynI)
        'syntype': "ExpSynI",       # synapse type
        'tau': param_dict['syn_tau'],                # syn. time constant
        'weight': param_dict['syn_weight'],            # syn. weight
        'record_current': False,
    }


    cell = ns.return_cell()
    input_pos = ['dend', "soma"]
    maxpos = 500
    minpos = -10000
    cell_input_idxs = cell.get_rand_idx_area_norm(section=input_pos,
                                                  nidx=num_synapses,
                                                  z_min=minpos,
                                                  z_max=maxpos)

    num_independent_spike_trains = int((num_synapses - 1) * (1 - correlation) + 1)

    spike_trains = LFPy.inputgenerators.get_activation_times_from_distribution(
                n=num_independent_spike_trains, tstart=cell.tstart, tstop=cell.tstop,
                rvs_args=dict(loc=0., scale=1000. / param_dict['input_firing_rate']))

    if correlation > 0:
        spike_trains = np.random.choice(spike_trains, num_synapses, replace=True)

    mod_spike_trains = []

    for tidx in range(len(spike_trains)):
        mod_spike_trains.append(np.zeros(len(spike_trains[tidx])))
        for sidx in range(len(spike_trains[tidx])):
            jit = np.random.normal(0, jitter)
            mod_spike_trains[tidx][sidx] = spike_trains[tidx][sidx] + jit
    spike_trains = mod_spike_trains

    synapses = ns.set_input_spiketrain(cell, cell_input_idxs, spike_trains, synapse_params)

    cell.simulate(rec_imem=True, rec_vmem=True)
    # center_electrode.calc_lfp()
    fig = plt.figure(figsize=[14, 8])

    fig.subplots_adjust(bottom=0.12, right=0.98, top=0.94)

    dt = cell.tvec[1] - cell.tvec[0]
    num_tsteps = len(cell.tvec)

    from matplotlib.mlab import window_hanning
    divide_into_welch = 8.
    welch_dict = {'Fs': 1000 / dt,
                  'NFFT': int(num_tsteps/divide_into_welch),
                  'noverlap': int(num_tsteps/divide_into_welch/2.),
                  'window': window_hanning,
                  'detrend': 'mean',
                  'scale_by_freq': True,
                  }

    lines = []
    line_names = []
    ax_sp = fig.add_subplot(3, 1, 1,
                          xlabel="Time", title="Independent spike trains: {}".format(num_independent_spike_trains),
                          ylabel="Input",)
    ax_vt = fig.add_subplot(3, 1, 2,
                          xlabel="Time",
                          ylabel="Membrane potential",)
    ax_v = fig.add_subplot(3, 4, 9, xlim=[1, 1000], aspect=0.5, ylim=[1e-7, 1e-2],
                          xlabel="frequency (Hz)",
                          ylabel="Vmem-PSD")
    ax_v_norm = fig.add_subplot(3, 4, 10, xlim=[1, 1000], aspect=0.5, ylim=[1e-4, 1e1],
                          xlabel="frequency (Hz)",
                          ylabel="Vmem-PSD")

    ax_i = fig.add_subplot(3, 4, 11, xlim=[1, 1000], aspect=0.5, ylim=[1e-14, 1e-8],
                          xlabel="frequency (Hz)",
                          ylabel="Imem-PSD",)
    ax_i_norm = fig.add_subplot(3, 4, 12, xlim=[1, 1000], aspect=0.5, ylim=[1e-4, 1e1],
                          xlabel="frequency (Hz)",
                          ylabel="Imem-PSD",)

    ax_i.grid(True)
    ax_i_norm.grid(True)
    ax_v.grid(True)
    ax_v_norm.grid(True)
    for train_idx, train in enumerate(spike_trains):
        ax_sp.plot(train, np.ones(len(train)) + train_idx, 'k.')

    ax_vt.plot(cell.tvec, cell.vmem[0])

    freq, psd_i = return_freq_and_psd_welch(cell.imem[0,:], welch_dict)
    freq, psd_v = return_freq_and_psd_welch(cell.vmem[0,:], welch_dict)
    l, = ax_i.loglog(freq, psd_i[0], lw=2)
    l, = ax_i_norm.loglog(freq, psd_i[0] / np.max(psd_i[0]), lw=2)
    l, = ax_v.loglog(freq, psd_v[0], lw=2)
    l, = ax_v_norm.loglog(freq, psd_v[0] / np.max(psd_v[0]), lw=2)

    simplify_axes(fig.axes)
    mark_subplots(fig.axes, ypos=1.11)
    fig.legend(lines, line_names, frameon=False, ncol=2, loc=(0.4, 0.01))
    fig.savefig(join(root_folder, param_dict["save_folder"],
     "intracellular_vm_jitter_c={}_jitter={}.pdf".format(correlation, jitter)))


if __name__ == '__main__':
    # init_example_pop(param_dict)
    simulate_populations(param_dict)
    # plot_LFP_PSDs(param_dict)