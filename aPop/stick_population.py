from __future__ import division
import param_dicts
import os
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg', warn=False)
    at_stallo = True
else:
    at_stallo = False
# from plotting_convention import *
import pylab as plt
import numpy as np
import os
from os.path import join
import LFPy
import neuron
import sys
# from NeuralSimulation import NeuralSimulation
# import tools

from param_dicts import stick_population_params as param_dict
#
# param_dict.update({'input_region': 'homogeneous'})
#
# ns = NeuralSimulation(**param_dict)

plt.seed(1234)

root_folder = param_dict['root_folder']
neuron_models = join(root_folder, 'neuron_models')

neuron.load_mechanisms(join(neuron_models))
sys.path.append(join(neuron_models, 'infinite_neurite'))
from infinite_neurite_active_declarations import active_declarations
mu_1 = 2.0
mu_2 = 2.0

args = [{'mu_factor_1': mu_1, 'mu_factor_2': mu_2}]
cell_params = {
    'morphology': join(neuron_models, param_dict['cell_name'], 'infinite_neurite.hoc'),
    'v_init': param_dict['holding_potential'],
    'passive': False,
    'nsegs_method': None,
    # 'lambda_f': None,
    'timeres_NEURON': param_dict['timeres_NEURON'],   # dt of LFP and NEURON simulation.
    'timeres_python': param_dict['timeres_python'],
    'tstartms': -param_dict['cut_off'],          # start time, recorders start at t=0
    'tstopms': param_dict['end_t'],
    'custom_fun': [active_declarations],
    'custom_fun_args': args,
}

cell = LFPy.Cell(**cell_params)
cell.set_pos(zpos=-cell.zstart[0])


def set_input_spiketrain(cell, cell_input_idxs, spike_trains, synapse_params):
    synapse_list = []
    for number, comp_idx in enumerate(cell_input_idxs):
        synapse_params.update({'idx': int(comp_idx)})
        s = LFPy.Synapse(cell, **synapse_params)
        s.set_spike_times(spike_trains[number])
        synapse_list.append(s)
    return synapse_list

synapse_params = {
    'e': 0.,                   # reversal potential
    'syntype': 'ExpSynI',       # synapse type
    'tau': param_dict['syn_tau'],                # syn. time constant
    'weight': param_dict['syn_weight'],            # syn. weight
    'record_current': False,
}

cell_input_idxs = cell.get_rand_idx_area_norm(section='allsec', nidx=1000, z_min=0)
firing_rate = 5
spike_trains = LFPy.inputgenerators.stationary_poisson(len(cell_input_idxs), firing_rate, cell.tstartms, cell.tstopms)

synapses = set_input_spiketrain(cell, cell_input_idxs, spike_trains, synapse_params)

cell.simulate(rec_imem=True, rec_vmem=True)

g_pas = []
for sec in neuron.h.allsec():
    for seg in sec:
        g_pas.append(seg.gamma_R_QA)
        # print seg.gamma_R_QA

g_pas = np.array(g_pas)

conductance_clr = lambda g: plt.cm.Greys(g / np.max(g_pas))

time_idx = len(cell.tvec)/1.5

fig = plt.figure(figsize=[19, 10])
fig.subplots_adjust(wspace=0.6)
ax_lfp = fig.add_subplot(151, ylim=[-100, 1100], xlim=[-100, 100], aspect=1)
ax_avrg_lfp = fig.add_subplot(154, ylim=[-100, 1100], xlim=[-100, 100], aspect=1)

[ax_lfp.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]], '-_',
             solid_capstyle='butt', ms=6, lw=cell.diam[idx], color=conductance_clr(g_pas[idx]), zorder=1)
 for idx in xrange(len(cell.xmid))]

ax_rc = fig.add_subplot(152, ylim=[-100, 1100], title='Transmembrane currents\nt=%d ms' % cell.tvec[time_idx])
ax_rc.plot([0, 0], [0, 1000], '--', c='gray')
ax_rc.plot(cell.imem[:, time_idx], cell.zmid)


ax_avrg_rc = fig.add_subplot(155, ylim=[-100, 1100], title='Average transmembrane currents')
ax_avrg_rc.plot([0, 0], [0, 1000], '--', c='gray')
ax_avrg_rc.plot(np.average(cell.imem, axis=1), cell.zmid)

x = np.linspace(-100, 100, 11)
z = np.linspace(-100, 1100, 25)
x, z = np.meshgrid(x, z)
elec_x = x.flatten()
elec_z = z.flatten()
elec_y = np.zeros(len(elec_x))
electrode_parameters = {
    'sigma': 0.3,              # extracellular conductivity
    'x': elec_x,        # x,y,z-coordinates of contact points
    'y': elec_y,
    'z': elec_z,
    'method': 'pointsource'
}
electrode = LFPy.RecExtElectrode(cell, **electrode_parameters)
electrode.calc_lfp()
sig_amp = 1000 * electrode.LFP[:, time_idx].reshape(x.shape)

color_lim = np.max(np.abs(sig_amp)) / 5
img_lfp = ax_lfp.imshow(sig_amp, origin='lower', extent=[np.min(x), np.max(x), np.min(z), np.max(z)],
              vmin=-color_lim, vmax=color_lim, interpolation='nearest', cmap=plt.cm.bwr, zorder=0)
plt.colorbar(img_lfp, ax=ax_lfp)
ax_v = fig.add_subplot(353, title='Membrane potential')
[ax_v.plot(cell.tvec, cell.vmem[idx]) for idx in range(cell.totnsegs)]
ax_v.plot([cell.tvec[time_idx], cell.tvec[time_idx]], [np.min(cell.vmem) - 0.1, np.max(cell.vmem) + 0.1], '--', c='gray')
ax_i = fig.add_subplot(358, title='Membrane current')
[ax_i.plot(cell.tvec, cell.imem[idx]) for idx in range(cell.totnsegs)]
ax_i.plot([cell.tvec[time_idx], cell.tvec[time_idx]], [np.min(cell.imem) - 0.001, np.max(cell.imem) + 0.001], '--', c='gray')

avrg_sig_amp = 1000 * np.average(electrode.LFP[:, :], axis=1).reshape(x.shape)

# color_lim = np.max(np.abs(avrg_sig_amp))
img_avrg = ax_avrg_lfp.imshow(avrg_sig_amp, origin='lower', extent=[np.min(x), np.max(x), np.min(z), np.max(z)],
              vmin=-color_lim, vmax=color_lim, interpolation='nearest', cmap=plt.cm.bwr, zorder=0)
plt.colorbar(img_avrg, ax=ax_avrg_lfp)

fig.savefig(join(root_folder, param_dict['save_folder'], 'stick_test_%+1.1f_%+1.1f.png' % (mu_1, mu_2)))
