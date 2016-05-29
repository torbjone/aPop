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
import matplotlib
# from NeuralSimulation import NeuralSimulation
# import tools

from param_dicts import stick_population_params as param_dict
#
# param_dict.update({'input_region': 'homogeneous'})
# ns = NeuralSimulation(**param_dict)

plt.seed(1234)

root_folder = param_dict['root_folder']
neuron_models = join(root_folder, 'neuron_models')

neuron.load_mechanisms(join(neuron_models))
sys.path.append(join(neuron_models, 'infinite_neurite'))
from infinite_neurite_active_declarations import active_declarations

mu = 0.0
g_w_bar_scaling = 5

input_region = 'homogeneous'
distribution = 'uniform'
cell_input_idxs = np.arange(00, 100)[::1] + 0
# spike_trains = np.array([[5], [5]])

args = [{'mu_factor': mu,
         'g_w_bar_scaling': g_w_bar_scaling,
         'distribution': distribution,
         }]

cell_params = {
    'morphology': join(neuron_models, param_dict['cell_name'], 'infinite_neurite.hoc'),
    'v_init': param_dict['holding_potential'],
    'passive': False,
    'nsegs_method': None,
    # 'lambda_f': None,
    'timeres_NEURON': param_dict['timeres_NEURON'],   # dt of LFP and NEURON simulation.
    'timeres_python': param_dict['timeres_python'],
    'tstartms': 0,          # start time, recorders start at t=0
    'tstopms': 10,
    'custom_fun': [active_declarations],
    'custom_fun_args': args,
}

cell = LFPy.Cell(**cell_params)
cell.set_pos(zpos=-cell.zstart[0])


def set_input_spiketrain(cell, cell_input_idxs, synapse_params):
    synapse_list = []
    for number, comp_idx in enumerate(cell_input_idxs):
        synapse_params.update({'idx': int(comp_idx)})
        s = LFPy.Synapse(cell, **synapse_params)
        s.set_spike_times(np.array([5]))
        synapse_list.append(s)
    return synapse_list

synapse_params = {
    'e': 0.,                   # reversal potential
    'syntype': 'ExpSynI',       # synapse type
    'tau': param_dict['syn_tau'],                # syn. time constant
    'weight': param_dict['syn_weight'],            # syn. weight
    'record_current': False,
}

synapses = set_input_spiketrain(cell, cell_input_idxs, synapse_params)

cell.simulate(rec_imem=True, rec_vmem=True)

time_idx = np.argmax(np.abs(cell.imem[cell_input_idxs[0], :]))
x = np.linspace(-400, 400, 14)
z = np.linspace(-400, 1400, 100)
x, z = np.meshgrid(x, z)
elec_x = x.flatten()
elec_z = z.flatten()
elec_y = np.ones(len(elec_x)) * 30

electrode_parameters = {
    'sigma': 0.3,              # extracellular conductivity
    'x': elec_x,        # x,y,z-coordinates of contact points
    'y': elec_y,
    'z': elec_z,
    'method': 'pointsource'
}
electrode = LFPy.RecExtElectrode(cell, **electrode_parameters)
electrode.calc_lfp()
sig_amp = 1e6 * electrode.LFP[:, time_idx].reshape(x.shape)

lfp_rms = np.sqrt(np.average((sig_amp)**2))

g_pas = []
for sec in neuron.h.allsec():
    for seg in sec:
        g_pas.append(seg.gamma_R_QA)
        # print seg.gamma_R_QA

g_pas = np.array(g_pas)

conductance_clr = lambda g: plt.cm.Greys(g / np.max(g_pas))

fig = plt.figure(figsize=[7, 5])
fig.subplots_adjust(wspace=0.6)
ax_lfp = fig.add_subplot(121, ylim=[np.min(elec_z), np.max(elec_z)], xlim=[np.min(elec_x), np.max(elec_x)],
                         aspect=1, frameon=False, xticks=[], yticks=[], title='LFP\nRMS: %1.3f nV' % lfp_rms)

[ax_lfp.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]], '-_',
             solid_capstyle='butt', ms=6, lw=cell.diam[idx], color=conductance_clr(g_pas[idx]), zorder=1)
 for idx in xrange(len(cell.xmid))]

ax_lfp.plot(cell.xmid[cell_input_idxs] - 40, cell.zmid[cell_input_idxs], '<', c='g', ms=6,  zorder=1)




ax_rc = fig.add_subplot(122, ylim=[np.min(elec_z), np.max(elec_z)], title='Transmembrane currents\nt=%d ms' % cell.tvec[time_idx], frameon=False, yticks=[], xticks=[]  )
ax_rc.plot([0, 0], [0, 1000], '--', c='gray')
ax_rc.plot(1000 * cell.imem[:, time_idx], cell.zmid)

ax_rc.plot([0, np.max(np.abs(1000 * cell.imem[:, time_idx]))], [-50, -50], lw=3, color='k')
ax_rc.text(np.max(np.abs(1000 * cell.imem[:, time_idx])) / 2, -200, '%1.3f pA' % np.max(np.abs(1000 * cell.imem[:, time_idx])))

logthresh = 2
color_lim = 10#np.max(np.abs(sig_amp)) / 1

vmax = color_lim
vmin = -color_lim
maxlog = int(np.ceil(np.log10(vmax)))
minlog = int(np.ceil(np.log10(-vmin)))

tick_locations = ([-(10**d) for d in xrange(minlog, -logthresh-1, -1)]
                + [0.0]
                + [(10**d) for d in xrange(-logthresh, maxlog+1)])

img_lfp = ax_lfp.imshow(sig_amp, origin='lower', extent=[np.min(x), np.max(x), np.min(z), np.max(z)],
                        cmap=plt.cm.bwr, norm=matplotlib.colors.SymLogNorm(10**-logthresh),
              vmin=vmin, vmax=vmax, interpolation='nearest', zorder=0)

cb = plt.colorbar(img_lfp, ax=ax_lfp, shrink=0.5, ticks=tick_locations)
cb.set_label('nV', labelpad=-10)

# ax_v = fig.add_subplot(353, title='Membrane potential')
# [ax_v.plot(cell.tvec, cell.vmem[idx]) for idx in range(cell.totnsegs)]
# ax_v.plot([cell.tvec[time_idx], cell.tvec[time_idx]], [np.min(cell.vmem) - 0.1, np.max(cell.vmem) + 0.1], '--', c='gray')
# ax_i = fig.add_subplot(358, title='Membrane current')
# [ax_i.plot(cell.tvec, cell.imem[idx]) for idx in range(cell.totnsegs)]
# ax_i.plot([cell.tvec[time_idx], cell.tvec[time_idx]], [np.min(cell.imem) - 0.001, np.max(cell.imem) + 0.001], '--', c='gray')


fig.savefig(join(root_folder, param_dict['save_folder'], 'illustration', 'stick_%+1.1f_%s_%s_%d.png' %
                 (mu, input_region, distribution, len(cell_input_idxs))))
