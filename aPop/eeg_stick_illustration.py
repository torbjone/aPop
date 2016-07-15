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
from Dipole import Dipole
from CalcLFP import CalcLFP
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

mu = 2.0
g_w_bar_scaling = 5
firing_rate = 5.

input_region = 'homogeneous'
distribution = 'increase'
if input_region == 'top':
    cell_input_idxs = np.arange(0, 50)[::-1] + 50
elif input_region == 'bottom':
    cell_input_idxs = np.arange(00, 50)[::1] + 0
elif input_region == 'homogeneous':
    cell_input_idxs = np.arange(00, 100)[::1] + 0
else:
    raise RuntimeError("Unrecognized")

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
    'tstartms': -100,          # start time, recorders start at t=0
    'tstopms': 100,
    'custom_fun': [active_declarations],
    'custom_fun_args': args,
}

cell = LFPy.Cell(**cell_params)
cell.set_pos(zpos=-cell.zstart[0])

spike_trains = LFPy.inputgenerators.stationary_poisson(len(cell_input_idxs), firing_rate, cell.tstartms, cell.tstopms)

def set_input_spiketrain(cell, cell_input_idxs, synapse_params, spike_trains):
    synapse_list = []
    for number, comp_idx in enumerate(cell_input_idxs):
        if len(spike_trains[number]) == 0:
            continue
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

synapses = set_input_spiketrain(cell, cell_input_idxs, synapse_params, spike_trains)

cell.simulate(rec_imem=True, rec_vmem=True)

dipole = Dipole(cell)
dlist, itrans = dipole.transmembrane_currents()

P3, P3_tot = dipole.current_dipole_moment(dlist, itrans)

r_syns = np.array([[cell.xmid[i], cell.ymid[i], cell.zmid[i]] for i in cell_input_idxs])

r_mid = np.average(r_syns, axis=0) + [0, 0, 500]
r_mid = r_mid/2.

eeg_x = np.linspace(-4000, 4000, 15)
eeg_y = np.linspace(-4000, 4000, 15)
eeg_x, eeg_y = np.meshgrid(eeg_x, eeg_y)

eeg_elec_x = eeg_x.flatten()
eeg_elec_y = eeg_y.flatten()
eeg_elec_z = np.ones(len(eeg_elec_x)) * 12000#np.linspace(20000, 1400, 100)

eeg_electrode_parameters = {
    'sigma': 0.3,              # extracellular conductivity
    'x': eeg_elec_x,        # x,y,z-coordinates of contact points
    'y': eeg_elec_y,
    'z': eeg_elec_z,
    'method': 'pointsource'
}
eeg_electrode = LFPy.RecExtElectrode(cell, **eeg_electrode_parameters)
eeg_electrode.calc_lfp()

eeg_dp = CalcLFP(cell, eeg_elec_x, eeg_elec_y, eeg_elec_z)
eeg_dp_sig, eeg_theta = eeg_dp.grid_lfp_theta(P3, 0.3)

center_eeg_elec = np.argmin(eeg_elec_x**2 + eeg_elec_y**2)

x = np.linspace(-4000, 4000, 15)
z = np.linspace(-4000, 14000, 100)
x, z = np.meshgrid(x, z)
elec_x = x.flatten()
elec_z = z.flatten()
elec_y = np.ones(len(elec_x)) * 30

electrode_parameters = {
    'sigma': 0.3,              # extracellular conductivi   ty
    'x': elec_x,        # x,y,z-coordinates of contact points
    'y': elec_y,
    'z': elec_z,
    'method': 'pointsource'
}
electrode = LFPy.RecExtElectrode(cell, **electrode_parameters)
electrode.calc_lfp()

logthresh = 3
color_lim = np.max(np.abs(1e6 * electrode.LFP)) / 1

vmax = color_lim
vmin = -color_lim
maxlog = int(np.ceil(np.log10(vmax)))
minlog = int(np.ceil(np.log10(-vmin)))

tick_locations = ([-(10**d) for d in xrange(minlog, -logthresh-1, -1)]
                + [0.0]
                + [(10**d) for d in xrange(-logthresh, maxlog+1)])


eeg_logthresh = 6
eeg_color_lim = np.max(np.abs(1e6 * eeg_electrode.LFP)) / 2
eeg_vmax = eeg_color_lim
eeg_vmin = -eeg_color_lim
eeg_maxlog = int(np.ceil(np.log10(eeg_vmax)))
eeg_minlog = int(np.ceil(np.log10(-eeg_vmin)))

eeg_tick_locations = ([-(10**d) for d in xrange(eeg_minlog, -eeg_logthresh-1, -1)]
                + [0.0]
                + [(10**d) for d in xrange(-eeg_logthresh, eeg_maxlog+1)])


for time_idx in np.arange(0, len(cell.tvec), 10):
    arrow = P3[time_idx] * 1e5  # overall, 0.05 for apical and basal

# time_idx = np.argmax(np.abs(cell.imem[cell_input_idxs[len(cell_input_idxs)/2], :]))
    eeg_sig_amp = 1e6 * eeg_electrode.LFP[:, time_idx].reshape(eeg_x.shape)
    eeg_dp_sig_amp = eeg_dp_sig[:, time_idx].reshape(eeg_x.shape)
    sig_amp = 1e6 * electrode.LFP[:, time_idx].reshape(x.shape)

    dp = CalcLFP(cell, elec_x, elec_y, elec_z)
    dp_sig, theta = dp.grid_lfp_theta(P3, 0.3)
    dp_sig_amp = dp_sig[:, time_idx].reshape(x.shape)
    plt.close('all')
    fig = plt.figure(figsize=[10, 6])
    fig.subplots_adjust(wspace=0.6, hspace=0.6)

    ax_lfp = fig.add_subplot(231, ylim=[np.min(elec_z), np.max(elec_z)], xlim=[np.min(elec_x), np.max(elec_x)],
                             aspect=1, frameon=False, xticks=[], yticks=[], title='True LFP')

    # [ax_lfp.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]], '-_',
    #              solid_capstyle='butt', ms=6, lw=cell.diam[idx], color='k', zorder=1)
    #  for idx in xrange(len(cell.xmid))]

    img_lfp = ax_lfp.imshow(sig_amp, origin='lower', extent=[np.min(x), np.max(x), np.min(z), np.max(z)],
                            cmap=plt.cm.bwr, norm=matplotlib.colors.SymLogNorm(10**-logthresh),
                  vmin=vmin, vmax=vmax, interpolation='nearest', zorder=0)

    cb = plt.colorbar(img_lfp, ax=ax_lfp, shrink=0.5, ticks=tick_locations)
    cb.set_label('nV', labelpad=-10)
    # ax_lfp.plot(cell.xmid[cell_input_idxs] - 40, cell.zmid[cell_input_idxs], '<', c='g', ms=6,  zorder=1)
    ax_lfp.arrow(r_mid[0] - arrow[0], r_mid[2] - arrow[2], arrow[0]*2, arrow[2]*2, fc='c', ec='c',
                 width=30, length_includes_head=True, zorder=10)

    ax_eeg = fig.add_subplot(233, ylim=[np.min(eeg_elec_y), np.max(eeg_elec_y)], xlim=[np.min(eeg_elec_x), np.max(eeg_elec_x)],
                             aspect=1, frameon=False, xticks=[], yticks=[], title='True EEG')

    # [ax_eeg.plot([cell.xstart[idx], cell.xend[idx]], [cell.ystart[idx], cell.yend[idx]], '-_',
    #              solid_capstyle='butt', ms=6, lw=cell.diam[idx], color='k', zorder=1)
    #  for idx in xrange(len(cell.xmid))]


    img_eeg = ax_eeg.imshow(eeg_sig_amp, origin='lower', extent=[np.min(eeg_x), np.max(eeg_x), np.min(eeg_y), np.max(eeg_y)],
                            cmap=plt.cm.bwr, norm=matplotlib.colors.SymLogNorm(10**-logthresh),
                  vmin=eeg_vmin, vmax=eeg_vmax, interpolation='nearest', zorder=0)

    cb = plt.colorbar(img_eeg, ax=ax_eeg, shrink=0.5, ticks=eeg_tick_locations)
    cb.set_label('nV', labelpad=-10)


    ax_dp = fig.add_subplot(234, ylim=[np.min(elec_z), np.max(elec_z)], xlim=[np.min(elec_x), np.max(elec_x)],
                             aspect=1, frameon=False, xticks=[], yticks=[], title='Dipole approximation')

    # [ax_dp.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]], '-_',
    #              solid_capstyle='butt', ms=6, lw=cell.diam[idx], color='k', zorder=1)
    #  for idx in xrange(len(cell.xmid))]


    img_dp = ax_dp.imshow(dp_sig_amp, origin='lower', extent=[np.min(x), np.max(x), np.min(z), np.max(z)],
                            cmap=plt.cm.bwr, norm=matplotlib.colors.SymLogNorm(10**-logthresh),
                  vmin=vmin, vmax=vmax, interpolation='nearest', zorder=0)

    cb = plt.colorbar(img_dp, ax=ax_dp, shrink=0.5, ticks=tick_locations)
    cb.set_label('nV', labelpad=-10)
    # ax_dp.plot(cell.xmid[cell_input_idxs] - 40, cell.zmid[cell_input_idxs], '<', c='g', ms=6,  zorder=1)
    ax_dp.arrow(r_mid[0] - arrow[0], r_mid[2] - arrow[2], arrow[0]*2, arrow[2]*2, fc='c', ec='c',
                 width=30, length_includes_head=True, zorder=10)

    ax_dp_eeg = fig.add_subplot(236, ylim=[np.min(eeg_elec_y), np.max(eeg_elec_y)], xlim=[np.min(eeg_elec_x), np.max(eeg_elec_x)],
                             aspect=1, frameon=False, xticks=[], yticks=[], title='Dipole approximation\nEEG')

    # [ax_dp_eeg.plot([cell.xstart[idx], cell.xend[idx]], [cell.ystart[idx], cell.yend[idx]], '-_',
    #              solid_capstyle='butt', ms=6, lw=cell.diam[idx], color='k', zorder=1)
    #  for idx in xrange(len(cell.xmid))]


    img_db_eeg = ax_dp_eeg.imshow(eeg_dp_sig_amp, origin='lower', extent=[np.min(eeg_x), np.max(eeg_x), np.min(eeg_y), np.max(eeg_y)],
                            cmap=plt.cm.bwr, norm=matplotlib.colors.SymLogNorm(10**-logthresh),
                  vmin=eeg_vmin, vmax=eeg_vmax, interpolation='nearest', zorder=0)

    cb = plt.colorbar(img_db_eeg, ax=ax_dp_eeg, shrink=0.5, ticks=eeg_tick_locations)
    cb.set_label('nV', labelpad=-10)

    ax_rc = fig.add_subplot(332, title='Dipole angle', ylim=[0, 2*np.pi])
    ax_rc.plot(cell.tvec, theta[center_eeg_elec])
    ax_rc.axvline(cell.tvec[time_idx], c='g')

    ax_rc = fig.add_subplot(335, title='Magnitude')
    ax_rc.plot(cell.tvec, P3_tot)
    ax_rc.axvline(cell.tvec[time_idx], c='g')

    ax_rc = fig.add_subplot(338, title='Center EEG signal', ylabel='nV')
    ax_rc.plot(cell.tvec, 1e6*eeg_electrode.LFP[center_eeg_elec], 'gray')
    ax_rc.plot(cell.tvec, eeg_dp_sig[center_eeg_elec], 'k--')

    ax_rc.axvline(cell.tvec[time_idx], c='g')

    # ax_v = fig.add_subplot(353, title='Membrane potential')
    # [ax_v.plot(cell.tvec, cell.vmem[idx]) for idx in range(cell.totnsegs)]
    # ax_v.plot([cell.tvec[time_idx], cell.tvec[time_idx]], [np.min(cell.vmem) - 0.1, np.max(cell.vmem) + 0.1], '--', c='gray')
    # ax_i = fig.add_subplot(358, title='Membrane current')
    # [ax_i.plot(cell.tvec, cell.imem[idx]) for idx in range(cell.totnsegs)]
    # ax_i.plot([cell.tvec[time_idx], cell.tvec[time_idx]], [np.min(cell.imem) - 0.001, np.max(cell.imem) + 0.001], '--', c='gray')


    fig.savefig(join(root_folder, param_dict['save_folder'], 'illustration', 'eeg_stick_%+1.1f_%s_%s_%d_%04d.png' %
                     (mu, input_region, distribution, len(cell_input_idxs), time_idx)))
