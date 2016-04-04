from __future__ import division
import os
import sys
from os.path import join

from plotting_convention import *
import numpy as np
import pylab as plt

import LFPy
import tools



def plot_all_shape_functions(params):
    
    secs = ['homogeneous', 'distal_tuft', 'basal']
    mus = [-0.5, 0.0, 2.0]
    dists = ['uniform', 'linear_increase', 'linear_decrease']

    for input_sec in secs:
        for distribution in dists:
            for mu in mus:
                plot_F(mu, distribution, input_sec, params)
                # sys.exit()
     
def _return_elec_row_col(elec_number, elec_x, elec_z):
    """ Return the subplot number for the distance study
    """
    num_elec_cols = len(set(elec_x))
    num_elec_rows = len(set(elec_z))
    elec_idxs = np.arange(len(elec_x)).reshape(num_elec_rows, num_elec_cols)
    row, col = np.array(np.where(elec_idxs == elec_number))[:, 0]
    return row, col


def plot_F(mu, distribution, input_sec, params):
    sim_folder = join(params['root_folder'], params['save_folder'], 'simulations')
    figure_folder = join(params['root_folder'], params['save_folder'])
    cell_name = params['cell_name']
    input_type = params['input_type']
    max_freq = params['max_freq']
    sim_name = '%s_%s_%s_%s_%+1.1f_%1.5fuS' % (cell_name, input_type,
                                           input_sec, distribution, mu, params['syn_weight'])
    
    F = np.load(join(sim_folder, 'F_%s.npy' % sim_name))
    freqs = np.load(join(sim_folder, 'welch_freqs.npy'))
    cut_idx_freq = np.argmin(np.abs(freqs - max_freq)) + 1

    plot_freqs = [1, 15, 50, 100, 500]
    plot_freq_idxs = [np.argmin(np.abs(freqs - plot_freq)) for plot_freq in plot_freqs]

    F = F[:, :cut_idx_freq]
    freqs = freqs[:cut_idx_freq]

    elec_x = params['electrode_parameters']['x']
    elec_z = params['electrode_parameters']['z']
    im_dict = {'cmap': 'hot', 'norm': LogNorm(), 'vmin': 1e-11, 'vmax': 1.e-7}
    ax_dict = {'xscale': 'log', 'yscale': 'log', 'xlim': [10, 1000], 'ylim': [1, 500],
               }
    plt.close('all')

    fig = plt.figure(figsize=[18, 9])
    fig.subplots_adjust(right=0.97, left=0.05, hspace=0.5, bottom=0.1, wspace=0.5)
    plt.seed(123*0)

    # apic_F = F.reshape(3, 30, cut_idx_freq)[0][:, plot_freq_idxs]
    # middle_F = F.reshape(3, 30, cut_idx_freq)[1][:, plot_freq_idxs]
    # soma_F = F.reshape(3, 30, cut_idx_freq)[2][:, plot_freq_idxs]
    f_clr = lambda idx: plt.cm.Blues(0.1 + idx / len(plot_freqs))

    ax_f_apic = fig.add_subplot(3, 7, 5, xlim=[10, 1000], ylim=[1e-12, 1e-6],
                                ylabel='mV$^2$/Hz', xlabel='distance ($\mu$m)')
    ax_f_middle = fig.add_subplot(3, 7, 12, xlim=[10, 1000], ylim=[1e-12, 1e-6],
                                  ylabel='mV$^2$/Hz', xlabel='distance ($\mu$m)')
    ax_f_soma = fig.add_subplot(3, 7, 19, xlim=[10, 1000], ylim=[1e-12, 1e-6],
                                ylabel='mV$^2$/Hz', xlabel='distance ($\mu$m)')
    lines = []
    line_names = []
    for id, f_idx in enumerate(plot_freq_idxs):
        c = f_clr(id)
        f_apic = F.reshape(3, 30, cut_idx_freq)[0][:, f_idx]
        f_middle = F.reshape(3, 30, cut_idx_freq)[1][:, f_idx]
        f_soma = F.reshape(3, 30, cut_idx_freq)[2][:, f_idx]
        ax_f_apic.loglog(elec_x[:30], f_apic, c=c, lw=2)
        ax_f_middle.loglog(elec_x[:30], f_middle, c=c, lw=2)
        l, = ax_f_soma.loglog(elec_x[:30], f_soma, c=c, lw=2)
        lines.append(l)
        line_names.append('%d Hz' % plot_freqs[id])
    plt.subplot(333, title='Apical region', **ax_dict)
    plt.pcolormesh(elec_x[:30], freqs, F.reshape(3, 30, cut_idx_freq)[0].T, **im_dict)
    plt.colorbar(label='mV$^2$/Hz')
    fig.legend(lines, line_names, frameon=False, loc='lower center', ncol=len(plot_freqs))

    plt.subplot(336, title='Middle region', **ax_dict)
    plt.pcolormesh(elec_x[:30], freqs, F.reshape(3, 30, cut_idx_freq)[1].T, **im_dict)
    plt.colorbar(label='mV$^2$/Hz')

    plt.subplot(339, title='Soma region', xlabel='distance ($\mu$m)', ylabel='frequency (Hz)', **ax_dict)
    plt.pcolormesh(elec_x[:30], freqs, F.reshape(3, 30, cut_idx_freq)[2].T, **im_dict)
    plt.colorbar(label='mV$^2$/Hz')

    scale = 'log'
    cell_ax = fig.add_subplot(1, 3, 1, aspect=1, frameon=False, xticks=[], yticks=[])

    xmid = np.load(join(sim_folder, 'xmid_hay_generic.npy'))
    zmid = np.load(join(sim_folder, 'zmid_hay_generic.npy'))
    xstart = np.load(join(sim_folder, 'xstart_hay_generic.npy'))
    zstart = np.load(join(sim_folder, 'zstart_hay_generic.npy'))
    xend = np.load(join(sim_folder, 'xend_hay_generic.npy'))
    zend = np.load(join(sim_folder, 'zend_hay_generic.npy'))
    synidx = np.load(join(sim_folder, 'synidx_%s_00000.npy' % sim_name))

    [cell_ax.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]],
             lw=1.5, color='0.5', zorder=0, alpha=1)
        for idx in xrange(len(xmid))]
    [cell_ax.plot(xmid[idx], zmid[idx], 'g.', ms=4) for idx in synidx]

    ax_top = fig.add_subplot(3, 7, 3, xlim=[1, max_freq], xscale=scale, yscale=scale, aspect=1)
    ax_top_n = fig.add_subplot(3, 7, 4, xlim=[1, max_freq], xscale=scale, yscale=scale, aspect=1)
    ax_mid = fig.add_subplot(3, 7, 10, xlim=[1, max_freq], xscale=scale, yscale=scale, aspect=1)
    ax_mid_n = fig.add_subplot(3, 7, 11, xlim=[1, max_freq], xscale=scale, yscale=scale, aspect=1)
    ax_bottom = fig.add_subplot(3, 7, 17, xlim=[1, max_freq], xscale=scale, yscale=scale, aspect=1)
    ax_bottom_n = fig.add_subplot(3, 7, 18, xlim=[1, max_freq], xscale=scale, yscale=scale, aspect=1)

    all_elec_ax = [ax_top, ax_mid, ax_bottom]
    all_elec_n_ax = [ax_top_n, ax_mid_n, ax_bottom_n]
    cutoff_dist = 1000
    dist_clr = lambda dist: plt.cm.Greys(np.log(dist) / np.log(cutoff_dist))

    for elec in xrange(len(elec_z)):
        if elec_x[elec] > cutoff_dist:
            continue
        clr = dist_clr(params['electrode_parameters']['x'][elec])
        cell_ax.plot(params['electrode_parameters']['x'][elec], elec_z[elec], 'o', c=clr)
    
        row, col = _return_elec_row_col(elec, elec_x, elec_z)

        all_elec_ax[row].loglog(freqs, F[elec, :], color=clr, lw=1)
        all_elec_n_ax[row].loglog(freqs, F[elec, :] / np.max(F[elec, :]), color=clr, lw=1)

    [ax.grid(True) for ax in all_elec_ax + all_elec_n_ax]
    [ax.set_ylabel('mV$^2$/Hz') for ax in all_elec_ax]
    [ax.set_xlabel('Hz') for ax in all_elec_ax + all_elec_n_ax]
    simplify_axes(all_elec_ax + all_elec_n_ax + [ax_f_apic, ax_f_middle, ax_f_soma])
    for ax in all_elec_ax + all_elec_n_ax:
        max_exponent = np.ceil(np.log10(np.max([np.max(l.get_ydata()) for l in ax.get_lines()])))
        ax.set_ylim([10**(max_exponent - 4), 10**max_exponent])
    fig.suptitle(sim_name)
    fig.savefig(join(figure_folder, 'F_%s.png' % sim_name))


if __name__ == '__main__':
    from param_dicts import distributed_delta_params
    plot_all_shape_functions(distributed_delta_params)