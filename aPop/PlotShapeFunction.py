from __future__ import division
import param_dicts
import os
import sys
from os.path import join
from plotting_convention import *
import numpy as np
import pylab as plt
import LFPy
import tools
from NeuralSimulation import NeuralSimulation

def plot_all_shape_functions(params):
    cell_name = params['cell_name']
    input_type = params['input_type']
    if params['conductance_type'] is 'generic':
        secs = ['homogeneous', 'distal_tuft', 'basal']
        mus = [-0.5, 0.0, 2.0]
        dists = ['uniform', 'linear_increase', 'linear_decrease']

        for input_region in secs:
            for distribution in dists:
                for mu in mus:
                    print input_region, distribution, mu
                    params.update({'input_region': input_region,
                                   'cell_number': 0,
                                   'mu': mu,
                                    'distribution': distribution,
                                   })
                    ns = NeuralSimulation(**params)
                    sim_name = ns.population_sim_name
                    plot_F(params, sim_name)
    else:
        secs = ['homogeneous', 'distal_tuft', 'basal']
        conductance_types = ['active', 'passive', 'Ih_frozen']
        holding_potentials = [-80, -60]

        for input_sec in secs:
            for conductance_type in conductance_types:
                for holding_potential in holding_potentials:
                    sim_name = '%s_%s_%s_%s_%+d_%1.5fuS' % (cell_name, input_type, input_sec,
                                                          conductance_type, holding_potential,
                                                          params['syn_weight'])
                    plot_F(params, sim_name)



def _return_elec_row_col(elec_number, elec_x, elec_z):
    """ Return the subplot number for the distance study
    """
    num_elec_cols = len(set(elec_x))
    num_elec_rows = len(set(elec_z))
    elec_idxs = np.arange(len(elec_x)).reshape(num_elec_rows, num_elec_cols)
    row, col = np.array(np.where(elec_idxs == elec_number))[:, 0]
    return row, col


def return_F(r_star, r, F0):
    F = np.zeros(len(r))
    F[np.where(r < r_star)] = np.sqrt(r[0] / r[np.where(r < r_star)])
    F[np.where(r >= r_star)] = np.sqrt(r[0] / r_star) * (r_star / r[np.where(r >= r_star)])**2
    return F0 * F


def return_r_star(F, r):
    r_stars = np.linspace(10, 1000, 1000)
    errors = np.zeros(len(r_stars))
    for idx, r_star in enumerate(r_stars):
        F_approx = return_F(r_star, r, F[0])
        errors[idx] = np.sum(((F - F_approx)/F)**2)
    return r_stars[np.argmin(errors)]


def plot_F(params, sim_name):
    figure_folder = join(params['root_folder'], params['save_folder'])
    sim_folder = join(params['root_folder'], params['save_folder'], 'simulations')
    F = np.load(join(sim_folder, 'F_%s.npy' % sim_name))
    freqs = np.load(join(sim_folder, 'welch_freqs.npy'))
    max_freq = params['max_freq']
    cut_idx_freq = np.argmin(np.abs(freqs - max_freq)) + 1

    plot_freqs = [1, 15, 50, 100, 500]
    plot_freq_idxs = [np.argmin(np.abs(freqs - plot_freq)) for plot_freq in plot_freqs]

    F = F[:, :cut_idx_freq]
    freqs = freqs[:cut_idx_freq]

    elec_x = params['electrode_parameters']['x']
    elec_z = params['electrode_parameters']['z']
    im_dict = {'cmap': 'hot', 'norm': LogNorm(), 'vmin': 1e-10, 'vmax': 1.e-3}
    ax_dict = {'xscale': 'log', 'yscale': 'log', 'xlim': [10, 1000], 'ylim': [1, 500],
               }
    plt.close('all')

    fig = plt.figure(figsize=[18, 9])
    fig.subplots_adjust(right=0.97, left=0.05, hspace=0.5, bottom=0.1, wspace=0.5)
    plt.seed(123*0)

    f_clr = lambda idx: plt.cm.Blues(0.1 + idx / len(plot_freqs))

    ax_f_apic = fig.add_subplot(3, 8, 5, xlim=[10, 1000], ylim=[1e-10, 1e-3],
                                ylabel='$\mu$V$^2$/Hz', xlabel='distance ($\mu$m)')
    ax_f_middle = fig.add_subplot(3, 8, 13, xlim=[10, 1000], ylim=[1e-10, 1e-3],
                                  ylabel='$\mu$V$^2$/Hz', xlabel='distance ($\mu$m)')
    ax_f_soma = fig.add_subplot(3, 8, 21, xlim=[10, 1000], ylim=[1e-10, 1e-3],
                                ylabel='$\mu$V$^2$/Hz', xlabel='distance ($\mu$m)')

    ax_r_apic = fig.add_subplot(3, 8, 6, xlim=[1, 500], ylim=[10, 400],
                                ylabel='r$_*$', xlabel='frequency (Hz)')
    ax_r_middle = fig.add_subplot(3, 8, 14, xlim=[1, 500], ylim=[10, 400],
                                  ylabel='r$_*$', xlabel='frequency (Hz)')
    ax_r_soma = fig.add_subplot(3, 8, 22, xlim=[1, 500], ylim=[10, 400],
                                ylabel='r$_*$', xlabel='frequency (Hz)')

    lines = []
    line_names = []
    r = elec_x[:30]
    r_star_soma = []
    r_star_middle = []
    r_star_apic = []
    f = []
    for id in range(0, len(freqs), 10):
        f.append(freqs[id])
        f_apic = F.reshape(3, 30, cut_idx_freq)[0][:, id]
        f_middle = F.reshape(3, 30, cut_idx_freq)[1][:, id]
        f_soma = F.reshape(3, 30, cut_idx_freq)[2][:, id]
        r_star_apic.append(return_r_star(np.sqrt(f_apic), r))
        r_star_middle.append(return_r_star(np.sqrt(f_middle), r))
        r_star_soma.append(return_r_star(np.sqrt(f_soma), r))

    ax_r_apic.plot(f, r_star_apic, 'k')
    ax_r_middle.plot(f, r_star_middle, 'k')
    ax_r_soma.plot(f, r_star_soma, 'k')

    for id, f_idx in enumerate(plot_freq_idxs):
        c = f_clr(id)
        f_apic = F.reshape(3, 30, cut_idx_freq)[0][:, f_idx]
        f_middle = F.reshape(3, 30, cut_idx_freq)[1][:, f_idx]
        f_soma = F.reshape(3, 30, cut_idx_freq)[2][:, f_idx]

        r_star_approx_soma = return_r_star(np.sqrt(f_soma), r)
        F_approx_soma = return_F(r_star_approx_soma, r, np.sqrt(f_soma[0]))**2

        r_star_approx_middle = return_r_star(np.sqrt(f_middle), r)
        F_approx_middle = return_F(r_star_approx_middle, r, np.sqrt(f_middle[0]))**2

        r_star_approx_apic = return_r_star(np.sqrt(f_apic), r)
        F_approx_apic = return_F(r_star_approx_apic, r, np.sqrt(f_apic[0]))**2

        ax_f_apic.loglog(r, f_apic, c=c, lw=2)
        ax_f_apic.loglog(r, F_approx_apic, c='r', lw=1)
        ax_f_middle.loglog(r, f_middle, c=c, lw=2)
        ax_f_middle.loglog(r, F_approx_middle, c='r', lw=1)
        l, = ax_f_soma.loglog(r, f_soma, c=c, lw=2)
        ax_f_soma.loglog(r, F_approx_soma, c='r', lw=1)
        lines.append(l)
        line_names.append('%d Hz' % plot_freqs[id])
    plt.subplot(344, title='Apical region', **ax_dict)
    plt.pcolormesh(elec_x[:30], freqs, F.reshape(3, 30, cut_idx_freq)[0].T, **im_dict)
    plt.colorbar(label='$\mu$V$^2$/Hz')
    fig.legend(lines, line_names, frameon=False, loc='lower center', ncol=len(plot_freqs))

    plt.subplot(348, title='Middle region', **ax_dict)
    plt.pcolormesh(elec_x[:30], freqs, F.reshape(3, 30, cut_idx_freq)[1].T, **im_dict)
    plt.colorbar(label='$\mu$V$^2$/Hz')

    plt.subplot(3, 4, 12, title='Soma region', xlabel='distance ($\mu$m)', ylabel='frequency (Hz)', **ax_dict)
    plt.pcolormesh(elec_x[:30], freqs, F.reshape(3, 30, cut_idx_freq)[2].T, **im_dict)
    plt.colorbar(label='$\mu$V$^2$/Hz')

    scale = 'log'
    cell_ax = fig.add_subplot(1, 4, 1, aspect=1, frameon=False, xticks=[], yticks=[])
    if params['conductance_type'] is 'generic':
        xmid = np.load(join(sim_folder, 'xmid_hay_generic.npy'))
        zmid = np.load(join(sim_folder, 'zmid_hay_generic.npy'))
        xstart = np.load(join(sim_folder, 'xstart_hay_generic.npy'))
        zstart = np.load(join(sim_folder, 'zstart_hay_generic.npy'))
        xend = np.load(join(sim_folder, 'xend_hay_generic.npy'))
        zend = np.load(join(sim_folder, 'zend_hay_generic.npy'))
    else:
        xmid = np.load(join(sim_folder, 'xmid_hay_active.npy'))
        zmid = np.load(join(sim_folder, 'zmid_hay_active.npy'))
        xstart = np.load(join(sim_folder, 'xstart_hay_active.npy'))
        zstart = np.load(join(sim_folder, 'zstart_hay_active.npy'))
        xend = np.load(join(sim_folder, 'xend_hay_active.npy'))
        zend = np.load(join(sim_folder, 'zend_hay_active.npy'))        
    synidx = np.load(join(sim_folder, 'synidx_%s_00000.npy' % sim_name))

    [cell_ax.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]],
             lw=1.5, color='0.5', zorder=0, alpha=1)
        for idx in xrange(len(xmid))]
    [cell_ax.plot(xmid[idx], zmid[idx], 'g.', ms=4) for idx in synidx]

    ax_top = fig.add_subplot(3, 8, 3, xlim=[1, max_freq], xscale=scale, yscale=scale, aspect=1)
    ax_top_n = fig.add_subplot(3, 8, 4, xlim=[1, max_freq], xscale=scale, yscale=scale, aspect=1)
    ax_mid = fig.add_subplot(3, 8, 11, xlim=[1, max_freq], xscale=scale, yscale=scale, aspect=1)
    ax_mid_n = fig.add_subplot(3, 8, 12, xlim=[1, max_freq], xscale=scale, yscale=scale, aspect=1)
    ax_bottom = fig.add_subplot(3, 8, 19, xlim=[1, max_freq], xscale=scale, yscale=scale, aspect=1)
    ax_bottom_n = fig.add_subplot(3, 8, 20, xlim=[1, max_freq], xscale=scale, yscale=scale, aspect=1)

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
        all_elec_n_ax[row].loglog(freqs, F[elec, :] / np.max(F[elec, 1:]), color=clr, lw=1)

    [ax.grid(True) for ax in all_elec_ax + all_elec_n_ax]
    [ax.set_ylabel('$\mu$V$^2$/Hz') for ax in all_elec_ax]
    [ax.set_xlabel('Hz') for ax in all_elec_ax + all_elec_n_ax]
    simplify_axes(all_elec_ax + all_elec_n_ax + [ax_f_apic, ax_f_middle, ax_f_soma] + [ax_r_apic, ax_r_middle, ax_r_soma])
    for ax in all_elec_ax + all_elec_n_ax:
        max_exponent = np.ceil(np.log10(np.max([np.max(l.get_ydata()[1:]) for l in ax.get_lines()])))
        ax.set_ylim([10**(max_exponent - 4), 10**max_exponent])
    fig.suptitle(sim_name)
    fig.savefig(join(figure_folder, 'F_%s.png' % sim_name))


if __name__ == '__main__':
    from param_dicts import shape_function_params
    plot_all_shape_functions(shape_function_params)