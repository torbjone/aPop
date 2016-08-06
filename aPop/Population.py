from __future__ import division
import os
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg', warn=False)
    at_stallo = True
else:
    at_stallo = False
from plotting_convention import *
import pylab as plt
import numpy as np
import os
import sys
from os.path import join
from suppress_print import suppress_stdout_stderr
with suppress_stdout_stderr():
    import LFPy
from NeuralSimulation import NeuralSimulation
import tools
import scipy.fftpack as sf


def plot_leski_6(param_dict):

    conductance_names = {-0.5: 'regenerative',
                         0.0: 'passive-frozen',
                         2.0: 'restorative'}

    correlations = [0.0, 0.1, 1.0]
    mus = [-0.5, 0.0, 2.0]
    folder = join(param_dict['root_folder'], param_dict['save_folder'], 'simulations')
    pop_sizes = [50, 250, 500]
    for input_region in param_dict['input_regions']:
        for distribution in param_dict['distributions']:
            param_dict.update({'input_region': input_region,
                               'cell_number': 0,
                               'distribution': distribution,
                               'correlation': 0.0,
                               'mu': 0.0,
                              })

            plt.close('all')
            fig = plt.figure(figsize=(10, 10))
            fig.suptitle('Soma region LFP\nconductance: %s\ninput region: %s'
                         % (param_dict['distribution'], param_dict['input_region']))
            fig.subplots_adjust(right=0.95, wspace=0.5, hspace=0.5, left=0.2, top=0.85, bottom=0.2)

            psd_ax_dict = {'xlim': [1e0, 5e2],
                           'xlabel': 'Frequency (Hz)',
                           'xticks': [1e0, 10, 100],
                           'ylim': [1e-6, 1e1]}
            lines = None
            line_names = None
            num_plot_cols = 3

            elec = 60
            for idx, pop_size in enumerate(pop_sizes):
                fig.text(0.01, 0.8 - idx / (len(pop_sizes) + 1), 'R = %d $\mu$m' % pop_size)
                for c, correlation in enumerate(correlations):
                    param_dict['correlation'] = correlation
                    plot_number = idx * num_plot_cols + c
                    ax_tuft = fig.add_subplot(3, num_plot_cols, plot_number + 1, **psd_ax_dict)
                    ax_tuft.set_title('c = %1.1f' % correlation)
                    ax_tuft.grid(True)
                    lines = []
                    line_names = []
                    for mu in mus:
                        param_dict['mu'] = mu

                        ns = NeuralSimulation(**param_dict)

                        name = 'summed_signal_%s_%dum' % (ns.population_sim_name, pop_size)
                        lfp = np.load(join(folder, '%s.npy' % name))
                        freq, psd = tools.return_freq_and_psd(ns.timeres_python/1000., lfp[elec])
                        l, = ax_tuft.loglog(freq, psd[0], c=qa_clr_dict[mu], lw=3)

                        lines.append(l)
                        line_names.append(conductance_names[mu])

                        ax_tuft.set_ylabel('LFP-PSD ($\mu$V$^2$/Hz)', labelpad=-5)
                        ax_tuft.set_xticklabels(['', '1', '10', '100'])
                        ax_tuft.set_yticks(ax_tuft.get_yticks()[1:-1][::2])

            fig.legend(lines, line_names, loc='lower center', frameon=False, ncol=3)
            simplify_axes(fig.axes)
            plt.savefig(join(param_dict['root_folder'], param_dict['save_folder'],
                             'Figure_population_size_%s_%s.png' % (param_dict['distribution'], param_dict['input_region'])))
            plt.close('all')


def plot_leski_13(param_dict):

    input_region_clr = {'distal_tuft': 'lightgreen',
                        'homogeneous': 'darkgreen'}
    correlations = [0.0, 0.1, 1.0]
    folder = join(param_dict['root_folder'], param_dict['save_folder'], 'simulations')
    pop_size = 500
    mu = 2.0

    param_dict.update({'input_region': 'homogeneous',
                       'cell_number': 0,
                       'distribution': 'linear_increase',
                       'correlation': 0.0,
                       'mu': mu,
                      })
    ns = NeuralSimulation(**param_dict)
    synidx_homo = np.load(join(folder, 'synidx_%s.npy' % ns.sim_name))

    param_dict['input_region'] = 'distal_tuft'
    ns = NeuralSimulation(**param_dict)
    synidx_tuft = np.load(join(folder, 'synidx_%s.npy' % ns.sim_name))

    xmid = np.load(join(folder, 'xmid_%s_generic.npy' % param_dict['cell_name']))
    zmid = np.load(join(folder, 'zmid_%s_generic.npy' % param_dict['cell_name']))
    xstart = np.load(join(folder, 'xstart_%s_generic.npy' % param_dict['cell_name']))
    zstart = np.load(join(folder, 'zstart_%s_generic.npy' % param_dict['cell_name']))
    xend = np.load(join(folder, 'xend_%s_generic.npy' % param_dict['cell_name']))
    zend = np.load(join(folder, 'zend_%s_generic.npy' % param_dict['cell_name']))

    elec_x = param_dict['lateral_electrode_parameters']['x']
    elec_z = param_dict['lateral_electrode_parameters']['z']
    z = np.linspace(elec_z[0], elec_z[60], 4)

    input_regions = ['distal_tuft', 'homogeneous', 'basal']
    distributions = ['uniform', 'linear_increase', 'linear_decrease']

    plt.close('all')
    fig = plt.figure(figsize=(18, 9))
    # fig.suptitle(param_dict['distribution'] + ' conductance')
    fig.subplots_adjust(right=0.95, wspace=0.2, hspace=0.5, left=0.05, top=0.85, bottom=0.2)

    # ax_morph_homogeneous = fig.add_subplot(1, 8, 2, aspect=1, frameon=False, xticks=[], yticks=[])
    # [ax_morph_homogeneous.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]], lw=2,
    #                  c=cell_color, zorder=0) for idx in xrange(len(xmid))]
    # ax_morph_homogeneous.plot(xmid[synidx_homo], zmid[synidx_homo], '.', c=input_region_clr['homogeneous'], ms=4)

    psd_ax_dict = {#'ylim': [-200, 1200],
                    'xlim': [1e0, 5e2],
                    'yticks': [],
                   # 'xlabel': 'Frequency (Hz)',
                   'xticks': [1e0, 10, 100],
                   'xscale': 'log'}
    lines = None
    line_names = None
    num_plot_cols = 12
    num_plot_rows = 3
    im_dict = {'cmap': 'hot', 'norm': LogNorm(), 'vmin': 1e-4, 'vmax': 10.,}
    fig.text(0.17, 0.95, 'Uniform conductance')
    fig.text(0.45, 0.95, 'Linear increasing conductance')
    fig.text(0.75, 0.95, 'Linear decreasing conductance')

    fig.text(0.05, 0.75, 'Distal tuft\ninput', ha='center')
    fig.text(0.05, 0.5, 'Homogeneous\ninput', ha='center')
    fig.text(0.05, 0.25, 'Basal\ninput', ha='center')

    for i, input_region in enumerate(input_regions):
        for d, distribution in enumerate(distributions):
            for c, correlation in enumerate(correlations):
                # print input_region, distribution, correlation
                plot_number = i * (num_plot_cols) + d * (1 + len(distributions)) + c + 2
                # print i, d, c, plot_number

                ax = fig.add_subplot(num_plot_rows, num_plot_cols, plot_number, **psd_ax_dict)
                ax.set_title('c=%1.1f' % correlation)

                lines = []
                line_names = []

                param_dict['correlation'] = correlation
                param_dict['input_region'] = input_region
                param_dict['distribution'] = distribution
                ns = NeuralSimulation(**param_dict)
                name = 'summed_signal_%s_%dum' % (ns.population_sim_name, pop_size)
                lfp = np.load(join(folder, '%s.npy' % name))[[0, 30, 60], :]
                freq, psd = tools.return_freq_and_psd(ns.timeres_python/1000., lfp)

                # plt.close('all')
                # plt.pcolormesh(freq[1:], z, psd[:, 1:])
                # plt.show()

                img = ax.pcolormesh(freq[1:500], z, psd[:, 1:500], **im_dict)
                if i==2 and d == 0 and c == 0:
                    ax.set_xlabel('frequency (Hz)', labelpad=-0)
                    ax.set_ylabel('z')
                    c_ax = fig.add_axes([0.37, 0.2, 0.005, 0.17])
                    cbar = plt.colorbar(img, cax=c_ax, label='$\mu$V$^2$/Hz')
                # plt.axis('auto')
                # plt.colorbar(img)
                # l, = ax.loglog(freq, psd[0], c=input_region_clr[input_region], lw=3)
                # lines.append(l)
                # line_names.append(input_region)

        # ax_tuft.set_ylabel('LFP-PSD ($\mu$V$^2$/Hz)', labelpad=-5)
        # ax_tuft.set_xticklabels(['', '1', '10', '100'])
        # ax_tuft.set_yticks(ax_tuft.get_yticks()[1:-1][::2])

    # fig.legend(lines, line_names, loc='lower center', frameon=False, ncol=3)
    simplify_axes(fig.axes)
    # mark_subplots([ax_morph_homogeneous], 'B', ypos=1.1, xpos=0.1)
    plt.savefig(join(param_dict['root_folder'], param_dict['save_folder'], 'Figure_Leski_13.png'))
    plt.close('all')


def plot_figure_5(param_dict):

    input_region_clr = {'distal_tuft': 'lightgreen',
                        'homogeneous': 'darkgreen'}
    correlations = [0.0, 0.1, 1.0]
    folder = join(param_dict['root_folder'], param_dict['save_folder'], 'simulations')
    pop_size = 500
    mu = 2.0

    param_dict.update({'input_region': 'homogeneous',
                       'cell_number': 0,
                       'distribution': 'linear_increase',
                       'correlation': 0.0,
                       'mu': mu,
                      })
    ns = NeuralSimulation(**param_dict)
    synidx_homo = np.load(join(folder, 'synidx_%s.npy' % ns.sim_name))


    param_dict['input_region'] = 'distal_tuft'
    ns = NeuralSimulation(**param_dict)
    synidx_tuft = np.load(join(folder, 'synidx_%s.npy' % ns.sim_name))

    xmid = np.load(join(folder, 'xmid_%s_generic.npy' % param_dict['cell_name']))
    zmid = np.load(join(folder, 'zmid_%s_generic.npy' % param_dict['cell_name']))
    xstart = np.load(join(folder, 'xstart_%s_generic.npy' % param_dict['cell_name']))
    zstart = np.load(join(folder, 'zstart_%s_generic.npy' % param_dict['cell_name']))
    xend = np.load(join(folder, 'xend_%s_generic.npy' % param_dict['cell_name']))
    zend = np.load(join(folder, 'zend_%s_generic.npy' % param_dict['cell_name']))

    elec_x = param_dict['lateral_electrode_parameters']['x']
    elec_z = param_dict['lateral_electrode_parameters']['z']

    plt.close('all')
    fig = plt.figure(figsize=(10, 5))
    # fig.suptitle(param_dict['distribution'] + ' conductance')
    fig.subplots_adjust(right=0.95, wspace=0.6, hspace=0.5, left=0., top=0.85, bottom=0.2)

    ax_morph_distal_tuft = fig.add_subplot(1, 4, 1, aspect=1, frameon=False, xticks=[], yticks=[])
    ax_morph_homogeneous = fig.add_subplot(1, 8, 2, aspect=1, frameon=False, xticks=[], yticks=[])

    [ax_morph_distal_tuft.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]], lw=2,
                     c=cell_color, zorder=0) for idx in xrange(len(xmid))]
    ax_morph_distal_tuft.plot(xmid[synidx_tuft], zmid[synidx_tuft], '.', c=input_region_clr['distal_tuft'], ms=4)

    [ax_morph_homogeneous.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]], lw=2,
                     c=cell_color, zorder=0) for idx in xrange(len(xmid))]
    ax_morph_homogeneous.plot(xmid[synidx_homo], zmid[synidx_homo], '.', c=input_region_clr['homogeneous'], ms=4)

    psd_ax_dict = {'xlim': [1e0, 5e2],
                   'xlabel': 'Frequency (Hz)',
                   'xticks': [1e0, 10, 100],
                   'ylim': [1e-5, 1e1]}
    lines = None
    line_names = None
    num_plot_cols = 4

    for idx, elec in enumerate([0, 60]):
        ax_morph_distal_tuft.plot(elec_x[elec], elec_z[elec], 'o', c=elec_color, ms=15, mec='none')

        ax_morph_distal_tuft.arrow(elec_x[elec], elec_z[elec], 100, 0, color=elec_color, zorder=50,
                         lw=2, head_width=30, head_length=50, clip_on=False)

        ax_morph_homogeneous.plot(elec_x[elec], elec_z[elec], 'o', c=elec_color, ms=15, mec='none')

        ax_morph_homogeneous.arrow(elec_x[elec], elec_z[elec], 100, 0, color=elec_color, zorder=50,
                         lw=2, head_width=30, head_length=50, clip_on=False)


        for c, correlation in enumerate(correlations):
            param_dict['correlation'] = correlation
            plot_number = idx * num_plot_cols + c
            ax_tuft = fig.add_subplot(2, num_plot_cols, plot_number + 2, **psd_ax_dict)

            if idx == 0:
                ax_tuft.set_title('c = %1.1f' % correlation)
                # mark_subplots(ax_tuft, 'BCD'[c])

            lines = []
            line_names = []
            for i, input_region in enumerate(['distal_tuft', 'homogeneous']):
                param_dict['input_region'] = input_region
                ns = NeuralSimulation(**param_dict)
                name = 'summed_signal_%s_%dum' % (ns.population_sim_name, pop_size)
                lfp = np.load(join(folder, '%s.npy' % name))
                freq, psd = tools.return_freq_and_psd(ns.timeres_python/1000., lfp[elec])
                l, = ax_tuft.loglog(freq, psd[0], c=input_region_clr[input_region], lw=3)
                lines.append(l)
                line_names.append(input_region)

            ax_tuft.set_ylabel('LFP-PSD ($\mu$V$^2$/Hz)', labelpad=-5)
            ax_tuft.set_xticklabels(['', '1', '10', '100'])
            ax_tuft.set_yticks(ax_tuft.get_yticks()[1:-1][::2])

    fig.legend(lines, line_names, loc='lower center', frameon=False, ncol=3)
    simplify_axes(fig.axes)
    mark_subplots([ax_morph_distal_tuft], 'A', ypos=1.1, xpos=0.1)
    mark_subplots([ax_morph_homogeneous], 'B', ypos=1.1, xpos=0.1)
    plt.savefig(join(param_dict['root_folder'], param_dict['save_folder'], 'Figure_5.png'))
    plt.close('all')


def plot_figure_3(param_dict):

    conductance_names = {-0.5: 'regenerative',
                         0.0: 'passive-frozen',
                         2.0: 'restorative'}

    correlations = [0.0, 0.1, 1.0]
    mus = [-0.5, 0.0, 2.0]
    folder = join(param_dict['root_folder'], param_dict['save_folder'], 'simulations')
    pop_size = 500

    param_dict.update({'input_region': 'homogeneous',
                       'cell_number': 0,
                       'distribution': 'uniform',
                       'correlation': 0.0,
                       'mu': 0.0,
                      })
    ns = NeuralSimulation(**param_dict)

    xmid = np.load(join(folder, 'xmid_%s_generic.npy' % param_dict['cell_name']))
    zmid = np.load(join(folder, 'zmid_%s_generic.npy' % param_dict['cell_name']))
    xstart = np.load(join(folder, 'xstart_%s_generic.npy' % param_dict['cell_name']))
    zstart = np.load(join(folder, 'zstart_%s_generic.npy' % param_dict['cell_name']))
    xend = np.load(join(folder, 'xend_%s_generic.npy' % param_dict['cell_name']))
    zend = np.load(join(folder, 'zend_%s_generic.npy' % param_dict['cell_name']))

    synidx = np.load(join(folder, 'synidx_%s.npy' % ns.sim_name))

    elec_x = param_dict['lateral_electrode_parameters']['x']
    elec_z = param_dict['lateral_electrode_parameters']['z']

    plt.close('all')
    fig = plt.figure(figsize=(10, 5))
    fig.suptitle(param_dict['distribution'] + ' conductance')
    fig.subplots_adjust(right=0.95, wspace=0.5, hspace=0.5, left=0., top=0.85, bottom=0.2)

    ax_morph_1 = fig.add_subplot(1, 4, 1, aspect=1, frameon=False, xticks=[], yticks=[])

    [ax_morph_1.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]], lw=2,
                     c=cell_color, zorder=0) for idx in xrange(len(xmid))]
    ax_morph_1.plot(xmid[synidx], zmid[synidx], '.', c=syn_color, ms=4)

    psd_ax_dict = {'xlim': [1e0, 5e2],
                   'xlabel': 'Frequency (Hz)',
                   'xticks': [1e0, 10, 100],
                   'ylim': [1e-5, 1e0]}
    lines = None
    line_names = None
    num_plot_cols = 4

    for idx, elec in enumerate([0, 60]):
        ax_morph_1.plot(elec_x[elec], elec_z[elec], 'o', c=elec_color, ms=15, mec='none')

        ax_morph_1.arrow(elec_x[elec], elec_z[elec], 100, 0, color=elec_color, zorder=50,
                         lw=2, head_width=30, head_length=50, clip_on=False)

        for c, correlation in enumerate(correlations):
            param_dict['correlation'] = correlation
            plot_number = idx * num_plot_cols + c
            ax_tuft = fig.add_subplot(2, num_plot_cols, plot_number + 2, **psd_ax_dict)

            if idx == 0:
                ax_tuft.set_title('c = %1.1f' % correlation)
                # mark_subplots(ax_tuft, 'BCD'[c])

            lines = []
            line_names = []
            for mu in mus:
                param_dict['mu'] = mu

                ns = NeuralSimulation(**param_dict)
                name = 'summed_signal_%s_%dum' % (ns.population_sim_name, pop_size)
                lfp = np.load(join(folder, '%s.npy' % name))
                freq, psd = tools.return_freq_and_psd(ns.timeres_python/1000., lfp[elec])
                l, = ax_tuft.loglog(freq, psd[0], c=qa_clr_dict[mu], lw=3)
                lines.append(l)
                line_names.append(conductance_names[mu])

                ax_tuft.set_ylabel('LFP-PSD ($\mu$V$^2$/Hz)', labelpad=-5)
                ax_tuft.set_xticklabels(['', '1', '10', '100'])
                ax_tuft.set_yticks(ax_tuft.get_yticks()[1:-1][::2])

    fig.legend(lines, line_names, loc='lower center', frameon=False, ncol=3)
    simplify_axes(fig.axes)
    mark_subplots([ax_morph_1], 'B', ypos=1.1, xpos=0.1)
    plt.savefig(join(param_dict['root_folder'], param_dict['save_folder'], 'Figure_3B.png'))
    plt.close('all')


def plot_figure_2(param_dict):

    conductance_names = {-0.5: 'regenerative',
                         0.0: 'passive-frozen',
                         2.0: 'restorative'}

    correlations = [0.0, 0.1, 1.0]
    mus = [-0.5, 0.0, 2.0]
    folder = join(param_dict['root_folder'], param_dict['save_folder'], 'simulations')
    pop_size = 500

    param_dict.update({'input_region': 'distal_tuft',
                       'cell_number': 0,
                       'distribution': 'linear_increase',
                       'correlation': 0.0,
                       'mu': 0.0,
                      })
    ns = NeuralSimulation(**param_dict)

    xmid = np.load(join(folder, 'xmid_%s_generic.npy' % param_dict['cell_name']))
    zmid = np.load(join(folder, 'zmid_%s_generic.npy' % param_dict['cell_name']))
    xstart = np.load(join(folder, 'xstart_%s_generic.npy' % param_dict['cell_name']))
    zstart = np.load(join(folder, 'zstart_%s_generic.npy' % param_dict['cell_name']))
    xend = np.load(join(folder, 'xend_%s_generic.npy' % param_dict['cell_name']))
    zend = np.load(join(folder, 'zend_%s_generic.npy' % param_dict['cell_name']))

    synidx = np.load(join(folder, 'synidx_%s.npy' % ns.sim_name))

    elec_x = param_dict['lateral_electrode_parameters']['x']
    elec_z = param_dict['lateral_electrode_parameters']['z']

    plt.close('all')
    fig = plt.figure(figsize=(10, 5))
    fig.subplots_adjust(right=0.95, wspace=0.5, hspace=0.5, left=0., top=0.9, bottom=0.2)

    ax_morph_1 = fig.add_subplot(1, 4, 1, aspect=1, frameon=False, xticks=[], yticks=[])

    [ax_morph_1.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]], lw=2,
                     c=cell_color, zorder=0) for idx in xrange(len(xmid))]
    ax_morph_1.plot(xmid[synidx], zmid[synidx], '.', c=syn_color, ms=4)

    psd_ax_dict = {'xlim': [1e0, 5e2],
                   'xlabel': 'Frequency (Hz)',
                   'xticks': [1e0, 10, 100],
                   'ylim': [1e-6, 1e1]}
    lines = None
    line_names = None
    num_plot_cols = 4

    for idx, elec in enumerate([0, 60]):
        ax_morph_1.plot(elec_x[elec], elec_z[elec], 'o', c=elec_color, ms=15, mec='none')

        ax_morph_1.arrow(elec_x[elec], elec_z[elec], 100, 0, color=elec_color, zorder=50,
                         lw=2, head_width=30, head_length=50, clip_on=False)

        for c, correlation in enumerate(correlations):
            param_dict['correlation'] = correlation
            plot_number = idx * num_plot_cols + c
            ax_tuft = fig.add_subplot(2, num_plot_cols, plot_number + 2, **psd_ax_dict)

            if idx == 0:
                ax_tuft.set_title('c = %1.1f' % correlation)
                mark_subplots(ax_tuft, 'BCD'[c])

            lines = []
            line_names = []
            for mu in mus:
                param_dict['mu'] = mu

                ns = NeuralSimulation(**param_dict)
                name = 'summed_signal_%s_%dum' % (ns.population_sim_name, pop_size)
                lfp = np.load(join(folder, '%s.npy' % name))
                freq, psd = tools.return_freq_and_psd(ns.timeres_python/1000., lfp[elec])
                l, = ax_tuft.loglog(freq, psd[0], c=qa_clr_dict[mu], lw=3)
                lines.append(l)
                line_names.append(conductance_names[mu])

                ax_tuft.set_ylabel('LFP-PSD ($\mu$V$^2$/Hz)', labelpad=-5)
                ax_tuft.set_xticklabels(['', '1', '10', '100'])
                ax_tuft.set_yticks(ax_tuft.get_yticks()[1:-1][::2])

    fig.legend(lines, line_names, loc='lower center', frameon=False, ncol=3)
    simplify_axes(fig.axes)
    mark_subplots([ax_morph_1], ypos=0.95, xpos=0.1)
    plt.savefig(join(param_dict['root_folder'], param_dict['save_folder'], 'Figure_2.png'))
    plt.close('all')


def plot_figure_1(param_dict):

    conductance_names = {-0.5: 'regenerative',
                         0.0: 'passive-frozen',
                         2.0: 'restorative'}
    folder = join(param_dict['root_folder'], param_dict['save_folder'], 'simulations')
    pop_size = 500

    distribution_1 = 'linear_increase'
    correlation = 0.0

    lfp_1 = {}
    lfp_2 = {}
    param_dict_1 = param_dict.copy()
    input_region_1 = 'distal_tuft'
    param_dict_1.update({'input_region': input_region_1,
                             'cell_number': 0,
                             'distribution': distribution_1,
                             'correlation': correlation,
                             })
    param_dict_2 = param_dict.copy()
    distribution_2 = 'uniform'
    input_region_2 = 'homogeneous'
    param_dict_2.update({'input_region': input_region_2,
                         'cell_number': 0,
                         'distribution': distribution_2,
                         'correlation': correlation,
                         })
    ns_1 = None
    ns_2 = None
    for mu in [-0.5, 0.0, 2.0]:
        param_dict_1['mu'] = mu
        ns_1 = NeuralSimulation(**param_dict_1)
        name_res = 'summed_signal_%s_%dum' % (ns_1.population_sim_name, pop_size)
        lfp_1[mu] = np.load(join(folder, '%s.npy' % name_res))

        param_dict_2['mu'] = mu
        ns_2 = NeuralSimulation(**param_dict_2)
        name_res = 'summed_signal_%s_%dum' % (ns_2.population_sim_name, pop_size)
        lfp_2[mu] = np.load(join(folder, '%s.npy' % name_res))

    xmid = np.load(join(folder, 'xmid_%s_generic.npy' % param_dict_1['cell_name']))
    zmid = np.load(join(folder, 'zmid_%s_generic.npy' % param_dict_1['cell_name']))
    xstart = np.load(join(folder, 'xstart_%s_generic.npy' % param_dict_1['cell_name']))
    zstart = np.load(join(folder, 'zstart_%s_generic.npy' % param_dict_1['cell_name']))
    xend = np.load(join(folder, 'xend_%s_generic.npy' % param_dict_1['cell_name']))
    zend = np.load(join(folder, 'zend_%s_generic.npy' % param_dict_1['cell_name']))

    synidx_1 = np.load(join(folder, 'synidx_%s.npy' % ns_1.sim_name))
    synidx_2 = np.load(join(folder, 'synidx_%s.npy' % ns_2.sim_name))

    elec_x = param_dict_1['lateral_electrode_parameters']['x']
    elec_z = param_dict_1['lateral_electrode_parameters']['z']

    plt.close('all')
    fig = plt.figure(figsize=(7, 5))
    fig.subplots_adjust(right=0.95, wspace=0.3, hspace=0.4, left=0., top=0.9, bottom=0.2)

    ax_morph_1 = fig.add_subplot(1, 4, 1, aspect=1, frameon=False, xticks=[], yticks=[])

    [ax_morph_1.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]], lw=2,
                     c=cell_color, zorder=0) for idx in xrange(len(xmid))]
    ax_morph_1.plot(xmid[synidx_1], zmid[synidx_1], '.', c=syn_color, ms=4)

    ax_morph_2 = fig.add_subplot(1, 4, 3, aspect=1, frameon=False, xticks=[], yticks=[])
    [ax_morph_2.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]], lw=2,
                     c=cell_color, zorder=0)
        for idx in xrange(len(xmid))]
    ax_morph_2.plot(xmid[synidx_2], zmid[synidx_2], '.', c=syn_color, ms=4)

    psd_ax_dict = {'xlim': [1e0, 5e2],
                   'xlabel': 'Frequency (Hz)',
                   'xticks': [1e0, 10, 100],}
    lines = None
    line_names = None
    for idx, elec in enumerate([0, 60]):
        ax_morph_1.plot(elec_x[elec], elec_z[elec], 'o', c=elec_color, ms=15, mec='none')
        ax_morph_2.plot(elec_x[elec], elec_z[elec], 'o', c=elec_color, ms=15, mec='none')

        ax_morph_1.arrow(elec_x[elec], elec_z[elec], 100, 0, color=elec_color, zorder=50,
                         lw=2, head_width=30, head_length=50, clip_on=False)
        ax_morph_2.arrow(elec_x[elec], elec_z[elec], 100, 0, color=elec_color, zorder=50,
                         lw=2, head_width=30, head_length=50, clip_on=False)

        num_plot_cols = 4
        plot_number = idx * num_plot_cols
        ax_tuft = fig.add_subplot(2, num_plot_cols, plot_number + 2, ylim=[1e-6, 1e-1], **psd_ax_dict)
        ax_homo = fig.add_subplot(2, num_plot_cols, plot_number + 4, ylim=[1e-6, 1e-1], **psd_ax_dict)
        lines = []
        line_names = []
        for mu in [-0.5, 0.0, 2.0]:
            freq, psd_tuft_res = tools.return_freq_and_psd(ns_1.timeres_python/1000., lfp_1[mu][elec])
            ax_tuft.loglog(freq, psd_tuft_res[0], c=qa_clr_dict[mu], lw=3)
            ax_tuft.set_ylabel('LFP-PSD ($\mu$V$^2$/Hz)', labelpad=-5)
            ax_tuft.set_xticklabels(['', '1', '10', '100'])
            ax_tuft.set_yticks(ax_tuft.get_yticks()[1:-1][::2])

            freq, psd_homogeneous = tools.return_freq_and_psd(ns_2.timeres_python/1000., lfp_2[mu][elec])
            l, = ax_homo.loglog(freq, psd_homogeneous[0], c=qa_clr_dict[mu], lw=3, solid_capstyle='round')
            lines.append(l)
            line_names.append(conductance_names[mu])
            ax_homo.set_xticklabels(['', '1', '10', '100'])
            ax_homo.set_ylabel('LFP-PSD ($\mu$V$^2$/Hz)', labelpad=-5)
            ax_homo.set_yticks(ax_homo.get_yticks()[1:-1][::2])

    fig.legend(lines, line_names, loc='lower center', frameon=False, ncol=3)
    simplify_axes(fig.axes)
    mark_subplots([ax_morph_1, ax_morph_2], ypos=0.95, xpos=0.1)
    plt.savefig(join(param_dict_1['root_folder'], param_dict_1['save_folder'], 'Figure_1.png'))
    plt.close('all')


def plot_linear_combination(param_dict):

    folder = join(param_dict['root_folder'], param_dict['save_folder'], 'simulations')
    pop_size = 500

    # for input_region in param_dict['input_regions']:
    for distribution in param_dict['distributions']:
        for correlation in param_dict['correlations']:

            if (distribution != 'linear_increase'):# or (correlation != 0.0):
                continue

            param_dict['mu'] = 2.0

            input_region = 'distal_tuft'
            param_dict.update({'input_region': input_region,
                           'cell_number': 0,
                           'distribution': distribution,
                           'correlation': correlation,
                           })

            ns = NeuralSimulation(**param_dict)
            name_res = 'summed_signal_%s_%dum' % (ns.population_sim_name, pop_size)
            print name_res
            lfp_tuft = np.load(join(folder, '%s.npy' % name_res))

            # P_res, c_phi_res, F2_tuft = return_simple_model(param_dict, pop_size, ns)
            sim_folder = join(param_dict['root_folder'], 'shape_function', 'simulations')
            F2_name = 'F2_shape_function_%s_%s_%s_%s_%1.1f_%1.2f' % (param_dict['cell_name'], param_dict['input_region'], ns.conductance_type,
                                                            param_dict['distribution'], param_dict['mu'], 0.0)
            #sim_folder = join(param_dict['root_folder'], param_dict['save_folder'], 'simulations')
            #F2_name = 'F2_stick_shape_function_%s_%s_%s_%s_%1.1f_%1.2f' % (param_dict['cell_name'], param_dict['input_region'], ns.conductance_type,
            #                                                          param_dict['distribution'], param_dict['mu'], 0.0)

            F2_tuft = np.load(join(sim_folder, '%s.npy' % F2_name))

            input_region = 'homogeneous'
            param_dict.update({'input_region': input_region,
                           'cell_number': 0,
                           'distribution': distribution,
                           'correlation': correlation,
                           })

            ns = NeuralSimulation(**param_dict)
            name_res = 'summed_signal_%s_%dum' % (ns.population_sim_name, pop_size)
            lfp_homogeneous = np.load(join(folder, '%s.npy' % name_res))

            F2_name = 'F2_shape_function_%s_%s_%s_%s_%1.1f_%1.2f' % (param_dict['cell_name'], param_dict['input_region'], ns.conductance_type,
                                                            param_dict['distribution'], param_dict['mu'], 0.0)
            #sim_folder = join(param_dict['root_folder'], param_dict['save_folder'], 'simulations')
            #F2_name = 'F2_stick_shape_function_%s_%s_%s_%s_%1.1f_%1.2f' % (param_dict['cell_name'], param_dict['input_region'], ns.conductance_type,
            #                                                          param_dict['distribution'], param_dict['mu'], 0.0)

            F2_homogeneous = np.load(join(sim_folder, '%s.npy' % F2_name))
            # P_res, c_phi_res, F2_homogeneous = return_simple_model(param_dict, pop_size, ns)

            xmid = np.load(join(folder, 'xmid_%s_generic.npy' % param_dict['cell_name']))
            zmid = np.load(join(folder, 'zmid_%s_generic.npy' % param_dict['cell_name']))
            xstart = np.load(join(folder, 'xstart_%s_generic.npy' % param_dict['cell_name']))
            zstart = np.load(join(folder, 'zstart_%s_generic.npy' % param_dict['cell_name']))
            xend = np.load(join(folder, 'xend_%s_generic.npy' % param_dict['cell_name']))
            zend = np.load(join(folder, 'zend_%s_generic.npy' % param_dict['cell_name']))

            synidx = np.load(join(folder, 'synidx_%s.npy' % ns.sim_name))

            elec_x = param_dict['lateral_electrode_parameters']['x']
            elec_z = param_dict['lateral_electrode_parameters']['z']

            plt.close('all')
            fig = plt.figure(figsize=(10, 10))
            fig.subplots_adjust(right=0.99, wspace=0.5, hspace=0.3)

            fig.suptitle('Channel distribution: %s        Correlation: %1.1f' % (distribution, correlation))
            ax_morph = fig.add_subplot(1, 3, 1, aspect=1, frameon=False, xticks=[])

            [ax_morph.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]], lw=1.5,
                           color='0.5', zorder=0, alpha=1)
                for idx in xrange(len(xmid))]
            ax_morph.plot(xmid[synidx], zmid[synidx], '.', c='green', ms=4)

            for row, elec in enumerate([0, 30, 60]):
                ax_morph.plot(elec_x[elec], elec_z[elec], 'kD')

                row, col = return_elec_row_col(elec, elec_x, elec_z)
                num_plot_cols = 3
                plot_number = row * num_plot_cols + col/4 + 2
                ax = fig.add_subplot(3, num_plot_cols, plot_number, ylim=[1e-5, 1e1], xlim=[1e0, 5e2], title='LFP')
                freq, psd_tuft, phase_res = tools.return_freq_and_psd_and_phase(ns.timeres_python/1000., lfp_tuft[elec])
                freq, psd_homogeneous, phase_res = tools.return_freq_and_psd_and_phase(ns.timeres_python/1000., lfp_homogeneous[elec])
                ax.loglog(freq, psd_tuft[0], 'b', zorder=0)
                ax.loglog(freq, psd_homogeneous[0], 'r', zorder=0)

                ax = fig.add_subplot(3, num_plot_cols, plot_number + 1, ylim=[1e-8, 1e-3], xlim=[1e0, 5e2], title='Shape function')
                l1, = ax.loglog(freq, F2_tuft[elec], 'b', zorder=1)
                l2, = ax.loglog(freq, F2_homogeneous[elec], 'r', zorder=1)

            fig.legend([l1, l2], ['Tuft input', 'Homogeneous input'], loc='lower center', frameon=False, ncol=2)
            simplify_axes(fig.axes)
            plt.savefig(join(param_dict['root_folder'], param_dict['save_folder'], 'linear_combination_%s_%1.1f.png' % (ns.name, correlation)))
            plt.close('all')


def return_elec_row_col(elec_number, elec_x, elec_z):
    """ Return the subplot number for the distance study
    """
    num_elec_cols = len(set(elec_x))
    num_elec_rows = len(set(elec_z))
    elec_idxs = np.arange(len(elec_x)).reshape(num_elec_rows, num_elec_cols)
    row, col = np.array(np.where(elec_idxs == elec_number))[:, 0]
    return row, col


def plot_sig_to_elec_grid(x, y):

    top_idxs = np.arange(0, 30, 5)
    middle_idxs = np.arange(30, 60, 5)
    bottom_idxs = np.arange(60, 90, 5)

    for col in range(len(top_idxs)):
        plt.subplot(3, len(top_idxs), col + 1, ylim=[1e-4, 1e1])
        plt.loglog(x, y[top_idxs[col]])
        # plt.loglog(x2, y2[top_idxs[col]])

        plt.subplot(3, len(top_idxs), len(top_idxs) + col + 1, ylim=[1e-4, 1e1])
        plt.loglog(x, y[middle_idxs[col]])
        # plt.loglog(x2, y2[middle_idxs[col]])

        plt.subplot(3, len(top_idxs), len(top_idxs) * 2 + col + 1, ylim=[1e-4, 1e1])
        plt.loglog(x, y[bottom_idxs[col]])
        # plt.loglog(x2, y2[bottom_idxs[col]])


def plot_coherence(param_dict):

    folder = join(param_dict['root_folder'], param_dict['save_folder'], 'simulations')
    # pop_size = 500

    sim_folder = join(param_dict['root_folder'], 'shape_function', 'simulations')
    freqs = np.load(join(sim_folder, 'freqs.npy'))


    for input_region in param_dict['input_regions']:
        for distribution in param_dict['distributions']:
            for correlation in param_dict['correlations']:
                param_dict.update({'input_region': input_region,
                               'cell_number': 0,
                               'distribution': distribution,
                               'correlation': correlation,
                               })

                param_dict['mu'] = 0.0

                ns = NeuralSimulation(**param_dict)
                name = 'summed_signal_%s' % (ns.population_sim_name)
                c_phi = np.load(join(ns.sim_folder, 'c_phi_%s.npy' % ns.population_sim_name))

                param_dict['mu'] = -0.5
                ns = NeuralSimulation(**param_dict)
                c_phi_reg = np.load(join(ns.sim_folder, 'c_phi_%s.npy' % ns.population_sim_name))

                # name_reg = 'summed_signal_%s_%dum' % (ns.population_sim_name, pop_size)

                param_dict['mu'] = 2.0
                ns = NeuralSimulation(**param_dict)
                # name_res = 'summed_signal_%s_%dum' % (ns.population_sim_name, pop_size)
                c_phi_res = np.load(join(ns.sim_folder, 'c_phi_%s.npy' % ns.population_sim_name))

                plt.close('all')
                plt.figure(figsize=[19, 10])
                plt.suptitle('Coherence\n%s' % name)

                top_idxs = np.arange(0, 30, 5)
                middle_idxs = np.arange(30, 60, 5)
                bottom_idxs = np.arange(60, 90, 5)

                for col in range(len(top_idxs)):
                    plt.subplot(3, len(top_idxs), col + 1, ylim=[1e-5, 1e1])
                    plt.loglog(freqs, c_phi[top_idxs[col]], 'k')
                    plt.loglog(freqs, c_phi_reg[top_idxs[col]], 'r')
                    plt.loglog(freqs, c_phi_res[top_idxs[col]], 'b')
                    # plt.loglog(x2, y2[top_idxs[col]])

                    plt.subplot(3, len(top_idxs), len(top_idxs) + col + 1, ylim=[1e-5, 1e1])
                    plt.loglog(freqs, c_phi[middle_idxs[col]], 'k')
                    plt.loglog(freqs, c_phi_reg[middle_idxs[col]], 'r')
                    plt.loglog(freqs, c_phi_res[middle_idxs[col]], 'b')
                    # plt.loglog(x2, y2[middle_idxs[col]])

                    plt.subplot(3, len(top_idxs), len(top_idxs) * 2 + col + 1, ylim=[1e-5, 1e1])
                    plt.loglog(freqs, c_phi[bottom_idxs[col]], 'k')
                    plt.loglog(freqs, c_phi_reg[bottom_idxs[col]], 'r')
                    plt.loglog(freqs, c_phi_res[bottom_idxs[col]], 'b')
                    # plt.loglog(x2, y2[bottom_idxs[col]])

                plt.savefig(join(param_dict['root_folder'], param_dict['save_folder'], 'coherenece_%s.png' % name))

                plt.close('all')


def return_simple_model(param_dict, pop_size, ns):

    c_phi = np.load(join(ns.sim_folder, 'c_phi_%s.npy' % ns.population_sim_name))
    # sim_folder = join(param_dict['root_folder'], 'shape_function', 'simulations')
    # F2_name = 'F2_shape_function_%s_%s_%s_%s_%1.1f_%1.2f' % (param_dict['cell_name'], param_dict['input_region'], ns.conductance_type,
    #                                                         param_dict['distribution'], param_dict['mu'], 0.0)
    # r_star_soma_name = 'r_star_soma_shape_function_%s_%s_%s_%s_%1.1f_%1.2f' % (param_dict['cell_name'], param_dict['input_region'],
    #                                     ns.conductance_type, param_dict['distribution'], param_dict['mu'], 0.0)
    # r_star_middle_name = 'r_star_middle_shape_function_%s_%s_%s_%s_%1.1f_%1.2f' % (param_dict['cell_name'], param_dict['input_region'],
    #                                     ns.conductance_type, param_dict['distribution'], param_dict['mu'], 0.0)
    # r_star_apic_name = 'r_star_apic_shape_function_%s_%s_%s_%s_%1.1f_%1.2f' % (param_dict['cell_name'], param_dict['input_region'],
    #                                     ns.conductance_type, param_dict['distribution'], param_dict['mu'], 0.0)


    sim_folder = join(param_dict['root_folder'], param_dict['save_folder'], 'simulations')
    F2_name = 'F2_stick_shape_function_%s_%s_%s_%s_%1.1f_%1.2f' % (param_dict['cell_name'], param_dict['input_region'], ns.conductance_type,
                                                            param_dict['distribution'], param_dict['mu'], 0.0)
    r_star_soma_name = 'r_star_soma_stick_shape_function_%s_%s_%s_%s_%1.1f_%1.2f' % (param_dict['cell_name'], param_dict['input_region'],
                                        ns.conductance_type, param_dict['distribution'], param_dict['mu'], 0.0)
    r_star_middle_name = 'r_star_middle_stick_shape_function_%s_%s_%s_%s_%1.1f_%1.2f' % (param_dict['cell_name'], param_dict['input_region'],
                                        ns.conductance_type, param_dict['distribution'], param_dict['mu'], 0.0)
    r_star_apic_name = 'r_star_apic_stick_shape_function_%s_%s_%s_%s_%1.1f_%1.2f' % (param_dict['cell_name'], param_dict['input_region'],
                                        ns.conductance_type, param_dict['distribution'], param_dict['mu'], 0.0)

    F2 = np.load(join(sim_folder, '%s.npy' % F2_name))
    freqs = np.load(join(sim_folder, 'freqs.npy'))
    cut_idx_freq = np.argmin(np.abs(freqs - param_dict['max_freq'])) + 1

    r_star_soma = np.load(join(sim_folder, '%s.npy' % r_star_soma_name))
    r_star_middle = np.load(join(sim_folder, '%s.npy' % r_star_middle_name))
    r_star_apic = np.load(join(sim_folder, '%s.npy' % r_star_apic_name))

    r_star = np.array([r_star_apic, r_star_middle, r_star_soma])

    G0 = np.zeros(F2.shape)
    G1 = np.zeros(F2.shape)
    rho = 3000. * 1e-6
    r_e = param_dict['lateral_electrode_parameters']['x'][0]
    R = pop_size
    for r_idx, dist in enumerate(param_dict['lateral_electrode_parameters']['x']):
        row_idx = r_idx // 30

        for f_idx, freq in enumerate(freqs[:cut_idx_freq]):
            if R <= r_star[row_idx, f_idx]:
                G0[r_idx, f_idx] = F2[r_idx, f_idx] * rho * np.pi * r_e * (2 * R - r_e)
            else:
                G0[r_idx, f_idx] = F2[r_idx, f_idx] * rho * np.pi * r_e * (3*r_star[row_idx, f_idx] -
                                                                           r_e - r_star[row_idx, f_idx]**3 / R**2)

            if R <= r_star[row_idx, f_idx]:
                G1[r_idx, f_idx] = 1./9 * F2[r_idx, f_idx] * rho ** 2 * np.pi**2 * \
                                   (r_e**2 - 4 * r_e**0.5 * R**(3./2))**2
            else:
                G1[r_idx, f_idx] = 1./9 * F2[r_idx, f_idx] * rho ** 2 * np.pi**2 * r_e * \
                                   (r_e**(3./2) - (4 + 6. * np.log(R/r_star[row_idx, f_idx])) *
                                    r_star[row_idx, f_idx]**(3./2))**2
    P = (1. - c_phi[:, :]) * G0 + c_phi[:, :] * G1
    return P, c_phi, F2


def plot_generic_population_LFP(param_dict):

    folder = join(param_dict['root_folder'], param_dict['save_folder'], 'simulations')
    pop_size = 500

    for input_region in param_dict['input_regions']:
        for distribution in param_dict['distributions']:
            for correlation in param_dict['correlations']:
                param_dict.update({'input_region': input_region,
                               'cell_number': 0,
                               'distribution': distribution,
                               'correlation': correlation,
                               })

                param_dict['mu'] = 0.0

                ns = NeuralSimulation(**param_dict)
                name = 'summed_signal_%s_%dum' % (ns.population_sim_name, pop_size)
                # P = return_simple_model(param_dict, pop_size, ns)
                # c_phi = np.load(join(ns.sim_folder, 'c_phi_%s.npy' % ns.population_sim_name))

                param_dict['mu'] = -0.5
                ns = NeuralSimulation(**param_dict)
                # c_phi_reg = np.load(join(ns.sim_folder, 'c_phi_%s.npy' % ns.population_sim_name))

                name_reg = 'summed_signal_%s_%dum' % (ns.population_sim_name, pop_size)
                # P_reg = return_simple_model(param_dict, pop_size, ns)

                param_dict['mu'] = 2.0
                ns = NeuralSimulation(**param_dict)
                name_res = 'summed_signal_%s_%dum' % (ns.population_sim_name, pop_size)
                # P_res = return_simple_model(param_dict, pop_size, ns)
                # c_phi_res = np.load(join(ns.sim_folder, 'c_phi_%s.npy' % ns.population_sim_name))

                xmid = np.load(join(folder, 'xmid_%s_generic.npy' % ns.cell_name))
                zmid = np.load(join(folder, 'zmid_%s_generic.npy' % ns.cell_name))
                xstart = np.load(join(folder, 'xstart_%s_generic.npy' % ns.cell_name))
                zstart = np.load(join(folder, 'zstart_%s_generic.npy' % ns.cell_name))
                xend = np.load(join(folder, 'xend_%s_generic.npy' % ns.cell_name))
                zend = np.load(join(folder, 'zend_%s_generic.npy' % ns.cell_name))

                synidx = np.load(join(folder, 'synidx_%s.npy' % ns.sim_name))


                lfp = np.load(join(folder, '%s.npy' % name))
                lfp_reg = np.load(join(folder, '%s.npy' % name_reg))
                lfp_res = np.load(join(folder, '%s.npy' % name_res))

                elec_x = param_dict['lateral_electrode_parameters']['x']
                elec_z = param_dict['lateral_electrode_parameters']['z']

                plt.close('all')
                fig = plt.figure(figsize=(18, 10))
                fig.suptitle(name)
                ax_morph = fig.add_subplot(1, 6, 1, aspect=1, frameon=False, xticks=[])
                [ax_morph.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]], lw=1.5,
                               color='0.5', zorder=0, alpha=1)
                    for idx in xrange(len(xmid))]
                ax_morph.plot(xmid[synidx], zmid[synidx], '.', c='green', ms=4)
                fig.subplots_adjust(right=0.99, wspace=0.5, hspace=0.3)
                col_cut_off = 20

                for elec in range(lfp.shape[0]):

                    row, col = return_elec_row_col(elec, elec_x, elec_z)
                    if col % 4 != 0 or col > col_cut_off:
                        continue
                    num_plot_cols = col_cut_off/4 + 2
                    plot_number = row * num_plot_cols + col/4 + 2
                    ax = fig.add_subplot(3, num_plot_cols, plot_number, aspect=0.5, ylim=[1e-9, 1e1], xlim=[1e0, 5e2],
                                         title='%1.1f $\mu$m' % (elec_x[elec]))

                    freq, psd = tools.return_freq_and_psd(ns.timeres_python/1000., lfp[elec])
                    freq, psd_reg = tools.return_freq_and_psd(ns.timeres_python/1000.,  lfp_reg[elec])
                    freq, psd_res = tools.return_freq_and_psd(ns.timeres_python/1000., lfp_res[elec])

                    ax.loglog(freq, psd[0], 'k', zorder=0)
                    #ax.loglog(freq, P[elec], 'gray', zorder=1)
                    ax.loglog(freq, psd_reg[0], 'r', zorder=0)
                    #ax.loglog(freq, P_reg[elec], 'pink', zorder=1)
                    ax.loglog(freq, psd_res[0], 'b', zorder=0)
                    #ax.loglog(freq, P_res[elec], 'cyan', zorder=1)

                plt.savefig(join(param_dict['root_folder'], param_dict['save_folder'], '%s.png' % name))
                plt.close('all')
                #sys.exit()


def plot_classic_population_LFP(param_dict):

    folder = join(param_dict['root_folder'], param_dict['save_folder'], 'simulations')
    pop_size = 350

    for input_region in param_dict['input_regions']:
        for correlation in param_dict['correlations']:
            param_dict.update({'input_region': input_region,
                           'cell_number': 0,
                           'correlation': correlation,
                           })

            param_dict['conductance_type'] = 'active'

            ns = NeuralSimulation(**param_dict)
            name_active = 'summed_signal_%s_%dum' % (ns.population_sim_name, pop_size)
            # P = return_simple_model(param_dict, pop_size, ns)
            # c_phi = np.load(join(ns.sim_folder, 'c_phi_%s.npy' % ns.population_sim_name))

            param_dict['conductance_type'] = 'passive'
            ns = NeuralSimulation(**param_dict)
            name_passive = 'summed_signal_%s_%dum' % (ns.population_sim_name, pop_size)
            # c_phi_reg = np.load(join(ns.sim_folder, 'c_phi_%s.npy' % ns.population_sim_name))
            # P_reg = return_simple_model(param_dict, pop_size, ns)

            param_dict['conductance_type'] = 'Ih'
            ns = NeuralSimulation(**param_dict)
            name_Ih = 'summed_signal_%s_%dum' % (ns.population_sim_name, pop_size)
            # P_res = return_simple_model(param_dict, pop_size, ns)
            # c_phi_res = np.load(join(ns.sim_folder, 'c_phi_%s.npy' % ns.population_sim_name))

            param_dict['conductance_type'] = 'Ih_frozen'
            ns = NeuralSimulation(**param_dict)
            name_Ih_frozen = 'summed_signal_%s_%dum' % (ns.population_sim_name, pop_size)
            # P_res = return_simple_model(param_dict, pop_size, ns)
            # c_phi_res = np.load(join(ns.sim_folder, 'c_phi_%s.npy' % ns.population_sim_name))

            xmid = np.load(join(folder, 'xmid_hay_active.npy'))
            zmid = np.load(join(folder, 'zmid_hay_active.npy'))
            xstart = np.load(join(folder, 'xstart_hay_active.npy'))
            zstart = np.load(join(folder, 'zstart_hay_active.npy'))
            xend = np.load(join(folder, 'xend_hay_active.npy'))
            zend = np.load(join(folder, 'zend_hay_active.npy'))

            synidx = np.load(join(folder, 'synidx_%s.npy' % ns.sim_name))
            try:
                lfp_active = np.load(join(folder, '%s.npy' % name_active))
                lfp_passive = np.load(join(folder, '%s.npy' % name_passive))
                lfp_Ih = np.load(join(folder, '%s.npy' % name_Ih))
                lfp_Ih_frozen = np.load(join(folder, '%s.npy' % name_Ih_frozen))
            except IOError:
                continue
            elec_x = param_dict['lateral_electrode_parameters']['x']
            elec_z = param_dict['lateral_electrode_parameters']['z']

            plt.close('all')
            fig = plt.figure(figsize=(18, 10))
            fig.suptitle(name_active)
            ax_morph = fig.add_subplot(1, 6, 1, aspect=1, frameon=False, xticks=[])
            [ax_morph.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]], lw=1.5,
                           color='0.5', zorder=0, alpha=1)
                for idx in xrange(len(xmid))]
            ax_morph.plot(xmid[synidx], zmid[synidx], '.', c='green', ms=4)
            fig.subplots_adjust(right=0.99, wspace=0.5, hspace=0.3)
            col_cut_off = 20

            for elec in range(lfp_active.shape[0]):

                row, col = return_elec_row_col(elec, elec_x, elec_z)
                if col % 4 != 0 or col > col_cut_off:
                    continue
                num_plot_cols = col_cut_off/4 + 2
                plot_number = row * num_plot_cols + col/4 + 2
                ax = fig.add_subplot(3, num_plot_cols, plot_number, aspect=0.5, ylim=[1e-9, 1e1], xlim=[1e0, 5e2],
                                     title='%1.1f $\mu$m' % (elec_x[elec]))

                freq, psd_active = tools.return_freq_and_psd(ns.timeres_python/1000., lfp_active[elec])
                freq, psd_passive = tools.return_freq_and_psd(ns.timeres_python/1000.,  lfp_passive[elec])
                freq, psd_Ih = tools.return_freq_and_psd(ns.timeres_python/1000., lfp_Ih[elec])
                freq, psd_Ih_frozen = tools.return_freq_and_psd(ns.timeres_python/1000., lfp_Ih_frozen[elec])

                la, = ax.loglog(freq, psd_active[0], 'r', zorder=0, lw=2)
                #ax.loglog(freq, P[elec], 'gray', zorder=1)
                lk, = ax.loglog(freq, psd_passive[0], 'k', zorder=0, lw=2)
                #ax.loglog(freq, P_reg[elec], 'pink', zorder=1)
                lb, = ax.loglog(freq, psd_Ih[0], 'b', zorder=0, lw=2)
                lf, = ax.loglog(freq, psd_Ih_frozen[0], 'c', zorder=0, lw=2)
                #ax.loglog(freq, P_res[elec], 'cyan', zorder=1)
            fig.legend([la, lk, lb, lf], ['active', 'passive', 'passive + Ih',  'passive + frozen Ih'], frameon=False, loc='lower center', ncol=4)

            simplify_axes(fig.axes)

            plt.savefig(join(param_dict['root_folder'], param_dict['save_folder'], '%s.png' % name_active))
            plt.close('all')
            sys.exit()


def plot_simple_model_LFP(param_dict):

    folder = join(param_dict['root_folder'], param_dict['save_folder'], 'simulations')
    pop_size = 500

    for input_region in param_dict['input_regions']:
        for distribution in param_dict['distributions']:
            for correlation in param_dict['correlations']:

                if (input_region != 'top') or (distribution != 'increase'):# or (correlation != 0.0):
                    continue
                print input_region, distribution, correlation

                param_dict.update({'input_region': input_region,
                               'cell_number': 0,
                               'distribution': distribution,
                               'correlation': correlation,
                               })

                param_dict['mu'] = 0.0
                ns = NeuralSimulation(**param_dict)
                name = 'summed_signal_%s_%dum' % (ns.population_sim_name, pop_size)
                P, c_phi, F2 = return_simple_model(param_dict, pop_size, ns)

                param_dict['mu'] = -0.5
                ns = NeuralSimulation(**param_dict)
                name_reg = 'summed_signal_%s_%dum' % (ns.population_sim_name, pop_size)
                P_reg, c_phi_reg, F2_reg = return_simple_model(param_dict, pop_size, ns)

                param_dict['mu'] = 2.0
                ns = NeuralSimulation(**param_dict)
                name_res = 'summed_signal_%s_%dum' % (ns.population_sim_name, pop_size)
                P_res, c_phi_res, F2_res = return_simple_model(param_dict, pop_size, ns)

                xmid = np.load(join(folder, 'xmid_%s_generic.npy' % param_dict['cell_name']))
                zmid = np.load(join(folder, 'zmid_%s_generic.npy' % param_dict['cell_name']))
                xstart = np.load(join(folder, 'xstart_%s_generic.npy' % param_dict['cell_name']))
                zstart = np.load(join(folder, 'zstart_%s_generic.npy' % param_dict['cell_name']))
                xend = np.load(join(folder, 'xend_%s_generic.npy' % param_dict['cell_name']))
                zend = np.load(join(folder, 'zend_%s_generic.npy' % param_dict['cell_name']))

                synidx = np.load(join(folder, 'synidx_%s.npy' % ns.sim_name))

                lfp = np.load(join(folder, '%s.npy' % name))
                lfp_reg = np.load(join(folder, '%s.npy' % name_reg))
                lfp_res = np.load(join(folder, '%s.npy' % name_res))

                elec_x = param_dict['lateral_electrode_parameters']['x']
                elec_z = param_dict['lateral_electrode_parameters']['z']

                plt.close('all')
                fig = plt.figure(figsize=(18, 10))
                fig.subplots_adjust(right=0.99, wspace=0.5, hspace=0.3)

                fig.suptitle('Input: %s     Channel distribution: %s        Correlation: %1.1f' %
                             (input_region, distribution, correlation))
                ax_morph = fig.add_subplot(1, 5, 1, aspect=1, frameon=False, xticks=[])

                [ax_morph.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]], lw=1.5,
                               color='0.5', zorder=0, alpha=1)
                    for idx in xrange(len(xmid))]
                ax_morph.plot(xmid[synidx], zmid[synidx], '.', c='green', ms=4)

                average_over = 3

                for row, elec in enumerate([0, 30, 60]):
                    ax_morph.plot(elec_x[elec], elec_z[elec], 'kD')

                    row, col = return_elec_row_col(elec, elec_x, elec_z)
                    num_plot_cols = 6
                    plot_number = row * num_plot_cols + col/4 + 2
                    ax = fig.add_subplot(3, num_plot_cols, plot_number, ylim=[1e-5, 1e1], xlim=[1e0, 5e2],
                                         title='Population LFP')

                    freq, psd, phase = tools.return_freq_and_psd_and_phase(ns.timeres_python/1000., lfp[elec])
                    freq, psd_reg, phase_reg = tools.return_freq_and_psd_and_phase(ns.timeres_python/1000.,  lfp_reg[elec])
                    freq, psd_res, phase_res = tools.return_freq_and_psd_and_phase(ns.timeres_python/1000., lfp_res[elec])

                    # smooth_psd = smooth_signal(freq[::average_over], freq, psd[0])
                    # smooth_psd_reg = smooth_signal(freq[::average_over], freq, psd_reg[0])
                    # smooth_psd_res = smooth_signal(freq[::average_over], freq, psd_res[0])

                    ax.loglog(freq, psd[0], 'k', zorder=0)
                    # ax.loglog(freq[::average_over], smooth_psd, 'k', zorder=0)
                    ax.loglog(freq, psd_reg[0], 'r', zorder=0)
                    # ax.loglog(freq[::average_over], smooth_psd_reg, 'r', zorder=0)
                    ax.loglog(freq, psd_res[0], 'b', zorder=0)
                    # ax.loglog(freq[::average_over], smooth_psd_res, 'b', zorder=0)


                    ax = fig.add_subplot(3, num_plot_cols, plot_number + 1, ylim=[1e-5, 1e1], xlim=[1e0, 5e2],
                                         title='Simple model')

                    smooth_P = tools.smooth_signal(freq[::average_over], freq, P[elec])
                    smooth_P_reg = tools.smooth_signal(freq[::average_over], freq, P_reg[elec])
                    smooth_P_res = tools.smooth_signal(freq[::average_over], freq, P_res[elec])

                    ax.loglog(freq, P[elec], 'k', zorder=1)
                    ax.loglog(freq, P_reg[elec], 'r', zorder=1)
                    ax.loglog(freq, P_res[elec], 'b', zorder=1)

                    # ax.loglog(freq[::average_over], smooth_P, 'k', zorder=1)
                    # ax.loglog(freq[::average_over], smooth_P_reg, 'r', zorder=1)
                    # ax.loglog(freq[::average_over], smooth_P_res, 'b', zorder=1)

                    ax = fig.add_subplot(3, num_plot_cols, plot_number + 2, ylim=[1e-5, 1e1], xlim=[1e0, 5e2],
                                         title='Coherence')

                    smooth_c = tools.smooth_signal(freq[::average_over], freq, c_phi[elec])
                    smooth_c_reg = tools.smooth_signal(freq[::average_over], freq, c_phi_reg[elec])
                    smooth_c_res = tools.smooth_signal(freq[::average_over], freq, c_phi_res[elec])

                    # ax.loglog(freq, c_phi[elec], 'k', zorder=1)
                    # ax.loglog(freq, c_phi_reg[elec], 'r', zorder=1)
                    # ax.loglog(freq, c_phi_res[elec], 'b', zorder=1)

                    ax.loglog(freq[::average_over], smooth_c, 'k', zorder=1)
                    ax.loglog(freq[::average_over], smooth_c_reg, 'r', zorder=1)
                    ax.loglog(freq[::average_over], smooth_c_res, 'b', zorder=1)

                    ax = fig.add_subplot(3, num_plot_cols, plot_number + 3, ylim=[1e-8, 1e-3], xlim=[1e0, 5e2],
                                         title='Shape function')
                    ax.loglog(freq, F2[elec], 'k', zorder=1)
                    ax.loglog(freq, F2_reg[elec], 'r', zorder=1)
                    ax.loglog(freq, F2_res[elec], 'b', zorder=1)

                    ax = fig.add_subplot(3, num_plot_cols, plot_number + 4, xlim=[1e0, 50], title='Phase')
                    ax.semilogx(freq, phase[0], 'k', zorder=1)
                    ax.semilogx(freq, phase_reg[0], 'r', zorder=1)
                    ax.semilogx(freq, phase_res[0], 'b', zorder=1)

                simplify_axes(fig.axes)
                plt.savefig(join(param_dict['root_folder'], param_dict['save_folder'], 'simple_model_%s.png' % name))
                plt.close('all')


def plot_LFP_time_trace(param_dict):

    param_dict.update({'input_region': 'distal_tuft',
                       'cell_number': 0,
                       'mu': 0.0,
                       'distribution': 'linear_increase',
                       'correlation': 1.0,
                       })
    folder = join(param_dict['root_folder'], param_dict['save_folder'], 'simulations')
    pop_size = 500

    param_dict['mu'] = 0.0
    ns = NeuralSimulation(**param_dict)
    name = 'summed_signal_%s_%dum' % (ns.population_sim_name, pop_size)

    param_dict['mu'] = -0.5
    ns = NeuralSimulation(**param_dict)

    name_reg = 'summed_signal_%s_%dum' % (ns.population_sim_name, pop_size)

    param_dict['mu'] = 2.0
    ns = NeuralSimulation(**param_dict)
    name_res = 'summed_signal_%s_%dum' % (ns.population_sim_name, pop_size)

    lfp = np.load(join(folder, '%s.npy' % name))
    lfp_reg = np.load(join(folder, '%s.npy' % name_reg))
    lfp_res = np.load(join(folder, '%s.npy' % name_res))
    tvec = np.arange(ns.num_tsteps) * ns.timeres_python

    elecs = np.array([0, 30, 60])
    plt.subplot(311, xlim=[000, 1000])
    plt.plot(tvec, lfp[0] - lfp[0, 0], 'k')
    plt.plot(tvec, lfp_reg[0] - lfp_reg[0, 0], 'r')
    plt.plot(tvec, lfp_res[0] - lfp_res[0, 0], 'b')

    plt.subplot(312, xlim=[00, 1000])
    plt.plot(tvec, lfp[30] - lfp[30, 0], 'k')
    plt.plot(tvec, lfp_reg[30] - lfp_reg[30, 0], 'r')
    plt.plot(tvec, lfp_res[30] - lfp_res[30, 0], 'b')

    plt.subplot(313, xlim=[00, 1000])
    plt.plot(tvec, lfp[60] - lfp[60, 0], 'k')
    plt.plot(tvec, lfp_reg[60] - lfp_reg[60, 0], 'r')
    plt.plot(tvec, lfp_res[60] - lfp_res[60, 0], 'b')

    plt.show()


def initialize_population(param_dict):
    print "Initializing cell positions and rotations ..."
    x_y_z_rot = np.zeros((param_dict['num_cells'], 4))
    for cell_number in range(param_dict['num_cells']):
        plt.seed(123 * cell_number)
        rotation = 2*np.pi*np.random.random()
        z = param_dict['layer_5_thickness'] * (np.random.random() - 0.5)
        x, y = 2 * (np.random.random(2) - 0.5) * param_dict['population_radius']
        while np.sqrt(x**2 + y**2) > param_dict['population_radius']:
            x, y = 2 * (np.random.random(2) - 0.5) * param_dict['population_radius']
        x_y_z_rot[cell_number, :] = [x, y, z, rotation]
    r = np.array([np.sqrt(d[0]**2 + d[1]**2) for d in x_y_z_rot])
    argsort = np.argsort(r)
    x_y_z_rot = x_y_z_rot[argsort]

    np.save(join(param_dict['root_folder'], param_dict['save_folder'],
                 'x_y_z_rot_%s.npy' % param_dict['name']), x_y_z_rot)

    print "Initializing input spike trains"
    plt.seed(123)
    num_synapses = param_dict['num_synapses']
    firing_rate = param_dict['input_firing_rate']
    num_trains = int(num_synapses/0.01)
    all_spiketimes = {}
    input_method = LFPy.inputgenerators.stationary_poisson
    for idx in xrange(num_trains):
        all_spiketimes[idx] = input_method(1, firing_rate, -param_dict['cut_off'], param_dict['end_t'])[0]
    np.save(join(param_dict['root_folder'], param_dict['save_folder'],
                         'all_spike_trains_%s.npy' % param_dict['name']), all_spiketimes)


def plot_population(param_dict):
    print "Plotting population"
    x_y_z_rot = np.load(join(param_dict['root_folder'], param_dict['save_folder'],
                         'x_y_z_rot_%s.npy' % param_dict['name']))
    all_spiketimes = np.load(join(param_dict['root_folder'], param_dict['save_folder'],
                         'all_spike_trains_%s.npy' % param_dict['name'])).item()

    plt.subplot(221, xlabel='x', ylabel='y', aspect=1, frameon=False)
    plt.scatter(x_y_z_rot[:, 0], x_y_z_rot[:, 1], edgecolor='none', s=2)

    plt.subplot(222, xlabel='x', ylabel='z', aspect=1, frameon=False)
    plt.scatter(x_y_z_rot[:, 0], x_y_z_rot[:, 2], edgecolor='none', s=2)

    plt.subplot(212, ylabel='Cell number', xlabel='Time', title='Raster plot', frameon=False)
    for idx in range(200):
        plt.scatter(all_spiketimes[idx], np.ones(len(all_spiketimes[idx])) * idx, edgecolors='none', s=4)

    plt.savefig(join(param_dict['root_folder'], param_dict['save_folder'],
                     'population_%s.png' % param_dict['name']))


def sum_one_population(param_dict, num_cells, num_tsteps):

    x_y_z_rot = np.load(os.path.join(param_dict['root_folder'], param_dict['save_folder'],
                             'x_y_z_rot_%s.npy' % param_dict['name']))
    summed_lateral_sig = np.zeros((len(param_dict['lateral_electrode_parameters']['x']), num_tsteps))
    summed_center_sig = np.zeros((len(param_dict['center_electrode_parameters']['x']), num_tsteps))
    summed_center_sig_half_density = np.zeros((len(param_dict['center_electrode_parameters']['x']), num_tsteps))
    # Cells are numbered with increasing distance from population center. We exploit this
    subpopulation_idx = 0
    ns = None
    pop_radius = 0

    sample_freq = sf.fftfreq(num_tsteps, d=param_dict['timeres_python'] / 1000.)
    pidxs = np.where(sample_freq >= 0)
    # freqs = sample_freq[pidxs]
    xfft_norm_sum_lateral = np.zeros((len(param_dict['lateral_electrode_parameters']['x']), 
                                      len(pidxs[0])), dtype=complex)
    xfft_norm_sum_center = np.zeros((len(param_dict['center_electrode_parameters']['x']), 
                                     len(pidxs[0])), dtype=complex)

    for cell_number in xrange(num_cells):
        param_dict.update({'cell_number': cell_number})

        r = np.sqrt(x_y_z_rot[cell_number, 0]**2 + x_y_z_rot[cell_number, 1]**2)
        pop_radius = param_dict['population_radii'][subpopulation_idx]

        ns = NeuralSimulation(**param_dict)
        sim_name = ns.sim_name
        lateral_lfp = 1000 * np.load(join(ns.sim_folder, 'lateral_sig_%s.npy' % sim_name))  # uV
        center_lfp = 1000 * np.load(join(ns.sim_folder, 'center_sig_%s.npy' % sim_name))  # uV
        s_lateral = sf.fft(lateral_lfp, axis=1)[:, pidxs[0]]
        s_center = sf.fft(center_lfp, axis=1)[:, pidxs[0]]
        s_norm_lateral = s_lateral / np.abs(s_lateral)
        s_norm_center = s_center / np.abs(s_center)

        # freqs, lfp_psd = tools.return_freq_and_psd_welch(lfp[:, :], ns.welch_dict)
        if not summed_lateral_sig.shape == lateral_lfp.shape:
            print "Reshaping LFP time signal", lateral_lfp.shape
            summed_lateral_sig = np.zeros(lateral_lfp.shape)

        if not xfft_norm_sum_lateral.shape == s_norm_lateral.shape:
            print "Reshaping coherence signal", xfft_norm_sum_lateral.shape, s_norm_lateral.shape
            xfft_norm_sum_lateral = np.zeros(s_norm_lateral.shape)

        if not summed_center_sig.shape == center_lfp.shape:
            print "Reshaping LFP time signal", center_lfp.shape
            summed_center_sig = np.zeros(center_lfp.shape)

        if not xfft_norm_sum_center.shape == s_norm_center.shape:
            print "Reshaping coherence signal", xfft_norm_sum_center.shape, s_norm_center.shape
            xfft_norm_sum_center = np.zeros(s_norm_center.shape)

        xfft_norm_sum_lateral[:] += s_norm_lateral[:]
        xfft_norm_sum_center[:] += s_norm_center[:]

        if r > pop_radius:
            print "Saving population of radius ", pop_radius
            print "r(-1): ", np.sqrt(x_y_z_rot[cell_number - 1, 0]**2 + x_y_z_rot[cell_number - 1, 1]**2)
            print "r: ", r
            np.save(join(ns.sim_folder, 'summed_lateral_signal_%s_%dum.npy' %
                         (ns.population_sim_name, pop_radius)), summed_lateral_sig)
            np.save(join(ns.sim_folder, 'summed_center_signal_%s_%dum.npy' %
                         (ns.population_sim_name, pop_radius)), summed_center_sig)
            np.save(join(ns.sim_folder, 'summed_center_signal_half_density_%s_%dum.npy' %
                         (ns.population_sim_name, pop_radius)), summed_center_sig_half_density)
            subpopulation_idx += 1
        summed_lateral_sig += lateral_lfp
        summed_center_sig += center_lfp
        if not cell_number % 2:
            summed_center_sig_half_density += center_lfp

    print "Saving population of radius ", pop_radius
    np.save(join(ns.sim_folder, 'summed_lateral_signal_%s_%dum.npy' %
                 (ns.population_sim_name, pop_radius)), summed_lateral_sig)
    np.save(join(ns.sim_folder, 'summed_center_signal_%s_%dum.npy' %
                 (ns.population_sim_name, pop_radius)), summed_center_sig)
    np.save(join(ns.sim_folder, 'summed_center_signal_half_density_%s_%dum.npy' %
                 (ns.population_sim_name, pop_radius)), summed_center_sig_half_density)
    c_phi_lateral = (np.abs(xfft_norm_sum_lateral) ** 2 - num_cells) / (num_cells * (num_cells - 1))
    c_phi_center = (np.abs(xfft_norm_sum_center) ** 2 - num_cells) / (num_cells * (num_cells - 1))
    np.save(join(ns.sim_folder, 'c_phi_lateral_%s.npy' % ns.population_sim_name), c_phi_lateral)
    np.save(join(ns.sim_folder, 'c_phi_center_%s.npy' % ns.population_sim_name), c_phi_center)


def sum_population_generic(param_dict):

    x_y_z_rot = np.load(os.path.join(param_dict['root_folder'], param_dict['save_folder'],
                         'x_y_z_rot_%s.npy' % param_dict['name']))
    # print param_dict['population_radii']
    num_cells = 2000
    num_tsteps = round(param_dict['end_t']/param_dict['timeres_python'] + 1)
    for input_region in param_dict['input_regions']:
        param_dict.update({'input_region': input_region})
        for distribution in param_dict['distributions']:
            param_dict.update({'distribution': distribution})
            for mu in param_dict['mus']:
                param_dict.update({'mu': mu})
                for correlation in param_dict['correlations']:
                    param_dict.update({'correlation': correlation})
                    print input_region, distribution, mu, correlation
                    sum_one_population(param_dict, num_cells, num_tsteps)


def sum_population_mpi_generic(param_dict):
    """ Run with
        mpirun -np 4 python example_mpi.py
    """
    from mpi4py import MPI

    class Tags:
        def __init__(self):
            self.READY = 0
            self.DONE = 1
            self.EXIT = 2
            self.START = 3
            self.ERROR = 4
    tags = Tags()
    # Initializations and preliminaries
    comm = MPI.COMM_WORLD   # get MPI communicator object
    size = comm.size        # total number of processes
    rank = comm.rank        # rank of this process
    status = MPI.Status()   # get MPI status object
    num_workers = size - 1
    num_cells = 2000 if at_stallo else 5
    num_tsteps = int(round(param_dict['end_t']/param_dict['timeres_python'] + 1))

    if size == 1:
        print "Can't do MPI with one core!"
        sys.exit()

    if rank == 0:

        print("\033[95m Master starting with %d workers\033[0m" % num_workers)

        task = 0
        num_tasks = (len(param_dict['input_regions']) * len(param_dict['mus']) *
                     len(param_dict['distributions']) * len(param_dict['correlations']))

        for input_region in param_dict['input_regions']:
            param_dict.update({'input_region': input_region})
            for distribution in param_dict['distributions']:
                param_dict.update({'distribution': distribution})
                for mu in param_dict['mus']:
                    param_dict.update({'mu': mu})
                    for correlation in param_dict['correlations']:
                        param_dict.update({'correlation': correlation})
                        print input_region, distribution, mu, correlation
                        task += 1
                        sent = False
                        while not sent:
                            data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                            source = status.Get_source()
                            tag = status.Get_tag()
                            if tag == tags.READY:
                                comm.send([param_dict], dest=source, tag=tags.START)
                                print "\033[95m Sending task %d/%d to worker %d\033[0m" % (task, num_tasks, source)
                                sent = True
                            elif tag == tags.DONE:
                                print "\033[95m Worker %d completed task %d/%d\033[0m" % (source, task, num_tasks)
                            elif tag == tags.ERROR:
                                print "\033[91mMaster detected ERROR at node %d. Aborting...\033[0m" % source
                                for worker in range(1, num_workers + 1):
                                    comm.send([None], dest=worker, tag=tags.EXIT)
                                sys.exit()

        for worker in range(1, num_workers + 1):
            comm.send([None], dest=worker, tag=tags.EXIT)
        print("\033[95m Master finishing\033[0m")
    else:
        while True:
            comm.send(None, dest=0, tag=tags.READY)
            [param_dict] = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()
            if tag == tags.START:
                # Do the work here
                # try:
                sum_one_population(param_dict, num_cells, num_tsteps)
                # except:
                #     print "\033[91mNode %d exiting with ERROR\033[0m" % rank
                #     comm.send(None, dest=0, tag=tags.ERROR)
                #     sys.exit()
                comm.send(None, dest=0, tag=tags.DONE)
            elif tag == tags.EXIT:
                print "\033[93m%d exiting\033[0m" % rank
                break
        comm.send(None, dest=0, tag=tags.EXIT)


def sum_population_mpi_classic(param_dict):
    """ Run with
        mpirun -np 4 python example_mpi.py
    """
    from mpi4py import MPI

    class Tags:
        def __init__(self):
            self.READY = 0
            self.DONE = 1
            self.EXIT = 2
            self.START = 3
            self.ERROR = 4
    tags = Tags()
    # Initializations and preliminaries
    comm = MPI.COMM_WORLD   # get MPI communicator object
    size = comm.size        # total number of processes
    rank = comm.rank        # rank of this process
    status = MPI.Status()   # get MPI status object
    num_workers = size - 1
    num_cells = 2000
    num_tsteps = int(round(param_dict['end_t']/param_dict['timeres_python'] + 1))

    if size == 1:
        print "Can't do MPI with one core!"
        sys.exit()

    if rank == 0:

        print("\033[95m Master starting with %d workers\033[0m" % num_workers)

        task = 0
        num_tasks = (len(param_dict['input_regions']) * len(param_dict['holding_potentials']) *
                     len(param_dict['conductance_types']) * len(param_dict['correlations']))

        # print param_dict['population_radii']
        for input_region in param_dict['input_regions']:
            param_dict.update({'input_region': input_region})
            for conductance_type in param_dict['conductance_types']:
                param_dict.update({'conductance_type': conductance_type})
                for correlation in param_dict['correlations']:
                    param_dict.update({'correlation': correlation})
                    for holding_potential in param_dict['holding_potentials']:
                        param_dict.update({'holding_potential': holding_potential})
                        print input_region, conductance_type, holding_potential, correlation
                        task += 1
                        sent = False
                        while not sent:
                            data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                            source = status.Get_source()
                            tag = status.Get_tag()
                            if tag == tags.READY:
                                comm.send([param_dict], dest=source, tag=tags.START)
                                print "\033[95m Sending task %d/%d to worker %d\033[0m" % (task, num_tasks, source)
                                sent = True
                            elif tag == tags.DONE:
                                print "\033[95m Worker %d completed task %d/%d\033[0m" % (source, task, num_tasks)
                            elif tag == tags.ERROR:
                                print "\033[91mMaster detected ERROR at node %d. Aborting...\033[0m" % source
                                for worker in range(1, num_workers + 1):
                                    comm.send([None], dest=worker, tag=tags.EXIT)
                                sys.exit()

        for worker in range(1, num_workers + 1):
            comm.send([None], dest=worker, tag=tags.EXIT)
        print("\033[95m Master finishing\033[0m")
    else:
        while True:
            comm.send(None, dest=0, tag=tags.READY)
            [param_dict] = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()
            if tag == tags.START:
                # Do the work here
                # try:
                sum_one_population(param_dict, num_cells, num_tsteps)
                # except:
                #     print "\033[91mNode %d exiting with ERROR\033[0m" % rank
                #     comm.send(None, dest=0, tag=tags.ERROR)
                #     sys.exit()
                comm.send(None, dest=0, tag=tags.DONE)
            elif tag == tags.EXIT:
                print "\033[93m%d exiting\033[0m" % rank
                break
        comm.send(None, dest=0, tag=tags.EXIT)


def PopulationMPIgeneric():
    """ Run with
        mpirun -np 4 python example_mpi.py
    """
    from mpi4py import MPI

    class Tags:
        def __init__(self):
            self.READY = 0
            self.DONE = 1
            self.EXIT = 2
            self.START = 3
            self.ERROR = 4
    tags = Tags()
    # Initializations and preliminaries
    comm = MPI.COMM_WORLD   # get MPI communicator object
    size = comm.size        # total number of processes
    rank = comm.rank        # rank of this process
    status = MPI.Status()   # get MPI status object
    num_workers = size - 1

    if size == 1:
        print "Can't do MPI with one core!"
        sys.exit()

    if rank == 0:

        print("\033[95m Master starting with %d workers\033[0m" % num_workers)
        task = 0
        num_cells = 2000 if at_stallo else 1
        num_tasks = (len(param_dict['input_regions']) * len(param_dict['mus']) *
                     len(param_dict['distributions']) * len(param_dict['correlations']) * (num_cells))

        for input_region in param_dict['input_regions']:
            for distribution in param_dict['distributions']:
                for mu in param_dict['mus']:
                    for correlation in param_dict['correlations']:
                        for cell_idx in xrange(0, num_cells):
                            task += 1
                            sent = False
                            while not sent:
                                data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                                source = status.Get_source()
                                tag = status.Get_tag()
                                if tag == tags.READY:
                                    comm.send([input_region, distribution, mu, correlation, cell_idx], dest=source, tag=tags.START)
                                    print "\033[95m Sending task %d/%d to worker %d\033[0m" % (task, num_tasks, source)
                                    sent = True
                                elif tag == tags.DONE:
                                    print "\033[95m Worker %d completed task %d/%d\033[0m" % (source, task, num_tasks)
                                elif tag == tags.ERROR:
                                    print "\033[91mMaster detected ERROR at node %d. Aborting...\033[0m" % source
                                    for worker in range(1, num_workers + 1):
                                        comm.send([None, None, None, None, None], dest=worker, tag=tags.EXIT)
                                    sys.exit()

        for worker in range(1, num_workers + 1):
            comm.send([None, None, None, None, None], dest=worker, tag=tags.EXIT)
        print("\033[95m Master finishing\033[0m")
    else:
        while True:
            comm.send(None, dest=0, tag=tags.READY)
            [input_region, distribution, mu, correlation, cell_idx] = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()
            if tag == tags.START:
                # Do the work here
                print "\033[93m%d put to work on %s %s %+1.1f %1.2f cell %d\033[0m" % (rank, input_region,
                                                                                distribution, mu, correlation, cell_idx)
                try:
                    os.system("python %s %s %1.2f %d %s %1.1f" % (sys.argv[0], input_region, correlation, cell_idx,
                                                                  distribution, mu))
                except:
                    print "\033[91mNode %d exiting with ERROR\033[0m" % rank
                    comm.send(None, dest=0, tag=tags.ERROR)
                    sys.exit()
                comm.send(None, dest=0, tag=tags.DONE)
            elif tag == tags.EXIT:
                print "\033[93m%d exiting\033[0m" % rank
                break
        comm.send(None, dest=0, tag=tags.EXIT)


def PopulationMPIclassic():
    """ Run with
        mpirun -np 4 python example_mpi.py
    """
    from mpi4py import MPI

    class Tags:
        def __init__(self):
            self.READY = 0
            self.DONE = 1
            self.EXIT = 2
            self.START = 3
            self.ERROR = 4
    tags = Tags()
    # Initializations and preliminaries
    comm = MPI.COMM_WORLD   # get MPI communicator object
    size = comm.size        # total number of processes
    rank = comm.rank        # rank of this process
    status = MPI.Status()   # get MPI status object
    num_workers = size - 1

    if size == 1:
        print "Can't do MPI with one core!"
        sys.exit()

    if rank == 0:

        print("\033[95m Master starting with %d workers\033[0m" % num_workers)
        task = 0
        num_cells = 2000 if at_stallo else 1
        num_tasks = (len(param_dict['input_regions']) *
                     len(param_dict['conductance_types']) * len(param_dict['correlations']) * (num_cells))

        for input_region in param_dict['input_regions']:
            for conductance_type in param_dict['conductance_types']:
                for correlation in param_dict['correlations']:
                    for holding_potential in param_dict['holding_potentials']:
                        for cell_idx in xrange(0, num_cells):
                            task += 1
                            sent = False
                            while not sent:
                                data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                                source = status.Get_source()
                                tag = status.Get_tag()
                                if tag == tags.READY:
                                    comm.send([input_region, conductance_type, holding_potential, correlation, cell_idx], dest=source, tag=tags.START)
                                    print "\033[95m Sending task %d/%d to worker %d\033[0m" % (task, num_tasks, source)
                                    sent = True
                                elif tag == tags.DONE:
                                    print "\033[95m Worker %d completed task %d/%d\033[0m" % (source, task, num_tasks)
                                elif tag == tags.ERROR:
                                    print "\033[91mMaster detected ERROR at node %d. Aborting...\033[0m" % source
                                    for worker in range(1, num_workers + 1):
                                        comm.send([None, None, None, None, None], dest=worker, tag=tags.EXIT)
                                    sys.exit()

        for worker in range(1, num_workers + 1):
            comm.send([None, None, None, None, None], dest=worker, tag=tags.EXIT)
        print("\033[95m Master finishing\033[0m")
    else:
        while True:
            comm.send(None, dest=0, tag=tags.READY)
            [input_region, conductance_type, holding_potential, correlation, cell_idx] = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()
            if tag == tags.START:
                # Do the work here
                print "\033[93m%d put to work on %s %s %+d %1.2f cell %d\033[0m" % (rank, input_region,
                                                                                conductance_type, holding_potential, correlation, cell_idx)
                try:
                    os.system("python %s %s %1.2f %d %s %d" % (sys.argv[0], input_region, correlation, cell_idx,
                                                                  conductance_type, holding_potential))
                except:
                    print "\033[91mNode %d exiting with ERROR\033[0m" % rank
                    comm.send(None, dest=0, tag=tags.ERROR)
                    sys.exit()
                comm.send(None, dest=0, tag=tags.DONE)
            elif tag == tags.EXIT:
                print "\033[93m%d exiting\033[0m" % rank
                break
        comm.send(None, dest=0, tag=tags.EXIT)


if __name__ == '__main__':
    conductance = 'generic'
    # conductance = 'stick_generic'
    # conductance = 'classic'

    if conductance == 'generic':
        from param_dicts import generic_population_params as param_dict
    elif conductance == 'stick_generic':
        from param_dicts import stick_population_params as param_dict
    else:
        from param_dicts import classic_population_params as param_dict

    # plot_coherence(param_dict)
    # plot_generic_population_LFP(param_dict)
    # plot_classic_population_LFP(param_dict)
    # plot_simple_model_LFP(param_dict)
    # plot_linear_combination(param_dict)
    # plot_figure_1(param_dict)
    # plot_figure_2(param_dict)
    # plot_figure_3(param_dict)
    # plot_figure_5(param_dict)
    # plot_leski_13(param_dict)
    # plot_leski_6(param_dict)
    # plot_LFP_time_trace(param_dict)
    # sys.exit()

    if len(sys.argv) >= 3:
        cell_number = int(sys.argv[3])
        input_region = sys.argv[1]
        correlation = float(sys.argv[2])
        if conductance == 'generic' or conductance == 'stick_generic':
            distribution = sys.argv[4]
            mu = float(sys.argv[5])
            param_dict.update({'input_region': input_region,
                               'cell_number': cell_number,
                               'mu': mu,
                               'distribution': distribution,
                               'correlation': correlation,
                               })
        else:
            conductance_type = sys.argv[4]
            holding_potential = int(sys.argv[5])
            param_dict.update({'input_region': input_region,
                               'cell_number': cell_number,
                               'conductance_type': conductance_type,
                               'holding_potential': holding_potential,
                               'correlation': correlation,
                               })
        ns = NeuralSimulation(**param_dict)
        ns.run_distributed_synaptic_simulation()
    elif len(sys.argv) == 2 and sys.argv[1] == 'initialize':
        initialize_population(param_dict)
        plot_population(param_dict)
    elif len(sys.argv) == 2 and sys.argv[1] == 'MPI':
        if conductance == 'generic' or conductance == 'stick_generic':
            PopulationMPIgeneric()
        else:
            PopulationMPIclassic()
    elif len(sys.argv) == 2 and sys.argv[1] == 'sum':
        if conductance == 'generic' or conductance == 'stick_generic':
            sum_population_mpi_generic(param_dict)
        else:
            sum_population_mpi_classic(param_dict)
        # sum_population_generic(param_dict)
