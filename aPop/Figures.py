from __future__ import division
import os
import matplotlib
matplotlib.use('Agg', warn=False)

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
import matplotlib


def plot_3d_rot_pop(param_dict):
    # plt.seed(1234)
    plt.seed(12345)
    num_cells = 100

    ns = NeuralSimulation(**param_dict)

    fig_dir = join(param_dict['root_folder'], param_dict['save_folder'], ns.population_sim_name)
    if not os.path.isdir(fig_dir):
        os.mkdir(fig_dir)
    else:
        os.system('rm %s/*.png' % fig_dir)
    x_y_z_rot = np.load(os.path.join(ns.param_dict['root_folder'],
                                     ns.param_dict['save_folder'],
                     'x_y_z_rot_%s.npy' % ns.param_dict['name']))

    # R = np.sqrt(np.sum(np.array(x_y_z_rot[num_cells - 1][:2])**2))
    R = 1000.
    cell_idxs = np.arange(num_cells)

    all_xstart = []
    all_xend = []
    all_ystart = []
    all_yend = []
    all_zstart = []
    all_zend = []
    all_diams = []
    all_vmems = []
    for cell_number in cell_idxs:#range(num_cells):
        param_dict['cell_number'] = cell_number
        ns = NeuralSimulation(**param_dict)
        cell = ns._return_cell(x_y_z_rot[cell_number])
        all_vmems.append(np.load(join(ns.sim_folder, 'vmem_%s.npy' % ns.sim_name)))
        all_xstart.append(cell.xstart)
        all_xend.append(cell.xend)
        all_ystart.append(cell.ystart)
        all_yend.append(cell.yend)
        all_zstart.append(cell.zstart)
        all_zend.append(cell.zend)
        all_diams.append(cell.diam)

    # print all_vmems
    vmin = np.min(all_vmems)
    vmax = np.max(all_vmems)
    print vmin, vmax
    # cmaps = [plt.cm.Greys_r, plt.cm.Blues_r, plt.cm.BuGn_r, plt.cm.BuPu_r, plt.cm.GnBu_r,
    #          plt.cm.Greens_r, plt.cm.Oranges_r, plt.cm.OrRd_r,
    #          plt.cm.PuBu_r, plt.cm.PuBuGn_r, plt.cm.PuRd_r, plt.cm.Purples_r, plt.cm.RdPu_r,
    #          plt.cm.Reds_r, plt.cm.YlGn_r, plt.cm.YlGnBu_r, plt.cm.YlOrBr_r, plt.cm.YlOrRd_r]
    #
    cmaps = [plt.cm.viridis]#plt.cm.Greys_r]

    vmem_clr = lambda vmem, cmap_func: cmap_func(0.0 + 1.5 * (vmem - vmin) / (vmax - vmin))
    # vmem_clr = lambda vmem, cmap_func: cmap_func(0+ 2 * (vmem - vmin) / (vmax - vmin))

    plt.close('all')
    img = plt.imshow([[]], vmin=vmin, vmax=vmax, origin='lower', cmap=cmaps[0])

    import matplotlib as mpl
    from mpl_toolkits.mplot3d import Axes3D
    plt.style.use('dark_background')
    fig = plt.figure(figsize=[12, 12])
    fig.subplots_adjust(top=1, bottom=0, left=0, right=1)
    ax = fig.gca(projection='3d', aspect=1)
    ax._axis3don = False
    ax.set_aspect('equal')

    plt.colorbar(img, shrink=0.5)
    num_tsteps = all_vmems[0].shape[1]

    lines = []
    t_idx = 0
    angle = t_idx / num_tsteps * 360

    cell_cmaps = []

    for cell_number, cell_idx in enumerate(cell_idxs):
        x,y,z,phi = x_y_z_rot[cell_idx]
        dist = 1. / np.sqrt((x - R * np.cos(np.deg2rad(angle)))**2 +
                       (y - R * np.sin(np.deg2rad(angle)))**2)
        cmap = np.random.choice(cmaps)
        cell_cmaps.append(cmap)
        lines.append([])
        for idx in xrange(len(all_xend[cell_number])):
            l = None
            if idx == 0:
                l, = ax.plot([x_y_z_rot[cell_idx][0]], [x_y_z_rot[cell_idx][1]],
                        [x_y_z_rot[cell_idx][2]], 'o', ms=8, mec='none', zorder=1./dist,
                         c=vmem_clr(all_vmems[cell_number][idx, t_idx], cmap))
            else:
                x = [all_xstart[cell_number][idx], all_xend[cell_number][idx]]
                z = [all_zstart[cell_number][idx], all_zend[cell_number][idx]]
                y = [all_ystart[cell_number][idx], all_yend[cell_number][idx]]
                l, = ax.plot(x, y, z, c=vmem_clr(all_vmems[cell_number][idx, t_idx], cmap),
                        lw=2, zorder=dist, clip_on=False)
            lines[cell_number].append(l)
    ax.auto_scale_xyz([-450, 450], [-450, 450], [0, 900])
    ax.view_init(0, angle)
    plt.draw()
    plt.savefig(join(fig_dir, 'vmem_%05d.png' % t_idx), dpi=150)

    for t_idx in range(num_tsteps)[::5][1:]:
        angle = t_idx / num_tsteps * 360
        for cell_number in range(num_cells):
            x,y,z,phi = x_y_z_rot[cell_number]
            dist = np.sqrt((x - R * np.cos(np.deg2rad(angle)))**2 +
                           (y - R * np.sin(np.deg2rad(angle)))**2)
            cmap = cell_cmaps[cell_number]
            for idx in xrange(len(all_xend[cell_number])):
                lines[cell_number][idx].set_color(vmem_clr(all_vmems[cell_number][idx, t_idx], cmap))
                lines[cell_number][idx].set_zorder(1./dist)

        ax.auto_scale_xyz([-450, 450], [-450, 450], [00, 900])

        print angle
        ax.view_init(0, angle)

        plt.draw()
        plt.savefig(join(fig_dir, 'vmem_%05d.png' % t_idx), dpi=150)

    os.system('mencoder "mf://%s/*.png" -o %s/../%s_%dcells.avi -ovc x264 -fps 30' %
              (fig_dir, fig_dir, ns.population_sim_name, num_cells))

def return_lfp_amp(param_dict, electrode_parameters, x, x_y_z_rot):
    plt.seed(1234 * param_dict['cell_number'])
    ns_1 = NeuralSimulation(**param_dict)
    cell = ns_1._return_cell(x_y_z_rot)
    cell, syn = ns_1._make_distributed_synaptic_stimuli(cell)
    cell.simulate(rec_imem=True)
    electrode = LFPy.RecExtElectrode(cell, **electrode_parameters)
    electrode.calc_lfp()
    sig_amp = 1e6 * electrode.LFP[:, -1].reshape(x.shape)
    return sig_amp, cell

def plot_asymetric_conductance_space_average_movie(param_dict):

    tot_num_cells = 100
    num_cells = 1
    x_y_z_rot = np.load(os.path.join(param_dict['root_folder'],
                                     param_dict['save_folder'],
                                     'x_y_z_rot_%s.npy' % param_dict['name']))
    x = np.linspace(-1500, 1500, 150)
    z = np.linspace(-1000, 2000, 150)
    x, z = np.meshgrid(x, z)
    elec_x = x.flatten()
    elec_z = z.flatten()
    elec_y = np.ones(len(elec_x)) * 0

    electrode_parameters = {
        'sigma': 0.3,              # extracellular conductivity
        'x': elec_x,        # x,y,z-coordinates of contact points
        'y': elec_y,
        'z': elec_z,
        'method': 'linesource'
    }
    param_dict.update({
                       'input_region': 'homogeneous',
                       'mu': 0.0,
                       'correlation': 0.0,
                       'cell_number': 0,
                       'cut_off': 10.0,
                       'end_t': 1,
                      })
    param_dict_1 = param_dict.copy()
    param_dict_2 = param_dict.copy()

    param_dict_1['distribution'] = 'uniform'
    param_dict_2['distribution'] = 'linear_increase'

    tot_sig_1 = np.zeros(x.shape)
    tot_sig_2 = np.zeros(x.shape)

    sig_amp_1, cell = return_lfp_amp(param_dict_1, electrode_parameters, x, x_y_z_rot[0])
    sig_amp_2, cell = return_lfp_amp(param_dict_2, electrode_parameters, x, x_y_z_rot[0])

    tot_sig_1 += sig_amp_1
    tot_sig_2 += sig_amp_2

    logthresh = 1
    color_lim = np.max(np.abs(tot_sig_2)) * 5

    vmax = color_lim
    vmin = -color_lim
    maxlog = int(np.ceil(np.log10(vmax)))
    minlog = int(np.ceil(np.log10(-vmin)))

    tick_locations = ([-(10**d) for d in xrange(minlog, -logthresh-1, -1)]
                    + [0.0]
                    + [(10**d) for d in xrange(-logthresh, maxlog+1)])

    plt.close('all')
    morph_line_dict = {'solid_capstyle': 'butt',
                       'lw': 0.5,
                       'color': 'k',
                       'zorder': 1}

    fig = plt.figure(figsize=[12, 6])
    name = fig.suptitle("Number of cells: %d" % num_cells)

    fig.subplots_adjust(wspace=0.6, hspace=0.6)
    ax_lfp_1 = fig.add_subplot(221, ylim=[np.min(elec_z), np.max(elec_z)], xlim=[np.min(elec_x), np.max(elec_x)],
                                 aspect=1, frameon=False, xticks=[], yticks=[], title='Symmetric conductance\nPopulation LFP')
    ax_lfp_2 = fig.add_subplot(222, ylim=[np.min(elec_z), np.max(elec_z)], xlim=[np.min(elec_x), np.max(elec_x)],
                                 aspect=1, frameon=False, xticks=[], yticks=[], title='Asymmetric conductance\nPopulation LFP')

    ax_lfp_1_single = fig.add_subplot(223, ylim=[np.min(elec_z), np.max(elec_z)], xlim=[np.min(elec_x), np.max(elec_x)],
                                 aspect=1, frameon=False, xticks=[], yticks=[], title='Singel cell LFP')
    ax_lfp_2_single = fig.add_subplot(224, ylim=[np.min(elec_z), np.max(elec_z)], xlim=[np.min(elec_x), np.max(elec_x)],
                                 aspect=1, frameon=False, xticks=[], yticks=[], title='single cell LFP')

    l_1 = []
    l_2 = []
    for idx in xrange(len(cell.xmid)):
        l_1.append(ax_lfp_1_single.plot([cell.xstart[idx], cell.xend[idx]],
                                        [cell.zstart[idx], cell.zend[idx]], **morph_line_dict)[0])
        ax_lfp_1.plot([cell.xstart[idx], cell.xend[idx]],
                      [cell.zstart[idx], cell.zend[idx]], **morph_line_dict)
        ax_lfp_2.plot([cell.xstart[idx], cell.xend[idx]],
                      [cell.zstart[idx], cell.zend[idx]], **morph_line_dict)
        l_2.append(ax_lfp_2_single.plot([cell.xstart[idx], cell.xend[idx]],
                                        [cell.zstart[idx], cell.zend[idx]], **morph_line_dict)[0])

    img_dict = {'origin': 'lower',
                'extent': [np.min(x), np.max(x), np.min(z), np.max(z)],
                'cmap': plt.cm.bwr,
                'norm': matplotlib.colors.SymLogNorm(10**-logthresh),
                'vmin': vmin,
                'vmax': vmax,
                'interpolation': 'nearest',
                'zorder': 0}

    img_lfp_1 = ax_lfp_1.imshow(tot_sig_1[:, :], **img_dict)
    img_lfp_2 = ax_lfp_2.imshow(tot_sig_2[:, :], **img_dict)
    img_lfp_1_single = ax_lfp_1_single.imshow(sig_amp_1[:, :], **img_dict)
    img_lfp_2_single = ax_lfp_2_single.imshow(sig_amp_2[:, :], **img_dict)

    cb = plt.colorbar(img_lfp_1_single, ax=ax_lfp_1_single, shrink=0.5, ticks=tick_locations)
    cb2 = plt.colorbar(img_lfp_2_single, ax=ax_lfp_2_single, shrink=0.5, ticks=tick_locations)
    cb.set_label('nV', labelpad=0)
    cb2.set_label('nV', labelpad=0)

    cb = plt.colorbar(img_lfp_1, ax=ax_lfp_1, shrink=0.5, ticks=tick_locations)
    cb2 = plt.colorbar(img_lfp_2, ax=ax_lfp_2, shrink=0.5, ticks=tick_locations)
    cb.set_label('nV', labelpad=0)
    cb2.set_label('nV', labelpad=0)
    fig.savefig(join(param_dict['root_folder'], 'asym_conductance_spatial',
                         'hay_%04d.png' % (num_cells)))

    for num_cells in np.arange(2, tot_num_cells):
        print num_cells
        name.set_text("Number of cells: %d" % num_cells)
        param_dict_1['cell_number'] = num_cells - 1
        param_dict_2['cell_number'] = num_cells - 1
        sig_amp_1, cell = return_lfp_amp(param_dict_1, electrode_parameters, x, x_y_z_rot[num_cells - 1])
        sig_amp_2, cell = return_lfp_amp(param_dict_2, electrode_parameters, x, x_y_z_rot[num_cells - 1])

        for idx in xrange(len(cell.xmid)):
            ax_lfp_1.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]], **morph_line_dict)
            ax_lfp_2.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]], **morph_line_dict)
        for idx in xrange(len(cell.xmid)):
            l_1[idx].set_data([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]])
            l_2[idx].set_data([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]])

        tot_sig_1 += sig_amp_1
        tot_sig_2 += sig_amp_2

        img_lfp_1.set_data(tot_sig_1[:, :])
        img_lfp_2.set_data(tot_sig_2[:, :])

        img_lfp_1_single.set_data(sig_amp_1[:, :])
        img_lfp_2_single.set_data(sig_amp_2[:, :])

        fig.savefig(join(param_dict['root_folder'], 'asym_conductance_spatial',
                         'hay_%04d.png' % (num_cells)))



def plot_asymetric_conductance_time_average_movie(param_dict):

    x = np.linspace(-2000, 2000, 150)
    z = np.linspace(-2000, 2000, 150)
    x, z = np.meshgrid(x, z)
    elec_x = x.flatten()
    elec_z = z.flatten()
    elec_y = np.ones(len(elec_x)) * 0

    electrode_parameters = {
        'sigma': 0.3,              # extracellular conductivi   ty
        'x': elec_x,        # x,y,z-coordinates of contact points
        'y': elec_y,
        'z': elec_z,
        'method': 'linesource'
    }

    param_dict.update({'distribution': 'uniform',
                       'input_region': 'homogeneous',
                       'mu': 0.0,
                       'correlation': 0.0,
                       'cell_number': 0,
                       # 'start_T': 0.0,
                       'end_t': 1000,
                      })
    plt.seed(1234)
    ns_1 = NeuralSimulation(**param_dict)
    cell = ns_1._return_cell([0, 0, 0, 0])
    cell, syn = ns_1._make_distributed_synaptic_stimuli(cell)

    cell.simulate(rec_imem=True)

    electrode = LFPy.RecExtElectrode(cell, **electrode_parameters)
    electrode.calc_lfp()
    sig_amp_1 = 1e6 * electrode.LFP.reshape([x.shape[0], x.shape[1], len(cell.tvec)])

    param_dict.update({'distribution': 'linear_increase',
                      })
    plt.seed(1234)
    ns_2 = NeuralSimulation(**param_dict)
    cell = ns_2._return_cell([0, 0, 0, 0])
    cell, syn = ns_2._make_distributed_synaptic_stimuli(cell)

    cell.simulate(rec_imem=True)

    electrode = LFPy.RecExtElectrode(cell, **electrode_parameters)
    electrode.calc_lfp()
    sig_amp_2 = 1e6 * electrode.LFP.reshape([x.shape[0], x.shape[1], len(cell.tvec)])

    logthresh = 1
    color_lim = np.max(np.abs(1e6 * electrode.LFP)) / 10
    print color_lim
    vmax = color_lim
    vmin = -color_lim
    maxlog = int(np.ceil(np.log10(vmax)))
    minlog = int(np.ceil(np.log10(-vmin)))

    tick_locations = ([-(10**d) for d in xrange(minlog, -logthresh-1, -1)]
                    + [0.0]
                    + [(10**d) for d in xrange(-logthresh, maxlog+1)])

    plt.close('all')
    fig = plt.figure(figsize=[12, 6])
    fig.subplots_adjust(wspace=0.6, hspace=0.6)
    ax_lfp_1 = fig.add_subplot(221, ylim=[np.min(elec_z), np.max(elec_z)], xlim=[np.min(elec_x), np.max(elec_x)],
                                 aspect=1, frameon=False, xticks=[], yticks=[], title='Symmetric conductance\nLFP')

    ax_lfp_1_mean = fig.add_subplot(222, ylim=[np.min(elec_z), np.max(elec_z)], xlim=[np.min(elec_x), np.max(elec_x)],
                                 aspect=1, frameon=False, xticks=[], yticks=[], title='Symmetric conductance\nLFP average')


    [ax_lfp_1.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]], '-',
                 solid_capstyle='butt', ms=6, lw=cell.diam[idx], color='k', zorder=1)
     for idx in xrange(len(cell.xmid))]

    [ax_lfp_1_mean.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]], '-',
                 solid_capstyle='butt', ms=6, lw=cell.diam[idx], color='k', zorder=1)
     for idx in xrange(len(cell.xmid))]


    ax_lfp_2 = fig.add_subplot(223, ylim=[np.min(elec_z), np.max(elec_z)], xlim=[np.min(elec_x), np.max(elec_x)],
                                 aspect=1, frameon=False, xticks=[], yticks=[], title='Asymmetric conductance\nLFP')

    ax_lfp_2_mean = fig.add_subplot(224, ylim=[np.min(elec_z), np.max(elec_z)], xlim=[np.min(elec_x), np.max(elec_x)],
                                 aspect=1, frameon=False, xticks=[], yticks=[], title='Asymmetric conductance\nLFP average')


    [ax_lfp_2.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]], '-',
                 solid_capstyle='butt', ms=6, lw=cell.diam[idx], color='k', zorder=1)
     for idx in xrange(len(cell.xmid))]

    [ax_lfp_2_mean.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]], '-',
                 solid_capstyle='butt', ms=6, lw=cell.diam[idx], color='k', zorder=1)
     for idx in xrange(len(cell.xmid))]

    time_idx = 0

    img_lfp_1 = ax_lfp_1.imshow(sig_amp_1[:, :, time_idx], origin='lower', extent=[np.min(x), np.max(x), np.min(z), np.max(z)],
                            cmap=plt.cm.bwr, norm=matplotlib.colors.SymLogNorm(10**-logthresh),
                  vmin=vmin, vmax=vmax, interpolation='nearest', zorder=0)

    img_lfp_1_mean = ax_lfp_1_mean.imshow(sig_amp_1[:, :, time_idx], origin='lower', extent=[np.min(x), np.max(x), np.min(z), np.max(z)],
                            cmap=plt.cm.bwr, norm=matplotlib.colors.SymLogNorm(10**-logthresh),
                  vmin=vmin, vmax=vmax, interpolation='nearest', zorder=0)

    img_lfp_2 = ax_lfp_2.imshow(sig_amp_2[:, :, time_idx], origin='lower', extent=[np.min(x), np.max(x), np.min(z), np.max(z)],
                            cmap=plt.cm.bwr, norm=matplotlib.colors.SymLogNorm(10**-logthresh),
                  vmin=vmin, vmax=vmax, interpolation='nearest', zorder=0)

    img_lfp_2_mean = ax_lfp_2_mean.imshow(sig_amp_2[:, :, time_idx], origin='lower', extent=[np.min(x), np.max(x), np.min(z), np.max(z)],
                            cmap=plt.cm.bwr, norm=matplotlib.colors.SymLogNorm(10**-logthresh),
                  vmin=vmin, vmax=vmax, interpolation='nearest', zorder=0)

    cb = plt.colorbar(img_lfp_1_mean, ax=ax_lfp_1_mean, shrink=0.5, ticks=tick_locations)
    cb2 = plt.colorbar(img_lfp_2_mean, ax=ax_lfp_2_mean, shrink=0.5, ticks=tick_locations)
    cb.set_label('nV', labelpad=0)
    fig.savefig(join(param_dict['root_folder'], 'asym_conductance_anim',
                         'hay_%04d.png' % (time_idx)))
    for time_idx in np.arange(0, len(cell.tvec), 5)[1:]:
        print time_idx
        img_lfp_1.set_data(sig_amp_1[:, :, time_idx])
        img_lfp_1_mean.set_data(np.average(sig_amp_1[:, :, :time_idx], axis=-1))

        img_lfp_2.set_data(sig_amp_2[:, :, time_idx])
        img_lfp_2_mean.set_data(np.average(sig_amp_2[:, :, :time_idx], axis=-1))


        fig.savefig(join(param_dict['root_folder'], 'asym_conductance_anim',
                         'hay_%04d.png' % (time_idx)))


def plot_cell_population(param_dict):

    num_cells = 2000
    param_dict.update({
                       'mu': 0.0,
                       'distribution': 'uniform',
                        'input_region': 'homogeneous',
                        'correlation': 0.0,
                        'cell_number': 0,
                      })

    ns = NeuralSimulation(**param_dict)
    x_y_z_rot = np.load(os.path.join(ns.param_dict['root_folder'],
                                     ns.param_dict['save_folder'],
                     'x_y_z_rot_%s.npy' % ns.param_dict['name']))

    plt.close('all')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, frameon=False, xticks=[], yticks=[], aspect=1, rasterized=True)
    fig.subplots_adjust(right=0.95, wspace=0.5, hspace=0.8, left=0.05, top=0.85, bottom=0.2)
    cell_clr = lambda cell_idx: plt.cm.rainbow(cell_idx / (num_cells - 1))
    ax.set_rasterization_zorder(0)
    for cell_number in range(num_cells):
        param_dict['cell_number'] = cell_number
        ns = NeuralSimulation(**param_dict)
        cell = ns._return_cell(x_y_z_rot[cell_number])
        if cell_number % 100 == 0:
            print "Plotting ", cell_number
        c = np.random.randint(0, num_cells)
        for idx in xrange(len(cell.xend)):
            if idx == 0:
                ax.plot(cell.xmid[0], cell.zmid[idx], 'o', ms=8, mec='none',
                         c=cell_clr(c), zorder=0, rasterized=True)
            else:
                ax.plot([cell.xstart[idx], cell.xend[idx]],
                        [cell.zstart[idx], cell.zend[idx]],
                        lw=cell.diam[idx]/2, rasterized=True,
                         c=cell_clr(c), zorder=0)
        if cell_number % 100 == 0:
            plt.savefig(join(param_dict['root_folder'], 'figures', 'Figure_cell_population_%d_%d.png' % (cell_number, num_cells)), dpi=300)
    plt.savefig(join(param_dict['root_folder'], 'figures', 'Figure_cell_population_%d.png' % num_cells), dpi=300)
    plt.savefig(join(param_dict['root_folder'], 'figures', 'Figure_cell_population_%d.pdf' % num_cells), dpi=300)
    plt.close('all')



def plot_all_soma_sigs(param_dict):

    folder = join(param_dict['root_folder'], param_dict['save_folder'], 'simulations')
    pop_size = 500

    param_dict.update({
                       'cell_number': 0,
                      })

    elec_z = param_dict['center_electrode_parameters']['z']
    elec = np.argmin(np.abs(elec_z - 0))
    correlations = param_dict['correlations']
    # input_regions = ['distal_tuft', 'homogeneous', 'basal']
    input_regions = ['top', 'homogeneous', 'bottom']
    # distributions = ['uniform', 'linear_increase', 'linear_decrease']
    distributions = ['uniform', 'increase']

    plt.close('all')
    fig = plt.figure(figsize=(18, 9))
    fig.subplots_adjust(right=0.95, wspace=0.5, hspace=0.8, left=0.05, top=0.85, bottom=0.2)

    psd_ax_dict = {'ylim': [1e-9, 1e-4],
                    'yscale': 'log',
                    'xscale': 'log',
                    'xlim': [1, 500],
                   #'xlabel': 'Frequency (Hz)',
                   'xticks': [1e0, 10, 100],
                    }
    num_plot_cols = 10
    num_plot_rows = 3
    fig.text(0.17, 0.95, 'Uniform conductance')
    fig.text(0.6, 0.95, 'Increasing conductance')
    # fig.text(0.45, 0.95, 'Linear increasing conductance')
    # fig.text(0.75, 0.95, 'Linear decreasing conductance')

    fig.text(0.05, 0.75, 'Distal tuft\ninput', ha='center')
    fig.text(0.05, 0.5, 'Homogeneous\ninput', ha='center')
    fig.text(0.05, 0.25, 'Basal\ninput', ha='center')
    lines = None
    line_names = None
    for i, input_region in enumerate(input_regions):
        for d, distribution in enumerate(distributions):
            for c, correlation in enumerate(correlations):
                print input_region, distribution, correlation
                plot_number = i * num_plot_cols + d * (3 + len(distributions)) + c + 2
                lines = []
                line_names = []

                ax = fig.add_subplot(num_plot_rows, num_plot_cols, plot_number, **psd_ax_dict)
                ax.set_title('c=%1.2f' % correlation)
                for mu in param_dict['mus']:
                    param_dict['correlation'] = correlation
                    param_dict['input_region'] = input_region
                    param_dict['distribution'] = distribution
                    param_dict['mu'] = mu

                    ns = NeuralSimulation(**param_dict)
                    name = 'summed_center_signal_%s_%dum' % (ns.population_sim_name, pop_size)
                    try:
                        lfp = np.load(join(folder, '%s.npy' % name))[elec, :]
                    except:
                        print name
                        continue
                    freq, psd = tools.return_freq_and_psd_welch(lfp, ns.welch_dict)
                    f_idx_max = np.argmin(np.abs(freq - param_dict['max_freq']))
                    f_idx_min = np.argmin(np.abs(freq - 1.))
                    l, = ax.plot(freq[f_idx_min:f_idx_max], psd[0][f_idx_min:f_idx_max],
                                 c=qa_clr_dict[mu], lw=3, clip_on=False, solid_capstyle='butt')
                    lines.append(l)
                    line_names.append(conductance_names[mu])
                    # img = ax.pcolormesh(freq[1:freq_idx], z, psd[:, 1:freq_idx], **im_dict)
                    if i==2 and d == 0 and c == 0:
                        ax.set_xlabel('frequency (Hz)', labelpad=-0)
                        ax.set_ylabel('LFP-PSD $\mu$V$^2$/Hz')
                    #     c_ax = fig.add_axes([0.37, 0.2, 0.005, 0.17])
                    #     cbar = plt.colorbar(img, cax=c_ax, label='$\mu$V$^2$/Hz')
                    # plt.axis('auto')
                    # plt.colorbar(img)
                    # l, = ax.loglog(freq, psd[0], c=input_region_clr[input_region], lw=3)
                    # lines.append(l)
                    # line_names.append(input_region)

                ax.set_yticks(ax.get_yticks()[1:-2][::2])
        # ax_tuft.set_ylabel('LFP-PSD ($\mu$V$^2$/Hz)', labelpad=-5)
        # ax_tuft.set_xticklabels(['', '1', '10', '100'])

    fig.legend(lines, line_names, loc='lower center', frameon=False, ncol=3)
    simplify_axes(fig.axes)
    # mark_subplots([ax_morph_homogeneous], 'B', ypos=1.1, xpos=0.1)
    plt.savefig(join(param_dict['root_folder'], param_dict['save_folder'], 'Figure_all_sigs_stick.png'))
    plt.close('all')


def plot_all_soma_sigs_classic(param_dict):

    folder = join(param_dict['root_folder'], param_dict['save_folder'], 'simulations')
    pop_size = 500

    param_dict.update({
                       'cell_number': 0,
                      })

    elec_z = param_dict['center_electrode_parameters']['z']
    elec = np.argmin(np.abs(elec_z - 0))
    correlations = param_dict['correlations']
    input_regions = ['distal_tuft', 'homogeneous']#, 'basal']
    # input_regions = ['top', 'homogeneous', 'bottom']
    # distributions = ['uniform', 'linear_increase', 'linear_decrease']
    # distributions = ['uniform', 'increase']
    holding_potentials = [-80, -70, -60]

    plt.close('all')
    fig = plt.figure(figsize=(18, 9))
    fig.subplots_adjust(right=0.95, wspace=0.5, hspace=0.8, left=0.06, top=0.85, bottom=0.2)

    psd_ax_dict = {'ylim': [1e-10, 1e-3],
                    'yscale': 'log',
                    'xscale': 'log',
                    'xlim': [1, 500],
                   #'xlabel': 'Frequency (Hz)',
                   'xticks': [1e0, 10, 100],
                    }
    num_plot_cols = 16
    num_plot_rows = 2
    fig.text(0.17, 0.95, '- 80 mV')
    fig.text(0.45, 0.95, '- 70 mV')
    fig.text(0.75, 0.95, '- 60 mV')

    fig.text(0.005, 0.75, 'Distal tuft\ninput', ha='left')
    fig.text(0.005, 0.3, 'Homogeneous\ninput', ha='left')
    # fig.text(0.05, 0.25, 'Basal\ninput', ha='center')
    lines = None
    line_names = None
    for i, input_region in enumerate(input_regions):
        for d, holding_potential in enumerate(holding_potentials):
            for c, correlation in enumerate(correlations):
                print input_region, holding_potential, correlation
                plot_number = i * num_plot_cols + d * (2 + len(holding_potentials)) + c + 2
                lines = []
                line_names = []

                ax = fig.add_subplot(num_plot_rows, num_plot_cols, plot_number, **psd_ax_dict)
                ax.set_title('c=%1.2f' % correlation)
                for conductance_type in param_dict['conductance_types']:
                    param_dict['correlation'] = correlation
                    param_dict['input_region'] = input_region
                    param_dict['holding_potential'] = holding_potential
                    param_dict['conductance_type'] = conductance_type

                    ns = NeuralSimulation(**param_dict)
                    name = 'summed_center_signal_%s_%dum' % (ns.population_sim_name, pop_size)
                    lfp = np.load(join(folder, '%s.npy' % name))[elec, :]
                    freq, psd = tools.return_freq_and_psd_welch(lfp, ns.welch_dict)
                    f_idx_max = np.argmin(np.abs(freq - param_dict['max_freq']))
                    f_idx_min = np.argmin(np.abs(freq - 1.))
                    l, = ax.plot(freq[f_idx_min:f_idx_max], psd[0][f_idx_min:f_idx_max],
                                 c=conductance_clr_dict[conductance_type], lw=3, clip_on=True)
                    lines.append(l)
                    line_names.append(conductance_type)
                    # img = ax.pcolormesh(freq[1:freq_idx], z, psd[:, 1:freq_idx], **im_dict)
                    if i==2 and d == 0 and c == 0:
                        ax.set_xlabel('frequency (Hz)', labelpad=-0)
                        ax.set_ylabel('LFP-PSD $\mu$V$^2$/Hz')
                    #     c_ax = fig.add_axes([0.37, 0.2, 0.005, 0.17])
                    #     cbar = plt.colorbar(img, cax=c_ax, label='$\mu$V$^2$/Hz')
                    # plt.axis('auto')
                    # plt.colorbar(img)
                    # l, = ax.loglog(freq, psd[0], c=input_region_clr[input_region], lw=3)
                    # lines.append(l)
                    # line_names.append(input_region)

                ax.set_yticks(ax.get_yticks()[1:-2][::2])
        # ax_tuft.set_ylabel('LFP-PSD ($\mu$V$^2$/Hz)', labelpad=-5)
        # ax_tuft.set_xticklabels(['', '1', '10', '100'])

    fig.legend(lines, line_names, loc='lower center', frameon=False, ncol=3)
    simplify_axes(fig.axes)
    # mark_subplots([ax_morph_homogeneous], 'B', ypos=1.1, xpos=0.1)
    plt.savefig(join(param_dict['root_folder'], param_dict['save_folder'], 'Figure_classic_all_sigs.png'))
    plt.close('all')




def plot_all_size_dependencies(param_dict):

    folder = join(param_dict['root_folder'], param_dict['save_folder'], 'simulations')
    pop_sizes = param_dict['population_radii'][param_dict['population_radii'] <= 500]

    mu = -0.5
    param_dict.update({#'input_region': 'homogeneous',
                       'cell_number': 0,
    #                    'distribution': 'linear_increase',
    #                    'correlation': 0.0,
                       'mu': mu,
                      })

    plot_freqs = np.array([1, 20, 100, 500])
    elec_x = param_dict['lateral_electrode_parameters']['x']
    elec = 60
    correlations = param_dict['correlations']
    input_regions = ['distal_tuft', 'homogeneous', 'basal']
    distributions = ['uniform', 'linear_increase', 'linear_decrease']
    freq_clr = lambda freq_idx: plt.cm.Greys((freq_idx + 1) / (len(plot_freqs) + 1))

    plt.close('all')
    fig = plt.figure(figsize=(18, 9))
    fig.subplots_adjust(right=0.95, wspace=0.2, hspace=0.5, left=0.05, top=0.85, bottom=0.2)


    psd_ax_dict = {'ylim': [1e-13, 1e-4],
                    'yscale': 'log',
                    'xlim': [50, 500],
                   # 'xlabel': 'Frequency (Hz)',
                   # 'xticks': [1e0, 10, 100],
                    }
    num_plot_cols = 12
    num_plot_rows = 3
    # im_dict = {'cmap': 'hot', 'norm': LogNorm(), 'vmin': 1e-8, 'vmax': 1e-2,}
    fig.text(0.17, 0.95, 'Uniform conductance')
    fig.text(0.45, 0.95, 'Linear increasing conductance')
    fig.text(0.75, 0.95, 'Linear decreasing conductance')

    fig.text(0.05, 0.75, 'Distal tuft\ninput', ha='center')
    fig.text(0.05, 0.5, 'Homogeneous\ninput', ha='center')
    fig.text(0.05, 0.25, 'Basal\ninput', ha='center')
    lines = None
    line_names = None
    for i, input_region in enumerate(input_regions):
        for d, distribution in enumerate(distributions):
            for c, correlation in enumerate(correlations):
                print input_region, distribution, correlation
                plot_number = i * (num_plot_cols) + d * (1 + len(distributions)) + c + 2

                ax = fig.add_subplot(num_plot_rows, num_plot_cols, plot_number, **psd_ax_dict)
                ax.set_title('c=%1.2f' % correlation)

                param_dict['correlation'] = correlation
                param_dict['input_region'] = input_region
                param_dict['distribution'] = distribution

                ns = NeuralSimulation(**param_dict)
                psd_at_freqs = np.zeros((len(plot_freqs), len(pop_sizes)))
                for idx, pop_size in enumerate(pop_sizes):
                    name = 'summed_lateral_signal_%s_%dum' % (ns.population_sim_name, pop_size)
                    lfp = np.load(join(folder, '%s.npy' % name))[elec, :]
                    freq, psd = tools.return_freq_and_psd_welch(lfp, ns.welch_dict)
                    freq_idxs = np.array([np.argmin(np.abs(freq - f)) for f in plot_freqs])
                    psd_at_freqs[:, idx] = psd[0, freq_idxs]
                    # print freq[freq_idxs]
                lines = []
                line_names = []
                for f in range(len(plot_freqs)):
                    l, = ax.plot(pop_sizes, psd_at_freqs[f, :], c=freq_clr(f), lw=3)
                    lines.append(l)
                    line_names.append('%d Hz' % plot_freqs[f])
                # img = ax.pcolormesh(freq[1:freq_idx], z, psd[:, 1:freq_idx], **im_dict)
                # if i==2 and d == 0 and c == 0:
                #     ax.set_xlabel('frequency (Hz)', labelpad=-0)
                #     ax.set_ylabel('z')
                #     c_ax = fig.add_axes([0.37, 0.2, 0.005, 0.17])
                #     cbar = plt.colorbar(img, cax=c_ax, label='$\mu$V$^2$/Hz')
                # plt.axis('auto')
                # plt.colorbar(img)
                # l, = ax.loglog(freq, psd[0], c=input_region_clr[input_region], lw=3)
                # lines.append(l)
                # line_names.append(input_region)

        # ax_tuft.set_ylabel('LFP-PSD ($\mu$V$^2$/Hz)', labelpad=-5)
        # ax_tuft.set_xticklabels(['', '1', '10', '100'])
        # ax_tuft.set_yticks(ax_tuft.get_yticks()[1:-1][::2])

    fig.legend(lines, line_names, loc='lower center', frameon=False, ncol=4)
    simplify_axes(fig.axes)
    # mark_subplots([ax_morph_homogeneous], 'B', ypos=1.1, xpos=0.1)
    plt.savefig(join(param_dict['root_folder'], param_dict['save_folder'], 'Figure_size_dep%+1.1f.png' % mu))
    plt.close('all')



def plot_population_density_effect(param_dict):

    conductance_names = {-0.5: 'regenerative',
                         0.0: 'passive-frozen',
                         2.0: 'restorative'}

    correlations = param_dict['correlations']
    mus = [-0.5, 0.0, 2.0]
    folder = join(param_dict['root_folder'], param_dict['save_folder'], 'simulations')
    pop_size = 500

    soma_layer_elec_idx = np.argmin(np.abs(param_dict['center_electrode_parameters']['z'] - 0))

    for input_region in param_dict['input_regions']:
        for distribution in param_dict['distributions']:
            print input_region, distribution
            param_dict.update({'input_region': input_region,
                               'cell_number': 0,
                               'distribution': distribution,
                               'correlation': 0.0,
                               'mu': 0.0,
                              })

            plt.close('all')
            fig = plt.figure(figsize=(10, 6))
            fig.suptitle('Soma region LFP\nconductance: %s\ninput region: %s'
                         % (param_dict['distribution'], param_dict['input_region']))
            fig.subplots_adjust(right=0.95, wspace=0.5, hspace=0.5, left=0.2, top=0.80, bottom=0.2)

            psd_ax_dict = {'xlim': [1e0, 5e2],
                           'xlabel': 'Frequency (Hz)',
                           'xticks': [1e0, 10, 100],
                           'ylim': [1e-10, 1e-4]}
            lines = None
            line_names = None
            num_plot_cols = 4

            elec = soma_layer_elec_idx

            fig.text(0.005, 0.75, 'Full density')
            fig.text(0.005, 0.35, 'Half density')
            for c, correlation in enumerate(correlations):
                param_dict['correlation'] = correlation

                plot_number_full = 0 * num_plot_cols + c
                plot_number_half = 1 * num_plot_cols + c
                ax_full = fig.add_subplot(2, num_plot_cols, plot_number_full + 1, **psd_ax_dict)
                ax_half = fig.add_subplot(2, num_plot_cols, plot_number_half + 1, **psd_ax_dict)
                ax_full.set_title('c = %1.2f' % correlation)
                ax_full.grid(True)
                ax_half.grid(True)
                lines = []
                line_names = []
                for mu in mus:
                    param_dict['mu'] = mu

                    ns = NeuralSimulation(**param_dict)

                    name_full_density = 'summed_center_signal_%s_%dum' % (ns.population_sim_name, pop_size)
                    name_half_density = 'summed_center_signal_half_density_%s_%dum' % (ns.population_sim_name, pop_size)
                    lfp_full = np.load(join(folder, '%s.npy' % name_full_density))
                    lfp_half = np.load(join(folder, '%s.npy' % name_half_density))
                    # freq, psd = tools.return_freq_and_psd(ns.timeres_python/1000., lfp[elec])
                    freq, psd_full = tools.return_freq_and_psd_welch(lfp_full[elec], ns.welch_dict)
                    freq, psd_half = tools.return_freq_and_psd_welch(lfp_half[elec], ns.welch_dict)
                    l, = ax_full.loglog(freq, psd_full[0], c=qa_clr_dict[mu], lw=3)
                    l, = ax_half.loglog(freq, psd_half[0], c=qa_clr_dict[mu], lw=3)

                    lines.append(l)
                    line_names.append(conductance_names[mu])

                    ax_full.set_ylabel('LFP-PSD ($\mu$V$^2$/Hz)', labelpad=-5)
                    ax_full.set_xticklabels(['', '1', '10', '100'])
                    ax_full.set_yticks(ax_full.get_yticks()[1:-1][::2])

                    ax_half.set_ylabel('LFP-PSD ($\mu$V$^2$/Hz)', labelpad=-5)
                    ax_half.set_xticklabels(['', '1', '10', '100'])
                    ax_half.set_yticks(ax_half.get_yticks()[1:-1][::2])

            fig.legend(lines, line_names, loc='lower center', frameon=False, ncol=3)
            simplify_axes(fig.axes)
            plt.savefig(join(param_dict['root_folder'], param_dict['save_folder'],
                             'Figure_population_density_%s_%s.png' % (param_dict['distribution'], param_dict['input_region'])))
            plt.close('all')



def plot_population_size_effect(param_dict):

    conductance_names = {-0.5: 'regenerative',
                         0.0: 'passive-frozen',
                         2.0: 'restorative'}

    correlations = param_dict['correlations']
    mus = [-0.5, 0.0, 2.0]
    folder = join(param_dict['root_folder'], param_dict['save_folder'], 'simulations')
    pop_sizes = [50, 250, 500]
    for input_region in param_dict['input_regions']:
        for distribution in param_dict['distributions']:
            print input_region, distribution
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
                           'ylim': [1e-10, 1e-4]}
            lines = None
            line_names = None
            num_plot_cols = 4

            elec = 60
            for idx, pop_size in enumerate(pop_sizes):
                fig.text(0.01, 0.8 - idx / (len(pop_sizes) + 1), 'R = %d $\mu$m' % pop_size)
                for c, correlation in enumerate(correlations):
                    param_dict['correlation'] = correlation
                    plot_number = idx * num_plot_cols + c
                    ax_tuft = fig.add_subplot(3, num_plot_cols, plot_number + 1, **psd_ax_dict)
                    ax_tuft.set_title('c = %1.2f' % correlation)
                    ax_tuft.grid(True)
                    lines = []
                    line_names = []
                    for mu in mus:
                        param_dict['mu'] = mu

                        ns = NeuralSimulation(**param_dict)

                        name = 'summed_lateral_signal_%s_%dum' % (ns.population_sim_name, pop_size)
                        lfp = np.load(join(folder, '%s.npy' % name))
                        # freq, psd = tools.return_freq_and_psd(ns.timeres_python/1000., lfp[elec])
                        freq, psd = tools.return_freq_and_psd_welch(lfp[elec], ns.welch_dict)
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


def plot_depth_resolved(param_dict):

    correlations = param_dict['correlations']
    folder = join(param_dict['root_folder'], param_dict['save_folder'], 'simulations')
    pop_size = 500
    mu = -0.5

    param_dict.update({'input_region': 'homogeneous',
                       'cell_number': 0,
                       'distribution': 'linear_increase',
                       'correlation': 0.0,
                       'mu': mu,
                      })

    param_dict['input_region'] = 'distal_tuft'
    elec_z = param_dict['center_electrode_parameters']['z']

    z = np.linspace(elec_z[0], elec_z[-1], len(elec_z) + 1)

    input_regions = ['distal_tuft', 'homogeneous', 'basal']
    distributions = ['uniform', 'linear_increase', 'linear_decrease']

    plt.close('all')
    fig = plt.figure(figsize=(18, 9))
    fig.subplots_adjust(right=0.95, wspace=0.2, hspace=0.5, left=0.05, top=0.85, bottom=0.2)


    psd_ax_dict = {#'ylim': [-200, 1200],
                    'xlim': [2e0, 5e2],
                    'yticks': [],
                   # 'xlabel': 'Frequency (Hz)',
                   'xticks': [1e0, 10, 100],
                   'xscale': 'log'}
    num_plot_cols = 15
    num_plot_rows = 3
    im_dict = {'cmap': 'hot', 'norm': LogNorm(), 'vmin': 1e-8, 'vmax': 1e-2,}
    fig.text(0.17, 0.95, 'Uniform conductance')
    fig.text(0.45, 0.95, 'Linear increasing conductance')
    fig.text(0.75, 0.95, 'Linear decreasing conductance')

    fig.text(0.5, 0.92, '$\mu$*=%+1.1f' % mu)
    fig.text(0.05, 0.75, 'Distal tuft\ninput', ha='center')
    fig.text(0.05, 0.5, 'Homogeneous\ninput', ha='center')
    fig.text(0.05, 0.25, 'Basal\ninput', ha='center')

    for i, input_region in enumerate(input_regions):
        for d, distribution in enumerate(distributions):
            for c, correlation in enumerate(correlations):
                # print input_region, distribution, correlation
                plot_number = i * (num_plot_cols) + d * (2 + len(distributions)) + c + 2
                # print i, d, c, plot_number

                ax = fig.add_subplot(num_plot_rows, num_plot_cols, plot_number, **psd_ax_dict)
                ax.set_title('c=%1.2f' % correlation)

                param_dict['correlation'] = correlation
                param_dict['input_region'] = input_region
                param_dict['distribution'] = distribution
                ns = NeuralSimulation(**param_dict)
                name = 'summed_center_signal_%s_%dum' % (ns.population_sim_name, pop_size)
                lfp = np.load(join(folder, '%s.npy' % name))[:, :]
                # freq, psd = tools.return_freq_and_psd(ns.timeres_python/1000., lfp)
                freq, psd = tools.return_freq_and_psd_welch(lfp, ns.welch_dict)


                freq_idx = np.argmin(np.abs(freq - param_dict['max_freq']))

                img = ax.pcolormesh(freq[1:freq_idx], z, psd[:, 1:freq_idx], **im_dict)
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
    plt.savefig(join(param_dict['root_folder'], param_dict['save_folder'], 'Figure_Leski_13_%+1.2f.png' % mu))
    plt.close('all')


def plot_figure_5(param_dict):

    input_region_clr = {'distal_tuft': 'lightgreen',
                        'homogeneous': 'darkgreen'}
    correlations = param_dict['correlations']
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

    ax_morph_distal_tuft = fig.add_subplot(1, 12, 1, aspect=1, frameon=False, xticks=[], yticks=[])
    ax_morph_homogeneous = fig.add_subplot(1, 12, 2, aspect=1, frameon=False, xticks=[], yticks=[])

    [ax_morph_distal_tuft.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]], lw=2,
                     c=cell_color, zorder=0) for idx in xrange(len(xmid))]
    plt.savefig('morph.pdf')

    ax_morph_distal_tuft.plot(xmid[synidx_tuft], zmid[synidx_tuft], '.', c=input_region_clr['distal_tuft'], ms=4)

    [ax_morph_homogeneous.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]], lw=2,
                     c=cell_color, zorder=0) for idx in xrange(len(xmid))]
    ax_morph_homogeneous.plot(xmid[synidx_homo], zmid[synidx_homo], '.', c=input_region_clr['homogeneous'], ms=4)

    psd_ax_dict = {'xlim': [1e0, 5e2],
                   'xlabel': 'Frequency (Hz)',
                   'xticks': [1e0, 10, 100],
                   'ylim': [1e-9, 1e-3]}
    lines = None
    line_names = None
    num_plot_cols = 5

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
                ax_tuft.set_title('c = %1.2f' % correlation)
                # mark_subplots(ax_tuft, 'BCD'[c])

            lines = []
            line_names = []
            sum = None
            for i, input_region in enumerate(['distal_tuft', 'homogeneous']):
                param_dict['input_region'] = input_region
                ns = NeuralSimulation(**param_dict)
                name = 'summed_lateral_signal_%s_%dum' % (ns.population_sim_name, pop_size)
                lfp = np.load(join(folder, '%s.npy' % name))
                # freq, psd = tools.return_freq_and_psd(ns.timeres_python/1000., lfp[elec])
                freq, psd = tools.return_freq_and_psd_welch(lfp[elec], ns.welch_dict)
                if sum is None:
                    sum = lfp[elec]
                else:
                    sum += lfp[elec]
                f_idx_max = np.argmin(np.abs(freq - param_dict['max_freq']))
                f_idx_min = np.argmin(np.abs(freq - 1.))
                l, = ax_tuft.loglog(freq[f_idx_min:f_idx_max], psd[0][f_idx_min:f_idx_max],
                                    c=input_region_clr[input_region], lw=3, solid_capstyle='round')
                lines.append(l)
                line_names.append(input_region)
            freq, sum_psd = tools.return_freq_and_psd_welch(sum, ns.welch_dict)
            l, = ax_tuft.loglog(freq[f_idx_min:f_idx_max], sum_psd[0][f_idx_min:f_idx_max],
                                '--', c='k', lw=2, solid_capstyle='round')
            lines.append(l)
            line_names.append("Sum")

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



    correlations = param_dict['correlations']
    mus = [-0.5, 0.0, 2.0]
    folder = join(param_dict['root_folder'], param_dict['save_folder'], 'simulations')
    pop_size = 500
    num_plot_cols = 5
    param_dict.update({'input_region': 'homogeneous',
                       'cell_number': 0,
                       'distribution': ['uniform', 'linear_increase'][0],
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

    ax_morph_1 = fig.add_subplot(1, num_plot_cols, 1, aspect=1, frameon=False, xticks=[], yticks=[])

    [ax_morph_1.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]], lw=2,
                     c=cell_color, zorder=0) for idx in xrange(len(xmid))]
    ax_morph_1.plot(xmid[synidx], zmid[synidx], '.', c=syn_color, ms=4)

    psd_ax_dict = {'xlim': [1e0, 5e2],
                   'xlabel': 'Frequency (Hz)',
                   'xticks': [1e0, 10, 100],
                   'ylim': [1e-9, 1e-5]}
    lines = None
    line_names = None


    for idx, elec in enumerate([0, 60]):
        ax_morph_1.plot(elec_x[elec], elec_z[elec], 'o', c=elec_color, ms=15, mec='none')

        ax_morph_1.arrow(elec_x[elec], elec_z[elec], 100, 0, color=elec_color, zorder=50,
                         lw=2, head_width=30, head_length=50, clip_on=False)

        for c, correlation in enumerate(correlations):
            param_dict['correlation'] = correlation
            plot_number = idx * num_plot_cols + c
            ax_tuft = fig.add_subplot(2, num_plot_cols, plot_number + 2, **psd_ax_dict)

            if idx == 0:
                ax_tuft.set_title('c = %1.2f' % correlation)
                # mark_subplots(ax_tuft, 'BCD'[c])

            lines = []
            line_names = []
            for mu in mus:
                param_dict['mu'] = mu

                ns = NeuralSimulation(**param_dict)
                name = 'summed_lateral_signal_%s_%dum' % (ns.population_sim_name, pop_size)
                lfp = np.load(join(folder, '%s.npy' % name))
                # freq, psd = tools.return_freq_and_psd(ns.timeres_python/1000., lfp[elec])
                freq, psd = tools.return_freq_and_psd_welch(lfp[elec], ns.welch_dict)
                f_idx_max = np.argmin(np.abs(freq - param_dict['max_freq']))
                f_idx_min = np.argmin(np.abs(freq - 1.))
                l, = ax_tuft.loglog(freq[f_idx_min:f_idx_max], psd[0][f_idx_min:f_idx_max],
                                    c=qa_clr_dict[mu], lw=3, clip_on=False, solid_capstyle='round')
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

    correlations = param_dict['correlations']
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

    ax_morph_1 = fig.add_subplot(1, 5, 1, aspect=1, frameon=False, xticks=[], yticks=[])

    [ax_morph_1.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]], lw=2,
                     c=cell_color, zorder=0) for idx in xrange(len(xmid))]
    ax_morph_1.plot(xmid[synidx], zmid[synidx], '.', c=syn_color, ms=4)

    psd_ax_dict = {'xlim': [1e0, 5e2],
                   'xlabel': 'Frequency (Hz)',
                   'xticks': [1e0, 10, 100],
                   'ylim': [1e-9, 1e-4]}
    lines = None
    line_names = None
    num_plot_cols = 5

    for idx, elec in enumerate([0, 60]):
        ax_morph_1.plot(elec_x[elec], elec_z[elec], 'o', c=elec_color, ms=15, mec='none')

        ax_morph_1.arrow(elec_x[elec], elec_z[elec], 100, 0, color=elec_color, zorder=50,
                         lw=2, head_width=30, head_length=50, clip_on=False)

        for c, correlation in enumerate(correlations):
            param_dict['correlation'] = correlation
            plot_number = idx * num_plot_cols + c
            ax_tuft = fig.add_subplot(2, num_plot_cols, plot_number + 2, **psd_ax_dict)

            if idx == 0:
                ax_tuft.set_title('c = %1.2f' % correlation)
                mark_subplots(ax_tuft, 'BCDE'[c])

            lines = []
            line_names = []
            for mu in mus:
                param_dict['mu'] = mu

                ns = NeuralSimulation(**param_dict)
                name = 'summed_lateral_signal_%s_%dum' % (ns.population_sim_name, pop_size)
                lfp = np.load(join(folder, '%s.npy' % name))
                # freq, psd = tools.return_freq_and_psd(ns.timeres_python/1000., lfp[elec])
                freq, psd = tools.return_freq_and_psd_welch(lfp[elec], ns.welch_dict)
                f_idx_max = np.argmin(np.abs(freq - param_dict['max_freq']))
                f_idx_min = np.argmin(np.abs(freq - 1.))
                l, = ax_tuft.loglog(freq[f_idx_min:f_idx_max], psd[0][f_idx_min:f_idx_max],
                                    c=qa_clr_dict[mu], lw=3, solid_capstyle='round')
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
        name_res = 'summed_lateral_signal_%s_%dum' % (ns_1.population_sim_name, pop_size)
        lfp_1[mu] = np.load(join(folder, '%s.npy' % name_res))

        param_dict_2['mu'] = mu
        ns_2 = NeuralSimulation(**param_dict_2)
        name_res = 'summed_lateral_signal_%s_%dum' % (ns_2.population_sim_name, pop_size)
        lfp_2[mu] = np.load(join(folder, '%s.npy' % name_res))

    # xmid = np.load(join(folder, 'xmid_%s_generic.npy' % param_dict_1['cell_name']))
    # zmid = np.load(join(folder, 'zmid_%s_generic.npy' % param_dict_1['cell_name']))
    # xstart = np.load(join(folder, 'xstart_%s_generic.npy' % param_dict_1['cell_name']))
    # zstart = np.load(join(folder, 'zstart_%s_generic.npy' % param_dict_1['cell_name']))
    # xend = np.load(join(folder, 'xend_%s_generic.npy' % param_dict_1['cell_name']))
    # zend = np.load(join(folder, 'zend_%s_generic.npy' % param_dict_1['cell_name']))

    # synidx_1 = np.load(join(folder, 'synidx_%s.npy' % ns_1.sim_name))
    # synidx_2 = np.load(join(folder, 'synidx_%s.npy' % ns_2.sim_name))

    # elec_x = param_dict_1['lateral_electrode_parameters']['x']
    # elec_z = param_dict_1['lateral_electrode_parameters']['z']

    plt.close('all')
    fig = plt.figure(figsize=(10, 5))
    fig.subplots_adjust(right=0.95, wspace=0.5, hspace=0.4, left=0., top=0.95, bottom=0.2)

    ax_morph_1 = fig.add_subplot(1, 4, 1, aspect=1, frameon=False, xticks=[], yticks=[])
    ax_morph_2 = fig.add_subplot(1, 4, 3, aspect=1, frameon=False, xticks=[], yticks=[])

    fig_folder = join(param_dict_1['root_folder'], 'figures')
    dist_image = plt.imread(join(fig_folder, 'linear_increase_distal_tuft.png'))
    homo_image = plt.imread(join(fig_folder, 'linear_increase_homogeneous.png'))

    ax_morph_1.imshow(dist_image)
    ax_morph_2.imshow(homo_image)
    # [ax_morph_1.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]], lw=2,
    #                  c=cell_color, zorder=0) for idx in xrange(len(xmid))]
    # ax_morph_1.plot(xmid[synidx_1], zmid[synidx_1], '.', c=syn_color, ms=4)

    # [ax_morph_2.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]], lw=2,
    #                  c=cell_color, zorder=0)
    #     for idx in xrange(len(xmid))]
    # ax_morph_2.plot(xmid[synidx_2], zmid[synidx_2], '.', c=syn_color, ms=4)

    psd_ax_dict = {'xlim': [1e0, 5e2],
                   'xlabel': 'Frequency (Hz)',
                   'xticks': [1e0, 10, 100],}
    lines = None
    line_names = None
    for idx, elec in enumerate([0, 60]):
        # ax_morph_1.plot(elec_x[elec], elec_z[elec], 'o', c=elec_color, ms=15, mec='none')
        # ax_morph_2.plot(elec_x[elec], elec_z[elec], 'o', c=elec_color, ms=15, mec='none')

        # ax_morph_1.arrow(elec_x[elec], elec_z[elec], 100, 0, color=elec_color, zorder=50,
        #                  lw=2, head_width=30, head_length=50, clip_on=False)
        # ax_morph_2.arrow(elec_x[elec], elec_z[elec], 100, 0, color=elec_color, zorder=50,
        #                  lw=2, head_width=30, head_length=50, clip_on=False)

        num_plot_cols = 4
        plot_number = idx * num_plot_cols
        ax_tuft = fig.add_subplot(2, num_plot_cols, plot_number + 2, ylim=[1e-10, 1e-6], **psd_ax_dict)
        ax_homo = fig.add_subplot(2, num_plot_cols, plot_number + 4, ylim=[1e-10, 1e-6], **psd_ax_dict)
        lines = []
        line_names = []
        for mu in [-0.5, 0.0, 2.0]:
            # freq, psd_tuft_res = tools.return_freq_and_psd(ns_1.timeres_python/1000., lfp_1[mu][elec])
            # smooth_psd_tuft_res = tools.smooth_signal(freq[::average_over], freq, psd_tuft_res[0])
            # ax_tuft.loglog(freq[::average_over], smooth_psd_tuft_res, c=qa_clr_dict[mu], lw=3)
            freq, psd_tuft_res = tools.return_freq_and_psd_welch(lfp_1[mu][elec], ns_1.welch_dict)
            f_idx_max = np.argmin(np.abs(freq - param_dict['max_freq']))
            f_idx_min = np.argmin(np.abs(freq - 1.))
            ax_tuft.loglog(freq[f_idx_min:f_idx_max], psd_tuft_res[0][f_idx_min:f_idx_max],
                           c=qa_clr_dict[mu], lw=3, solid_capstyle='round')
            ax_tuft.set_ylabel('LFP-PSD ($\mu$V$^2$/Hz)', labelpad=-5)
            ax_tuft.set_xticklabels(['', '1', '10', '100'])
            ax_tuft.set_yticks(ax_tuft.get_yticks()[1:-1][::2])

            # freq, psd_homogeneous = tools.return_freq_and_psd(ns_2.timeres_python/1000., lfp_2[mu][elec])
            # smooth_psd_homogeneous = tools.smooth_signal(freq[::average_over], freq, psd_homogeneous[0])
            # l, = ax_homo.loglog(freq[::average_over], smooth_psd_homogeneous, c=qa_clr_dict[mu], lw=3, solid_capstyle='round')

            freq, psd_homogeneous = tools.return_freq_and_psd_welch(lfp_2[mu][elec], ns_2.welch_dict)
            l, = ax_homo.loglog(freq, psd_homogeneous[0], c=qa_clr_dict[mu], lw=3, solid_capstyle='round')

            lines.append(l)
            line_names.append(conductance_names[mu])
            ax_homo.set_xticklabels(['', '1', '10', '100'])
            ax_homo.set_ylabel('LFP-PSD ($\mu$V$^2$/Hz)', labelpad=-5)
            ax_homo.set_yticks(ax_homo.get_yticks()[1:-1][::2])

    fig.legend(lines, line_names, loc='lower center', frameon=False, ncol=3)
    simplify_axes(fig.axes)
    mark_subplots([ax_morph_1, ax_morph_2], ypos=1., xpos=0.1)
    plt.savefig(join(param_dict_1['root_folder'], 'figures', 'Figure_1_population_classic.png'))
    plt.close('all')

def plot_figure_1_classic(param_dict):

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
                             'correlation': correlation,
                             })
    param_dict_2 = param_dict.copy()
    input_region_2 = 'homogeneous'
    param_dict_2.update({'input_region': input_region_2,
                         'cell_number': 0,
                         'correlation': correlation,
                         })
    ns_1 = None
    ns_2 = None
    for conductance_type in param_dict['conductance_types']:
        param_dict_1['conductance_type'] = conductance_type
        ns_1 = NeuralSimulation(**param_dict_1)
        name_res = 'summed_lateral_signal_%s_%dum' % (ns_1.population_sim_name, pop_size)
        lfp_1[conductance_type] = np.load(join(folder, '%s.npy' % name_res))

        param_dict_2['conductance_type'] = conductance_type

        ns_2 = NeuralSimulation(**param_dict_2)
        name_res = 'summed_lateral_signal_%s_%dum' % (ns_2.population_sim_name, pop_size)
        lfp_2[conductance_type] = np.load(join(folder, '%s.npy' % name_res))

    # xmid = np.load(join(folder, 'xmid_%s_generic.npy' % param_dict_1['cell_name']))
    # zmid = np.load(join(folder, 'zmid_%s_generic.npy' % param_dict_1['cell_name']))
    # xstart = np.load(join(folder, 'xstart_%s_generic.npy' % param_dict_1['cell_name']))
    # zstart = np.load(join(folder, 'zstart_%s_generic.npy' % param_dict_1['cell_name']))
    # xend = np.load(join(folder, 'xend_%s_generic.npy' % param_dict_1['cell_name']))
    # zend = np.load(join(folder, 'zend_%s_generic.npy' % param_dict_1['cell_name']))

    # synidx_1 = np.load(join(folder, 'synidx_%s.npy' % ns_1.sim_name))
    # synidx_2 = np.load(join(folder, 'synidx_%s.npy' % ns_2.sim_name))

    # elec_x = param_dict_1['lateral_electrode_parameters']['x']
    # elec_z = param_dict_1['lateral_electrode_parameters']['z']

    plt.close('all')
    fig = plt.figure(figsize=(10, 5))
    fig.subplots_adjust(right=0.95, wspace=0.5, hspace=0.4, left=0., top=0.95, bottom=0.2)

    ax_morph_1 = fig.add_subplot(1, 4, 1, aspect=1, frameon=False, xticks=[], yticks=[])
    ax_morph_2 = fig.add_subplot(1, 4, 3, aspect=1, frameon=False, xticks=[], yticks=[])

    fig_folder = join(param_dict_1['root_folder'], 'figures')
    dist_image = plt.imread(join(fig_folder, 'linear_increase_distal_tuft.png'))
    homo_image = plt.imread(join(fig_folder, 'linear_increase_homogeneous.png'))

    ax_morph_1.imshow(dist_image)
    ax_morph_2.imshow(homo_image)
    # [ax_morph_1.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]], lw=2,
    #                  c=cell_color, zorder=0) for idx in xrange(len(xmid))]
    # ax_morph_1.plot(xmid[synidx_1], zmid[synidx_1], '.', c=syn_color, ms=4)

    # [ax_morph_2.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]], lw=2,
    #                  c=cell_color, zorder=0)
    #     for idx in xrange(len(xmid))]
    # ax_morph_2.plot(xmid[synidx_2], zmid[synidx_2], '.', c=syn_color, ms=4)

    psd_ax_dict = {'xlim': [1e0, 5e2],
                   'xlabel': 'Frequency (Hz)',
                   'xticks': [1e0, 10, 100],}
    lines = None
    line_names = None
    for idx, elec in enumerate([0, 60]):
        # ax_morph_1.plot(elec_x[elec], elec_z[elec], 'o', c=elec_color, ms=15, mec='none')
        # ax_morph_2.plot(elec_x[elec], elec_z[elec], 'o', c=elec_color, ms=15, mec='none')

        # ax_morph_1.arrow(elec_x[elec], elec_z[elec], 100, 0, color=elec_color, zorder=50,
        #                  lw=2, head_width=30, head_length=50, clip_on=False)
        # ax_morph_2.arrow(elec_x[elec], elec_z[elec], 100, 0, color=elec_color, zorder=50,
        #                  lw=2, head_width=30, head_length=50, clip_on=False)

        num_plot_cols = 4
        plot_number = idx * num_plot_cols
        ax_tuft = fig.add_subplot(2, num_plot_cols, plot_number + 2, ylim=[1e-10, 1e-6], **psd_ax_dict)
        ax_homo = fig.add_subplot(2, num_plot_cols, plot_number + 4, ylim=[1e-10, 1e-6], **psd_ax_dict)
        lines = []
        line_names = []
        for conductance_type in param_dict['conductance_types']:

            # freq, psd_tuft_res = tools.return_freq_and_psd(ns_1.timeres_python/1000., lfp_1[mu][elec])
            # smooth_psd_tuft_res = tools.smooth_signal(freq[::average_over], freq, psd_tuft_res[0])
            # ax_tuft.loglog(freq[::average_over], smooth_psd_tuft_res, c=qa_clr_dict[mu], lw=3)
            freq, psd_tuft_res = tools.return_freq_and_psd_welch(lfp_1[conductance_type][elec], ns_1.welch_dict)
            f_idx_max = np.argmin(np.abs(freq - param_dict['max_freq']))
            f_idx_min = np.argmin(np.abs(freq - 1.))
            ax_tuft.loglog(freq[f_idx_min:f_idx_max], psd_tuft_res[0][f_idx_min:f_idx_max],
                           c=conductance_clr_dict[conductance_type], lw=3, solid_capstyle='round')
            ax_tuft.set_ylabel('LFP-PSD ($\mu$V$^2$/Hz)', labelpad=-5)
            ax_tuft.set_xticklabels(['', '1', '10', '100'])
            ax_tuft.set_yticks(ax_tuft.get_yticks()[1:-1][::2])

            # freq, psd_homogeneous = tools.return_freq_and_psd(ns_2.timeres_python/1000., lfp_2[mu][elec])
            # smooth_psd_homogeneous = tools.smooth_signal(freq[::average_over], freq, psd_homogeneous[0])
            # l, = ax_homo.loglog(freq[::average_over], smooth_psd_homogeneous, c=qa_clr_dict[mu], lw=3, solid_capstyle='round')

            freq, psd_homogeneous = tools.return_freq_and_psd_welch(lfp_2[conductance_type][elec], ns_2.welch_dict)
            l, = ax_homo.loglog(freq, psd_homogeneous[0], c=conductance_clr_dict[conductance_type], lw=3, solid_capstyle='round')

            lines.append(l)
            line_names.append(conductance_type)
            ax_homo.set_xticklabels(['', '1', '10', '100'])
            ax_homo.set_ylabel('LFP-PSD ($\mu$V$^2$/Hz)', labelpad=-5)
            ax_homo.set_yticks(ax_homo.get_yticks()[1:-1][::2])

    fig.legend(lines, line_names, loc='lower center', frameon=False, ncol=4)
    simplify_axes(fig.axes)
    mark_subplots([ax_morph_1, ax_morph_2], ypos=1., xpos=0.1)
    plt.savefig(join(param_dict_1['root_folder'], 'figures', 'Figure_1_population_classic.png'))
    plt.close('all')



def plot_figure_1_single_cell(param_dict):

    conductance_names = {-0.5: 'regenerative',
                         0.0: 'passive-frozen',
                         2.0: 'restorative'}
    folder = join(param_dict['root_folder'], param_dict['save_folder'], 'simulations')
    pop_size = 500

    distribution_1 = 'linear_increase'
    correlation = 0.0
    average_over = 8

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
    distribution_2 = 'linear_increase'
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
        name_res = 'lateral_sig_%s' % (ns_1.sim_name)
        lfp_1[mu] = 1000 * np.load(join(folder, '%s.npy' % name_res))

        param_dict_2['mu'] = mu
        ns_2 = NeuralSimulation(**param_dict_2)
        name_res = 'lateral_sig_%s' % (ns_2.sim_name)
        lfp_2[mu] = 1000 * np.load(join(folder, '%s.npy' % name_res))

    # xmid = np.load(join(folder, 'xmid_%s_generic.npy' % param_dict_1['cell_name']))
    # zmid = np.load(join(folder, 'zmid_%s_generic.npy' % param_dict_1['cell_name']))
    # xstart = np.load(join(folder, 'xstart_%s_generic.npy' % param_dict_1['cell_name']))
    # zstart = np.load(join(folder, 'zstart_%s_generic.npy' % param_dict_1['cell_name']))
    # xend = np.load(join(folder, 'xend_%s_generic.npy' % param_dict_1['cell_name']))
    # zend = np.load(join(folder, 'zend_%s_generic.npy' % param_dict_1['cell_name']))

    # synidx_1 = np.load(join(folder, 'synidx_%s.npy' % ns_1.sim_name))
    # synidx_2 = np.load(join(folder, 'synidx_%s.npy' % ns_2.sim_name))

    # elec_x = param_dict_1['lateral_electrode_parameters']['x']
    # elec_z = param_dict_1['lateral_electrode_parameters']['z']

    plt.close('all')
    fig = plt.figure(figsize=(10, 5))
    fig.subplots_adjust(right=0.95, wspace=0.5, hspace=0.4, left=0., top=0.95, bottom=0.2)

    ax_morph_1 = fig.add_subplot(1, 4, 1, aspect=1, frameon=False, xticks=[], yticks=[])
    ax_morph_2 = fig.add_subplot(1, 4, 3, aspect=1, frameon=False, xticks=[], yticks=[])

    # [ax_morph_1.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]], lw=2,
    #                  c=cell_color, zorder=0) for idx in xrange(len(xmid))]
    # ax_morph_1.plot(xmid[synidx_1], zmid[synidx_1], '.', c=syn_color, ms=4)

    # [ax_morph_2.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]], lw=2,
    #                  c=cell_color, zorder=0)
    #     for idx in xrange(len(xmid))]
    # ax_morph_2.plot(xmid[synidx_2], zmid[synidx_2], '.', c=syn_color, ms=4)

    fig_folder = join(param_dict_1['root_folder'], 'figures')
    dist_image = plt.imread(join(fig_folder, 'single_cell_distal_tuft.png'))
    homo_image = plt.imread(join(fig_folder, 'single_cell_homogeneous.png'))

    ax_morph_1.imshow(dist_image)
    ax_morph_2.imshow(homo_image)

    psd_ax_dict = {'xlim': [1e0, 5e2],
                   'xlabel': 'Frequency (Hz)',
                   'xticks': [1e0, 10, 100],}
    lines = None
    line_names = None
    for idx, elec in enumerate([0, 60]):
        # ax_morph_1.plot(elec_x[elec], elec_z[elec], 'o', c=elec_color, ms=15, mec='none')
        # ax_morph_2.plot(elec_x[elec], elec_z[elec], 'o', c=elec_color, ms=15, mec='none')
        #
        # ax_morph_1.arrow(elec_x[elec], elec_z[elec], 100, 0, color=elec_color, zorder=50,
        #                  lw=2, head_width=30, head_length=50, clip_on=False)
        # ax_morph_2.arrow(elec_x[elec], elec_z[elec], 100, 0, color=elec_color, zorder=50,
        #                  lw=2, head_width=30, head_length=50, clip_on=False)

        num_plot_cols = 4
        plot_number = idx * num_plot_cols
        ax_tuft = fig.add_subplot(2, num_plot_cols, plot_number + 2, ylim=[1e-12, 1e-8], **psd_ax_dict)
        ax_homo = fig.add_subplot(2, num_plot_cols, plot_number + 4, ylim=[1e-12, 1e-8], **psd_ax_dict)
        lines = []
        line_names = []
        for mu in [-0.5, 0.0, 2.0]:
            # freq, psd_tuft_res = tools.return_freq_and_psd(ns_1.timeres_python/1000., lfp_1[mu][elec])
            # smooth_psd_tuft_res = tools.smooth_signal(freq[::average_over], freq, psd_tuft_res[0])
            # ax_tuft.loglog(freq[::average_over], 1000 * smooth_psd_tuft_res, c=qa_clr_dict[mu], lw=3)

            freq, psd_tuft_res = tools.return_freq_and_psd_welch(lfp_1[mu][elec], ns_1.welch_dict)
            f_idx_max = np.argmin(np.abs(freq - param_dict['max_freq']))
            f_idx_min = np.argmin(np.abs(freq - 1.))
            ax_tuft.loglog(freq[f_idx_min:f_idx_max],
                           psd_tuft_res[0][f_idx_min:f_idx_max],
                           c=qa_clr_dict[mu], lw=3, solid_capstyle='round')


            ax_tuft.set_ylabel('LFP-PSD ($\mu$V$^2$/Hz)', labelpad=-5)
            ax_tuft.set_xticklabels(['', '1', '10', '100'])
            ax_tuft.set_yticks(ax_tuft.get_yticks()[1:-1][::2])

            # freq, psd_homogeneous = tools.return_freq_and_psd(ns_2.timeres_python/1000., lfp_2[mu][elec])
            # smooth_psd_homogeneous = tools.smooth_signal(freq[::average_over], freq, psd_homogeneous[0])
            # l, = ax_homo.loglog(freq[::average_over], 1000 * smooth_psd_homogeneous, c=qa_clr_dict[mu], lw=3, solid_capstyle='round')

            freq, psd_homogeneous = tools.return_freq_and_psd_welch(lfp_2[mu][elec], ns_2.welch_dict)
            l, = ax_homo.loglog(freq, psd_homogeneous[0], c=qa_clr_dict[mu], lw=3, solid_capstyle='round')

            lines.append(l)
            line_names.append(conductance_names[mu])
            ax_homo.set_xticklabels(['', '1', '10', '100'])
            ax_homo.set_ylabel('LFP-PSD ($\mu$V$^2$/Hz)', labelpad=-5)
            ax_homo.set_yticks(ax_homo.get_yticks()[1:-1][::2])

    fig.legend(lines, line_names, loc='lower center', frameon=False, ncol=3)
    simplify_axes(fig.axes)
    mark_subplots([ax_morph_1, ax_morph_2], ypos=1., xpos=0.1)
    plt.savefig(join(param_dict_1['root_folder'], 'figures', 'Figure_1_single_cell.png'))
    plt.close('all')

def plot_figure_1_single_cell_difference(param_dict):

    conductance_names = {-0.5: 'regenerative',
                         0.0: 'passive-frozen',
                         2.0: 'restorative'}
    folder = join(param_dict['root_folder'], param_dict['save_folder'], 'simulations')

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
    param_dict_1['mu'] = 0.0
    ns_0 = NeuralSimulation(**param_dict_1)
    name_pas = 'lateral_sig_%s' % ns_0.sim_name

    lfp_pas = 1000 * np.load(join(folder, '%s.npy' % name_pas))

    for mu in [-0.5, 0.0, 2.0]:
        param_dict_1['mu'] = mu
        ns_1 = NeuralSimulation(**param_dict_1)
        name_res = 'lateral_sig_%s' % ns_1.sim_name
        lfp = 1000 * np.load(join(folder, '%s.npy' % name_res))
        # freq, psd_tuft_res = tools.return_freq_and_psd(ns_1.timeres_python/1000., lfp[[0, 60]])
        freq, psd_tuft_res = tools.return_freq_and_psd_welch(lfp, ns_1.welch_dict)
        lfp_1[mu] = psd_tuft_res

        param_dict_2['mu'] = mu
        ns_2 = NeuralSimulation(**param_dict_2)
        name_res = 'lateral_sig_%s' % ns_2.sim_name
        lfp = 1000 * np.load(join(folder, '%s.npy' % name_res))
        # freq, psd_homogeneous = tools.return_freq_and_psd(ns_2.timeres_python/1000., lfp[[0, 60]])
        freq, psd_homogeneous = tools.return_freq_and_psd_welch(lfp, ns_2.welch_dict)
        lfp_2[mu] = psd_homogeneous

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

    psd_ax_dict = {'xlim': [1e0, 1e2],
                   'ylim': [-2, 2],
                   'xlabel': 'Frequency (Hz)',
                   'xticks': [1e0, 10, 100],
                   'xscale': 'log'}
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
        ax_tuft = fig.add_subplot(2, num_plot_cols, plot_number + 2, **psd_ax_dict)
        ax_homo = fig.add_subplot(2, num_plot_cols, plot_number + 4, **psd_ax_dict)
        lines = []
        line_names = []
        for mu in [-0.5, 2.0]:
            f_idx_max = np.argmin(np.abs(freq - param_dict['max_freq']))
            f_idx_min = np.argmin(np.abs(freq - 1.))
            ax_tuft.plot(freq[f_idx_min:f_idx_max],
                         np.log(lfp_1[mu][elec][f_idx_min:f_idx_max] /lfp_1[0.0][elec][f_idx_min:f_idx_max]),
                         c=qa_clr_dict[mu], lw=3, solid_capstyle='round')

            ax_tuft.set_ylabel(r'$\log(\frac{\rm{QA}}{\rm{PASSIVE}}$)', labelpad=-5)
            ax_tuft.set_xticklabels(['', '1', '10', '100'])
            ax_tuft.set_yticks(ax_tuft.get_yticks()[::2])

            l, = ax_homo.plot(freq[f_idx_min:f_idx_max], np.log(lfp_2[mu][elec][f_idx_min:f_idx_max] / lfp_2[0.0][elec][f_idx_min:f_idx_max]), c=qa_clr_dict[mu],
                                lw=3, solid_capstyle='round')
            lines.append(l)
            line_names.append(conductance_names[mu])
            ax_homo.set_xticklabels(['', '1', '10', '100'])
            ax_homo.set_ylabel(r'$\log(\frac{\rm{QA}}{\rm{PASSIVE}})$', labelpad=-5)
            ax_homo.set_yticks(ax_homo.get_yticks()[::2])

    fig.legend(lines, line_names, loc='lower center', frameon=False, ncol=3)
    simplify_axes(fig.axes)
    mark_subplots([ax_morph_1, ax_morph_2], ypos=0.95, xpos=0.1)
    plt.savefig(join(param_dict_1['root_folder'], param_dict_1['save_folder'], 'Figure_1_single_cell_difference.png'))
    plt.close('all')


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


    # plot_asymetric_conductance_time_average_movie(param_dict)
    # plot_asymetric_conductance_space_average_movie(param_dict)

    # from param_dicts import vmem_3D_population_params as param_dict
    # input_regions = ['homogeneous']
    # distributions = ['uniform']
    # correlations = [0.0]
    # for i, input_region in enumerate(input_regions):
    #     for d, distribution in enumerate(distributions):
    #         for c, correlation in enumerate(correlations):
    #             param_dict.update({
    #                                'mu': 0.0,
    #                                'distribution': distribution,
    #                                 'input_region': input_region,
    #                                 'correlation': correlation,
    #                                 'cell_number': 0,
    #                               })
    #
    #             plot_3d_rot_pop(param_dict)

    # plot_coherence(param_dict)
    # plot_generic_population_LFP(param_dict)
    # plot_classic_population_LFP(param_dict)
    # plot_simple_model_LFP(param_dict)

    # plot_linear_combination(param_dict)
    # plot_figure_1_classic(param_dict)
    # plot_figure_1_single_cell(param_dict)
    # plot_figure_1_single_cell_difference(param_dict)
    # plot_depth_resolved(param_dict)
    # plot_figure_2(param_dict)
    # plot_figure_3(param_dict)
    # plot_figure_5(param_dict)
    # plot_leski_13(param_dict)
    # plot_population_size_effect(param_dict)
    # plot_population_density_effect(param_dict)
    # plot_all_size_dependencies(param_dict)
    # plot_all_soma_sigs_classic(param_dict)
    # plot_all_soma_sigs(param_dict)
    # plot_LFP_time_trace(param_dict)
    plot_cell_population(param_dict)

    # sys.exit()