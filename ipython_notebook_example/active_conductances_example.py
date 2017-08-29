#!/usr/bin/env python
'''
Simple Hay simulation with one synaptic input, and extracellular electrodes
'''
import os
import sys
from os.path import join
import numpy as np
import pylab as plt
import neuron
import LFPy
h = neuron.h
import scipy.fftpack as ff
import zipfile

if sys.version < '3':
    from urllib2 import urlopen
else:
    from urllib.request import urlopen


def remove_active_mechanisms(remove_list):
    mt = h.MechanismType(0)
    for sec in h.allsec():
        for seg in sec:
            for mech in remove_list:
                mt.select(mech)
                mt.remove()

def make_Ih_frozen(cell):
    """Rescales the passive leak conductance of the cell object, so that the
    membrane conductance of the Ih channel is included.
    Note that Ih must be removed afterwards."""

    h.t = 0
    h.finitialize(cell.v_init)
    h.fcurrent()
    i = 0
    ehcn_Ih = -45.
    for sec in h.allsec():
        for seg in sec:
            if hasattr(seg, "gIh_Ih"):
                seg.e_pas = (seg.g_pas * seg.e_pas + seg.gIh_Ih * ehcn_Ih) / (seg.g_pas + seg.gIh_Ih)
                seg.g_pas += seg.gIh_Ih
            i += 1


def make_cell_uniform(Vrest=-80):
    """ Shifts the passive leak reversal potential of the cell e_pas, so that
    each cell compartment will have the resting potential given by Vrest
    """
    h.t = 0
    h.finitialize(Vrest)
    h.fcurrent()
    for sec in h.allsec():
        for seg in sec:
            seg.e_pas = seg.v
            if h.ismembrane("na_ion"):
                seg.e_pas += seg.ina/seg.g_pas
            if h.ismembrane("k_ion"):
                seg.e_pas += seg.ik/seg.g_pas
            if h.ismembrane("ca_ion"):
                seg.e_pas += seg.ica/seg.g_pas
            if h.ismembrane("Ih"):
                seg.e_pas += seg.ihcn_Ih/seg.g_pas

def fetch_hay_model():
    """ Downloads the Hay model and tries to compile the mechanisms
    """
    #get the model files:
    u = urlopen('http://senselab.med.yale.edu/ModelDB/eavBinDown.asp?o=139653&a=23&mime=application/zip')
    localFile = open('L5bPCmodelsEH.zip', 'w')
    localFile.write(u.read())
    localFile.close()
    #unzip:
    myzip = zipfile.ZipFile('L5bPCmodelsEH.zip', 'r')
    myzip.extractall('.')
    myzip.close()

    #compile mod files every time, because of incompatibility with Mainen96 files:
    os.system('''
              cd L5bPCmodelsEH/mod/
              nrnivmodl
              ''')


def return_freq_and_psd(tvec, sig):
    """ Returns the freqency and Power Spectral Density (PSD) of sig"""
    sig = np.array(sig)
    if len(sig.shape) == 1:
        sig = np.array([sig])
    elif len(sig.shape) == 2:
        pass
    else:
        raise RuntimeError("Not compatible with given array shape!")
    timestep = (tvec[1] - tvec[0])/1000. if type(tvec) in [list, np.ndarray] else tvec
    sample_freq = ff.fftfreq(sig.shape[1], d=timestep)
    pidxs = np.where(sample_freq >= 0)
    freqs = sample_freq[pidxs]
    Y = ff.fft(sig, axis=1)[:, pidxs[0]]
    power = np.abs(Y)**2/Y.shape[1]
    return freqs, power


def return_cell(sim_time, conductance_type, holding_potential=None):
    """ Returns an LFPy Cell object, based on the cortical pyramidal cell model
    from Hay et al. (2011).

    Parameters
    ----------
    sim_time: float, int
        Duration of the simulation in ms.
    conductance_type: str
        Either 'active', 'passive', 'Ih' or 'Ih_frozen'. If 'active' all
        original ion-channels are included, if 'passive' they are all removed.
        'Ih' will include a single active Ih channel. 'Ih_frozen' results in
        a passive cell with the passive membrane conductance scaled to include
        the increased membrane conductance that comes with the Ih conductance
    holding_potential: int, float, None
        If a number is given (e.g. - 70), the passive leak reversal potential
        is modified to make the given number the resting potential of the cell,
        i.e. the resting potential will be uniform.
        Defaults to None, which does not enforce a uniform potential.

    """

    # Dict of which ion-channels to remove for a given conductance_type
    remove_lists = {'active': [],
                    'passive': ["Nap_Et2", "NaTa_t", "NaTs2_t", "SKv3_1",
                                "SK_E2", "K_Tst", "K_Pst", "Im", "Ih",
                                "CaDynamics_E2", "Ca_LVAst", "Ca"],
                    'Ih': ["Nap_Et2", "NaTa_t", "NaTs2_t", "SKv3_1",
                           "SK_E2", "K_Tst", "K_Pst", "Im",
                           "CaDynamics_E2", "Ca_LVAst", "Ca"],

                    'Ih_frozen': ["Nap_Et2", "NaTa_t", "NaTs2_t", "SKv3_1",
                           "SK_E2", "K_Tst", "K_Pst", "Im", "Ih",
                           "CaDynamics_E2", "Ca_LVAst", "Ca"]
                    }

    if not os.path.isfile('L5bPCmodelsEH/morphologies/cell1.asc'):
        fetch_hay_model()

    h("forall delete_section()")
    neuron.load_mechanisms('L5bPCmodelsEH/mod/')

    ##define cell parameters used as input to cell-class
    cellParameters = {
        'morphology': 'L5bPCmodelsEH/morphologies/cell1.asc',
        'templatefile': ['L5bPCmodelsEH/models/L5PCbiophys3.hoc',
                           'L5bPCmodelsEH/models/L5PCtemplate.hoc'],
        'templatename': 'L5PCtemplate',
        'templateargs': 'L5bPCmodelsEH/morphologies/cell1.asc',
        'passive': False,
        'nsegs_method': None,
        'timeres_NEURON': 2**-4,
        'timeres_python': 2**-4,
        'tstartms': -200,
        'tstopms': sim_time,
        'v_init': -70 if holding_potential is None else holding_potential,
        'celsius': 34,
        'pt3d': True,
    }
    cell = LFPy.TemplateCell(**cellParameters)

    if conductance_type == "Ih_frozen":
        make_Ih_frozen(cell)
    remove_active_mechanisms(remove_lists[conductance_type])
    if holding_potential is not None:
        make_cell_uniform(holding_potential)
    cell.set_rotation(x=4.729, y=-3.166)
    return cell

def example_synapse(synaptic_z_pos=0, conductance_type='active',
                    weight=0.001, holding_potential=-70):
    """
    Runs a NEURON simulation with a cortical pyramidal cell that receives a
    single synaptic input. The Local Field Potential is calculated and
    plotted.

    Parameters:
    ----------
    synaptic_z_pos: float, int
        position along the apical dendrite where the synapse is inserted.
    conductance_type: str
        Either 'active', 'passive', 'Ih' or 'Ih_frozen'. If 'active' all
        original ion-channels are included, if 'passive' they are all removed.
        'Ih' will include a single active Ih channel. 'Ih_frozen' results in
        a passive cell with the passive membrane conductance scaled to include
        the increased membrane conductance that comes with the Ih conductance
    weight: float
        Strength of synaptic input.
    holding_potential: int, float, None
        If a number is given (e.g. - 70), the passive leak reversal potential
        is modified to make the given number the resting potential of the cell,
        i.e. the resting potential will be uniform.
        Defaults to None, which does not enforce a uniform potential.
    """

    #  Making cell
    sim_time = 100

    cell = return_cell(sim_time, conductance_type, holding_potential)
    input_idx = cell.get_closest_idx(x=0., y=0., z=synaptic_z_pos)

    cell, synapse = make_synapse(cell, weight, input_idx)
    cell.simulate(rec_imem=True, rec_vmem=True)

    plot_electrode_signal(cell, input_idx, conductance_type, False)


def example_white_noise(synaptic_z_pos=0, conductance_type='active',
                        weight=0.001, holding_potential=None):
    """
    Runs a NEURON simulation with a cortical pyramidal cell that receives a
    single white noise input. The Local Field Potential is calculated and
    plotted.

    Parameters:
    ----------
    synaptic_z_pos: float, int
        position along the apical dendrite where the synapse is inserted.
    conductance_type: str
        Either 'active', 'passive', 'Ih' or 'Ih_frozen'. If 'active' all
        original ion-channels are included, if 'passive' they are all removed.
        'Ih' will include a single active Ih channel. 'Ih_frozen' results in
        a passive cell with the passive membrane conductance scaled to include
        the increased membrane conductance that comes with the Ih conductance
    weight: float
        Strength of synaptic input.
    holding_potential: int, float, None
        If a number is given (e.g. - 70), the passive leak reversal potential
        is modified to make the given number the resting potential of the cell,
        i.e. the resting potential will be uniform.
        Defaults to None, which does not enforce a uniform potential.
    """

    # Repeat same stimuli and use only the last repetition. This is to avoid
    # initiation artifacts. The original Hay model uses a long time to reach
    # equilibrium, mostly because of a calcium mechanism
    # that is not well initialized.
    repeats = 2
    sim_time = 500

    cell = return_cell(repeats * sim_time, conductance_type, holding_potential)
    input_idx = cell.get_closest_idx(x=0., y=0., z=synaptic_z_pos)
    cell, synapse, noiseVec = make_white_noise(cell, weight, input_idx)

    cell.simulate(rec_imem=True, rec_vmem=True)
    if repeats is not None:
        cut_off_idx = (len(cell.tvec) - 1) / repeats
        cell.tvec = cell.tvec[-cut_off_idx:] - cell.tvec[-cut_off_idx]
        cell.imem = cell.imem[:, -cut_off_idx:]
        cell.vmem = cell.vmem[:, -cut_off_idx:]

    plot_electrode_signal(cell, input_idx, conductance_type, True)


def make_synapse(cell, weight, input_idx):

    input_spike_train = np.array([10.])
    synapse_parameters = {
        'idx': input_idx,
        'e': 0.,
        'syntype': 'ExpSyn',
        'tau': 10.,
        'weight': weight,
        'record_current': True,
    }
    synapse = LFPy.Synapse(cell, **synapse_parameters)
    synapse.set_spike_times(input_spike_train)
    return cell, synapse


def make_white_noise(cell, weight, input_idx):
    """Makes white noise input to the cell. Dependent on the file stim.mod being
    compiled. The white noise is a sum of integer frequency sinusoids.
    """
    max_freq = 600
    min_freq = 2
    plt.seed(1234)
    tot_ntsteps = int(round((cell.tstopms - cell.tstartms) / cell.timeres_NEURON + 1))
    input_array = np.zeros(tot_ntsteps)
    tvec = np.arange(tot_ntsteps) * cell.timeres_NEURON
    for freq in xrange(min_freq, max_freq + 1, min_freq):
        input_array += np.sin(2 * np.pi * freq * tvec/1000. + 2*np.pi*np.random.random())
    input_array *= weight
    noiseVec = h.Vector(input_array)

    i = 0
    syn = None
    for sec in cell.allseclist:
        for seg in sec:
            if i == input_idx:
                print "Input inserted in ", sec.name()
                syn = h.ISyn(seg.x, sec=sec)
            i += 1
    if syn is None:
        raise RuntimeError("Wrong stimuli index")
    syn.dur = 1E9
    syn.delay = 0
    noiseVec.play(syn._ref_amp, cell.timeres_NEURON)
    return cell, syn, noiseVec


def simplify_axes(axes):
    """Removes top and right axis line"""
    if not type(axes) is list:
        axes = [axes]

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()


def plot_electrode_signal(cell, input_idx, conductance_type, psd=False):
    #  Making extracellular electrode
    elec_z = np.array([0, 1000])
    elec_x = np.ones(len(elec_z)) * 50
    elec_y = np.zeros(len(elec_z))

    electrode_parameters = {
        'sigma': 0.3,              # extracellular conductivity
        'x': elec_x,        # x,y,z-coordinates of contact points
        'y': elec_y,
        'z': elec_z,
    }
    electrode = LFPy.RecExtElectrode(cell, **electrode_parameters)
    electrode.calc_lfp()

    num_elecs = len(electrode.x)
    elec_clr = lambda elec_idx: plt.cm.jet(1.0 * elec_idx / num_elecs)

    cell_plot_idxs = [0, input_idx]

    plt.close("all")

    num_columns = 3
    fig = plt.figure(figsize=[8, 8])
    plt.subplots_adjust(hspace=0.6, wspace=0.6, left=0.)  # Adjusts the vertical distance between panels.
    plt.subplot(1, num_columns, 1, aspect='equal')
    plt.axis('off')
    [plt.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]], c='k') for idx in xrange(cell.totnsegs)]

    lines = []
    line_names = []
    for idx in range(num_elecs):
        l_elec, = plt.plot(electrode.x[idx], electrode.z[idx], 'D', ms=12, c=elec_clr(idx))
        lines.append(l_elec)
        line_names.append("Extracellular electrode")

    l_input, = plt.plot(cell.xmid[input_idx], cell.zmid[input_idx], 'y*', markersize=10)
    lines.append(l_input)
    line_names.append("Input")

    fig.legend(lines, line_names, loc="upper left", frameon=False)

    if psd:
        x, y_v = return_freq_and_psd(cell.tvec, cell.vmem[cell_plot_idxs, :])
        # x, y_i = return_freq_and_psd(cell.tvec, cell.imem[cell_plot_idxs, :])
        x, y_lfp = return_freq_and_psd(cell.tvec, 1000*electrode.LFP)

        plot_dict = {'xlabel': 'Hz',
                     'xscale': 'log',
                     'yscale': 'log',
                     'xlim': [2, 500],
                     "aspect": 1,

                     }
        max = 10**np.ceil(np.log10(np.max(y_v[:, 1:], axis=1)))
        min = max / 1e3
    else:
        x = cell.tvec
        y_v = cell.vmem[cell_plot_idxs, :]
        y_lfp = 1000*electrode.LFP
        plot_dict = {'xlabel': 'Time [ms]',
                     'xscale': 'linear',
                     'yscale': 'linear',
                     'xlim': [0, cell.tvec[-1]],
        }
        max = np.max(y_v, axis=1)
        min = np.min(y_v, axis=1)

    print "Max membrane potential STD: ", np.max(np.std(cell.vmem, axis=1))

    # print min, max
    # print y_v[0]
    plt.subplot(236, title='Somatic\nmembrane potential', ylim=[min[0], max[0]],
                ylabel='mV$^2$/Hz' if psd else "mV",
                **plot_dict)
    if psd:
        plt.grid(True)
    plt.plot(x, y_v[0, :], color='k', lw=2)

    plt.subplot(233, title='Input site\nmembrane potential', ylim=[min[1], max[1]],
                ylabel='mV$^2$/Hz' if psd else "mV",
                **plot_dict)
    if psd:
        plt.grid(True)
    plt.plot(x, y_v[1, :], color='k', lw=2)

    for idx in range(num_elecs):
        if psd:
            max = 10**np.ceil(np.log10(np.max(y_lfp[idx, 1:])))
            min = max / 1e3
        else:
            max = np.max(y_lfp[idx, :])
            min = np.min(y_lfp[idx, :])

        plot_number = num_elecs * num_columns - num_columns * idx - 1
        plt.subplot(num_elecs, num_columns, plot_number,
                    title='Extracellular potential',
                    ylabel='$\mu$V$^2$/Hz' if psd else "$\mu$V",
                    ylim=[min, max], **plot_dict)
        plt.plot(x, y_lfp[idx, :], c=elec_clr(idx), lw=2)

        if psd:
            plt.grid(True)

    simplify_axes(fig.axes)
    plt.savefig('example_fig_{}.png'.format(conductance_type))

if __name__ == '__main__':
    # example_synapse(synaptic_z_pos=1000, conductance_type='active',
    #                 weight=-0.0005, holding_potential=-70)
    example_white_noise(synaptic_z_pos=1000, conductance_type='active',
                        weight=0.0005, holding_potential=-70)
