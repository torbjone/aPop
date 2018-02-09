from __future__ import division
#!/usr/bin/env python
'''
Implementation using cell models of the Blue Brain Project with LFPy.
The example assumes that the complete set of cell models available from
https://bbpnmc.epfl.ch/nmc-portal/downloads is unzipped in this folder.
'''

import os
from os.path import join
import sys
from glob import glob
import numpy as np
import matplotlib
if not 'DISPLAY' in os.environ:
    matplotlib.use('Agg', warn=False)
import pylab as plt
import neuron
import LFPy
from neuron import h


hbp_models_folder = os.path.split(__file__)[0]

neuron.h.load_file("stdrun.hoc")
neuron.h.load_file("import3d.hoc")
neuron.load_mechanisms(join(hbp_models_folder, 'mods'))


def get_templatename(f):
    '''
    Assess from hoc file the templatename being specified within

    Arguments
    ---------
    f : file, mode 'r'

    Returns
    -------
    templatename : str

    '''
    f = open("template.hoc", 'r')
    for line in f.readlines():
        if 'begintemplate' in line.split():
            templatename = line.split()[-1]
            # print 'template {} found!'.format(templatename)
            continue

    return templatename


def return_cell(cell_folder, end_T, dt, start_T, v_init=-65.):

    cell_name = os.path.split(cell_folder)[1]
    cwd = os.getcwd()
    os.chdir(cell_folder)

    f = open("template.hoc", 'r')
    templatename = get_templatename(f)
    f.close()

    f = open("biophysics.hoc", 'r')
    biophysics = get_templatename(f)
    f.close()

    f = open("morphology.hoc", 'r')
    morphology = get_templatename(f)
    f.close()

    f = open(os.path.join("synapses", "synapses.hoc"), 'r')
    synapses = get_templatename(f)
    f.close()
    print('Loading constants')
    neuron.h.load_file('constants.hoc')
    # with suppress_stdout_stderr():

    if not hasattr(neuron.h, morphology):
        neuron.h.load_file(1, "morphology.hoc")

    if not hasattr(neuron.h, biophysics):
        neuron.h.load_file(1, "biophysics.hoc")
    #get synapses template name
    if not hasattr(neuron.h, synapses):
        # load synapses
        neuron.h.load_file(1, os.path.join('synapses', 'synapses.hoc'))

    if not hasattr(neuron.h, templatename):
        # Load main cell template
        neuron.h.load_file(1, "template.hoc")

    morphologyfile = os.listdir('morphology')[0]#glob('morphology\\*')[0]

    # Instantiate the cell(s) using LFPy
    cell = LFPy.TemplateCell(morphology=join('morphology', morphologyfile),
                     templatefile=os.path.join('template.hoc'),
                     templatename=templatename,
                     passive=False,
                     templateargs=0,
                     tstop=end_T,
                     tstart=start_T,
                     dt=dt,
                     celsius=34,
                     v_init=v_init,
                     )
    os.chdir(cwd)
    cell.set_rotation(z=np.pi/2, x=np.pi/2)
    return cell


def plot_morphology(cell_folder, cell_name, figure_folder):
    cell = return_cell(cell_folder, end_T=1, start_T=0, dt=2**-3)
    i = 0
    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(111, frameon=False, xticks=[], yticks=[], aspect=1, xlim=[-300, 300], ylim=[-300, 1200])
    for sec in cell.allseclist:
        c = 'g' if 'axon' in sec.name() else 'k'
        for seg in sec:
            # if i == 0:
            #     ax.plot(cell.xmid[i], cell.zmid[i], 'o', c=c, ms=10)
            # else:
            ax.plot([cell.xstart[i], cell.xend[i]], [cell.zstart[i], cell.zend[i]], c=c, clip_on=False)
            i += 1
    ax.plot([200, 300], [-100, -100], lw=3, c='b')
    ax.text(200, -160, '100 $\mu$m')
    fig.savefig(join(figure_folder, 'morph_%s.png' % cell_name))


def plot_cell_soma_response(cell_folder, figure_folder):
    neuron.h('forall delete_section()')
    cell_name = os.path.split(cell_folder)[1]
    cell = return_cell(cell_folder, end_T=50, start_T=-200, dt=2**-5)

    synapse_parameters_soma = {
        'idx' : cell.get_closest_idx(x=0., y=0., z=0.),
        'e' : 0.,                   # reversal potential
        'syntype' : 'ExpSyn',       # synapse type
        'tau' : 2.,                 # synaptic time constant
        'weight' : 0.3,            # synaptic weight
        'record_current' : True,    # record synapse current
    }

    # Create synapse and set time of synaptic input
    synapse = LFPy.Synapse(cell, **synapse_parameters_soma)
    synapse.set_spike_times(np.array([10., 14., 18., 22]))

    cell.simulate()

    i = 0
    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(121, frameon=False, xticks=[], yticks=[], aspect=1,
                         xlim=[-300, 300], ylim=[-300, 1200])
    for sec in cell.allseclist:
        c = 'g' if 'axon' in sec.name() else 'k'
        for seg in sec:
            ax.plot([cell.xstart[i], cell.xend[i]], [cell.zstart[i], cell.zend[i]], c=c, clip_on=False)
            i += 1
    ax.plot([200, 300], [-100, -100], lw=3, c='b')
    ax.text(200, -160, '100 $\mu$m')

    ax = fig.add_subplot(122, title='Somatic membrane potential')
    ax.plot(cell.tvec, cell.somav)

    fig.savefig(join(figure_folder, 'single_synapse_%s_%d.png' % (cell_name, neuron.h.celsius)))


if __name__ == '__main__':
    plot_cell_soma_response(join('L5_TTPC2_cADpyr232_2'), '.')


