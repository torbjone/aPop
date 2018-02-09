__author__ = 'torbjone'
#!/usr/bin/env python

import os
from os.path import join
import sys
import numpy as np
import pylab as plt
import neuron
nrn = neuron.h
import LFPy


def _get_longest_distance():

    nrn.distance(0, 0.5)
    max_dist = 0
    for sec in nrn.allsec():
        for seg in sec:
            max_dist = np.max([max_dist, nrn.distance(seg.x)])
    return max_dist

def _get_total_area():

    total_area = 0
    for sec in nrn.allsec():
        for seg in sec:
           total_area += nrn.area(seg.x)
    return total_area

def _get_linear_increase_factor(increase_factor, max_dist, total_conductance):
    normalization = 0
    for sec in neuron.h.allsec():
        for seg in sec:
            normalization += nrn.area(seg.x) * (1 + (increase_factor - 1) * nrn.distance(seg.x)/max_dist)
    return total_conductance / normalization

def _get_linear_decrease_factor(decrease_factor, max_dist, total_conductance):
    normalization = 0
    for sec in neuron.h.allsec():
        for seg in sec:
            normalization += nrn.area(seg.x) * (decrease_factor - (decrease_factor - 1) * nrn.distance(seg.x)/max_dist)
    return total_conductance / normalization


def biophys_generic(**kwargs):
    # print "Inserting generic conductances"
    # v = kwargs['hold_potential']
    if 'auto' in kwargs['tau_w']:
        # mAlpha = 0.001 * 6.43 * (v + 154.9)/(np.exp((v + 154.9) / 11.9) - 1.)
        # mBeta = 0.001 * 193. * np.exp(v / 33.1)
        tau = 50 #1/(mAlpha + mBeta)
        if '0.1' in kwargs['tau_w']:
            print("1/10th of auto")
            tau_w = tau * 0.1
        elif '10' in kwargs['tau_w']:
            print("10-fold of auto")
            tau_w = tau * 10
        else:
            tau_w = tau
        #print "Ih calculated tau_w: ", tau_w
    else:
        tau_w = kwargs['tau_w']

    for sec in neuron.h.allsec():
        sec.insert("QA")
        sec.V_r_QA = kwargs['hold_potential']
        sec.tau_w_QA = tau_w
        sec.Ra = 100
        sec.cm = 1.0
        sec.g_pas_QA = kwargs['g_pas']

    if kwargs['mu_factor'] is None:
        # print "Not inserting qasi-active conductance, only passive!"
        for sec in neuron.h.allsec():
            for seg in sec:
                seg.g_w_bar_QA = 0
                seg.mu_QA = 0
        return


    total_area = _get_total_area()
    total_w_conductance = kwargs['avrg_w_bar'] * total_area
    max_dist = _get_longest_distance()
    # print "Max dist: ", max_dist

    if kwargs['distribution'] == 'uniform':
        for sec in neuron.h.allsec():
            for seg in sec:
                seg.g_w_bar_QA = total_w_conductance / total_area
    elif kwargs['distribution'] == 'linear_increase':
        increase_factor = 60
        conductance_factor = _get_linear_increase_factor(increase_factor, max_dist, total_w_conductance)
        nrn.distance(0, 0.5)
        for sec in neuron.h.allsec():
            for seg in sec:
                seg.g_w_bar_QA = conductance_factor * (1 + (increase_factor - 1.) * nrn.distance(seg.x) / max_dist)
                # if "soma" in sec.name():
                #     print seg.g_w_bar_QA, conductance_factor


        # print 'Linear increase:  %1.8f + %1.10f * x' % (conductance_factor, conductance_factor*(increase_factor - 1.) / max_dist)

    elif kwargs['distribution'] == 'linear_decrease':
        decrease_factor = 60
        conductance_factor = _get_linear_decrease_factor(decrease_factor, max_dist, total_w_conductance)
        nrn.distance(0, 0.5)
        for sec in neuron.h.allsec():
            for seg in sec:
                seg.g_w_bar_QA = conductance_factor*(decrease_factor - (decrease_factor - 1.) * nrn.distance(seg.x) / max_dist)
                # if "soma" in sec.name():
                #     print seg.g_w_bar_QA, conductance_factor * 60
        # print 'Linear decrease: %1.8f - %1.10f * x' % (conductance_factor * decrease_factor, conductance_factor * (decrease_factor - 1.) / max_dist)
    else:
        raise RuntimeError("Unknown distribution...")
    cond_check = 0
    for sec in neuron.h.allsec():
        for seg in sec:
            cond_check += seg.g_w_bar_QA * nrn.area(seg.x)
            seg.mu_QA = seg.g_w_bar_QA / seg.g_pas_QA * kwargs['mu_factor']

    if np.abs(cond_check - total_w_conductance) < 1e-6:
        pass
        #print "Works", cond_check, total_w_conductance, cond_check - total_w_conductance
    else:
        raise RuntimeError("Total conductance not as expected")


def make_syaptic_stimuli(cell, input_idx):
    # Define synapse parameters
    synapse_parameters = {
        'idx': input_idx,
        'e': 0.,                   # reversal potential
        'syntype': 'ExpSyn',       # synapse type
        'tau': 10.,                # syn. time constant
        'weight': 0.00145,            # syn. weight
        'record_current': True,
    }
    synapse = LFPy.Synapse(cell, **synapse_parameters)
    synapse.set_spike_times(np.array([5.]))
    return cell, synapse


def active_declarations(**kwargs):
    ''' set active conductances for Hay model 2011 '''

    exec('biophys_%s(**kwargs)' % kwargs['conductance_type'])

def inspect_cells():

        morph_top_folder = join(".")
        from aPop.rotation_lastis import find_major_axes, alignCellToAxes
        fldrs = [f for f in os.listdir(morph_top_folder) if os.path.isdir(join(morph_top_folder, f))
                 and not f.startswith("__")]
        all_morphs = []
        for f in fldrs:
            files = os.listdir(join(morph_top_folder, f, "CNG version"))
            morphs = [join(morph_top_folder, f,  "CNG version", mof) for mof in files if mof.endswith("swc")]
            all_morphs.extend(morphs)
        zs = []
        diams = []
        for m in all_morphs:
            cell_params = {
                'morphology': m,
                'v_init': -70,
                'passive': True,           # switch on passive mechs
                'nsegs_method': 'lambda_f',  # method for setting number of segments,
                'lambda_f': 100,           # segments are isopotential at this frequency
                'dt': 2**-2,
                'tstart': 0,          # start time, recorders start at t=0
                'tstop': 20,
            }
            # try:
            cell = LFPy.Cell(**cell_params)

            axes = find_major_axes(cell)
            alignCellToAxes(cell, axes[2], axes[1])
            zs.append(cell.zmid)
            diams.append(cell.diam)

        for idx in range(len(zs)):
            plt.plot(zs[idx], diams[idx], '.')
        plt.show()
        morph = all_morphs[np.random.randint(0, len(all_morphs) - 1)]
        print(morph)
if __name__ == '__main__':
    inspect_cells()