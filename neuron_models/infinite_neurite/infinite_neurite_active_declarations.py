

import neuron
import numpy as np

def active_declarations(**kwargs):
    neuron.h.distance(0, 0)
    print "Infinite neurite"
    for sec in neuron.h.allsec():
        sec.nseg = 100

        sec.insert("QA")
        sec.V_r_QA = -80.

        sec.tau_w_QA = 50.0
        sec.w_inf_QA = 0.5
        sec.Ra = 100.
        sec.cm = 1.0
        baseline_g = 0.00005

        if kwargs['distribution'] == 'increase':
            g_w_top = baseline_g * kwargs['g_w_bar_scaling']
        elif kwargs['distribution'] == 'uniform':
            g_w_top = baseline_g
        else:
            raise ValueError('Distribution not recognized!')

        for seg in sec:
            if neuron.h.distance(seg.x) <= 500:
                seg.g_pas_QA = baseline_g
                seg.g_w_bar_QA = baseline_g
                seg.mu_QA = seg.g_w_bar_QA / seg.g_pas_QA * kwargs['mu_factor']
            else:
                seg.g_pas_QA = baseline_g
                seg.g_w_bar_QA = g_w_top
                seg.mu_QA = seg.g_w_bar_QA / seg.g_pas_QA * kwargs['mu_factor']
