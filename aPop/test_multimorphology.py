import matplotlib
matplotlib.use("AGG")
import sys
import os
from os.path import join
import LFPy
import neuron
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


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

neuron_models = join("..", "neuron_models")
dt = 2**-4
end_t = 100
cut_off = 0
holding_potential = -80
conductance_type = "generic"
mu = 0.0
distribution = "linear_increase"
neuron.load_mechanisms(join(neuron_models))

morph_top_folder = join(neuron_models, "neuron_nmo_adult")
sys.path.append(morph_top_folder)
from active_declarations import active_declarations

fldrs = [f for f in os.listdir(morph_top_folder) if os.path.isdir(join(morph_top_folder, f))
         and not f.startswith("__")]
print(fldrs)
all_morphs = []
for f in fldrs:
    files = os.listdir(join(morph_top_folder, f, "CNG version"))
    morphs = [join(morph_top_folder, f,  "CNG version", mof) for mof in files if mof.endswith("swc")
              or mof.endswith("hoc")]
    all_morphs.extend(morphs)
    # all_morphs.append()

for mi, morph in enumerate(all_morphs):
    # if not mi == 0:
    #     continue
    print(mi, morph)
    cellname = os.path.split(morph)[-1]

    cell_params = {
        'morphology': morph,
        'v_init': holding_potential,
        'passive': False,           # switch on passive mechs
        'nsegs_method': 'lambda_f',  # method for setting number of segments,
        'lambda_f': 100,           # segments are isopotential at this frequency
        'dt': dt,
        'tstart': -cut_off,          # start time, recorders start at t=0
        'tstop': end_t,
        'pt3d': True,
        # 'custom_code': [join(morph_top_folder, 'custom_codes.hoc')],
        'custom_fun': [active_declarations],  # will execute this function
        'custom_fun_args': [{'conductance_type': conductance_type,
                             'mu_factor': mu,
                             'g_pas': 0.00005,#0.0002, # / 5,
                             'distribution': distribution,
                             'tau_w': 'auto',
                             'avrg_w_bar': 0.00005, # Half of "original"!!!
                             'hold_potential': holding_potential}]
    }
    # try:
    cell = LFPy.Cell(**cell_params)
    # except:
    #     cell_params['custom_code'] = []
    #     cell = LFPy.Cell(**cell_params)
    # for sec in cell.allseclist:
    #     print(sec.name())
    from aPop.rotation_lastis import find_major_axes, alignCellToAxes
    axes = find_major_axes(cell)
    alignCellToAxes(cell, axes[2], axes[1])
    cell.set_pos(z=1200 - np.max(cell.zend))

    make_syaptic_stimuli(cell, cell.get_closest_idx(z=1000))
    cell.simulate()
    plt.close("all")
    fig = plt.figure(figsize=[9, 9])
    fig.subplots_adjust(top=0.99, bottom=0.05)
    ax = fig.add_subplot(111, aspect=1, xlim=[-300, 300], ylim=[-300, 1400])
    ax2 = fig.add_axes([0.77, 0.3, 0.17, 0.3], title="Soma Vm")
    ax2.plot(cell.tvec, cell.somav)
    #plot morphology
    zips = []
    for x, z in cell.get_idx_polygons():
        zips.append(list(zip(x, z)))
    polycol = PolyCollection(zips,
                             edgecolors='none',
                             facecolors='k')
    ax.add_collection(polycol)

    plt.savefig(join(morph_top_folder, "{}_{}.png".format(mi, cellname)))

