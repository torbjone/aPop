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

from sklearn.decomposition import PCA


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

def find_major_axes(cell):
    """
    Based on code from: https://github.com/lastis/LFPy_util/
    Find the principal geometrical components of the neuron currently loaded
    with Neuron. Uses :class:`sklearn.decomposition.PCA`.
    If used with :class:`LFPy.Cell`, the parameter **pt3d** must be set to True.
    :returns:
        Matrix (3 x 3) where each row is a principal component.
    :rtype: :class:`~numpy.ndarray`
    Example:
        .. code-block:: python
            # Find the principal component axes and rotate cell.
            axes = LFPy_util.data_extraction.findMajorAxes()
            LFPy_util.rotation.alignCellToAxes(cell,axes[0],axes[1])
    """
    points = np.array([cell.xmid, cell.ymid, cell.zmid])
    pca = PCA(n_components=3)
    pca.fit(points[:3].T)
    return pca.components_

def alignCellToAxes(cell, y_axis, x_axis=None):
    """
    Based on code from: https://github.com/lastis/LFPy_util/
    Rotates the cell such that **y_axis** is paralell to the global y-axis and
    **x_axis** will be aligned to the global x-axis as well as possible.
    **y_axis** and **x_axis** should be orthogonal, but need not be.
    :param `~LFPy.Cell` cell:
        Initialized Cell object to rotate.
    :param `~numpy.ndarray` y_axis:
        Vector to be aligned to the global y-axis.
    :param `~numpy.ndarray` x_axis:
        Vector to be aligned to the global x-axis.
    Example:
        .. code-block:: python
            # Find the principal component axes and rotate cell.
            axes = LFPy_util.data_extraction.findMajorAxes()
            LFPy_util.rotation.alignCellToAxes(cell,axes[0],axes[1])
    """
    y_axis = np.asarray(y_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    dx = y_axis[0]
    dy = y_axis[1]
    dz = y_axis[2]

    x_angle = -np.arctan2(dz, dy)
    z_angle = np.arctan2(dx, np.sqrt(dy * dy + dz * dz))

    cell.set_rotation(x_angle, None, z_angle)
    if x_axis is None:
        return

    x_axis = np.asarray(x_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    Rx = rotation_matrix([1, 0, 0], x_angle)
    Rz = rotation_matrix([0, 0, 1], z_angle)

    x_axis = np.dot(x_axis, Rx)
    x_axis = np.dot(x_axis, Rz)

    dx = x_axis[0]
    dz = x_axis[2]

    y_angle = np.arctan2(dz, dx)
    cell.set_rotation(None, y_angle, None)

    if np.abs(np.min(cell.zmid)) > np.abs(np.max(cell.zmid)):
        cell.set_rotation(x=np.pi)


def rotation_matrix(axis, theta):
    """
    Based on code from: https://github.com/lastis/LFPy_util/
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    Uses the Euler-rodrigues formula
    """
    theta = -theta
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2)
    b, c, d = -axis * np.sin(theta / 2)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

neuron_models = join("..", "neuron_models")
dt = 2**-4
end_t = 100
cut_off = 0
holding_potential = -80
conductance_type = "generic"
mu = 0.0
distribution = "linear_increase"
neuron.load_mechanisms(join(neuron_models))

morph_top_folder = join(neuron_models, "neuron_nmo")
sys.path.append(morph_top_folder)
from active_declarations import active_declarations

fldrs = [f for f in os.listdir(morph_top_folder) if os.path.isdir(join(morph_top_folder, f))
         and not f.startswith("__")]
print(fldrs)
all_morphs = []
for f in fldrs:
    files = os.listdir(join(morph_top_folder, f, "CNG version"))
    morphs = [join(morph_top_folder, f,  "CNG version", mof) for mof in files if mof.endswith("swc")]
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
        'custom_code': [join(morph_top_folder, 'custom_codes.hoc')],
        'custom_fun': [active_declarations],  # will execute this function
        'custom_fun_args': [{'conductance_type': conductance_type,
                             'mu_factor': mu,
                             'g_pas': 0.00005,#0.0002, # / 5,
                             'distribution': distribution,
                             'tau_w': 'auto',
                             'total_w_conductance': 6.23843378791,# / 5,
                             'avrg_w_bar': 0.00005, # Half of "original"!!!
                             'hold_potential': holding_potential}]
    }
    try:
        cell = LFPy.Cell(**cell_params)
    except:
        cell_params['custom_code'] = []
        cell = LFPy.Cell(**cell_params)
    # for sec in cell.allseclist:
    #     print(sec.name())
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

