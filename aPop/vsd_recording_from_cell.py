from __future__ import division
import os
import sys
from os.path import join
if not 'DISPLAY' in os.environ:
    import matplotlib
    matplotlib.use('Agg')
    at_stallo = True
else:
    at_stallo = False
from plotting_convention import mark_subplots, simplify_axes
import numpy as np
import pylab as plt
import neuron
import LFPy
import tools
from NeuralSimulation import NeuralSimulation

from param_dicts import vsd_params as params

ns = NeuralSimulation(**params)

plt.seed(123 * 0)
cell = ns._return_cell(mu=0.0, distribution='uniform')
cell, syn = ns._make_distributed_synaptic_stimuli(cell, 'homogeneous')
cell.simulate(rec_vmem=True)

t_idx = np.argmin(np.abs(cell.tvec - 50))

dx = 50  # um
dy = 50
dz = 50

x_box = np.arange(np.min(cell.xend) - dx/2, np.max(cell.xend) + dx, dx)
y_box = np.arange(np.min(cell.yend) - dy/2, np.max(cell.yend) + dy, dy)
z_box = np.arange(np.min(cell.zend) - dz/2, np.max(cell.zend) + dz, dz)

center_x_idx = np.argmin(np.abs(x_box - 0))
center_y_idx = np.argmin(np.abs(y_box - 0))
center_z_idx = np.argmin(np.abs(z_box - 0))

vsd = np.zeros((len(x_box), len(y_box), len(z_box), len(cell.tvec)))
cell_area_in_box = np.zeros((len(x_box), len(y_box), len(z_box)))
print vsd.shape
for idx in range(cell.totnsegs):
    x_idx = np.argmin(np.abs(x_box - cell.xmid[idx]))
    y_idx = np.argmin(np.abs(y_box - cell.ymid[idx]))
    z_idx = np.argmin(np.abs(z_box - cell.zmid[idx]))
    cell_area_in_box[x_idx, y_idx, z_idx] += cell.area[idx]
    vsd[x_idx, y_idx, z_idx, :] += cell.vmem[idx, :] * cell.area[idx]

for x in range(len(x_box)):
    for y in range(len(y_box)):
        for z in range(len(z_box)):
            if not cell_area_in_box[x, y, z] == 0.0:
                vsd[x, y, z, :] /= 1.0 * cell_area_in_box[x, y, z]
            else:
                vsd[x, y, z, :] = np.nan
                cell_area_in_box[x, y, z] = np.nan

# print np.sum(cell_area_in_box), np.sum(cell.area)

fig = plt.figure(figsize=[18, 8])
fig.subplots_adjust(wspace=0.6)
ax_dict = {'aspect': 1, 'frameon': False, 'xticks': [], 'xlim': [-300, 300]}
ax1 = fig.add_subplot(241, ylim=[-300, 1200], xlabel='x ($\mu$m)', ylabel='z ($\mu$m)', title='Morphology and\nvoxel centers', **ax_dict)
ax2 = fig.add_subplot(245, ylim=[-300, 300], xlabel='x ($\mu$m)', ylabel='y ($\mu$m)', **ax_dict)
ax3 = fig.add_subplot(242, ylim=[-300, 1200], xlabel='x ($\mu$m)', ylabel='z ($\mu$m)', title='t = %1.1f ms, y=0' % cell.tvec[t_idx], **ax_dict)
ax4 = fig.add_subplot(246, ylim=[-300, 300], xlabel='x ($\mu$m)', ylabel='y ($\mu$m)', title='t = %1.1f ms, z=0' % cell.tvec[t_idx], **ax_dict)
ax5 = fig.add_subplot(243, ylim=[-300, 1200], xlabel='x ($\mu$m)', ylabel='z ($\mu$m)', title='y=0', **ax_dict)
ax6 = fig.add_subplot(247, ylim=[-300, 300], xlabel='x ($\mu$m)', ylabel='y ($\mu$m)', title='z=0', **ax_dict)
ax7 = fig.add_subplot(144, title='membrane potential')

v_idxs = [cell.get_closest_idx(z=z) for z in [0, 500, 1000]]
v_clr = ['c', 'm', 'olive']
[ax1.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]], lw=1., color='0.5', zorder=0)
        for idx in xrange(len(cell.xmid))]
[ax1.plot(cell.xmid[idx], cell.zmid[idx], 'D', c=v_clr[i]) for i, idx in enumerate(v_idxs)]
grid_xz = np.meshgrid(x_box, z_box)
ax1.scatter(grid_xz[0], grid_xz[1], s=1, edgecolor='none')
ax1.plot([np.min(x_box), np.max(x_box)], [z_box[center_z_idx], z_box[center_z_idx]], 'r')
ax1.plot([x_box[center_x_idx], x_box[center_x_idx]], [np.min(z_box), np.max(z_box)], 'r')

[ax2.plot([cell.xstart[idx], cell.xend[idx]], [cell.ystart[idx], cell.yend[idx]], lw=1., color='0.5', zorder=0)
        for idx in xrange(len(cell.ymid))]
grid_xy = np.meshgrid(x_box, y_box)
ax2.scatter(grid_xy[0], grid_xy[1], s=1, edgecolor='none')
ax2.plot([np.min(x_box), np.max(x_box)], [y_box[center_y_idx], y_box[center_y_idx]], 'r')
ax2.plot([x_box[center_x_idx], x_box[center_x_idx]], [np.min(y_box), np.max(y_box)], 'r')

# print vsd[:, center_y_idx, :, t_idx].shape, grid_xz[0].shape
palette = plt.cm.hot_r
palette.set_bad('gray', 1.0) # Bad values (i.e., masked, set to grey 0.8
VSD = np.ma.array(vsd, mask=np.isnan(vsd))
area = np.ma.array(cell_area_in_box, mask=np.isnan(cell_area_in_box))
p_dict = {'cmap': palette}#, 'vmin': -81, 'vmax': -79}

img3 = ax3.pcolormesh(grid_xz[0], grid_xz[1], VSD[:, center_y_idx, :, t_idx].T, **p_dict)
plt.colorbar(img3, ax=ax3, label='VSD')

img5 = ax5.pcolormesh(grid_xz[0], grid_xz[1], area[:, center_y_idx, :].T, **p_dict)
plt.colorbar(img5, ax=ax5, label='Membrane area ($\mu$m$^2$)')

img6 = ax6.pcolormesh(grid_xy[0], grid_xy[1], area[:, :, center_z_idx].T, **p_dict)
plt.colorbar(img6, ax=ax6, label='Membrane area ($\mu$m$^2$)')

img4 = ax4.pcolormesh(grid_xy[0], grid_xy[1], VSD[:, :, center_z_idx, t_idx].T, **p_dict)
plt.colorbar(img4, ax=ax4, label='VSD')


[ax7.plot(cell.tvec, cell.vmem[idx, :], c=v_clr[i], lw=2) for i, idx in enumerate(v_idxs)]

ax7.plot([cell.tvec[t_idx], cell.tvec[t_idx]], [np.min(cell.vmem), np.max(cell.vmem)], '--', c='b')

plt.savefig('VSD.png')