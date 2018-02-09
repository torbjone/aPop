
import os
import sys
from os.path import join
if not 'DISPLAY' in os.environ:
    import matplotlib
    matplotlib.use('Agg', warn=False)
    at_stallo = True
else:
    at_stallo = False
import aPop
from aPop.plotting_convention import mark_subplots, simplify_axes, LogNorm
import numpy as np
import pylab as plt
import neuron
import LFPy
from aPop import tools
h = neuron.h


class NeuralSimulation:
    """
    Main class for the neural simulation. It makes the synaptic inputs,
    sets up the cell models.

    """
    def __init__(self, **kwargs):
        self.input_type = kwargs['input_type']
        self.input_region = kwargs['input_region']
        self.name = kwargs['name']
        self.cell_number = kwargs['cell_number'] if 'cell_number' in kwargs else None
        self.mu = kwargs['mu'] if 'mu' in kwargs else None
        self.distribution = kwargs['distribution'] if 'distribution' in kwargs else None
        self.correlation = kwargs['correlation'] if 'correlation' in kwargs else None

        self.param_dict = kwargs
        self.cell_name = kwargs['cell_name'] if 'cell_name' in kwargs else 'hay'
        self.max_freq = kwargs['max_freq']
        self.holding_potential = kwargs['holding_potential']
        self.conductance_type = kwargs['conductance_type']
        self.username = os.getenv('USER')

        self._set_input_params()
        self.sim_name = self.get_simulation_name()
        self.population_sim_name = self.sim_name[:-6]  # Just remove the cell number

        self.center_electrode_parameters = kwargs['center_electrode_parameters']
        self.elec_x_center = self.center_electrode_parameters['x']
        self.elec_y_center = self.center_electrode_parameters['y']
        self.elec_z_center = self.center_electrode_parameters['z']

        self.root_folder = kwargs['root_folder']
        self.neuron_models = join(self.root_folder, 'aPop', 'neuron_models')
        self.figure_folder = join(self.root_folder, self.param_dict['save_folder'])
        self.sim_folder = join(self.root_folder, self.param_dict['save_folder'], 'simulations')

        if not os.path.isdir(self.figure_folder):
            os.mkdir(self.figure_folder)
        if not os.path.isdir(self.sim_folder):
            os.mkdir(self.sim_folder)

        self.divide_into_welch = 8.
        self.welch_dict = {'Fs': 1000 / self.dt,
                           'NFFT': int(self.num_tsteps/self.divide_into_welch),
                           'noverlap': int(self.num_tsteps/self.divide_into_welch/2.),
                           'window': plt.window_hanning,
                           'detrend': plt.detrend_mean,
                           'scale_by_freq': True,
                           }

    def get_simulation_name(self):
        if self.conductance_type == 'generic':
            if self.mu is None:
                conductance = '%s_%s_None' % (self.conductance_type, self.distribution)
            else:
                conductance = '%s_%s_%1.1f' % (self.conductance_type, self.distribution, self.mu)
        else:
            if self.holding_potential is None:
                conductance = '%s_None' % (self.conductance_type)
            else:
                conductance = '%s_%+d' % (self.conductance_type, self.holding_potential)
        sim_name = '%s_%s_%s_%s_%1.2f_%05d' % (self.name, self.cell_name, self.input_region, conductance,
                                               self.correlation, self.cell_number)
        return sim_name

    def _set_input_params(self):
        self.dt = self.param_dict['dt']
        self.cut_off = self.param_dict['cut_off']
        self.end_t = self.param_dict['end_t']
        self.max_freq = self.param_dict['max_freqs'] if 'max_freqs' in self.param_dict else 500
        self.num_tsteps = round(self.end_t/self.dt + 1)

    def remove_active_mechanisms(self, remove_list):
        mt = h.MechanismType(0)
        for sec in h.allsec():
            for seg in sec:
                for mech in remove_list:
                    mt.select(mech)
                    mt.remove()

    def make_Ih_frozen(self, cell):
        neuron.h.t = 0
        neuron.h.finitialize(cell.v_init)
        neuron.h.fcurrent()
        i = 0
        ehcn_Ih = -45.
        for sec in neuron.h.allsec():
            for seg in sec:
                if hasattr(seg, "gIh_Ih"):
                    seg.e_pas = (seg.g_pas * seg.e_pas + seg.gIh_Ih * ehcn_Ih) / (seg.g_pas + seg.gIh_Ih)
                    seg.g_pas += seg.gIh_Ih
                i += 1

    def return_gIhbar_at_distance(self, cell, dist):

        gIhbar = np.zeros(cell.totnsegs)
        all_dist = np.zeros(cell.totnsegs)
        h.distance(0, 0.5)
        comp = 0
        for sec in neuron.h.allsec():
            for seg in sec:
                try:
                    gIhbar[comp] = seg.gIhbar_Ih
                except:
                    gIhbar[comp] = 0
                all_dist[comp] = h.distance(seg.x)
                comp += 1
        return gIhbar[np.argmin(np.abs(all_dist - dist))]

    def make_Ih_plateau(self, cell, plateau_distance, plot=False):

        plateau_value = self.return_gIhbar_at_distance(cell, plateau_distance)
        orig_gIhbar = []
        dist = []
        h.distance(0, 0.5)
        for sec in neuron.h.allsec():
            for seg in sec:
                if not "apic" in sec.name():
                    continue
                try:
                    orig_gIhbar.append(seg.gIhbar_Ih)
                except:
                    orig_gIhbar.append(0)
                dist.append(h.distance(seg.x))


        h.distance(0, 0.5)
        for sec in neuron.h.allsec():
            for seg in sec:
                if h.distance(seg.x) >= plateau_distance:
                    try:
                        seg.gIhbar_Ih = plateau_value
                    except:
                        pass

        plateau_gIhbar = []
        dist = []
        h.distance(0, 0.5)
        for sec in neuron.h.allsec():
            for seg in sec:
                if not "apic" in sec.name():
                    continue
                try:
                    plateau_gIhbar.append(seg.gIhbar_Ih)
                except:
                    plateau_gIhbar.append(0)
                dist.append(h.distance(seg.x))

        if plot:

            dist_norm = np.array(dist / np.max(dist))
            fit = 0.0002 * (-0.8696 + 2.0870*np.exp(3.6161*(np.sort(dist_norm)-0.0)))
            fit_plateau = 0.0002 * (-0.8696 + 2.0870*np.exp(3.6161*(np.sort(dist_norm)-0.0)))
            fit_plateau[np.where(np.sort(dist) > plateau_distance)] = plateau_value

            harnett_plateau_value = 0.00566
            fig = plt.figure()
            fig.subplots_adjust(left=0.15, bottom=0.15)

            ax = fig.add_subplot(111, xlabel="Distance ($\mu$m)",
                                 ylabel="mS/cm$^2$",
                                 )
            l1, = plt.plot(dist, 1000 * np.array(orig_gIhbar), 'k.')
            l2, = plt.plot(dist, 1000 * np.array(plateau_gIhbar), 'r.')
            l3, = plt.plot(np.sort(dist), 1000 * np.array(fit), c='gray', ls='-')
            l4, = plt.plot(np.sort(dist), 1000 * np.array(fit_plateau), c='y', ls='-')
            ax.axhline(1000 * harnett_plateau_value, ls=':')
            ax.text(0, 1000 * harnett_plateau_value,
                "Harnett et al. (2015)\nplateau value = {:1.2f} mS/cm$^2$".format(1000*harnett_plateau_value),
                    fontsize=9, va="bottom")

            ax.text(1000, 1000 * plateau_value - 0.5,
                "Plateau value =\n{:1.2f} mS/cm$^2$".format(1000*plateau_value),
                    fontsize=9, va="top")

            simplify_axes(ax)
            plt.legend([l1, l2], ["Original peak Ih conductance",
                                  "Plateau peak Ih conductance"], frameon=False)
            plt.savefig("Ih_plateau_{}um.png".format(plateau_distance))
            plt.savefig("Ih_plateau_{}um.pdf".format(plateau_distance))
            plt.close("all")

    def make_cell_uniform(self, Vrest=-80):
        """
        Adjusts e_pas to enforce a uniform resting membrane potential at Vrest
        """
        neuron.h.t = 0
        neuron.h.finitialize(Vrest)
        neuron.h.fcurrent()
        for sec in neuron.h.allsec():
            for seg in sec:
                seg.e_pas = seg.v
                if neuron.h.ismembrane("na_ion"):
                    seg.e_pas += seg.ina/seg.g_pas
                if neuron.h.ismembrane("k_ion"):
                    seg.e_pas += seg.ik/seg.g_pas
                if neuron.h.ismembrane("ca_ion"):
                    seg.e_pas = seg.e_pas + seg.ica/seg.g_pas
                if neuron.h.ismembrane("Ih"):
                    seg.e_pas += seg.ihcn_Ih/seg.g_pas
                if neuron.h.ismembrane("Ih_frozen"):
                    seg.e_pas += seg.ihcn_Ih_frozen/seg.g_pas

    def return_cell(self, cell_x_y_z_rotation=None):
        """
        Returns the cell model, based on the population parameters already
        supplied to the class.
        :param cell_x_y_z_rotation: contains array with position and rotation
        :return: LFPy cell object
        """
        remove_lists = {'active': [],
                        'passive': ["Nap_Et2", "NaTa_t", "NaTs2_t", "SKv3_1",
                                    "SK_E2", "K_Tst", "K_Pst", "Im", "Ih",
                                    "CaDynamics_E2", "Ca_LVAst", "Ca"],
                        'Ih': ["Nap_Et2", "NaTa_t", "NaTs2_t", "SKv3_1",
                               "SK_E2", "K_Tst", "K_Pst", "Im",
                               "CaDynamics_E2", "Ca_LVAst", "Ca"],
                        'Ih_plateau': ["Nap_Et2", "NaTa_t", "NaTs2_t", "SKv3_1",
                               "SK_E2", "K_Tst", "K_Pst", "Im",
                               "CaDynamics_E2", "Ca_LVAst", "Ca"],
                        'Ih_plateau2': ["Nap_Et2", "NaTa_t", "NaTs2_t", "SKv3_1",
                               "SK_E2", "K_Tst", "K_Pst", "Im",
                               "CaDynamics_E2", "Ca_LVAst", "Ca"],
                        'Ih_frozen': ["Nap_Et2", "NaTa_t", "NaTs2_t", "SKv3_1",
                               "SK_E2", "K_Tst", "K_Pst", "Im", "Ih",
                               "CaDynamics_E2", "Ca_LVAst", "Ca"]
                        }

        if 'i_QA' not in list(neuron.h.__dict__.keys()):
            neuron.load_mechanisms(join(self.neuron_models))
        cell = None
        if self.cell_name == 'hay':
            if self.conductance_type == 'generic':

                from aPop.neuron_models.hay.hay_active_declarations import active_declarations
                cell_params = {
                    'morphology': join(self.neuron_models, 'hay', 'cell1.hoc'),
                    'v_init': self.holding_potential,
                    'passive': False,           # switch on passive mechs
                    'nsegs_method': 'lambda_f',  # method for setting number of segments,
                    'lambda_f': 100,           # segments are isopotential at this frequency
                    'dt': self.dt,
                    'tstart': -self.cut_off,          # start time, recorders start at t=0
                    'tstop': self.end_t,
                    'custom_code': [join(self.neuron_models, 'hay', 'custom_codes.hoc')],
                    'custom_fun': [active_declarations],  # will execute this function
                    'custom_fun_args': [{'conductance_type': self.conductance_type,
                                         'mu_factor': self.mu,
                                         'g_pas': 0.00005,#0.0002, # / 5,
                                         'distribution': self.distribution,
                                         'tau_w': 'auto',
                                         #'total_w_conductance': 6.23843378791,# / 5,
                                         'avrg_w_bar': 0.00005, # Half of "original"!!!
                                         'hold_potential': self.holding_potential}]
                }
                cell = LFPy.Cell(**cell_params)
            else:

                neuron.load_mechanisms(join(self.neuron_models, 'hay', 'mod'))

                cell_params = {
                    'morphology': join(self.neuron_models, 'hay', 'cell1.hoc'),
                    'v_init': self.holding_potential if self.holding_potential is not None else -70,
                    'passive': False,           # switch on passive mechs
                    'nsegs_method': 'lambda_f',  # method for setting number of segments,
                    'lambda_f': 100,           # segments are isopotential at this frequency
                    'dt': self.dt,   # dt of LFP and NEURON simulation.
                    'tstart': -self.cut_off,          # start time, recorders start at t=0
                    'tstop': self.end_t,
                    'custom_code': [join(self.neuron_models, 'hay', 'custom_codes.hoc'),
                                    join(self.neuron_models, 'hay', 'biophys3.hoc')],
                }

                cell = LFPy.Cell(**cell_params)
                if self.conductance_type == "Ih_frozen":
                    self.make_Ih_frozen(cell)
                if self.conductance_type == "Ih_plateau":
                    self.make_Ih_plateau(cell, 600, True)
                if self.conductance_type == "Ih_plateau2":
                    self.make_Ih_plateau(cell, 945, True)
                self.remove_active_mechanisms(remove_lists[self.conductance_type])
                self.make_cell_uniform(self.holding_potential)

        elif 'L5_' in self.cell_name or 'L4_' in self.cell_name:
            remove_lists = {'active': [],
                            'passive': ["Nap_Et2", "NaTa_t", "NaTs2_t", "SKv3_1",
                                        "SK_E2", "K_Tst", "K_Pst", "Im", "Ih",
                                        "CaDynamics_E2", "Ca_LVAst", "Ca"],
                            'Ih': ["Nap_Et2", "NaTa_t", "NaTs2_t", "SKv3_1",
                                   "SK_E2", "K_Tst", "K_Pst", "Im",
                                   "CaDynamics_E2", "Ca_LVAst", "Ca"],
                            'Ih_frozen': ["Nap_Et2", "NaTa_t", "NaTs2_t", "SKv3_1",
                               "SK_E2", "K_Tst", "K_Pst", "Im", "Ih",
                               "CaDynamics_E2", "Ca_LVAst", "Ca"]}

            from aPop.neuron_models.hbp_cells.hbp_cells import return_cell
            cell_folder = join(self.param_dict['model_folder'], 'models', self.cell_name)
            cell = return_cell(cell_folder, self.end_t, self.dt, -self.cut_off)
            if self.conductance_type == "Ih_frozen":
                self.make_Ih_frozen(cell)
            self.remove_active_mechanisms(remove_lists[self.conductance_type])

        elif self.cell_name == 'stick_cell':
            # sys.path.append(join(self.neuron_models, 'stick_cell'))
            from aPop.neuron_models.stick_cell.stick_cell import active_declarations

            args = [{'mu_factor': self.param_dict['mu'],
                     'g_w_bar_scaling': self.param_dict['g_w_bar_scaling'],
                     'distribution': self.distribution,
                     }]
            cell_params = {
                'morphology': join(self.neuron_models, self.cell_name,
                                   'stick_cell_morph.hoc'),
                'v_init': self.holding_potential,
                'passive': False,
                'nsegs_method': None,
                'dt': self.dt,
                'tstart': -self.cut_off,          # start time, recorders start at t=0
                'tstop': self.end_t,
                'custom_fun': [active_declarations],
                'custom_fun_args': args,
            }
            cell = LFPy.Cell(**cell_params)

        elif self.cell_name == 'multimorph':

            morph_top_folder = join(self.neuron_models, "neuron_nmo_adult")
            sys.path.append(morph_top_folder)
            from active_declarations import active_declarations

            fldrs = [f for f in os.listdir(morph_top_folder) if os.path.isdir(join(morph_top_folder, f))
                     and not f.startswith("__")]
            all_morphs = []
            for f in fldrs:
                files = os.listdir(join(morph_top_folder, f, "CNG version"))
                morphs = [join(morph_top_folder, f,  "CNG version", mof) for mof in files
                          if mof.endswith("swc") or mof.endswith("hoc")]
                all_morphs.extend(morphs)

            morph = all_morphs[np.random.randint(0, len(all_morphs) - 1)]
            print(param_dict["cell_number"], morph)
            # sys.exit()
            cell_params = {
                'morphology': morph,
                'v_init': self.holding_potential,
                'passive': False,           # switch on passive mechs
                'nsegs_method': 'lambda_f',  # method for setting number of segments,
                'lambda_f': 100,           # segments are isopotential at this frequency
                'dt': self.dt,
                'tstart': -self.cut_off,          # start time, recorders start at t=0
                'tstop': self.end_t,
                'pt3d': False,
                'custom_code': [join(morph_top_folder, 'custom_codes.hoc')],
                'custom_fun': [active_declarations],  # will execute this function
                'custom_fun_args': [{'conductance_type': self.conductance_type,
                                     'mu_factor': self.mu,
                                     'g_pas': 0.00005,#0.0002, # / 5,
                                     'distribution': self.distribution,
                                     'tau_w': 'auto',
                                     'avrg_w_bar': 0.00005, # Half of "original"!!!
                                     'hold_potential': self.holding_potential}]
            }
            # try:
            cell = LFPy.Cell(**cell_params)
            # except:
            #     cell_params['custom_code'] = []
            #     cell = LFPy.Cell(**cell_params)

            # Rotate cell so the apical dendrite lies along the z-axis
            from rotation_lastis import find_major_axes, alignCellToAxes
            axes = find_major_axes(cell)
            alignCellToAxes(cell, axes[2], axes[1])

        else:
            raise NotImplementedError("Cell name is not recognized")
        if cell_x_y_z_rotation is not None:
            cell.set_rotation(z=cell_x_y_z_rotation[3])
            zpos = 1200 - np.max(cell.zend) if self.cell_name == 'multimorph' else cell_x_y_z_rotation[2]
            cell.set_pos(x=cell_x_y_z_rotation[0],
                         y=cell_x_y_z_rotation[1],
                         z=zpos)
        return cell

    def save_neural_sim_single_input_data(self, cell):

        name = self.sim_name
        if 'split_sim' in self.param_dict:
            name += '_%s' % self.param_dict['split_sim']

        center_electrode = LFPy.RecExtElectrode(cell, **self.center_electrode_parameters)
        center_electrode.calc_lfp()
        np.save(join(self.sim_folder, 'center_sig_%s.npy' % name), center_electrode.LFP)
        del center_electrode.LFP

        if self.cell_number == 0:
            np.save(join(self.sim_folder, 'tvec_%s_%s.npy' % (self.cell_name, self.input_type)), cell.tvec)
            if hasattr(cell, 'vmem'):
                np.save(join(self.sim_folder, 'vmem_%s.npy' % name), cell.vmem)
            np.save(join(self.sim_folder, 'imem_%s.npy' % name), cell.imem)
            np.save(join(self.sim_folder, 'synidx_%s.npy' % name), cell.synidx)

            np.save(join(self.sim_folder, 'center_elec_x_%s.npy' % self.cell_name), center_electrode.x)
            np.save(join(self.sim_folder, 'center_elec_y_%s.npy' % self.cell_name), center_electrode.y)
            np.save(join(self.sim_folder, 'center_elec_z_%s.npy' % self.cell_name), center_electrode.z)

            np.save(join(self.sim_folder, 'xstart_%s_%s.npy' %
                         (self.cell_name, self.param_dict['conductance_type'])), cell.xstart)
            np.save(join(self.sim_folder, 'ystart_%s_%s.npy' %
                         (self.cell_name, self.param_dict['conductance_type'])), cell.ystart)
            np.save(join(self.sim_folder, 'zstart_%s_%s.npy' %
                         (self.cell_name, self.param_dict['conductance_type'])), cell.zstart)
            np.save(join(self.sim_folder, 'xend_%s_%s.npy' %
                         (self.cell_name, self.param_dict['conductance_type'])), cell.xend)
            np.save(join(self.sim_folder, 'yend_%s_%s.npy' %
                         (self.cell_name, self.param_dict['conductance_type'])), cell.yend)
            np.save(join(self.sim_folder, 'zend_%s_%s.npy' %
                         (self.cell_name, self.param_dict['conductance_type'])), cell.zend)
            np.save(join(self.sim_folder, 'xmid_%s_%s.npy' %
                         (self.cell_name, self.param_dict['conductance_type'])), cell.xmid)
            np.save(join(self.sim_folder, 'ymid_%s_%s.npy' %
                         (self.cell_name, self.param_dict['conductance_type'])), cell.ymid)
            np.save(join(self.sim_folder, 'zmid_%s_%s.npy' %
                         (self.cell_name, self.param_dict['conductance_type'])), cell.zmid)
            np.save(join(self.sim_folder, 'diam_%s_%s.npy' %
                         (self.cell_name, self.param_dict['conductance_type'])), cell.diam)

    def run_single_simulation(self):
        """
        Main function to run single-cell simulation

        """
        if os.path.isfile(join(self.sim_folder, 'center_sig_%s.npy' % self.sim_name)):
            print("Skipping ", self.sim_name)
            return

        plt.seed(123 * self.cell_number)
        x_y_z_rot = np.load(join(self.param_dict['root_folder'],
                                 self.param_dict['save_folder'],
                         'x_y_z_rot_%s.npy' % self.param_dict['name']))[self.cell_number]

        cell = self.return_cell(x_y_z_rot)
        cell, syn = self._make_synaptic_stimuli(cell)

        rec_vmem = False if at_stallo else True
        cell.simulate(rec_imem=True, rec_vmem=rec_vmem)
        # if not at_stallo:
        #     print("Max vmem STD: ", np.max(np.std(cell.vmem, axis=1)))

        self.save_neural_sim_single_input_data(cell)
        if self.cell_number < 5 or (self.cell_number % 50) == 0:
            self._plot_results(cell)

    def _make_synaptic_stimuli(self, cell):

        syntype = 'ExpSynI' if not 'syntype' in self.param_dict else self.param_dict['syntype']

        # Define synapse parameters
        synapse_params = {
            'e': 0.,                   # reversal potential (not used for ExpSynI)
            'syntype': syntype,       # synapse type
            'tau': self.param_dict['syn_tau'],                # syn. time constant
            'weight': self.param_dict['syn_weight'],            # syn. weight
            'record_current': False,
        }

        if self.input_region == 'distal_tuft':
            input_pos = ['apic']
            maxpos = 10000
            minpos = 900
        elif self.input_region == 'tuft':
            input_pos = ['apic']
            maxpos = 10000
            minpos = 600
        elif self.input_region == 'homogeneous':
            input_pos = ['apic', 'dend', "soma"]
            maxpos = 10000
            minpos = -10000
        elif self.input_region == 'basal':
            input_pos = ['dend', "soma"]
            maxpos = 10000
            minpos = -10000

        elif self.input_region == 'top':
            input_pos = ['dend', "soma"]
            maxpos = 10000
            minpos = 500
        elif self.input_region == 'bottom':
            input_pos = ['dend', "soma"]
            maxpos = 500
            minpos = -1000
        elif 'dual' in self.input_region:
            input_pos_tuft = ['apic']
            maxpos_tuft = 10000
            minpos_tuft = 900
            input_pos_homo = ['apic', 'dend', "soma"]
            maxpos_homo = 10000
            minpos_homo = -10000
            dual_fraction = float(self.input_region[4:])
            print("Dual fraction: ", dual_fraction)
        else:
            raise RuntimeError("Use other input section")

        num_synapses = self.param_dict['num_synapses']
        firing_rate = self.param_dict['input_firing_rate']
        if 'dual' in self.input_region:
            num_synapses_tuft = int(num_synapses * dual_fraction)
            num_synapses_homo = num_synapses - num_synapses_tuft
            cell_input_idxs_tuft = cell.get_rand_idx_area_norm(section=input_pos_tuft,
                                                               nidx=num_synapses_tuft,
                                                               z_min=minpos_tuft,
                                                               z_max=maxpos_tuft)
            cell_input_idxs_homo = cell.get_rand_idx_area_norm(section=input_pos_homo,
                                                               nidx=num_synapses_homo,
                                                               z_min=minpos_homo,
                                                               z_max=maxpos_homo)
            cell_input_idxs = np.r_[cell_input_idxs_homo, cell_input_idxs_tuft]

        else:
            cell_input_idxs = cell.get_rand_idx_area_norm(section=input_pos,
                                                          nidx=num_synapses,
                                                          z_min=minpos,
                                                          z_max=maxpos)

        if self.correlation < 1e-6:
            spike_trains = LFPy.inputgenerators.get_activation_times_from_distribution(
                n=num_synapses, tstart=cell.tstart, tstop=cell.tstop,
                rvs_args=dict(loc=0., scale=1000. / firing_rate))
            synapses = self.set_input_spiketrain(cell, cell_input_idxs, spike_trains, synapse_params)

        else:
            all_spiketimes = np.load(join(self.param_dict['root_folder'], self.param_dict['save_folder'],
                             'all_spike_trains_%s.npy' % self.param_dict['name']), encoding = 'latin1').item()

            spike_train_idxs = np.random.choice(np.arange(int(num_synapses/self.correlation)),
                                                num_synapses, replace=False)
            spike_trains = [all_spiketimes[idx] for idx in spike_train_idxs]
            synapses = self.set_input_spiketrain(cell, cell_input_idxs, spike_trains, synapse_params)

        return cell, synapses

    def set_input_spiketrain(self, cell, cell_input_idxs,
                             spike_trains, synapse_params):
        synapse_list = []
        for number, comp_idx in enumerate(cell_input_idxs):
            synapse_params.update({'idx': int(comp_idx)})
            if 'split_sim' in self.param_dict:
                if self.param_dict['split_sim'] == 'top' and cell.zmid[comp_idx] < 500:
                    continue
                if self.param_dict['split_sim'] == 'bottom' and cell.zmid[comp_idx] > 500:
                    continue

            s = LFPy.Synapse(cell, **synapse_params)
            s.set_spike_times(spike_trains[number])
            synapse_list.append(s)
        return synapse_list

    def plot_cell_to_ax(self, cell, ax):
        [ax.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]],
                 lw=1.5, color='0.5', zorder=0, alpha=1)
            for idx in range(len(cell.xmid))]

        # [ax.plot(cell.xmid[idx], cell.zmid[idx], 'g.', ms=4) for idx in cell.synidx]
        for syn in cell.synapses:
            ax.plot(cell.xmid[syn.idx], cell.zmid[syn.idx], marker='.', ms=4, c='r')

    def _plot_results(self, cell):

        plt.close('all')
        fig = plt.figure(figsize=[18, 9])
        fig.subplots_adjust(right=0.97, left=0.05)

        scale = 'log'

        cell_ax = fig.add_subplot(1, 6, 1, aspect=1, frameon=False, xticks=[], yticks=[])
        ax_center = fig.add_subplot(1, 6, 5, xlim=[1, self.max_freq], xscale=scale, xlabel='Hz', ylabel='z')
        ax_lfp = fig.add_subplot(1, 6, 4, xlim=[0, cell.tvec[-1]], xlabel="Time [ms]", ylabel='z',
                                 title="LFP")
        ax_psd = fig.add_subplot(2, 6, 6, xlim=[1, self.max_freq], xlabel="Hz", ylabel='PSD')
        ax_psd_n = fig.add_subplot(2, 6, 12, xlim=[1, self.max_freq], ylim=[1e-4, 1e1],
                                   xlabel="Hz", ylabel='Normalized PSD')

        ax_somav = fig.add_subplot(1, 6, 2, xlabel='Time', ylabel='V$_m$')
        ax_imem = fig.add_subplot(1, 6, 3, xlabel='Time', ylabel='I$_m$',
                                  title="Imem close to electrode")
        if not hasattr(cell, "vmem"):
            ax_somav.plot(cell.tvec, cell.somav)
        else:
            ax_somav.set_title("Vm (max STD: %1.2f)" % np.max(np.std(cell.vmem, axis=1)))
        self.plot_cell_to_ax(cell, cell_ax)
        center_electrode = LFPy.RecExtElectrode(cell, **self.center_electrode_parameters)
        center_electrode.calc_lfp()
        center_LFP = 1000 * center_electrode.LFP
        freq, psd = tools.return_freq_and_psd_welch(center_LFP, self.welch_dict)
        elec_clr = lambda elec_idx: plt.cm.rainbow(elec_idx / (len(center_LFP) - 1))

        h = center_electrode.z[1] - center_electrode.z[0]
        for idx in range(len(center_LFP)):
            closest_cell_idx = cell.get_closest_idx(x=center_electrode.x[idx],
                                                    y=center_electrode.y[idx],
                                                    z=center_electrode.z[idx])
            im_y = cell.imem[closest_cell_idx] / np.max(cell.imem) * h + center_electrode.z[idx]
            ax_imem.plot(cell.tvec, im_y, c=elec_clr(idx))

            if hasattr(cell, "vmem"):
                vm_y = cell.vmem[closest_cell_idx] / np.max(cell.vmem) * h + center_electrode.z[idx]
                ax_somav.plot(cell.tvec, vm_y, c=elec_clr(idx))

            y = center_LFP[idx]/np.max(np.abs(center_LFP)) * h + center_electrode.z[idx]
            ax_lfp.plot(cell.tvec, y, c=elec_clr(idx))
            ax_psd.loglog(freq, psd[idx], c=elec_clr(idx))
            ax_psd_n.loglog(freq, psd[idx]/np.max(psd[idx]), c=elec_clr(idx))
            cell_ax.plot(center_electrode.x[idx], center_electrode.z[idx], 'D', c=elec_clr(idx))

        vmax = 10**np.ceil(np.log10(np.max(psd)))
        im_dict = {'cmap': 'hot', 'norm': LogNorm(), 'vmin': vmax/1e4, 'vmax': vmax}
        try:
            img = ax_center.pcolormesh(freq, center_electrode.z, psd, **im_dict)
            cbar = plt.colorbar(img, ax=ax_center)
        except:
            pass
        fig.suptitle(self.sim_name)

        fig.savefig(join(self.figure_folder, '%s.png' % self.sim_name))
        plt.close("all")

if __name__ == '__main__':

    # from param_dicts import classic_population_params as param_dict
    # from param_dicts import stick_population_params as param_dict
    from param_dicts import generic_population_params as param_dict
    # from param_dicts import multimorph_population_params as param_dict

    param_dict.update({'input_region': sys.argv[1],
                       'correlation': float(sys.argv[2]),
                       'distribution': "linear_increase",
                       'conductance_type': 'generic',
                       'mu': None if sys.argv[3] == "None" else float(sys.argv[3]),
                       'holding_potential': -80.,
                       'cell_number': int(sys.argv[4])})
    ns = NeuralSimulation(**param_dict)
    ns.run_single_simulation()

