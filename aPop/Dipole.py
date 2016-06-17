# -*- coding: utf-8 -*-
"""
Created on Mon May 11 09:28:16 2015

@author: solveig
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 14:53:06 2014

@author: solveig

"""
import numpy as np
import neuron as nrn

class Dipole:
    """Calculate currents and current dipole moment from cell."""
    
    def __init__(self, cell, first_sp = None):
        """Initialize cell, # timesteps, # sections in cell."""
        self.cell = cell
        self.time = len(cell.tvec)
        self.numsecs = len(cell.allsecnames)
        self.vlist = cell.vmem
        self.children_dict = self.children_dictionary()
        self.ri_list = self.axial_resistance()
        if not first_sp:
            first_sp = 0#cell.sptimeslist[0][0]
        self.startstep = 0#int((self.time - 1)/(self.cell.tstopms-
                        #self.cell.tstartms)*np.floor(first_sp) + 1)
        
    def transmembrane_currents(self):
        """Return midpoint locations and transmembrane currents.
           
           Returns
           _______
           r_seg : ndarray [microm]
               Array containing location vectors for midpoints of 
               all segments in cell.
           
           i_trans : ndarray [nA]
               Array of transmembrane currents through midpoints of
               all compartments in cell.
        """
               
        r_seg = np.array([self.cell.get_intersegment_vector(idx0 = 
                        0, idx1 = i) for i in range(
                        self.cell.totnsegs)])
        r_seg += self.cell.somapos
        
        i_trans = self.cell.imem        
        
        return r_seg, i_trans

    def axial_currents(self):
        """Return magnitude and distance traveled by axial currents.
           
           Returns
           -------
           i_axial : ndarray [nA]
               Array of axial currents, I(t) going from compartment
               end to mid/ compartment mid to start for all
               comparment halves in cell.
           d_list : ndarray [microm]
               Array of distance vectors traveled by each axial 
               current in i_axial.
        """

        iaxial = []
        d_list = []

        dseg = zip((self.cell.xmid - self.cell.xstart),
                   (self.cell.ymid - self.cell.ystart),
                   (self.cell.zmid - self.cell.zstart))
        dpar = zip((self.cell.xend - self.cell.xmid),
                   (self.cell.yend - self.cell.ymid),
                   (self.cell.zend - self.cell.zmid))
        
        for secnum, sec in enumerate(nrn.h.allsec()):
            print sec.name()

            # if not nrn.h.SectionRef(sec.name()).has_parent():
            #     skip soma, since soma is an orphan
                # print "skip"
                # continue
            # else:
            #     print "keep"

            bottom_seg = True

            parentseg = nrn.h.SectionRef(sec.name()).parent()
            parentsec = parentseg.sec

            branch = len(self.children_dict[
                              parentsec.name()])> 1
            
            parent_idx = self.cell.get_idx(section = 
                                           parentsec.name())[-1]
            seg_idx = self.cell.get_idx(section=sec.name())[0]


            # we only need parent_ri calculated for bottom-segments
            # that aren't children of soma.
            parent_ri = (nrn.h.ri(0) if bottom_seg and 
                         not 'soma' in parentsec.name() else 0)

            for seg in sec:
                iseg, ipar = self.parent_and_segment_i(seg_idx, 
                               parent_idx, parent_ri,bottom_seg,
                               branch, sec, parentsec)

                if bottom_seg and 'soma' in parentsec.name():
                    # if a seg is connencted to soma, it is 
                    # connected to the middle of soma,
                    # and dpar needs to be altered.
                    dpar[parent_idx] = [(self.cell.xstart[seg_idx] - 
                                        self.cell.xmid[parent_idx]),
                                        (self.cell.ystart[seg_idx] - 
                                        self.cell.ymid[parent_idx]),
                                        (self.cell.zstart[seg_idx] - 
                                        self.cell.zmid[parent_idx])]

                d_list.append(dpar[parent_idx])
                d_list.append(dseg[seg_idx])
                iaxial.append(ipar)
                iaxial.append(iseg)
                
                parent_idx = seg_idx
                seg_idx += 1
#                counter += 2
                branch = False
                bottom_seg = False
                parent_ri = 0
        return np.array(d_list), np.array(iaxial)

    def axial_resistance(self):
        """Return NEURON axial resistance for all cell compartments.
        
           Returns
           -------
           ri_list : ndarray [MOhm]
               Array containing nrn.h.ri(seg.x) for all segments in
               cell. nrn.h.ri(seg.x) is the axial resistance from
               the middle of the segment to the middle of its parent
               segment. If seg is the first/ bottom segment in a
               section, nrn.h.ri(seg.x) is the axial resistance from
               the middle to the start of the segment, seg, only.
        """

        ri_list = np.zeros(self.cell.totnsegs)
        comp = 0
        for sec in nrn.h.allsec():
            for seg in sec:
                ri_list[comp] = nrn.h.ri(seg.x)
                comp += 1
        return ri_list
    
    def children_dictionary(self):
        """Return dictionary with children seg indices for all secs.
        
           Returns
           -------
           children_dict : dictionary
               Dictionary containing a list for each section,
               with the segment index of all the section's children.
               The dictionary is needed to find the 
               sibling of a segment.
        """
        
        children_dict = {}
        for sec in nrn.h.allsec():
            children_dict[sec.name()] = []
            for child in nrn.h.SectionRef(sec.name()).child:
                # add index of first segment of each child
                children_dict[sec.name()].append(self.cell.get_idx(
                                        section = child.name())[0])
        
        return children_dict

    def parent_and_segment_i(self, seg_idx, parent_idx,
                             parent_ri, bottom_seg, branch,
                             sec, parentsec):
        """Return current from segmid to start and parentend to mid.
           
           Parameters
           ----------
           seg_idx : int
           parent_idx : int
           parent_ri : float [MOhm]
           bottom_seg : boolean
           branch : boolean
           sec : nrn.Section object
           parentsec : nrn.Section object
           
           Returns
           -------
           iseg : ndarray [nA]
               ndarray containing axial currents from segment middle
               to segment start for all segments in sec.
           ipar : ndarray [nA]
               ndarray containing axial currents from
               parent segment end to parent segment middle
               forall parent segments in cell.
        
        """
        seg_ri = self.ri_list[seg_idx]
        vpar = self.vlist[parent_idx]
        vseg = self.vlist[seg_idx]
        if bottom_seg and branch and not 'soma' in parentsec.name():
            # segment is a bottom_seg with siblings and a parent
            # hat is not soma. need to calculate ipar and iseg
            # separately.

            [[sib_idx]] = np.take(self.children_dict[
                                parentsec.name()], 
                                np.where(self.children_dict[
                                parentsec.name()]
                              != seg_idx))                
            sib_ri = self.ri_list[sib_idx]
            vsib = self.vlist[sib_idx]

            if np.abs(parent_ri) < 1e-8:
                raise RuntimeError("Zero parent ri")
                
            v_branch = (vpar/parent_ri + vseg/seg_ri +
                        vsib/sib_ri)*(1./(1./parent_ri + 
                        1./seg_ri + 1./sib_ri))
            # only a fraction of ipar is added for each parent,
            # since children can have the same parent
            # and ipar should only be counted once.
            ipar = (vpar - 
                v_branch)/parent_ri/len(self.children_dict[
                                        parentsec.name()])
            iseg = (v_branch - vseg)/seg_ri
        else:
            ri = (parent_ri + seg_ri)
            iseg = (vpar - vseg)/ri
            ipar = iseg
            
        return iseg, ipar

    def current_dipole_moment(self, dist, current):
        """Return current dipole moment vector P and P_tot.
           
           Parameters
           ----------
           current : ndarray [nA]
               Either an array containing all transmembrane currents
               from all compartments of the cell. Or an array of all
               axial currents between compartments in cell.
           dist : ndarray [microm]
               When input current is an array of axial currents,
               the dist is the length of each axial current.
               When current is the an array of transmembrane
               currents, dist is the position vector of each
               compartment middle.
        
           Returns
           -------
           P : ndarray [10^-15 mA]
               Array containing the current dipole moment for all
               timesteps in the x-, y- and z-direction.
           P_tot : ndarray [10^-15 mA]
               Array containing the magnitude of the
               current dipole moment vector for all timesteps.
        """

        P = np.dot(current.T, dist)
        # print P[:self.startstep]
        P[:self.startstep] = 0.
        P_tot = np.sqrt(np.sum(P**2, axis=1))
        return P, P_tot
