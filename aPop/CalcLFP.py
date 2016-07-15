# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 12:20:17 2015

@author: solveig
"""

import numpy as np

class CalcLFP:
    '''Calculate extracellular potentials from cell.'''
    
    def __init__(self, cell, X, Y, Z, ppidx = [], first_sp = None):
        '''Initialize cell, X, Y, Z and dipole midpoint r_mid.'''
        # need ppidx and first_sp if pointprocess input        
        # conversion factors:
        self.k1 = 1E6 # from mV to nV        
        
        self.cell = cell
        self.time = len(cell.tvec)
        self.totnsegs = cell.totnsegs
        self.X = X
        self.Y = Y
        self.Z = Z
        syninds = cell.synidx + ppidx
        
        r_soma_syns = [self.cell.get_intersegment_vector(idx0 = 0,
                       idx1 = i) for i in syninds]
        self.r_mid = np.average(r_soma_syns, axis = 0)
        self.r_mid = self.r_mid/2. + self.cell.somapos
        if not first_sp:
            first_sp = cell.sptimeslist[0][0]
            
        self.startstep = int((self.time - 1)/(
                             self.cell.tstopms -
                             self.cell.tstartms)*np.floor(
                             first_sp) + 1)


    def grid_lfp_theta(self, P, sigma):
        '''Return array phi(t) for points in XYZ-grid,timedep theta.

           Parameters
           ----------
           P : ndarray [1E-15 mA]
               Array containing the current dipole moment for 
               all timesteps in the x-, y- and z-direction.
           sigma : float [ohm/m]
               Extracellular Conductivity.
          
           Returns
           -------
           theta : ndarray [radians]
               Angle between phi(t) and distance vector from
               electrode to current dipole location, 
               calculated for all timesteps.
           grid_LFP : ndarray [nV]
               Array containing the current dipole moment at all
               points in X-, Y-, Z-grid for all timesteps.
        '''
        gridpoints = zip(self.X.flatten(), self.Y.flatten(), self.Z.flatten())
        grid_LFP = np.zeros((len(gridpoints), self.time))
        grid_theta = np.zeros((len(gridpoints), self.time))
        for j in range(len(gridpoints)):
            dist = gridpoints[j] - self.r_mid
            cos_theta = np.dot(P, dist)/(np.linalg.norm(dist)*np.linalg.norm(P, axis=1))
            cos_theta = np.nan_to_num(cos_theta)
            theta = np.arccos(cos_theta)
            grid_theta[j, :] = theta
            grid_LFP[j, :] = 1./(4*np.pi*sigma)*np.linalg.norm(P, axis=1)*cos_theta/np.sum(dist**2)*self.k1

        return grid_LFP, grid_theta
