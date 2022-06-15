#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lidar Scatterer Augmentation (LISA)
    - Given lidar point cloud and a rain rate generates corresponding noisy signal
    - Reflection data must be normalized to range [0 1] and
      range must be in units of meters
"""
import numpy as np
from scipy.special import gamma
from scipy.integrate import trapz
import PyMieScatt as ps


class LISA():
    # def __init__(self,m,lam,rmax,rmin,bdiv,dst,dR,atm_model,saved_model=False,mode='strongest'):

    def __init__(self,atm_model,m=1.328,lam=903,rmax=120,rmin=1.5,bdiv=3e-3,dst=0.05,
                 dR=0.04,saved_model=False,mode='strongest'):
        '''
        Initialize LISA class
        Parameters
        ----------
        m           : refractive index contrast
        lam         : wavelength (nm)
        rmax        : max lidar range (m)
        rmin        : min lidar range (m)
        bdiv        : beam divergence angle (rad)
        dst         : droplet diameter starting point (mm)
        dR          : range accuracy (m)
        saved_model : use saved mie coefficients (bool)
        atm_model   : atmospheric model type
        mode        : lidar return mode: "strongest" or "last"

        Returns
        -------
        None.

        '''
        self.m    = m
        self.lam  = lam
        self.rmax = rmax   # max range (m)
        self.bdiv = bdiv  # beam divergence (rad)
        self.dst  = dst   # min rain drop diameter to be sampled (mm)
        self.rmin = rmin   # min lidar range (bistatic)
        self.dR   = dR
        self.mode = mode
        self.atm_model = atm_model
        
        
        if saved_model:
            # If Mie parameters are saved, use those
            dat = np.load('mie_q.npz')
            self.D     = dat['D']
            self.qext  = dat['qext']
            self.qback = dat['qback']
        else:
            try:
                dat = np.load('mie_q.npz')
                self.D     = dat['D']
                self.qext  = dat['qext']
                self.qback = dat['qback']
            except:
                # else calculate Mie parameters
                print('Calculating Mie coefficients... \nThis might take a few minutes')
                self.D,self.qext,self.qback = self.calc_Mie_params()
                print('Mie calculation done...')
        
        # Diameter distribution function based on user input
        if atm_model=='rain':
            self.N_model = lambda D, Rr    : self.N_MP_rain(D,Rr)
            self.N_tot   = lambda Rr,dst   : self.N_MP_tot_rain(Rr,dst)
            self.N_sam   = lambda Rr,N,dst : self.MP_Sample_rain(Rr,N,dst)
            
            # Augmenting function: hybrid Monte Carlo
            self.augment  = lambda pc,Rr : self.augment_mc(pc,Rr)
        
        elif atm_model=='snow':
            self.N_model = lambda D, Rr    : self.N_MG_snow(D,Rr)
            self.N_tot   = lambda Rr,dst   : self.N_MG_tot_snow(Rr,dst)
            self.N_sam   = lambda Rr,N,dst : self.MG_Sample_snow(Rr,N,dst)
            self.m       = 1.3031 # refractive index of ice
            
            # Augmenting function: hybrid Monte Carlo
            self.augment  = lambda pc,Rr : self.augment_mc(pc,Rr)
        

        
        elif atm_model=='strong_advection_fog':
            self.N_model = lambda D : self.Nd_strong_advection_fog(D)
            
            # Augmenting function: average effects
            self.augment  = lambda pc : self.augment_avg(pc)
        
        elif atm_model=='mod_strong_advection_fog':
            self.N_model = lambda D : self.Nd_mod_strong_advection_fog(D)
            
            # Augmenting function: average effects
            self.augment  = lambda pc : self.augment_avg(pc)

        elif atm_model=='moderate_advection_fog':
            self.N_model = lambda D : self.Nd_moderate_advection_fog(D)
            
            # Augmenting function: average effects
            self.augment  = lambda pc : self.augment_avg(pc)

        elif atm_model=='easy_advection_fog':
            self.N_model = lambda D : self.Nd_easy_advection_fog(D)
            
            # Augmenting function: average effects
            self.augment  = lambda pc : self.augment_avg(pc)

        elif atm_model=='chu_hogg_fog':
            self.N_model = lambda D : self.Nd_chu_hogg(D)
            
            # Augmenting function: average effects
            self.augment  = lambda pc : self.augment_avg(pc)
    
    def augment_mc(self,pc,Rr):
        '''
        Augment clean pointcloud for a given rain rate
        Parameters
        ----------
        pc : pointcloud (N,4) -> x,y,z,reflectivity
        Rr : rain rate (mm/hr)

        Returns
        -------
        pc_new : new noisy point cloud (N,5) -> x,y,z,reflectivity,label
                        label 0 -> lost point
                        label 1 -> randomly scattered point
                        label 2 -> not-scattered 
        '''
        shp    = pc.shape
        pc_new = np.zeros((shp[0],shp[1]+1))
        leng = len(pc)
        for i in range(leng):
            x    = pc[i,0]
            y    = pc[i,1]
            z    = pc[i,2]
            ref  = pc[i,3]
            if ref!=0:
                pc_new[i,:]  = self.lisa_mc(x,y,z,ref,Rr) 
            else:
                pc_new[i,:]  = self.lisa_mc(x,y,z,self.rmax**(-2)*1.2,Rr)         
        return pc_new
    
    def lisa_mc(self,x,y,z,ref,Rr):
        '''
        For a single lidar return, performs a hybrid Monte-Carlo experiment

        Parameters
        ----------
        x,y,z : coordinates of the point
        ref   : reflectivity [0 1]
        Rr    : rain rate (mm/hr)

        Returns
        -------
        x,y,z   : new coordinates of the noisy lidar point
        ref_new : new reflectivity
        '''
        rmax = self.rmax                      # max range (m)
        Pmin = 0.9*rmax**(-2)                 # min measurable power (arb units)
        bdiv = self.bdiv                      # beam divergence (rad)
        Db   = lambda x: 1e3*np.tan(bdiv)*x   # beam diameter (mm) for a given range (m)
        
        dst  = self.dst                       # min rain drop diameter to be sampled (mm)
        n    = self.m                         # refractive index of scatterer
        rmin = self.rmin                      # min lidar range (bistatic)
        
        
        Nd          = self.N_model(self.D,Rr) # density of rain droplets (m^-3)
        alpha, beta = self.alpha_beta(Nd)     # extinction coeff. (1/m)  
        
        ran   = np.sqrt(x**2 + y**2 + z**2)                               # range in m
        if ran>rmin:
            bvol  = (np.pi/3)*ran*(1e-3*Db(ran)/2)**2                         # beam volume in m^3 (cone)
            Nt    = self.N_tot(Rr,dst) * bvol                                 # total number of particles in beam path
            Nt    = np.int32(np.floor(Nt) + (np.random.rand() < Nt-int(Nt)))  # convert to integer w/ probabilistic rounding
        else:
            Nt = 0
            
        ran_r = ran*(np.random.rand(Nt))**(1/3) # sample distances from a quadratic pdf
        indx  = np.where(ran_r>rmin)[0]         # keep points where ranges larger than rmin
        Nt    = len(indx)                       # new particle number
        
        P0  = ref*np.exp(-2*alpha*ran)/(ran**2) # power
        snr = P0/Pmin # signal noise ratio
        if Nt>0:
            Dr    = self.N_sam(Rr,Nt,dst) # randomly sample Nt particle diameters
            ref_r = abs((n-1)/(n+1))**2   # Fresnel reflection at normal incidence
            ran_r = ran_r[indx]
            
            # Calculate powers for all particles       
            Pr = ref_r*np.exp(-2*alpha*ran_r)*np.minimum((Dr/Db(ran_r))**2,np.ones(Dr.shape))/(ran_r**2)
            if (self.mode=='strongest'):
                ind_r = np.argmax(Pr) # index of the max power
                
                if P0<Pmin and Pr[ind_r]<Pmin: # if all smaller than Pmin, do nothing
                    ran_new = 0
                    ref_new = 0
                    labl    = 0 # label for lost point
                    #print(P0, Pr[ind_r])
                elif P0<Pr[ind_r]: # scatterer has larger power
                    ran_new = ran_r[ind_r] # new range is scatterer range
                    ref_new = ref_r*np.exp(-2*alpha*ran_new)*np.minimum((Dr[ind_r]/Db(ran_r[ind_r]))**2,1) # new reflectance biased by scattering
                    labl    = 1 # label for randomly scattered point 
                else: # object return has larger power
                    sig     = self.dR/np.sqrt(2*snr)        # std of range uncertainty
                    ran_new = ran + np.random.normal(0,sig) # range with uncertainty added
                    ref_new = ref*np.exp(-2*alpha*ran)      # new reflectance modified by scattering
                    labl    = 2                             # label for a non-scattering point
            elif (self.mode=='last'):
                # if object power larger than Pmin, then nothing is scattered
                if P0>Pmin:
                    sig     = self.dR/np.sqrt(2*snr)        # std of range uncertainty
                    ran_new = ran + np.random.normal(0,sig) # range with uncertainty added
                    ref_new = ref*np.exp(-2*alpha*ran)      # new reflectance modified by scattering
                    labl    = 2                             # label for a non-scattering point
                # otherwise find the furthest point above Pmin
                else:
                    inds = np.where(Pr>Pmin)[0]
                    if len(inds) == 0:
                        ran_new = 0
                        ref_new = 0
                        labl    = 0 # label for lost point
                        # print(self.mode)
                    else:
                        ind_r   = np.where(ran_r == np.max(ran_r[inds]))[0]
                        ran_new = ran_r[ind_r] # new range is scatterer range
                        ref_new = ref_r*np.exp(-2*alpha*ran_new)*np.minimum((Dr[ind_r]/Db(ran_r[ind_r]))**2,1) # new reflectance biased by scattering
                        labl    = 1 # label for randomly scattered point 
                    
            else:
                print("Invalid lidar return mode")
            
        else:
            if P0<Pmin:
                ran_new = 0
                ref_new = 0
                labl    = 0 # label for lost point
                #print(P0, 'Nt',Nt)
            else:
                sig     = self.dR/np.sqrt(2*snr)        # std of range uncertainty
                ran_new = ran + np.random.normal(0,sig) # range with uncertainty added
                ref_new = ref*np.exp(-2*alpha*ran)      # new reflectance modified by scattering
                labl    = 2                             # label for a non-scattering point
        
        # Angles are same
        if ran>0:
            phi = np.arctan2(y,x)  # angle in radians
            the = np.arccos(z/ran) # angle in radians
        else:
            phi,the=0,0
        
        # Update new x,y,z based on new range
        x = ran_new*np.sin(the)*np.cos(phi)
        y = ran_new*np.sin(the)*np.sin(phi)
        z = ran_new*np.cos(the)
        
        return x,y,z,ref_new,labl
    
    def augment_avg(self,pc):

        shp    = pc.shape      # data shape
        pc_new = np.zeros(shp) # init new point cloud
        leng   = shp[0]        # data length
        
        # Rename variables for better readability
        x    = pc[:,0]
        y    = pc[:,1]
        z    = pc[:,2]
        ref  = pc[:,3]          
        
        # Get parameters from class init
        rmax = self.rmax       # max range (m)
        Pmin = 0.9*rmax**(-2)  # min measurable power (arb units)
        rmin = self.rmin       # min lidar range (bistatic)
        
        # Calculate extinction coefficient from the particle distribution
        Nd          = self.N_model(self.D) # density of rain droplets (m^-3)
        alpha, beta = self.alpha_beta(Nd)  # extinction coeff. (1/m)  
        
        ran   = np.sqrt(x**2 + y**2 + z**2)  # range in m
        indx  = np.where(ran>rmin)[0]         # keep points where ranges larger than rmin
        
        P0        = np.zeros((leng,))                                  # init back reflected power
        P0[indx]  = ref[indx]*np.exp(-2*alpha*ran[indx])/(ran[indx]**2) # calculate reflected power
        snr       = P0/Pmin                                             # signal noise ratio
        
        indp = np.where(P0>Pmin)[0] # keep points where power is larger than Pmin
        
        sig        = np.zeros((leng,))                         # init sigma - std of range uncertainty
        sig[indp]  = self.dR/np.sqrt(2*snr[indp])               # calc. std of range uncertainty
        ran_new    = np.zeros((leng,))                         # init new range
        ran_new[indp]    = ran[indp] + np.random.normal(0,sig[indp])  # range with uncertainty added, keep range 0 if P<Pmin
        ref_new    = ref*np.exp(-2*alpha*ran)                   # new reflectance modified by scattering
        
        # Init angles
        phi = np.zeros((leng,))
        the = np.zeros((leng,))
        
        phi[indx] = np.arctan2(y[indx],x[indx])   # angle in radians
        the[indx] = np.arccos(z[indx]/ran[indx])  # angle in radians
        
        # Update new x,y,z based on new range
        pc_new[:,0] = ran_new*np.sin(the)*np.cos(phi)
        pc_new[:,1] = ran_new*np.sin(the)*np.sin(phi)
        pc_new[:,2] = ran_new*np.cos(the)
        pc_new[:,3] = ref_new
        
        return pc_new
    
    def calc_Mie_params(self):
        '''
        Calculate scattering efficiencies
        Returns
        -------
        D     : Particle diameter (mm)
        qext  : Extinction efficiency
        qback : Backscattering efficiency

        '''
        out   = ps.MieQ_withDiameterRange(self.m, self.lam, diameterRange=(1,1e7),
                                        nd=2000, logD=True)
        D     = out[0]*1e-6
        qext  = out[1]
        qback = out[6]
        
        # Save for later use since this function takes long to run
        np.savez('mie_q.npz',D=D,qext=qext,qback=qback)
        
        return D,qext,qback
    
    
    def alpha_beta(self,Nd):
        '''
        Calculates extunction and backscattering coefficients
        Parameters
        ----------
        Nd : particle size distribution, m^-3 mm^-1

        Returns
        -------
        alpha : extinction coefficient
        beta  : backscattering coefficient
        '''
        D  = self.D
        qe = self.qext
        qb = self.qback
        alpha = 1e-6*trapz(D**2*qe*Nd,D)*np.pi/4 # m^-1
        beta  = 1e-6*trapz(D**2*qb*Nd,D)*np.pi/4 # m^-1
        return alpha, beta
    
    # RAIN
    def N_MP_rain(self,D,Rr):
        '''
        Marshall - Palmer rain model

        Parameters
        ----------
        D  : rain droplet diameter (mm)
        Rr : rain rate (mm h^-1)

        Returns
        -------
        number of rain droplets for a given diameter (m^-3 mm^-1)
        '''
        return 8000*np.exp(-4.1*Rr**(-0.21)*D)
    
    def N_MP_tot_rain(self,Rr,dstart):
        '''
        Integrated Marshall - Palmer Rain model

        Parameters
        ----------
        Rr     : rain rate (mm h^-1)
        dstart : integral starting point for diameter (mm)

        Returns
        -------
        rain droplet density (m^-3) for a given min diameter
        '''
        lam = 4.1*Rr**(-0.21)
        return 8000*np.exp(-lam*dstart)/lam

    def MP_Sample_rain(self,Rr,N,dstart):
        '''
        Sample particle diameters from Marshall Palmer distribution

        Parameters
        ----------
        Rr     : rain rate (mm/hr)
        N      : number of samples
        dstart : Starting diameter (min diameter sampled)

        Returns
        -------
        diameters : diameter of the samples

        '''
        lmda      = 4.1*Rr**(-0.21)
        r         = np.random.rand(N)
        diameters = -np.log(1-r)/lmda + dstart
        return diameters
    
    # SNOW
    def N_MG_snow(self,D,Rr):
        '''
        Marshall - Palmer snow model

        Parameters
        ----------
        D  : snow diameter (mm)
        Rr : water equivalent rain rate (mm h^-1)

        Returns
        -------
        number of snow particles for a given diameter (m^-3 mm^-1)
        '''
        N0   = 7.6e3* Rr**(-0.87)
        lmda = 2.55* Rr**(-0.48)
        return N0*np.exp(-lmda*D)
    
    def N_MG_tot_snow(self,Rr,dstart):
        '''
        Integrated Marshall - Gunn snow model

        Parameters
        ----------
        Rr     : rain rate (mm h^-1)
        dstart : integral starting point for diameter (mm)

        Returns
        -------
        snow particle density (m^-3) for a given min diameter
        '''
        N0   = 7.6e3* Rr**(-0.87)
        lmda = 2.55* Rr**(-0.48)
        return N0*np.exp(-lmda*dstart)/lmda

    def MG_Sample_snow(self,Rr,N,dstart):
        '''
        Sample particle diameters from Marshall Palmer distribution

        Parameters
        ----------
        Rr     : rain rate (mm/hr)
        N      : number of samples
        dstart : Starting diameter (min diameter sampled)

        Returns
        -------
        diameters : diameter of the samples

        '''
        lmda      = 2.55* Rr**(-0.48)
        r         = np.random.rand(N)
        diameters = -np.log(1-r)/lmda + dstart
        return diameters
    # FOG
    def N_GD(self,D,rho,alpha,g,Rc):
        '''
        Gamma distribution model
        Note the parameters are NOT normalized to unitless values
        For example D^alpha term will have units Length^alpha
        It is therefore important to use exactly the same units for D as those
        cited in the paper by Rasshofer et al. and then perform unit conversion
        after an N(D) curve is generated
    
        D  : rain diameter
        Outputs number of rain droplets for a given diameter
        '''
        b = alpha/(g*Rc**g)
        
        Nd = g*rho*b**((alpha+1)/g)*(D/2)**alpha*np.exp(-b*(D/2)**g)/gamma((alpha+1)/g)
        
        return Nd
    # Coastal fog distribution
    # With given parameters, output has units cm^-3 um^-1 which is
    # then converted to m^-3 mm^-1 which is what alpha_beta() expects
    # so whole quantity is multiplied by (100 cm/m)^3 (1000 um/mm)
    def Nd_haze_coast(self,D):
        return 1e9*self.N_GD(D*1e3,rho=100,alpha=1,g=0.5,Rc=0.05e-3)
    
    # Continental fog distribution
    def Nd_haze_continental(self,D):
        return 1e9*self.N_GD(D*1e3,rho=100,alpha=2,g=0.5,Rc=0.07)
    
    # Strong advection fog
    def Nd_strong_advection_fog(self,D):
        return 1e9*self.N_GD(D*1e3,rho=20,alpha=3,g=1.,Rc=10)

    # Moderate-Strong advection fog
    def Nd_mod_strong_advection_fog(self,D):
        return 1e9*self.N_GD(D*1e3,rho=20,alpha=3,g=1.,Rc=9)
    
    # Moderate advection fog
    def Nd_moderate_advection_fog(self,D):
        return 1e9*self.N_GD(D*1e3,rho=20,alpha=3,g=1.,Rc=8)

    # Easy advection fog
    def Nd_easy_advection_fog(self,D):
        return 1e9*self.N_GD(D*1e3,rho=20,alpha=3,g=1.,Rc=6)
    
    # # Strong spray
    # def Nd_strong_spray(self,D):
    #     return 1e9*self.N_GD(D*1e3,rho=100,alpha=6,g=1.,Rc=4)
    
    # # Moderate spray
    # def Nd_moderate_spray(self,D):
    #     return 1e9*self.N_GD(D*1e3,rho=100,alpha=6,g=1.,Rc=2)
    
    # Chu/Hogg
    def Nd_chu_hogg(self,D):
        return 1e9*self.N_GD(D*1e3,rho=20,alpha=2,g=0.5,Rc=1)
    
