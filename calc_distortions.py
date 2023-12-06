#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 16:42:02 2023

@author: arest

Calculdate the distortions of detector/filter/pupil using a list of images and their catalogs

"""

import jwst
import os,re,sys,copy
from jwst.datamodels import ImageModel
from jwst import datamodels
import astropy.units as u
import pysiaf
from pdastro import makepath,rmfile,pdastroclass,pdastrostatsclass,AnotB,AorB,unique,makepath4file
from astropy.io import fits
import numpy as np
from astropy.modeling.models import Polynomial2D, Mapping, Shift
from scipy import linalg
import pandas as pd
import glob
import random
import matplotlib.pyplot as plt
import argparse
from calc_fpa2fpa_alignment import calc_v2v3center_info

from astropy.coordinates import SkyCoord

def v2v3_idl_model(from_sys, to_sys, par, angle):
    """
    Creates an astropy.modeling.Model object
    for the undistorted ("ideal") to V2V3 coordinate translation
    """
    if from_sys != 'v2v3' and to_sys != 'v2v3':
        raise ValueError("This function is designed to generate the transformation either to or from V2V3.")

    # Cast the transform functions as 1st order polynomials
    xc = {}
    yc = {}
    if to_sys == 'v2v3':
        xc['c1_0'] = par * np.cos(angle)
        xc['c0_1'] = np.sin(angle)
        yc['c1_0'] = (0.-par) * np.sin(angle)
        yc['c0_1'] = np.cos(angle)

    if from_sys == 'v2v3':
        xc['c1_0'] = par * np.cos(angle)
        xc['c0_1'] = par * (0. - np.sin(angle))
        yc['c1_0'] = np.sin(angle)
        yc['c0_1'] = np.cos(angle)

    #0,0 coeff should never be used.
    xc['c0_0'] = 0
    yc['c0_0'] = 0

    xmodel = Polynomial2D(1, **xc)
    ymodel = Polynomial2D(1, **yc)

    return xmodel, ymodel

# https://pysiaf.readthedocs.io/en/latest/_modules/pysiaf/utils/polynomial.html
def polyfit0(u, x, y, order,fit_coeffs0=False,weight=None):
    """Fit polynomial to a set of u values on an x,y grid.
    u is a function u(x,y) being a polynomial of the form
    u = a[i, j] x**(i-j) y**j. x and y can be on a grid or be arbitrary values
    This version uses scipy.linalg.solve instead of matrix inversion.
    u, x and y must have the same shape and may be 2D grids of values.
    
    **** NOTE: in this routine coeffs[0] is kept at 0 if fit_coeffs0=False! ****
    
    Parameters
    ----------
    u : array
        an array of values to be the results of applying the sought after
        polynomial to the values (x,y)
    x : array
        an array of x values
    y : array
        an array of y values
    order : int
        the polynomial order
    Returns
    -------
    coeffs: array
        polynomial coefficients being the solution to the fit.
    """
    # First set up x and y powers for each coefficient
    px = []
    py = []

    if fit_coeffs0 == False:
        ### Do not fit coeffs[0]!!!!
        startindex = 1
    else:
        startindex = 0
        
    for i in range(startindex,order + 1):
        for j in range(i + 1):
            px.append(i - j)
            py.append(j)
    terms = len(px)
    
    # Make up matrix and vector
    vector = np.zeros((terms))
    mat = np.zeros((terms, terms))
    for i in range(terms):
        vector[i] = (u * x ** px[i] * y ** py[i]).sum()  # Summing over all x,y
        for j in range(terms):
            mat[i, j] = (x ** px[i] * y ** py[i] * x ** px[j] * y ** py[j]).sum()

    coeffs = linalg.solve(mat, vector)
    
    if fit_coeffs0 == False:
        ### Add coeffs[0]=0.0
        coeffs0 = [0.0]
        coeffs0.extend(coeffs)  
    else:
        coeffs0 = coeffs
    
    return coeffs0

def initplot(nrows=1, ncols=1, xfigsize4subplot=5, yfigsize4subplot=None, **kwargs):
    sp=[]
    if yfigsize4subplot is None: yfigsize4subplot=xfigsize4subplot
    xfigsize=xfigsize4subplot*ncols
    yfigsize=yfigsize4subplot*nrows
    plt.figure(figsize=(xfigsize,yfigsize))
    counter=1
    for row in range(nrows):
        for col in range(ncols):
            sp.append(plt.subplot(nrows, ncols, counter,**kwargs))
            counter+=1

    for i in range(len(sp)):
        plt.setp(sp[i].get_xticklabels(),'fontsize',12)
        plt.setp(sp[i].get_yticklabels(),'fontsize',12)
        sp[i].set_xlabel(sp[i].get_xlabel(),fontsize=14)
        sp[i].set_ylabel(sp[i].get_ylabel(),fontsize=14)
        sp[i].set_title(sp[i].get_title(),fontsize=14)

    return(sp)

def get_files(filepatterns,directory=None,verbose=1):
    filenames=[]
    for filepattern in filepatterns:
        if directory is not None:
            filepattern=os.path.join(directory,filepattern)
        if verbose>2: print(f'Looking for filepattern {filepattern}')
        filenames.extend(glob.glob(filepattern))
    
    for i in range(len(filenames)):
        filenames[i] = os.path.abspath(filenames[i])
    filenames=unique(filenames)
    filenames.sort()

    if verbose: print(f'Found {len(filenames)} files matching filepatterns {filepatterns}')
    return(filenames)



class calc_distortions_class(pdastrostatsclass):
    def __init__(self):
        pdastroclass.__init__(self)
        
        # define colnames of input tables
        self.colnames={}
        self.colnames['x']='x'
        self.colnames['y']='y'
        self.colnames['mag']='mag'
        self.colnames['dmag']='dmag'
        self.colnames['ra']='gaia_ra'
        self.colnames['dec']='gaia_dec'

        # output directory
        self.outdir = None
        self.outbasename = None
        
        self.instrument = None
        self.apername = None
        self.filtername = None
        self.pupilname = None
        self.verbose = 0

        self.siaf_instrument = None
        self.siaf_aperture = None

        self.imtable = pdastroclass(columns=['imID','progID','fullimage','instrument','apername',
                                             'filter','pupil',
                                             'V2_REF','V3_REF','V3I_YANG',                                             
                                             'V2cen','V3cen','V3IdlYAnglecen'])
        
        self.coeffs = pdastroclass()
        
        self.poly_degree = 5
        self.converged = False
        self.iteration = 0

        self.ixs_use = None
        self.ixs_excluded = None
        self.ixs_cut_3sigma = None
        
        # plot style for residual plots
        self.plot_style={}
        self.plot_style['good']={'style':'o','color':'blue', 'ms':5 ,'alpha':0.5}
        self.plot_style['cut']={'style':'o','color':'red', 'ms':5 ,'alpha':0.3}
        self.plot_style['excluded']={'style':'o','color':'gray', 'ms':3 ,'alpha':0.3}

        self.showplots = 0
        self.saveplots = 0

    def define_options(self,parser=None,usage=None,conflict_handler='resolve'):
        if parser is None:
            parser = argparse.ArgumentParser(usage=usage,conflict_handler=conflict_handler)

        parser.add_argument('-v','--verbose', default=0, action='count')
        parser.add_argument('-p','--showplots', default=0, action='count')
        parser.add_argument('-s','--saveplots', default=0, action='count')

        return(parser)
    
    def set_outdir(self,outrootdir=None,outsubdir=None):
        self.outdir = outrootdir
        if self.outdir is None: self.outdir = '.'
        
        if outsubdir is not None:
            self.outdir+=f'/{outsubdir}'
        return(self.outdir) 
         
    def set_outbasename(self,outrootdir=None,outsubdir=None,outbasename=None):
        """
        This sets self.outbasename and self.outdir
        
        if outbasename is not None, then self.outdir is set to the directory of outbasename
        if outbasename is None:
            self.outdir combines outrootdir and outsubdir (if exists)
            self.outbasename combines self.outdir and puts together the filename as {self.apername}_{self.filtername}_{self.pupilname}'

        Parameters
        ----------
        outrootdir : string, optional
            ouput rootdir. The default is None.
        outsubdir : string, optional
            output subdir. The default is None.
        outbasename : string, optional
            The default is None.

        Returns
        -------
        None.

        """

        if outbasename is not None:
            if outrootdir is not None or outsubdir is not None:
                raise RuntimeError(f'Cannot specify both, outbasename={outbasename} and outrootdir/outsubdir={outrootdir}/{outsubdir}')
            self.set_outdir(outrootdir=os.path.dirname(outbasename))
            self.outbasename = f'{self.outdir}/{os.path.basename(outbasename)}'
        else:
            self.set_outdir(outrootdir=outrootdir,outsubdir=outsubdir)
            self.outbasename = f'{self.outdir}/{self.apername}_{self.filtername}_{self.pupilname}'
        if self.verbose: print(f'Output basename is set to {self.outbasename}')
        return(self.outbasename)


    def get_inputfiles_info(self):
        """
        Populate the self.imtable with the relevant fits keywords 

        Returns
        -------
        None.

        """
        for ix in self.imtable.getindices():     
            hdr0 = fits.getheader(self.imtable.t.loc[ix,'fullimage'])
            hdr1 = fits.getheader(self.imtable.t.loc[ix,'fullimage'],ext=1)
            self.imtable.t.loc[ix,['instrument','apername']]=[hdr0["INSTRUME"].lower(),hdr0["APERNAME"].lower()]
            if "FILTER" in hdr0: 
                self.imtable.t.loc[ix,'filter']=hdr0["FILTER"].lower()
            else:
                self.imtable.t.loc[ix,'filter']=None
            if "PUPIL" in hdr0: 
                self.imtable.t.loc[ix,'pupil']=hdr0["PUPIL"].lower()
            else:
                self.imtable.t.loc[ix,'pupil']=None
            if "V2_REF" in hdr1: 
                self.imtable.t.loc[ix,['V2_REF','V3_REF','V3I_YANG']]=[float(hdr1["V2_REF"]),float(hdr1["V3_REF"]),float(hdr1["V3I_YANG"])]
            else:
                self.imtable.t.loc[ix,['V2_REF','V3_REF','V3I_YANG']]=[np.nan,np.nan,np.nan]
                
            m = re.search('^jw(\d\d\d\d\d)',os.path.basename(self.imtable.t.loc[ix,'fullimage']))
            if m is not None:
                progID = int(m.groups()[0])
            else:
                progID = 0
            self.imtable.t.loc[ix,'progID']=progID
        
    def get_inputfiles_imtable(self,filepatterns,directory=None):
        """
        Find all images that fulfill the filepatterns, and obtain important info like aperture name, filter, pupil etc.
        

        Parameters
        ----------
        filepatterns : array of filepattern strings
            image filepatterns.
        directory : string, optional
            directory in which the filepatterns should be applied in. The default is None.

        Returns
        -------
        None.

        """
        filenames = get_files(filepatterns,directory=directory,verbose=self.verbose)
        self.imtable.t['fullimage']=filenames
        self.imtable.t['imID']=range(len(filenames))
        self.get_inputfiles_info()
        if self.verbose>1:
            distortions.imtable.write()

        return(0)
    
    def init_coefftable(self):
        self.coeffs = pdastroclass(columns=['AperName','siaf_index','exponent_x','exponent_y','filter','pupil','Sci2IdlX','Sci2IdlY','Idl2SciX','Idl2SciY'])
        for i in range(self.poly_degree+1):
            exp_x = i
            for j in range(0,i+1):
                siaf_index = i*10+j
                self.coeffs.newrow({'AperName':self.apername,
                                    'siaf_index':siaf_index,
                                    'exponent_x':exp_x,
                                    'exponent_y':j,
                                    'filter':self.filtername,
                                    'pupil':self.pupilname,
                                    })
                exp_x -= 1
        
        self.coeffs.default_formatters['Sci2IdlX']='{:.10e}'.format
        self.coeffs.default_formatters['Sci2IdlY']='{:.10e}'.format
        self.coeffs.default_formatters['Idl2SciX']='{:.10e}'.format
        self.coeffs.default_formatters['Idl2SciY']='{:.10e}'.format

    
    def initialize(self,apername,filtername,pupilname, progIDs=None, raiseErrorFlag=True):
        """
        Get all the image in the self.imtable for the given aperture, filter, and pupil.
        Optionally constraint it to certain program IDs

        self.ixs_im: indices of self.imtable for the images to be usedused
        self.apername
        self.filtername
        self.pupilname
        self.instrument

        The following siaf apertures are also set:        
        self.siaf_instrument 
        self.siaf_aperture 


        Parameters
        ----------
        apername : string
        filtername : string
        pupilname : string
        progIDs : list of program IDs, optional
            list pf program IDs that are allowed. The default is None.
        raiseErrorFlag: boolean
            If raiseErrorFlag, then an error is raised if no images are found

        Raises
        ------
        RuntimeError
            an error is raised if no images are found and raiseErrorFlag==True.

        Returns
        -------
        None.

        """
        
        
        # first, get all the entries that match apername,filtername,pupilname      
        ixs_im = self.imtable.ix_equal('apername', apername)
        if self.verbose>1: print(f'{len(ixs_im)} images left with aperture={apername}')

        if filtername!='None' and (filtername is not None): 
            ixs_im = self.imtable.ix_equal('filter', filtername,indices=ixs_im)
            if self.verbose>1: print(f'{len(ixs_im)} images left with filter={filtername}')

        if pupilname!='None' and (pupilname is not None): 
            ixs_im = self.imtable.ix_equal('pupil', pupilname,indices=ixs_im)
            if self.verbose>1: print(f'{len(ixs_im)} images left with pupil={pupilname}')
        
        # check for program IDs
        if progIDs is not None:
            if isinstance(progIDs,str):
                progIDs = [progIDs]
            ixs_tmp = []
            for progID in progIDs:
               ixs_tmp.extend(self.imtable.ix_equal('progID', int(progID), indices=ixs_im))

            ixs_im=ixs_tmp
            if self.verbose>1: print(f'{len(ixs_im)} images left with progIDs={progIDs}')
            
        # Make sure all the selected images have exactly one value for these columns. Otherwise something is wrong!!!
        for col in ['apername','filter','pupil','V2_REF','V3_REF','V3I_YANG']:
            vals = unique(self.imtable.t.loc[ixs_im,col])
            if len(vals)>1:
                self.imtable.write(indices=ixs_im)
                raise RuntimeError(f'More than one value for column {col}: {vals}')
        
        # Any images that pass all cuts? If not raise error or return empty list
        if len(ixs_im)==0:
            self.apername=self.filtername=self.pupilname=self.instrument=self.siaf_instrument=self.siaf_aperture=self.ixs_im=None
            if raiseErrorFlag:
                raise RuntimeError('No images!!!')
            return(ixs_im)
        
        
        self.apername=self.imtable.t.loc[ixs_im[0],'apername']
        self.filtername=self.imtable.t.loc[ixs_im[0],'filter']
        self.pupilname=self.imtable.t.loc[ixs_im[0],'pupil']
        self.instrument=self.imtable.t.loc[ixs_im[0],'instrument']
        if self.verbose: print(f'Instrument {self.instrument} with aperture {self.apername}: filter={self.filtername} pupil={self.pupilname}')
        
        self.siaf_instrument = pysiaf.Siaf(self.instrument) 
        self.siaf_aperture = self.siaf_instrument.apertures[self.apername.upper()]
        
        self.ixs_im = ixs_im
        
        # set up the distortion coefficient table with the correct format!
        self.init_coefftable()
        
        return(ixs_im)
    
    def load_catalogs(self, ixs_im=None, cat_suffix = '.good.phot.txt', catcolname=None):
        if ixs_im is None:
            ixs_im = self.ixs_im
            
        frames = {}
        for ix_im in ixs_im:
            imID = self.imtable.t.loc[ix_im,'imID']
            if catcolname is not None:
                catfilename = self.imtable.t.loc[ix_im,catcolname]
            else:
                catfilename = re.sub('\.fits$',cat_suffix,self.imtable.t.loc[ix_im,'fullimage'])
            print(catfilename)

            # Load the table into hash
            frames[imID] = pd.read_csv(catfilename,sep='\s+')
            if len(frames[imID])<1:
                raise RuntimeError(f'file {catfilename} has no data!')
            
            # only keep relevant columns
            frames[imID] = frames[imID][list(self.colnames.values())].copy()

            # save the filename and assign ID
            #frames[imID]['imagename']=self.imtable.t.loc[ix_im,'fullimage']
            frames[imID]['imID']=imID
    
            # Get detector info and save in table
            image_model = ImageModel(self.imtable.t.loc[ix_im,'fullimage'])

            # Calculate v2/v3 from refcat ra/dec
            # This should be independent of the distortions, it just depends if the WCS alignment is good
            # enough with default distortions
            world_to_v2v3 = image_model.meta.wcs.get_transform('world', 'v2v3')
            (frames[ix_im]['refcat_v2'], frames[ix_im]['refcat_v3']) = world_to_v2v3(frames[ix_im][self.colnames['ra']],frames[ix_im][self.colnames['dec']])

            # get the center v2/v3 and the angle.
            # This is used to get from v2v3 to xy_idl!
            v2_0,v3_0,V3IdlYAngle = calc_v2v3center_info(image_model,self.siaf_aperture)
            self.imtable.t.loc[ix_im,['V2cen','V3cen','V3IdlYAnglecen']]= v2_0,v3_0,V3IdlYAngle
            
            print(f'Loaded {len(frames[ix_im])} rows for imID={ix_im}')
    
        # Merge the tables into one mastertable
        print(f'Loaded {len(frames)} tables')
        self.t = pd.concat(frames,ignore_index=True)
        print(f'{len(self.t)} rows total')
    
    def calc_refcat_xy_idl(self, ixs_im=None, save_imtable=True):
        if ixs_im is None:
            ixs_im = self.ixs_im

        parity = self.siaf_aperture.VIdlParity
        v3_ideal_y_angle = self.siaf_aperture.V3IdlYAngle * np.pi / 180.
        
        #refcat_v2_col4fit = 'refcat_v2'
        #refcat_v3_col4fit = 'refcat_v3'
        
        # Loop through the images
        for ix_im in ixs_im:
            imID=self.imtable.t.loc[ix_im,'imID']
            # Get all indices in the catalog table that belong to imID
            ixs_imID =self.ix_equal('imID',imID)

            # define the rotation to get to xy_idl from v2v3
            v2v32idlx, v2v32idly = v2v3_idl_model('v2v3', 'ideal', parity, self.imtable.t.loc[ix_im,'V3IdlYAnglecen']* np.pi / 180.)
        
            # calculate xy_idl
            self.t.loc[ixs_imID,'refcat_x_idl'] = v2v32idlx(self.t.loc[ixs_imID,'refcat_v2']-self.imtable.t.loc[ix_im,'V2cen'],
                                                            self.t.loc[ixs_imID,'refcat_v3']-self.imtable.t.loc[ix_im,'V3cen'])
            self.t.loc[ixs_imID,'refcat_y_idl'] = v2v32idly(self.t.loc[ixs_imID,'refcat_v2']-self.imtable.t.loc[ix_im,'V2cen'],
                                                            self.t.loc[ixs_imID,'refcat_v3']-self.imtable.t.loc[ix_im,'V3cen'])
        
        if save_imtable:
            outfilename = f'{self.outbasename}.images.txt'
            print(f'Saving {outfilename}')
            self.imtable.write(outfilename,indices=ixs_im)

        return(0)
        
    def calc_xyprime(self):

        self.t['xprime'] =  self.t['x'] + 1 - self.siaf_aperture.XSciRef 
        self.t['yprime'] =  self.t['y'] + 1 - self.siaf_aperture.YSciRef 

        return(0)
        
    def fit_Sci2Idl(self, ixs_use = None, Nsigma=3.0,percentile_cut_firstiteration=80,save_matches=True):
        if self.verbose: print('### fitting Sci2Idl...')
        if ixs_use is None:
            ixs_use = self.getindices()
        self.ixs_use = ixs_use
        self.ixs_excluded =  AnotB(self.getindices(),ixs_use)
        if self.verbose: print(f'{len(ixs_use)} entries used for fit, {len(self.ixs_excluded)} excluded')  

        self.converged = False
        self.iteration = 0

        # start fresh!
        ixs4fit=copy.deepcopy(self.ixs_use)

        while not self.converged:
            coeff_Sci2IdlX = polyfit0(self.t.loc[ixs4fit,'refcat_x_idl'], self.t.loc[ixs4fit,'xprime'], self.t.loc[ixs4fit,'yprime'],order=self.poly_degree)
            coeff_Sci2IdlY = polyfit0(self.t.loc[ixs4fit,'refcat_y_idl'], self.t.loc[ixs4fit,'xprime'], self.t.loc[ixs4fit,'yprime'],order=self.poly_degree)
        
            self.t['x_idl_fit'] = pysiaf.utils.polynomial.poly(coeff_Sci2IdlX, self.t['xprime'], self.t['yprime'], order=self.poly_degree)
            self.t['y_idl_fit'] = pysiaf.utils.polynomial.poly(coeff_Sci2IdlY, self.t['xprime'], self.t['yprime'], order=self.poly_degree)
            self.t['dx_idl_pix']  = (self.t['refcat_x_idl'] - self.t['x_idl_fit']) / coeff_Sci2IdlX[1]
            self.t['dy_idl_pix']  = (self.t['refcat_y_idl'] - self.t['y_idl_fit']) / coeff_Sci2IdlY[2]
        
            self.calcaverage_sigmacutloop('dx_idl_pix',indices=ixs4fit,verbose=0,Nsigma=Nsigma,percentile_cut_firstiteration=percentile_cut_firstiteration)
            #print(self.statparams)
            Nclip = self.statparams['Nclip']
            ixs4fit = self.statparams['ix_good']
        
            self.calcaverage_sigmacutloop('dy_idl_pix',indices=ixs4fit,verbose=0,Nsigma=Nsigma,percentile_cut_firstiteration=percentile_cut_firstiteration)
            #print(self.statparams)
            # add the Nclip from y to the Nclip from x
            Nclip += self.statparams['Nclip']
            ixs4fit = self.statparams['ix_good']
        
            self.iteration+=1
            if Nclip<1:
                if self.verbose>1: print('CONVERGED!!!!!!')
                self.converged = True
            else:
                if self.verbose>1: print(f'iteraton {self.iteration}: {Nclip} clipped, {len(ixs4fit)} kept')
                
            if self.iteration>20:
                break
            
        self.coeffs.t['Sci2IdlX']=coeff_Sci2IdlX
        self.coeffs.t['Sci2IdlY']=coeff_Sci2IdlY
        self.ixs4fit = ixs4fit
            
        self.ixs_cut_3sigma = AnotB(self.ixs_use,self.ixs4fit)    
        if self.verbose: print(f'3-sigma clip result: {len(self.ixs_cut_3sigma)} ({len(self.ixs_cut_3sigma)/len(self.ixs_use)*100.0:.1f}%) out of {len(self.ixs_use)} clipped')
        self.t['cutflag']=8   ### All 8 should be overwritten by the following commands! If not it's a bug!!
        self.t.loc[self.ixs4fit,'cutflag']=0
        self.t.loc[self.ixs_cut_3sigma,'cutflag']=1
        self.t.loc[self.ixs_excluded,'cutflag']=2
        
        if self.verbose>2:
            self.coeffs.write()
        
        if save_matches:
            outfilename = f'{self.outbasename}.matches.txt'
            print(f'Saving {outfilename}')
            self.write(outfilename)
        
        return(coeff_Sci2IdlX,coeff_Sci2IdlY)
        
    def fit_Idl2Sci(self, fit_coeffs0=False,gridbinsize=16):
        if self.verbose: print('### fitting Idl2Sci...')
        
        # set up the grid of xyprime
        nx, ny = (int(self.siaf_aperture.XSciSize/gridbinsize), int(self.siaf_aperture.XSciSize/gridbinsize))
        x = np.linspace(1, self.siaf_aperture.XSciSize, nx)
        y = np.linspace(1, self.siaf_aperture.YSciSize, ny)
        xgprime, ygprime = np.meshgrid(x-self.siaf_aperture.XSciRef, y-self.siaf_aperture.YSciRef)
        
        # Calculate xy_idl for grid using the fitted polynomials
        xg_idl = pysiaf.utils.polynomial.poly(self.coeffs.t['Sci2IdlX'], xgprime, ygprime, order=self.poly_degree)
        yg_idl = pysiaf.utils.polynomial.poly(self.coeffs.t['Sci2IdlY'], xgprime, ygprime, order=self.poly_degree)
    
        # fit Idl2Sci using the grid!
        coeff_Idl2SciX = polyfit0(xgprime, xg_idl, yg_idl, order=self.poly_degree, fit_coeffs0=fit_coeffs0)
        coeff_Idl2SciY = polyfit0(ygprime, xg_idl, yg_idl, order=self.poly_degree, fit_coeffs0=fit_coeffs0)
    
        # calculate residuals
        xgprime_fit = pysiaf.utils.polynomial.poly(coeff_Idl2SciX, xg_idl, yg_idl, order=self.poly_degree)
        ygprime_fit = pysiaf.utils.polynomial.poly(coeff_Idl2SciY, xg_idl, yg_idl, order=self.poly_degree)
        dxgprime_pix  = (xgprime - xgprime_fit)
        dygprime_pix  = (ygprime - ygprime_fit)

        if self.showplots>1 or self.saveplots:    

            sp = initplot(2,1,xfigsize4subplot=9,yfigsize4subplot=3)
            sp[0].plot(ygprime,dygprime_pix,'bo',alpha=0.2)
            sp[0].set_xlabel('yprime',fontsize=14)
            sp[0].set_ylabel('dyprime_pix',fontsize=14)
            sp[0].set_title(f'{self.apername} {self.filtername} {self.pupilname}: IDL to SCI residuals')
        
            sp[1].plot(xgprime,dxgprime_pix,'bo',alpha=0.2)
            sp[1].set_xlabel('xprime',fontsize=14)
            sp[1].set_ylabel('dxprime_pix',fontsize=14)
            plt.tight_layout()
            if self.saveplots:
                outfilename = f'{self.outbasename}.Idl2Sci.residuals.png'
                print(f'Saving IDL to SCI residual plot to {outfilename}')
                plt.savefig(outfilename)
            if self.showplots:
                plt.show()
            plt.close()
            
        return(coeff_Idl2SciX,coeff_Idl2SciY)

    
if __name__ == '__main__':
    
    distortions = calc_distortions_class()

    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('aperture', type=str, help='aperture name, e.g. nrca1_full')
    parser.add_argument('filter', type=str, help='filter name, e.g. f200w. Can be "None" if the instrument does not have filters like FGS')
    parser.add_argument('pupil', type=str, help='pupil name, e.g. clear. Can be "None" if the instrument does not have pupils like FGS')
    parser.add_argument('input_filepatterns', nargs='+', type=str, help='list of input file(pattern)s. These get added to input_dir if input_dir is not None')
    parser.add_argument('--input_dir', type=str, default=None, help='input_dir is the directory in which the input images are located located (default=%(default)s)')
    parser.add_argument('--outrootdir', default='.', help='output root directory. The output directoy is the output root directory + the outsubdir if not None. (default=%(default)s)')
    parser.add_argument('--outsubdir', default=None, help='outsubdir added to output root directory (default=%(default)s)')
    parser.add_argument('--progIDs', type=int, default=None, nargs="+", help='list of progIDs (default=%(default)s)')

    parser = distortions.define_options(parser=parser)

    args = parser.parse_args()
    distortions.verbose=args.verbose
    distortions.showplots=args.showplots
    distortions.saveplots=args.saveplots
    
    # Prepare the fitting
    distortions.get_inputfiles_imtable(args.input_filepatterns,
                                       directory=args.input_dir)
    distortions.initialize(args.aperture,args.filter,args.pupil,progIDs=args.progIDs)
    distortions.set_outbasename(outrootdir=args.outrootdir,outsubdir=args.outsubdir)
    distortions.load_catalogs()
    
    distortions.imtable.write()
    
    # Calculate xyprime and refcat_xy_idl: this is what is used to fit the distortions!
    distortions.calc_refcat_xy_idl()
    distortions.calc_xyprime()
    
    # Now do the fitting!
    distortions.fit_Sci2Idl()
    distortions.fit_Idl2Sci()
    
    print('Distortions finished!')
    