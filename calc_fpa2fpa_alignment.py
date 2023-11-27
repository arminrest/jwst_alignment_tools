#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 22:01:53 2023

@author: arest
"""

from jwst import datamodels
import pysiaf
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
import sys,os,re
import numpy as np
import pandas as pd
import argparse
from astropy.modeling.models import Polynomial2D, Mapping, Shift
from astropy.modeling.rotations import Rotation2D

from pdastro import pdastroclass
from v2v3ref import v2v3refclass

def calc_V3IdlYAngle(v2_1,v3_1,v2_2,v3_2):
    V3IdlYAngle = np.degrees(np.arctan2(v2_2 - v2_1, v3_2 - v3_1))
    if V3IdlYAngle>90.0: V3IdlYAngle-=180.0
    if V3IdlYAngle<-90.0: V3IdlYAngle+=180.0
    return(V3IdlYAngle)
    

class fpa2fpa_alignmentclass(pdastroclass):
    def __init__(self):
        pdastroclass.__init__(self)
        
        # the source and target siaf aperture.
        # We don't pu this into clear() since we often can re-use them
        self.src_siaf = None
        self.trg_siaf = None

        self.src_aperture= None
        self.trg_aperture= None
        
        self.dy = 0.02 # difference in pixel to go into +- y-direction in the center for calculating the angle

        self.clear()
        
    def clear(self):
        """
        Clear all parameters related to the source and target image, and the associated v2/v3 table.
        After clear, a new set of source and target image can be analyzed.
        Clear does ***not*** clear self.src_siaf, self.trg_siaf, 
        self.src_aperture, self.trg_aperture, which means that they do not
        get re-initialized if they are still consistent with the new source and
        target image.

        Returns
        -------
        None.

        """
        self.src_model = None
        self.trg_model = None
        
        self.src_apername = None
        self.src_filter = None
        self.src_pupil = None
        
        self.trg_apername = None
        self.trg_filter = None
        self.trg_pupil = None
        
        self.src_nominal_V2ref = None
        self.src_nominal_V3ref = None
        self.src_nominal_V3IdlYAngle = None

        self.new_trg_V2ref = None
        self.new_trg_V3ref = None
        self.new_trg_V3IdlYAngle = None
        
        self.src_ixs = None
        self.trg_ixs = None

        self.t = pd.DataFrame(columns=self.t.columns)
        
    def define_options(self,parser=None,usage=None,conflict_handler='resolve'):
        if parser is None:
            parser = argparse.ArgumentParser(usage=usage,conflict_handler=conflict_handler)

        parser.add_argument('--siaf_file', default=None, help='pass the siaf file for the nominal SRC V2/V3ref info for the source image. This can be a standard xml file or one of the siaf txt files. If None, then the V2/V3ref info is determined using the siaf aperture python module')
        parser.add_argument('--v2v3refvalues', type=float, nargs=3, default=None, help='pass the desired, nominal V2ref, V3ref, and V3IdlYAngle for the source image. This supercedes --siaf_file')

        parser.add_argument('-v','--verbose', default=0, action='count')

        return(parser)

    def set_src_siaf(self, src_siaf = None, src_aperture = None):
        """
        
        This sets the siaf and siaf aperture of the source detector, based on what is loaded into self.src_model
        you can pass the source siaf and aperture, which means they won't need to be initalized again
        if self.src_siaf and self.src_aperture are already set, and IF they are consistent with the src_model, then they are not changed
        

        Parameters
        ----------
        src_siaf : TYPE, optional
            siaf of source instrument. The default is None.
        src_aperture : TYPE, optional
            siaf of source aperture. The default is None.

        Raises
        ------
        RuntimeError
            DESCRIPTION.

        Returns
        -------
        None.

        """

        # first get the SRC siaf for the instrument
        # If the correct siaf is already set, just use it
        if src_siaf is not None:
            self.src_siaf = src_siaf
            if self.verbose: print(f'Setting SRC siaf to passed siaf {self.src_siaf.instrument}.')
        else:
            if (self.src_siaf is not None) and (self.src_siaf.instrument.lower() == self.src_model.meta.instrument.name.lower()):
                if self.verbose: print(f'Keeping SRC siaf {self.src_siaf.instrument}.')
            else:
                if self.verbose: print(f'Setting SRC siaf to {self.src_model.meta.instrument.name}')
                self.src_siaf = pysiaf.Siaf(self.src_model.meta.instrument.name) 
        # make sure the SRC siaf is consistent with the SRC imagemodel!
        if  self.src_siaf.instrument.lower() != self.src_model.meta.instrument.name.lower():
            raise RuntimeError('Inconsistent instruments {self.src_siaf.instrument.lower()}!={self.src_model.meta.instrument.name.lower()}')
            
        # now get the aperture
        # if the correct aperture is already set, just use it!
        if src_aperture is not None:
            self.src_aperture = src_aperture
            if self.verbose: print(f'Setting SRC siaf aperture to passed aperture {src_aperture.AperName}.')
        else:
            if (self.src_aperture is not None) and (self.src_aperture.AperName.lower() == self.src_model.meta.aperture.name.lower()):
                if self.verbose: print(f'Keeping SRC siaf aperture {self.src_aperture.AperName}.')
            else:
                if self.verbose: print(f'Setting SRC siaf aperture to {self.src_model.meta.aperture.name}')
                self.src_aperture = self.src_siaf[self.src_model.meta.aperture.name]
        # make sure the SRC aperture is consistent with the SRC imagemodel!
        if  self.src_aperture.AperName.lower() != self.src_model.meta.aperture.name.lower():
            raise RuntimeError('Inconsistent apertures {self.src_aperture.AperName.lower()}!={self.src_model.meta.aperture.name.lower()}')            
        
        self.src_apername = self.src_aperture.AperName.lower()
            
        return(0)
    
    def set_trg_siaf(self, trg_siaf = None, trg_aperture = None):
        """
        
        This sets the siaf and siaf aperture of the target detector, based on what is loaded into self.trg_model
        you can pass the target siaf and aperture, which means they won't need to be initalized again
        if self.trg_siaf and self.trg_aperture are already set, and IF they are consistent with the trg_model, then they are not changed
        

        Parameters
        ----------
        trg_siaf : TYPE, optional
            siaf of target instrument. The default is None.
        trg_aperture : TYPE, optional
            siaf of target aperture. The default is None.

        Raises
        ------
        RuntimeError
            DESCRIPTION.

        Returns
        -------
        None.

        """

        # first get the SRC siaf for the instrument
        # If the correct siaf is already set, just use it
        if trg_siaf is not None:
            self.trg_siaf = trg_siaf
            if self.verbose: print(f'Setting TRG siaf to passed siaf {self.trg_siaf.instrument}.')
        else:
            if (self.trg_siaf is not None) and (self.trg_siaf.instrument.lower() == self.trg_model.meta.instrument.name.lower()):
                if self.verbose: print(f'Keeping TRG siaf {self.trg_siaf.instrument}.')
            else:
                if self.verbose: print(f'Setting TRG siaf to {self.trg_model.meta.instrument.name}')
                self.trg_siaf = pysiaf.Siaf(self.trg_model.meta.instrument.name) 
        # make sure the SRC siaf is consistent with the SRC imagemodel!
        if  self.trg_siaf.instrument.lower() != self.trg_model.meta.instrument.name.lower():
            raise RuntimeError('Inconsistent instruments {self.trg_siaf.instrument.lower()}!={self.trg_model.meta.instrument.name.lower()}')
            
        # now get the aperture
        # if the correct aperture is already set, just use it!
        if trg_aperture is not None:
            self.trg_aperture = trg_aperture
            if self.verbose: print(f'Setting TRG siaf aperture to passed aperture {trg_aperture.AperName}.')
        else:
            if (self.trg_aperture is not None) and (self.trg_aperture.AperName.lower() == self.trg_model.meta.aperture.name.lower()):
                if self.verbose: print(f'Keeping TRG siaf aperture {self.trg_aperture.AperName}.')
            else:
                if self.verbose: print(f'Setting TRG siaf aperture to {self.trg_model.meta.aperture.name}')
                self.trg_aperture = self.trg_siaf[self.trg_model.meta.aperture.name]
        # make sure the TRG aperture is consistent with the TRG imagemodel!
        if  self.trg_aperture.AperName.lower() != self.trg_model.meta.aperture.name.lower():
            raise RuntimeError('Inconsistent apertures {self.trg_aperture.AperName.lower()}!={self.trg_model.meta.aperture.name.lower()}')            
        
        self.trg_apername = self.trg_aperture.AperName.lower()

        return(0)
    
    def get_nominal_v2v3info(self,siaf_file = None, v2v3refvalues=None):
        """
        Get the desired, nominal V2ref,V3ref,V3IdlYAngle for the source image.
        These values can either be taken from siaf, or alternatively from a siaf file.
        The nominal values are saved in the following variables:

        self.src_nominal_V2ref
        self.src_nominal_V3ref
        self.src_nominal_V3IdlYAngle


        Parameters
        ----------
        siaf_file : string, optional, the default is None.
            if no siaf_file is passed, then the V2V3info is taken from the siaf aperture.
            siaf_file can be a siaf xml file of one of the siaf text files in the format from this package
            note: if the v2v3info is read in from a siaf text file, then it will get it for the source filter and pupil!!!

        Returns
        -------
        None.

        """
        if v2v3refvalues is not None:
            (self.src_nominal_V2ref,self.src_nominal_V3ref,self.src_nominal_V3IdlYAngle)=v2v3refvalues
        elif siaf_file is not None:
            if self.verbose>0:
                print(f'Loading SRC siaf file {siaf_file}')
            v2v3ref = v2v3refclass()
            v2v3ref.load_v2v3ref(siaf_file)
            (self.src_nominal_V2ref,self.src_nominal_V3ref,self.src_nominal_V3IdlYAngle,ix)=v2v3ref.get_v2v3info(self.src_aperture.AperName,filtername=self.src_filter,pupilname=self.src_pupil)
        else:
            if self.verbose>0:
                print(f'Using SRC siaf aperture {self.src_aperture.AperName} for nominal v2/v3ref info')
            self.src_nominal_V2ref = self.src_aperture.V2Ref
            self.src_nominal_V3ref = self.src_aperture.V3Ref
            self.src_nominal_V3IdlYAngle = self.src_aperture.V3IdlYAngle

        if self.verbose:
            print(f'NOMINAL v2v3ref values for source {self.src_aperture.AperName} filter={self.src_filter} pupil={self.src_pupil}')
            print(f'(V2Ref,V3Ref,V3IdlYAngle)=({self.src_nominal_V2ref},{self.src_nominal_V3ref},{self.src_nominal_V3IdlYAngle})')
            
    def initialize_src(self,srcfilename, 
                       src_siaf = None, src_aperture = None):
        """
        Load the source image into the image model self.src_model, 
        save the filter and pupil into self.src_filter and self.src_pupil,
        and then initial a siaf instance of for the instrument and aperture
        you can pass the source siaf and aperture, which means they won't need 
        to be initalized again if self.src_siaf and self.src_aperture are already 
        set, and IF they are consistent with the src_model, then they are not changed

        parameter set:
            self.src_model
            self.src_filter
            self.src_pupil
            self.src_siaf
            self.src_aperture
            self.src_apername
           
        Parameters
        ----------
        srcfilename : string
            source image filename.
        src_siaf : siaf instance of the instrument, optional
            src_siaf is used If not None and consistent with self.src_model. The default is None.
        src_aperture : TYPE, optional
            src_aperture is used If not None and consistent with self.src_model. The default is None.

        Returns
        -------
        None.

        """
        
        if self.verbose:
            print(f'\n### Initializing source image {srcfilename}')
        
        self.src_model = datamodels.ImageModel(srcfilename)
        self.src_filter = self.src_model.meta.instrument.filter
        self.src_pupil = self.src_model.meta.instrument.pupil
        if self.verbose: print(f'SRC {srcfilename} loaded (FILTER={self.src_filter}, PUPIL={self.src_pupil})')
        
        self.set_src_siaf(src_siaf = src_siaf, src_aperture = src_aperture)
        
        return(0)

    def initialize_trg(self, trgfilename,
                       trg_siaf = None, trg_aperture = None):
        """
        Load the target image into the image model self.trg_model, 
        save the filter and pupil into self.trg_filter and self.trg_pupil,
        and then initial a siaf instance of for the instrument and aperture
        you can pass the target siaf and aperture, which means they won't need 
        to be initalized again if self.trg_siaf and self.trg_aperture are already 
        set, and IF they are consistent with the trg_model, then they are not changed

        parameter set:
            self.trg_model
            self.trg_filter
            self.trg_pupil
            self.trg_siaf
            self.trg_aperture
            self.trg_apername
            
        Parameters
        ----------
        srcfilename : string
            target image filename.
        trg_siaf : siaf instance of the instrument, optional
            trg_siaf is used If not None and consistent with self.trg_model. The default is None.
        trg_aperture : TYPE, optional
            trg_aperture is used If not None and consistent with self.trg_model. The default is None.

        Returns
        -------
        None.

        """
       
        if self.verbose:
            print(f'\n### Initializing target image {trgfilename}')

        self.trg_model = datamodels.ImageModel(trgfilename)
        self.trg_filter = self.trg_model.meta.instrument.filter
        self.trg_pupil = self.trg_model.meta.instrument.pupil
        if self.verbose: print(f'TRG {trgfilename} loaded (FILTER={self.trg_filter}, PUPIL={self.trg_pupil})')

        self.set_trg_siaf(trg_siaf = trg_siaf, trg_aperture = trg_aperture)
        
        return(0)
    
    def calc_src_v2v3info(self):
        """
        calculate the 3  v2/v3 positions of the source.
        These are used to calculate the actual V2ref, V3ref, V3YIdlangle of the
        source image WCS in the asdf extension, which can be compared to the siaf one.
            

        Returns
        -------
        array with 3 elements: indices to position 0, 1 and 2 of the source image in the self.t table.

        """
        if self.verbose:
            print('\n### Calculating v2/v3 positions in the source image, and determine V2ref, V3ref, V3YIdlangle of the source image')


        src_detector_to_v2v3=self.src_model.meta.wcs.get_transform('detector', 'v2v3') 

        # get the x/y coordinates of the center
        src_x0=self.src_aperture.XSciRef-1.0
        src_y0=self.src_aperture.YSciRef-1.0

        src_v2_0,src_v3_0 = src_detector_to_v2v3(src_x0,src_y0)
        src_v2_1,src_v3_1 = src_detector_to_v2v3(src_x0,src_y0+self.dy)
        src_v2_2,src_v3_2 = src_detector_to_v2v3(src_x0,src_y0-self.dy)

        src_ix_0 = self.newrow({'name':'src_v2v3_0','v2':src_v2_0,'v3':src_v3_0})
        src_ix_1 = self.newrow({'name':'src_v2v3_1','v2':src_v2_1,'v3':src_v3_1})
        src_ix_2 = self.newrow({'name':'src_v2v3_2','v2':src_v2_2,'v3':src_v3_2})

        src_V3IdlYAngle = calc_V3IdlYAngle(self.t.loc[src_ix_1,'v2'],self.t.loc[src_ix_1,'v3'],
                                           self.t.loc[src_ix_2,'v2'],self.t.loc[src_ix_2,'v3'])

        if self.verbose:
            print('Calculated v2/v3 info at center of SRC image')
            print(f'V2ref {src_v2_0:10.4f} (siaf: {self.src_aperture.V2Ref:10.4f}, diff={src_v2_0-self.src_aperture.V2Ref:7.4f})')
            print(f'V3ref {src_v3_0:10.4f} (siaf: {self.src_aperture.V3Ref:10.4f}, diff={src_v3_0-self.src_aperture.V3Ref:7.4f})')
            print(f'V3YIdlangle {src_V3IdlYAngle:.8f} (siaf: {self.src_aperture.V3IdlYAngle:.8f}, diff={src_V3IdlYAngle-self.src_aperture.V3IdlYAngle:.8f})')
        if self.verbose>=2:
            self.write()
            
        return(src_ix_0,src_ix_1,src_ix_2)
            
    def calc_trg_v2v3info_in_src_system(self):
        """
        calculate the 3  v2/v3 positions of the target in the source v2/v3 system!
        These will be ultimately used to calculate the actual V2ref, V3ref, V3YIdlangle of the
        target image
            

        Returns
        -------
        array with 3 elements: indices to position 0, 1 and 2 of the target image in the self.t table.

        """

        if self.verbose:
            print('\n### Calculating v2/v3 positions in the target image in the source image v2/v3 system')

        src_world_to_v2v3=self.src_model.meta.wcs.get_transform('world','v2v3') 
        trg_detector_to_world=self.trg_model.meta.wcs.get_transform('detector', 'world') 

        # get the x/y coordinates of the center
        trg_x0=self.trg_aperture.XSciRef-1.0
        trg_y0=self.trg_aperture.YSciRef-1.0

        trg_ra_0,trg_dec_0 = trg_detector_to_world(trg_x0,trg_y0)
        trg_ra_1,trg_dec_1 = trg_detector_to_world(trg_x0,trg_y0+self.dy)
        trg_ra_2,trg_dec_2 = trg_detector_to_world(trg_x0,trg_y0-self.dy)
        trg_v2_0,trg_v3_0 = src_world_to_v2v3(trg_ra_0,trg_dec_0)
        trg_v2_1,trg_v3_1 = src_world_to_v2v3(trg_ra_1,trg_dec_1)
        trg_v2_2,trg_v3_2 = src_world_to_v2v3(trg_ra_2,trg_dec_2)
        trg_ix_0 = self.newrow({'name':'trg_v2v3_0','v2':trg_v2_0,'v3':trg_v3_0})
        trg_ix_1 = self.newrow({'name':'trg_v2v3_1','v2':trg_v2_1,'v3':trg_v3_1})
        trg_ix_2 = self.newrow({'name':'trg_v2v3_2','v2':trg_v2_2,'v3':trg_v3_2})

        if self.verbose>1:
            self.write()
            
        return(trg_ix_0,trg_ix_1,trg_ix_2)
    
    def rotate_v2v3(self,src_ixs):
        """
        rotate and shift the values in the v2 and v3 columns so that they agree with the nominal V2ref, V3ref, and V3IdlYAngle of the source
        This corrects for any differences between the actual v2/v3ref and angle values in the source image WCS, and teh desired, nomminal 
        v2/v3ref and angle values. These differences can happen if the siaf values are different to the ones used in the distortion
        asdf files, or if there are new v2/v3ref values that have not been put into siaf yet.

        Parameters
        ----------
        src_ixs : array with 3 values.
            The 3 indices to the 3 v2/v3 positions of the source.
    
        Returns
        -------
        None.

        """
        if self.verbose:
            print('\n### rotate and shift the values in the v2 and v3 columns so that they agree with the nominal V2ref, V3ref, and V3IdlYAngle of the source')


        src_V3IdlYAngle = calc_V3IdlYAngle(self.t.loc[src_ixs[1],'v2'],self.t.loc[src_ixs[1],'v3'],
                                           self.t.loc[src_ixs[2],'v2'],self.t.loc[src_ixs[2],'v3'])

        R = Rotation2D()
        # calculate the differences in v2/v3 with respect to the center pixel v2/v3
        v2diff=np.array(self.t['v2']-self.t.loc[src_ixs[0],'v2'])
        v3diff=np.array(self.t['v3']-self.t.loc[src_ixs[0],'v3'])
        # calculate the correction angle between the true src_V3IdlYAngle and the nominal one
        angle = src_V3IdlYAngle-self.src_nominal_V3IdlYAngle
        # Rotate the v2/v3 values, and add the ***nominal*** v2/v3 values
        (v2diffrot,v3diffrot) = R.evaluate(v2diff,v3diff,angle*u.deg)
        self.t['v2rot']=v2diffrot+self.src_nominal_V2ref
        self.t['v3rot']=v3diffrot+self.src_nominal_V3ref
        if self.verbose:
            self.write()
        
        src_V3IdlYAngle_rot = calc_V3IdlYAngle(self.t.loc[src_ixs[1],'v2rot'],self.t.loc[src_ixs[1],'v3rot'],
                                               self.t.loc[src_ixs[2],'v2rot'],self.t.loc[src_ixs[2],'v3rot'])
#            self.t.loc[src_ixs[2],'v2rot']-self.t.loc[src_ixs[1],'v2rot'], 
#                                               self.t.loc[src_ixs[2],'v3rot']-self.t.loc[src_ixs[1],'v3rot'])
        if self.verbose: print(f'SRC rotated V3YIdlangle {src_V3IdlYAngle_rot:.8f} from {src_V3IdlYAngle:.8f} (nominal: {self.src_nominal_V3IdlYAngle:.8f}, siaf: {self.src_aperture.V3IdlYAngle:.8f})')

        # error checking: did the rotation and offset work, i.e., are the 3 v2v3 positions now consistent with the desired nominal v2ref, v3ref and V3IdlYAngle?
        tolerance = 0.0000001
        if self.verbose:
            print(f'Checking if rotated source v2ref, v3ref, and V3IdlYAngle is consistent with nominal, desired v2ref, v3ref, and V3IdlYAngle values, tolerance={tolerance}')
        if self.verbose>1:
            print(f'Source image rotated v2ref, v3ref, and V3IdlYAngle differences to desired, nominal values: {self.t.loc[src_ixs[0],"v2rot"]-self.src_nominal_V2ref} {self.t.loc[src_ixs[0],"v3rot"]-self.src_nominal_V3ref} {src_V3IdlYAngle_rot-self.src_nominal_V3IdlYAngle} respectively')
        if np.fabs(src_V3IdlYAngle_rot-self.src_nominal_V3IdlYAngle)>tolerance:
            raise RuntimeError(f'rotated V3IdlYAngle {src_V3IdlYAngle_rot:.8f} is different from nominal {self.src_nominal_V3IdlYAngle:.8f}, diff={src_V3IdlYAngle_rot-self.src_nominal_V3IdlYAngle}')
        # another check, make sure FGS V2/V3ref are equal to nominal V2/V3ref after rotation!
        if np.fabs(self.t.loc[src_ixs[0],'v2rot']-self.src_nominal_V2ref)>tolerance:
            raise RuntimeError(f'rotated v2 {self.t.loc[src_ixs[0],"v2rot"]:.7f} is different from nominal {self.src_nominal_V2ref:.7f}, diff={self.t.loc[src_ixs[0],"v2rot"]-self.src_nominal_V2ref}')
        if np.fabs(self.t.loc[src_ixs[0],'v3rot']-self.src_nominal_V3ref)>tolerance:
            raise RuntimeError(f'rotated v2 {self.t.loc[src_ixs[0],"v3rot"]:.7f} is different from nominal {self.src_nominal_V3ref:.7f}, diff={self.t.loc[src_ixs[0],"v3rot"]-self.src_nominal_V3ref}')
        if self.verbose:
            print(f'All differences within tolerance of {tolerance}!!!')
        return(0)        

    def calc_new_v2v3info(self,src_ixs,trg_ixs):

        self.new_trg_V3IdlYAngle = calc_V3IdlYAngle(self.t.loc[trg_ixs[1],'v2rot'],self.t.loc[trg_ixs[1],'v3rot'],
                                                                      self.t.loc[trg_ixs[2],'v2rot'],self.t.loc[trg_ixs[2],'v3rot'])
        self.new_trg_V2ref = self.t.loc[trg_ixs[0],"v2rot"]
        self.new_trg_V3ref = self.t.loc[trg_ixs[0],"v3rot"]
        
        if self.verbose:
            print('\n######\n###### Results:\n######')
            print(f'{self.trg_aperture.AperName} {self.trg_filter} {self.trg_pupil} V2Ref: {self.new_trg_V2ref:10.4f} (siaf: {self.trg_aperture.V2Ref:10.4f}, difference {self.new_trg_V2ref-self.trg_aperture.V2Ref:7.4f})')
            print(f'{self.trg_aperture.AperName} {self.trg_filter} {self.trg_pupil} V3Ref: {self.new_trg_V3ref:10.4f} (siaf: {self.trg_aperture.V3Ref:10.4f}, difference {self.new_trg_V3ref-self.trg_aperture.V3Ref:7.4f})')
            #self.t.loc[trg_ixs[2],'v2rot']-self.t.loc[trg_ixs[1],'v2rot'], 
            #                                   self.t.loc[trg_ixs[2],'v3rot']-self.t.loc[trg_ixs[1],'v3rot'])
            print(f'{self.trg_aperture.AperName} {self.trg_filter} {self.trg_pupil} V3YIdlangle {self.new_trg_V3IdlYAngle:.8f} (siaf: {self.trg_aperture.V3IdlYAngle:.8f}, difference {self.new_trg_V3IdlYAngle-self.trg_aperture.V3IdlYAngle:.8f})')

        return(self.new_trg_V2ref,self.new_trg_V3ref,self.new_trg_V3IdlYAngle)
    
    def calc_trg_v2v3info(self,srcfilename,trgfilename,siaf_file=None,v2v3refvalues=None):
        
        ###
        ### initialize source and target images and apertures
        ###
        
        # first initialise the source
        self.initialize_src(srcfilename)
        # set the source V2/V3ref and V3IdlYAngle
        # if no siaf_file is passed, then the V2V3info is taken from the siaf aperture.
        # siaf_file can be a siaf xml file of one of the siaf text files in the format from this package
        # note: if the v2v3info is read in from a siaf text file, then it will get it for the source filter and pupil!!!
        self.get_nominal_v2v3info(siaf_file = siaf_file, v2v3refvalues=v2v3refvalues)
        # initialize the target
        self.initialize_trg(trgfilename)
        
        
        
        ###
        ### Calculate the v2/v3 positions, and derive v2v3info for target
        ###
        
        # There are 3 v2/v3 positions for each image:
        # 0: v2/v3 for the central pixel
        # 1: v2/v3 for the central pixel with +dy added to y, where dy is a snall fraction of a pixel
        # 2: v2/v3 for the central pixel with -dy added to y, where dy is a snall fraction of a pixel
        # position 1+2 are used to calculate V3IdlYAngle
        # These source positions will be in the pandas table self.t
        
        
        # first calculate the 3 v2/v3 positions for the source
        self.src_ixs = self.calc_src_v2v3info()
        
        # now calculate the 3 v2/v3 positions for the target *** in the source v2/v3 system ***
        self.trg_ixs = self.calc_trg_v2v3info_in_src_system()
        
        # Now rotate the v2/v3 position so that the source image v2/v3 system 
        # agrees with the desired nominal v2/v3ref and V3IdlYAngle specified in 
        # self.src_nominal_V2ref
        # self.src_nominal_V3ref
        # self.src_nominal_V3IdlYAngle
        self.rotate_v2v3(self.src_ixs)

        # get the new v2v3 info for the target image
        # These are also saved in 
        # self.new_trg_V2ref
        # self.new_trg_V3ref
        # self.new_trg_V3IdlYAngle
        (new_trg_V2ref,new_trg_V3ref,new_trg_V3IdlYAngle) = self.calc_new_v2v3info(self.src_ixs,self.trg_ixs)
        
        return(new_trg_V2ref,new_trg_V3ref,new_trg_V3IdlYAngle)


if __name__ == '__main__':

    fpa2fpa = fpa2fpa_alignmentclass()

    parser = argparse.ArgumentParser(conflict_handler="resolve")
    parser.add_argument('srcfilename', help='source filename on which the v2v3 alignment is based on.')
    parser.add_argument('trgfilename', help='target filename for which the v2v3ref info is calculated for.')
    parser = fpa2fpa.define_options(parser)
    args = parser.parse_args()
    
    fpa2fpa.verbose=args.verbose

    fpa2fpa.calc_trg_v2v3info(args.srcfilename,
                              args.trgfilename,
                              args.siaf_file,
                              args.v2v3refvalues)
