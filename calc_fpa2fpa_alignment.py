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
import argparse
from astropy.modeling.models import Polynomial2D, Mapping, Shift
from pdastro import pdastroclass


def calc_V3IdlYAngle(v2_1,v3_1,v2_2,v3_2):
    V3IdlYAngle = np.degrees(np.arctan2(v2_2 - v2_1, v3_2 - v3_1))
    if V3IdlYAngle>90.0: V3IdlYAngle-=180.0
    if V3IdlYAngle<-90.0: V3IdlYAngle+=180.0
    return(V3IdlYAngle)
    

class fpa2fpa_alignmentclass(pdastroclass):
    def __init__(self):
        pdastroclass.__init__(self)
        
        self.src_model = None
        self.trg_model = None
        
        #self.src_instrument = None
        #self.src_apername = None
        #self.trg_instrument = None
        #self.trg_apername = None
        
        self.src_siaf = None
        self.trg_siaf = None

        self.src_aperture= None
        self.trg_aperture= None
        
        self.dy = 0.02 # difference in pixel to go into +- y-direction in the center for calculating the angle

        
    def define_options(self,parser=None,usage=None,conflict_handler='resolve'):
        if parser is None:
            parser = argparse.ArgumentParser(usage=usage,conflict_handler=conflict_handler)

        parser.add_argument('--siaf_file', default=None, help='pass the siaf file for the nominal SRC v2/v3ref info. This can be a standard xml file or one of the siaf txt files. If None, then the siaf info is determined using the siaf aperture python module')
        parser.add_argument('--filter', type=str, help='select filter. Only if siaf_file is defined, and the file has a "filter" column')
        parser.add_argument('--pupil', type=str, help='select pupil. Only if siaf_file is defined, and the file has a "pupil" column')
        parser.add_argument('--progID', type=str, help='select progID. Only if siaf_file is defined, and the file has a "progID" column')

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
            if self.verbose>2: print(f'Setting SRC siaf to passed siaf {self.src_siaf.instrument}.')
        else:
            if (self.src_siaf is not None) and (self.src_siaf.instrument.lower() == self.src_model.meta.instrument.name.lower()):
                if self.verbose>2: print(f'Keeping SRC siaf {self.src_siaf.instrument}.')
            else:
                print(f'Setting SRC siaf to {self.src_model.meta.instrument.name}')
                self.src_siaf = pysiaf.Siaf(self.src_model.meta.instrument.name) 
        # make sure the SRC siaf is consistent with the SRC imagemodel!
        if  self.src_siaf.instrument.lower() != self.src_model.meta.instrument.name.lower():
            raise RuntimeError('Inconsistent instruments {self.src_siaf.instrument.lower()}!={self.src_model.meta.instrument.name.lower()}')
            
        # now get the aperture
        # if the correct aperture is already set, just use it!
        if src_aperture is not None:
            self.src_aperture = src_aperture
            if self.verbose>2: print(f'Setting SRC siaf aperture to passed aperture {src_aperture.AperName}.')
        else:
            if (self.src_aperture is not None) and (self.src_aperture.AperName.lower() == self.src_model.meta.aperture.name.lower()):
                if self.verbose>2: print(f'Keeping SRC siaf aperture {self.src_aperture.AperName}.')
            else:
                print(f'Setting SRC siaf aperture to {self.src_model.meta.aperture.name}')
                self.src_aperture = self.src_siaf[self.src_model.meta.aperture.name]
        # make sure the SRC aperture is consistent with the SRC imagemodel!
        if  self.src_aperture.AperName.lower() != self.src_model.meta.aperture.name.lower():
            raise RuntimeError('Inconsistent apertures {self.src_aperture.AperName.lower()}!={self.src_model.meta.aperture.name.lower()}')            
        
            
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
            if self.verbose>2: print(f'Setting TRG siaf to passed siaf {self.trg_siaf.instrument}.')
        else:
            if (self.trg_siaf is not None) and (self.trg_siaf.instrument.lower() == self.trg_model.meta.instrument.name.lower()):
                if self.verbose>2: print(f'Keeping TRG siaf {self.trg_siaf.instrument}.')
            else:
                print(f'Setting TRG siaf to {self.trg_model.meta.instrument.name}')
                self.trg_siaf = pysiaf.Siaf(self.trg_model.meta.instrument.name) 
        # make sure the SRC siaf is consistent with the SRC imagemodel!
        if  self.trg_siaf.instrument.lower() != self.trg_model.meta.instrument.name.lower():
            raise RuntimeError('Inconsistent instruments {self.trg_siaf.instrument.lower()}!={self.trg_model.meta.instrument.name.lower()}')
            
        # now get the aperture
        # if the correct aperture is already set, just use it!
        if trg_aperture is not None:
            self.trg_aperture = trg_aperture
            if self.verbose>2: print(f'Setting TRG siaf aperture to passed aperture {trg_aperture.AperName}.')
        else:
            if (self.trg_aperture is not None) and (self.trg_aperture.AperName.lower() == self.trg_model.meta.aperture.name.lower()):
                if self.verbose>2: print(f'Keeping TRG siaf aperture {self.trg_aperture.AperName}.')
            else:
                print(f'Setting TRG siaf aperture to {self.trg_model.meta.aperture.name}')
                self.trg_aperture = self.trg_siaf[self.trg_model.meta.aperture.name]
        # make sure the TRG aperture is consistent with the TRG imagemodel!
        if  self.trg_aperture.AperName.lower() != self.trg_model.meta.aperture.name.lower():
            raise RuntimeError('Inconsistent apertures {self.trg_aperture.AperName.lower()}!={self.trg_model.meta.aperture.name.lower()}')            
        
        return(0)
    
    def get_nominal_v2v3info(self,siaf_file = None, filtername=None,pupilname=None,progID=None):
        if siaf_file is None:
            if self.verbose>0:
                print(f'Using SRC siaf aperture {self.src_aperture.AperName} for nominal v2/v3ref info')
            self.src_nominal_V2ref = self.src_aperture.V2Ref
            self.src_nominal_V3ref = self.src_aperture.V3Ref
            self.src_nominal_V3IdlYAngle = self.src_aperture.V3IdlYAngle
        else:
            if self.verbose>0:
                print(f'Loading SRC siaf aperture {self.src_aperture.AperName}')
            v2v3ref = v2v3refclass()
            v2v3ref.load_v2v3ref(siaf_file)
            (self.src_nominal_V2ref,self.src_nominal_V3ref,self.src_nominal_V3IdlYAngle,ix)=v2v3ref.get_v2v3info(self.src_aperture.AperName,filtername=filtername,pupilname=pupilname,progID=progID)
        if self.verbose>0:
            print(f'SRC {self.src_aperture.AperName} filter={filtername} progID={pupilname} progID={progID}, nominal values: (V2Ref,V3Ref,V3IdlYAngle)=({self.src_nominal_V2ref},{self.src_nominal_V3ref},{self.src_nominal_V3IdlYAngle})')
            

    def calc_src_v2v3info(self):
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

        if self.verbose>1:
            print(f'Calculated v2/v3 info at center of SRC image')
            print(f'V2ref {src_v2_0} (siaf: {self.src_aperture.V2Ref})')
            print(f'V3ref {src_v3_0} (siaf: {self.src_aperture.V3Ref})')
            print(f'V3YIdlangle {src_V3IdlYAngle} (siaf: {self.src_aperture.V3IdlYAngle})')
        if self.verbose>2:
            self.write()
            
    def calc_trg_v2v3info_in_src_system(self):
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

        if self.verbose>2:
            self.write()
    
    def initialize_src(self,srcfilename, 
                       src_siaf = None, src_aperture = None):
        
        self.src_model = datamodels.ImageModel(srcfilename)
        if self.verbose>2: print(f'SRC {srcfilename} loaded')
        
        self.set_src_siaf(src_siaf = src_siaf, src_aperture = src_aperture)
        
        return(0)

    def initialize_trg(self, trgfilename,
                       trg_siaf = None, trg_aperture = None):
        
        self.trg_model = datamodels.ImageModel(trgfilename)
        if self.verbose>2: print(f'TRG {trgfilename} loaded')

        self.set_trg_siaf(trg_siaf = trg_siaf, trg_aperture = trg_aperture)
        
        return(0)
    
    def calc_trg_v2v3info(self):
        self.calc_src_v2v3info()
        self.calc_trg_v2v3info_in_src_system()


if __name__ == '__main__':

    fpa2fpa = fpa2fpa_alignmentclass()

    parser = argparse.ArgumentParser(conflict_handler="resolve")
    parser.add_argument('srcfilename', help='source filename on which the v2v3 alignment is based on.')
    parser.add_argument('trgfilename', help='target filename for which the v2v3ref info is calculated for.')
    parser = fpa2fpa.define_options(parser)
    args = parser.parse_args()
    
    fpa2fpa.verbose=args.verbose
    
    ######################################################
    ### initialize source and target images and apertures
    ######################################################
    fpa2fpa.initialize_src(args.srcfilename)
    fpa2fpa.get_nominal_v2v3info(siaf_file = args.siaf_file, 
                                 filtername=args.filter,
                                 pupilname=args.pupil,
                                 progID=args.progID)
    
    fpa2fpa.initialize_trg(args.trgfilename)
    
    ######################################################
    ### Now run the main routine which calculates v2v3 info for target image
    ######################################################
    fpa2fpa.calc_trg_v2v3info()