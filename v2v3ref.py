#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 12:41:05 2023

@author: arest
"""

import argparse,glob,re,sys,os
from pdastro import pdastroclass,makepath4file,unique,AnotB,AorB,AandB,rmfile
import pandas as pd

class v2v3refclass(pdastroclass):
    def __init__(self):
        pdastroclass.__init__(self)
        
        #  param2col is a dictionary that contains the mapping from the parameters to the column names
        # This needs to be set depending on what file is loaded
        # the key parameters are 'apername','V2ref','V3ref','V3IdlYAngle'
        # However, other parameters can be added, like 'filter' or 'pupil', if appropriate
        self.param2col = {}
        for param in ['apername','V2ref','V3ref','V3IdlYAngle']:
            self.param2col[param]=None

        
    def init_colnames_siaf(self):
        self.param2col={'apername':'AperName',
                         'V2Ref':'V2Ref',
                         'V3Ref':'V3Ref',
                         'V3IdlYAngle':'V3IdlYAngle'
                         }
        return(0)

    def init_colnames_txt_meanvals(self):
        self.param2col={'apername':'aperture',
                         'V2Ref':'V2ref_mean',
                         'V3Ref':'V3ref_mean',
                         'V3IdlYAngle':'V3IdlYAngle_mean'
                         }
        if 'filter' in self.t.columns:
            self.param2col['filter']='filter'
        if 'pupil' in self.t.columns:
            self.param2col['pupil']='pupil'
        if 'progID' in self.t.columns:
            self.param2col['progID']='progID'


    def init_colnames_txt(self):
        self.param2col={'apername':'aperture',
                         'V2Ref':'V2Ref',
                         'V3Ref':'V3Ref',
                         'V3IdlYAngle':'V3IdlYAngle'
                         }
        if 'filter' in self.t.columns:
            self.param2col['filter']='filter'
        if 'pupil' in self.t.columns:
            self.param2col['pupil']='pupil'
        if 'progID' in self.t.columns:
            self.param2col['progID']='filter'
        return(0)

    
    def load_xml(self,xmlfilename):
        self.t = pd.read_xml(xmlfilename)
        self.init_colnames_siaf()
        self.t[self.param2col['apername']]=self.t[self.param2col['apername']].str.lower()
        
    def load_txt(self,txtfilename):
        self.load(txtfilename)
        if 'V2ref_mean' in self.t.columns:
            self.init_colnames_txt_meanvals()
            if 'filter' in self.param2col:
                self.t[self.param2col['filter']]=self.t[self.param2col['filter']].str.lower()
        else:
            self.init_colnames_txt()
        self.t[self.param2col['apername']]=self.t[self.param2col['apername']].str.lower()
            
    
    def load_v2v3ref(self,filename,**kwargs):
        if re.search('\.xml$',filename) is not None:
            self.load_xml(filename,**kwargs)
        elif re.search('\.txt$',filename) is not None:            
            self.load_txt(filename,**kwargs)
        else:
            raise RuntimeError(f'cannot load file {filename}, only xml and txt files are allowed!')
        return(0)
    
    def define_options(self,parser=None,usage=None,conflict_handler='resolve'):
        if parser is None:
            parser = argparse.ArgumentParser(usage=usage,conflict_handler=conflict_handler)

        parser.add_argument('--filter', type=str, help='select filter')
        parser.add_argument('--pupil', type=str, help='select pupil')
        parser.add_argument('--progID', type=str, help='select progID')

        return(parser)

    
    def get_v2v3info(self,aperture,filtername=None,pupilname=None,progID=None,verbose=0):
        # get all entries for aperture
        ixs = self.ix_equal(self.param2col['apername'],aperture.lower())
        if verbose:print(f'{len(ixs)} with aperture {aperture.lower()}')
        
        # check for filter?
        if filtername is not None:
            if not ('filter' in  self.param2col):
                raise RuntimeError("column name for filter is not defined!")
            ixs = self.ix_equal(self.param2col['filter'],filtername.lower(),indices=ixs)
            if verbose:print(f'{len(ixs)} left with aperture {aperture.lower()}')

        # check for pupil?
        if pupilname is not None:
            if not ('pupil' in  self.param2col):
                raise RuntimeError("column name for pupil is not defined!")
            ixs = self.ix_equal(self.param2col['pupil'],pupilname.lower(),indices=ixs)

        if progID is not None:
            if not ('progID' in  self.param2col):
                raise RuntimeError("column name for progID is not defined!")
            ixs = self.ix_equal(self.param2col['progID'],progID.lower(),indices=ixs)
            
        # error checking: make sure there is exactly one entrie
        if len(ixs)==0:
            raise RuntimeError(f'no v2v3 entries found for aperture {aperture}')
        elif len(ixs)>1:
            print('### SELECTED entries:')
            self.write(columns=self.param2col.values(),indices=ixs)
            raise RuntimeError(f'more than one v2v3 entries found for aperture {aperture}')
        
        #print('vvvvv',self.param2col['V2Ref'])
        
        return(float(self.t.loc[ixs[0],self.param2col['V2Ref']]),
               float(self.t.loc[ixs[0],self.param2col['V3Ref']]),
               float(self.t.loc[ixs[0],self.param2col['V3IdlYAngle']]),
               ixs[0]
            )
                

if __name__ == '__main__':
    v2v3ref = v2v3refclass()

    parser = argparse.ArgumentParser()
    parser.add_argument('aperture', type=str, help='select aperture')
    parser.add_argument('v2v3ref_filename', type=str, help='filename with v2v3ref info. Can be a .xml or .txt file')
    
    parser = v2v3ref.define_options(parser)
    #parser.add_argument('--filter', type=str, help='select filter')
    #parser.add_argument('--progID', type=str, help='select progID')
    args = parser.parse_args()
    
    v2v3ref.load_v2v3ref(args.v2v3ref_filename)
    cols = v2v3ref.param2col.values()
    v2v3ref.write(columns=cols)
    
    (V2Ref,V3Ref,V3IdlYAngle,ix)=v2v3ref.get_v2v3info(args.aperture,filtername=args.filter,pupilname=args.pupil,progID=args.progID)
    print(f'{args.aperture} filter={args.filter} pupil={args.pupil} progID={args.progID}: (V2Ref,V3Ref,V3IdlYAngle)=({V2Ref},{V3Ref},{V3IdlYAngle})')
