#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 16:28:28 2024

@author: arest
"""

import argparse,re,sys,os,random
from calc_distortions import calc_distortions_class
from pdastro import makepath,rmfile,pdastroclass,AnotB,AorB,unique,makepath4file

class calc_distortions_coron_class(calc_distortions_class):
    def __init__(self):
        calc_distortions_class.__init__(self)
        self.coron_info = pdastroclass()
        self.ix_coron_info = None
    
    def define_optional_arguments(self,parser=None,usage=None,conflict_handler='resolve'):
        calc_distortions_class.define_optional_arguments(self,parser=parser,usage=usage,conflict_handler=conflict_handler)
        parser.add_argument('--coron_info_filename', type=str, default='./CoronInfo.txt', help='')
        return(parser)
    
    def load_coron_info(self,coron_info_filename):
        # Load the coronograph info
        if not os.path.isfile(coron_info_filename): raise RuntimeError(f'coron info file {coron_info_filename} does not exist!')
        self.coron_info.load(coron_info_filename,verbose=1)
    
    #def initialize(self,coron_info_filename,*args,**kwargs):
    def initialize(self, *args,
                   #coron_info_filename,
                   #apername, filtername, pupilname,
                   **kwargs):
        # standard initialization
        #calc_distortions_class.initialize(self, apername, filtername, pupilname,**kwargs)
        calc_distortions_class.initialize(self, *args,**kwargs)
        
        # make sure this is is coron mask!
        if (re.search('^mask',self.pupilname) is None):
            raise RuntimeError(f'Looks like pupil={pupilname} is not a mask!')
            
        # Load the coronograph info
        #if not os.path.isfile(coron_info_filename): raise RuntimeError(f'coron info file {coron_info_filename} does not exist!')
        #self.coron_info.load(coron_info_filename,verbose=1)
        
        
        # get the index of the detector/filter/pupil if it exists in the coron info file
        self.ix_coron_info = self.coron_info.getindices()
        for (colname,value) in zip(("apername","filter","pupil"),(self.apername, self.filtername, self.pupilname)):
            #print(value,colname,len(self.ix_coron_info))
            self.ix_coron_info = self.coron_info.ix_equal(colname,value,indices=self.ix_coron_info)
        
        # Make sure there is not more than one entry for the given apername/filter/pupil in Coron info file
        if len(self.ix_coron_info)<1:
            self.ix_coron_info = -1
            print(f'WARNING: Could not find entry for {(apername, filtername, pupilname)}. This means that this will be treated like any normal imaging!')
        elif len(self.ix_coron_info)>1:
            self.coron_info.write(indices=self.ix_coron_info)
            self.ix_coron_info = None
            raise RuntimeError(f'More than one entry for {(apername, filtername, pupilname)}')
        else:
            self.ix_coron_info = self.ix_coron_info[0]
            print(f'Entry for {(self.apername, self.filtername, self.pupilname)} found:')
            self.coron_info.write(indices=[self.ix_coron_info])
            
        # Copy the coron info into the fit summary table
        if self.ix_fitsum is None: raise RuntimeError("BUG! the fit summary table should have been already initialized in the initialize function!")
        self.fitsummary.t.loc[self.ix_fitsum,self.coron_info.t.columns]=self.coron_info.t.loc[self.ix_coron_info,self.coron_info.t.columns]

        return(0)
    
    def calc_xyprime(self):
        
        # standard calculation of x/yprime
        calc_distortions_class.calc_xyprime(self)

        # Is there coron info for this apername/filter/mask?
        if self.ix_coron_info is None: raise RuntimeError('BUG! The index ix_coron_info should be set! Is the Coron Info file loaded?')
        if self.ix_coron_info==-1:
            print('WARNING: No adding of a step, even though this is a mask image!')
            return(0)
        
        # add step to yprime
        y_transition = self.coron_info.t.loc[self.ix_coron_info,'y_transition']
        y_step_pixels = self.coron_info.t.loc[self.ix_coron_info,'y_step_pixels']
        print(f'### Adding {y_step_pixels} for all pixels with y>{y_transition}!')
        ixs_top = self.ix_inrange('y',y_transition,None)
        self.t.loc[ixs_top,'yprime'] +=  y_step_pixels

        # populate the fit summary table the with values used for y_transition and y_step_pixels
        # this already has been done in the initialize function, but just do it here again
        # to make 100% sure the correct values are in the fit summary table
        if self.ix_fitsum is None: raise RuntimeError("BUG! the fit summary table should have been already initialized in the initialize function!")
        self.fitsummary.t.loc[self.ix_fitsum,['y_transition','y_step_pixels']]=[y_transition,y_step_pixels]

    def get_ixs_use(self, ixs_use = None, downsample = None):
        if ixs_use is None:
            ixs_use = self.getindices()
        
        if self.ix_coron_info is None: raise RuntimeError('BUG! The index ix_coron_info should be set! Is the Coron Info file loaded?')

        if self.ix_coron_info==-1:
            print('WARNING: No additional cutting in x and y, even though this is a mask file!')
        else:
            # get the indices for the top (mask) part 
            ixs_top = self.ix_inrange(self.colnames['y'],self.coron_info.t.loc[self.ix_coron_info,'ymin1'],self.coron_info.t.loc[self.ix_coron_info,'ymax1'],indices = ixs_use)
            ixs_top = self.ix_inrange(self.colnames['x'],self.coron_info.t.loc[self.ix_coron_info,'xmin1'],self.coron_info.t.loc[self.ix_coron_info,'xmax1'],indices = ixs_top)
    
            # get the indices for the bottom (non-mask) part 
            ixs_bottom = self.ix_inrange(self.colnames['y'],self.coron_info.t.loc[self.ix_coron_info,'ymin2'],self.coron_info.t.loc[self.ix_coron_info,'ymax2'],indices = ixs_use)
            ixs_bottom = self.ix_inrange(self.colnames['x'],self.coron_info.t.loc[self.ix_coron_info,'xmin2'],self.coron_info.t.loc[self.ix_coron_info,'xmax2'],indices = ixs_bottom)

            # Copy the coron info into the fit summary table
            # this already has been done in the initialize function, but just do it here again
            # to make 100% sure the correct values are in the fit summary table
            if self.ix_fitsum is None: raise RuntimeError("BUG! the fit summary table should have been already initialized in the initialize function!")
            cols2copy = ['xmin1','xmax1','ymin1','ymax1','xmin2','xmax2','ymin2','ymax2']
            self.fitsummary.t.loc[self.ix_fitsum,cols2copy]=self.coron_info.t.loc[self.ix_coron_info,cols2copy]
            
            if downsample is None:
                downsample = self.coron_info.t.loc[self.ix_coron_info,'downsample']
            # Downsample if downsample!=0.0
            if isinstance(downsample,float) and downsample!=0.0:
                Nkeep = int(len(ixs_top)*downsample)
                print(f'Downsampling bottom part ({len(ixs_bottom)}) to {downsample} of top part ({len(ixs_top)}): keeping {Nkeep}')
                ixs_use = AorB(ixs_top,random.choices(ixs_bottom,k=Nkeep))
                # copy downsample value into fit summary table
                self.fitsummary.t.loc[self.ix_fitsum,'downsample']=downsample
            else:
                ixs_use = AorB(ixs_top,ixs_bottom)
    
        self.ixs_use = ixs_use
        self.ixs_excluded =  AnotB(self.getindices(),ixs_use)
        if self.verbose: print(f'{len(ixs_use)} entries used for fit, {len(self.ixs_excluded)} excluded')  
         

if __name__ == '__main__':
    
    distortions = calc_distortions_coron_class()

    # get the arguments
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser = distortions.define_arguments(parser=parser)
    parser = distortions.define_optional_arguments(parser=parser)
    args = parser.parse_args()
    
    # set some verbose, plot, and save flags
    distortions.verbose=args.verbose
    distortions.savecoeff=not (args.skip_savecoeff)
    distortions.showplots=args.showplots
    distortions.saveplots=args.saveplots
    
    # define the x/y columns and phot cat suffix based on what photometry is used
    distortions.define_xycols(xypsf=args.xypsf,xy1pass=args.xy1pass,date4suffix=args.date4suffix)
        
    # get the coron info table!
    distortions.load_coron_info(args.coron_info_filename)
    
    # get the input file list
    distortions.get_inputfiles_imtable(args.input_filepatterns,
                                       directory=args.input_dir,
                                       progIDs=args.progIDs)

    # fit the distortions!
    (errorflag,coefffilename) = distortions.fit_distortions(args.aperture, args.filter, args.pupil,
                                               outrootdir=args.outrootdir, 
                                               outsubdir=args.outsubdir,
                                               outbasename=args.outbasename,
                                               skip_if_exists=args.skip_if_exists)
    if errorflag:
        print('ERROR! Something went wrong!')
    