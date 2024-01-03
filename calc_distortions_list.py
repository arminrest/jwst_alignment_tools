#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 10:11:00 2023

@author: arest
"""

import sys,argparse
from pdastro import unique,AandB
from calc_distortions import calc_distortions_class

class calc_distortions_list_class(calc_distortions_class):
    def __init__(self):
        calc_distortions_class.__init__(self)
        
        self.dict_aper_filt_pupil = {}

    def fit_all_distortions(self, apertures=None, filters=None, pupils=None, 
                            outrootdir=None, outsubdir=None,
                            outbasename=None,
                            ixs_im=None, progIDs=None,
                            raiseErrorflag=True):
        
        # get the image indices
        ixs_im = self.imtable.getindices(ixs_im)
        
        if apertures is None: 
            apertures = sorted(unique(self.imtable.t.loc[ixs_im,'apername']))
        if self.verbose: print(f'Apertures: {apertures}')
        # check if there are images?
        if len(apertures)==0:
            if raiseErrorflag: raise RuntimeError(f'No images for Apertures: {apertures}!')
            print(f'WARNING: No images for Apertures: {apertures}!')
            return(0)

        for apername in apertures:
            
            ixs_aperture = self.imtable.ix_equal('apername', apername, indices=ixs_im)
            
            filters4aperture = unique(self.imtable.t.loc[ixs_aperture,'filter'])
            if filters is not None:
                filters4aperture = AandB(filters, filters4aperture)
            filters4aperture = sorted(filters4aperture)
            if self.verbose: print(f'##########################################\n### Aperture {apername}: filters = {" ".join(filters4aperture)}\n##########################################')
            
            if len(filters4aperture)==0:
                print(f'WARNING: No images for aperture {apername}!')
                continue
            
            
            for filtname in filters4aperture:
                ixs_filter = self.imtable.ix_equal('filter', filtname, indices=ixs_aperture)
                
                pupils4filter = unique(self.imtable.t.loc[ixs_filter,'pupil'])
                if pupils is not None:
                    pupils4filter = AandB(pupils,pupils4filter)
                pupils4filter = sorted(pupils4filter)
                if self.verbose and len(pupils4filter)>1: print(f'Pupils for {apername} {filtname}: pupils={" ".join(pupils4filter)}')
                
                if len(pupils4filter)==0:
                    print(f'WARNING: No filters/images for {apername} {filtname}!')
                    continue
                
                for pupilname in pupils4filter:
                    ixs_pupil = self.imtable.ix_equal('pupil', pupilname, indices=ixs_filter)
                    if self.verbose: print(f'###\n### {apername} {filtname} {pupilname}\n###')
                    self.imtable.write(indices=ixs_pupil)
                    
                    errorflag = self.fit_distortions(apername, filtname, pupilname,
                                                     outrootdir=outrootdir, outsubdir=outsubdir,
                                                     outbasename=outbasename,
                                                     ixs_im=ixs_pupil, progIDs=progIDs)
                    
                    
                
    

if __name__ == '__main__':
    
    distortions = calc_distortions_list_class()

    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('input_filepatterns', nargs='+', type=str, help='list of input file(pattern)s. These get added to input_dir if input_dir is not None')
    parser.add_argument('--apertures', default=None, nargs='+', type=str, help='list of aperture names, e.g. nrca1_full')
    parser.add_argument('--filters', default=None, nargs='+', type=str, help='list of filter names, e.g. f200w.')
    parser.add_argument('--pupils', default=None, nargs='+', type=str, help='list of pupil names, e.g. clear. ')
    parser.add_argument('--savecoeff', default=False, action='store_true', help='Save the coefficients. If not overridden with --coefffilename, the default name is {apername}_{filtername}_{pupilname}.polycoeff.txt')
    #parser.add_argument('--input_dir', type=str, default=None, help='input_dir is the directory in which the input images are located located (default=%(default)s)')
    #parser.add_argument('--outrootdir', default='.', help='output root directory. The output directoy is the output root directory + the outsubdir if not None. (default=%(default)s)')
    #parser.add_argument('--outsubdir', default=None, help='outsubdir added to output root directory (default=%(default)s)')
    #parser.add_argument('--progIDs', type=int, default=None, nargs="+", help='list of progIDs (default=%(default)s)')

    parser = distortions.define_options(parser=parser)

    args = parser.parse_args()
    
    distortions.verbose=args.verbose
    distortions.savecoeff=not (args.skip_savecoeff)
    distortions.showplots=args.showplots
    distortions.saveplots=args.saveplots
    
    # get all the files
    distortions.get_inputfiles_imtable(args.input_filepatterns,
                                       directory=args.input_dir,
                                       progIDs=args.progIDs)
    
    distortions.fit_all_distortions(apertures=args.apertures, filters=args.filters, pupils=args.pupils,
                                    outrootdir=args.outrootdir, outsubdir=args.outsubdir,
                                    outbasename=args.outbasename)
        
