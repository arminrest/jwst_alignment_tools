#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 10:11:00 2023

@author: arest
"""

import sys,argparse,os
from pdastro import unique,AandB,pdastroclass
from calc_distortions import calc_distortions_class

class calc_distortions_list_class(calc_distortions_class):
    def __init__(self):
        calc_distortions_class.__init__(self)
        
        self.dict_aper_filt_pupil = {}
        
        self.summary = pdastroclass()
        
    def remove_summary_entry(self,apername,filtname,pupilname):
        if len(self.summary.t)>0:
            ixs = self.summary.ix_equal('apername', apername)
            ixs = self.summary.ix_equal('filter', filtname,indices=ixs)
            ixs = self.summary.ix_equal('pupil', pupilname,indices=ixs)
            if len(ixs)>1:
                self.summary.write(indices=ixs)
                raise RuntimeError(f'more than 1 entry for {apername} {filtname} {pupilname}!')
            elif len(ixs)==1:
                if self.verbose: print(f'Removing previous entry in summary table for  {apername} {filtname} {pupilname}')
                self.summary.t.drop(index=ixs,inplace=True)
        return(0)

    def fit_all_distortions(self, apertures=None, filters=None, pupils=None, 
                            outrootdir=None, outsubdir=None,
                            outbasename=None,
                            summaryfilename=None,
                            ixs_im=None, progIDs=None,
                            raiseErrorflag=True):

        if (summaryfilename is not None) and self.save_coeffs:
            if os.path.dirname(summaryfilename)=='':
                self.set_outbasename(outrootdir=outrootdir,outsubdir=outsubdir)
                summaryfilename = f'{os.path.dirname(self.outbasename)}/{summaryfilename}'    
            if os.path.isfile(summaryfilename):
                print(f'Loading summary file {summaryfilename} ...')
                self.summary.load(summaryfilename)
            #self.summary.write()
            #sys.exit(0)
        
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
                    
                    print('VVVVVVV')
                    self.summary.write()
                    self.remove_summary_entry(apername,filtname,pupilname)
                    print('VVVVVVV')
                    self.summary.write()
                    
                    ixs_pupil = self.imtable.ix_equal('pupil', pupilname, indices=ixs_filter)
                    if self.verbose: print(f'###\n### {apername} {filtname} {pupilname}\n###')
                    self.imtable.write(indices=ixs_pupil)
                    
                    (errorflag,coefffilename) = self.fit_distortions(apername, filtname, pupilname,
                                                                     outrootdir=outrootdir, outsubdir=outsubdir,
                                                                     outbasename=outbasename,
                                                                     ixs_im=ixs_pupil, progIDs=progIDs)
                    
                    ix_result = self.summary.newrow({'apername':apername,'filter':filtname,'pupil':pupilname,
                                                     'Nim':len(self.ixs_im),
                                                    'errorflag':errorflag,'coefffilename':os.path.basename(coefffilename)})
                    cols=['dx_mean','dx_mean_err','dx_stdev','dy_mean','dy_mean_err','dy_stdev','Ngood','Nclip']
                    self.summary.t.loc[ix_result,cols] = self.Sci2Idl_residualstats.t.loc[0,cols]

        ixs_summary_sorted = self.summary.ix_sort_by_cols(['apername','filter','pupil']) 
        self.summary.write(indices=ixs_summary_sorted)
        # Only save into summary files if coefficients were saved as well.
        if (summaryfilename is not None) and self.save_coeffs:
            print(f'Saving summary into {summaryfilename}')
            self.summary.write(summaryfilename,indices=ixs_summary_sorted) 
                
    

if __name__ == '__main__':
    
    distortions = calc_distortions_list_class()

    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('input_filepatterns', nargs='+', type=str, help='list of input file(pattern)s. These get added to input_dir if input_dir is not None')
    parser.add_argument('--apertures', default=None, nargs='+', type=str, help='list of aperture names, e.g. nrca1_full')
    parser.add_argument('--filters', default=None, nargs='+', type=str, help='list of filter names, e.g. f200w.')
    parser.add_argument('--pupils', default=None, nargs='+', type=str, help='list of pupil names, e.g. clear. ')
    parser.add_argument('--savecoeff', default=False, action='store_true', help='Save the coefficients. If not overridden with --coefffilename, the default name is {apername}_{filtername}_{pupilname}.polycoeff.txt')
    parser.add_argument('--summaryfilename', default='distortion_summary.txt', help='filename that contains a summary of the distortion fits. If filename has not path, it is saved in the output directory (default=%(default)s).')
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
                                    outbasename=args.outbasename,
                                    summaryfilename=args.summaryfilename)
        
