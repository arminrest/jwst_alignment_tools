#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 10:11:00 2023

@author: arest
"""

import sys,argparse,os,re
from pdastro import unique,AandB,pdastroclass
from calc_distortions import calc_distortions_class
from astropy.time import Time

class calc_distortions_list_class(calc_distortions_class):
    def __init__(self):
        calc_distortions_class.__init__(self)
        
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
                            skip_if_exists=False,
                            ixs_im=None, progIDs=None,
                            raiseErrorflag=True):


        # prepare summary file: load it if it exists, so that it can be updated with the new results.
        if (summaryfilename is not None) and self.save_coeffs:
            if os.path.dirname(summaryfilename)=='':
                self.set_outbasename(outrootdir=outrootdir,outsubdir=outsubdir)
                summaryfilename = f'{os.path.dirname(self.outbasename)}/{summaryfilename}'    
            if os.path.isfile(summaryfilename):
                print(f'Loading summary file {summaryfilename} ...')
                self.summary.load(summaryfilename)
            #self.summary.write()
            #sys.exit(0)
        
        # get the image indices. If the passed ixs_im==None, then all of them are used.
        ixs_im = self.imtable.getindices(ixs_im)
        
        # get the apertures
        if apertures is None:
            # No apertures passed, so use all of them
            apertures = unique(self.imtable.t.loc[ixs_im,'apername'])
        else:
            # Only use apertures that are also in the passed list!
            apertures = AandB(apertures, unique(self.imtable.t.loc[ixs_im,'apername']))
        apertures = sorted(apertures)
        # check if there are images?
        if len(apertures)==0:
            if raiseErrorflag: raise RuntimeError(f'No images for Apertures: {apertures}!')
            print(f'WARNING: No images for Apertures: {apertures}!')
            return(1)

        # initialize some bookkeeping!
        goodcounter=errorcounter=skipcounter=nodata_counter=0
        errorlist=[]
        allfilters=[]
        allpupils=[]

        ### Loop through apertures
        for apername in apertures:
            
            # get image indices for aperture
            ixs_aperture = self.imtable.ix_equal('apername', apername, indices=ixs_im)
            
            # get the filters for given aperture
            filters4aperture = unique(self.imtable.t.loc[ixs_aperture,'filter'])
            if filters is not None:
                filters4aperture = AandB(filters, filters4aperture)
            filters4aperture = sorted(filters4aperture)
            if self.verbose: print(f'##########################################\n### Aperture {apername}: filters = {" ".join(filters4aperture)}\n##########################################')
            
            if len(filters4aperture)==0:
                print(f'WARNING: No images for aperture {apername}!')
                nodata_counter+=1
                continue
            
            
            ### Loop through filters for given aperture
            for filtname in filters4aperture:
                
                # get image indices for aperture/filter pair
                ixs_filter = self.imtable.ix_equal('filter', filtname, indices=ixs_aperture)
                
                # get the pupils for given  aperture/filter pair
                pupils4filter = unique(self.imtable.t.loc[ixs_filter,'pupil'])
                if pupils is not None:
                    pupils4filter = AandB(pupils,pupils4filter)
                pupils4filter = sorted(pupils4filter)
                if self.verbose and len(pupils4filter)>1: print(f'Pupils for {apername} {filtname}: pupils={" ".join(pupils4filter)}')
                
                if len(pupils4filter)==0:
                    print(f'WARNING: No filters/images for {apername} {filtname}!')
                    nodata_counter+=1
                    continue
                
                allfilters.append(filtname)
                allpupils.extend(pupils4filter)
                
                ### Loop through pupils for given aperture/filter
                for pupilname in pupils4filter:
                                        
                    # get image indices for aperture/filter/pupil tuple
                    ixs_pupil = self.imtable.ix_equal('pupil', pupilname, indices=ixs_filter)
                    if self.verbose: print(f'###\n### {apername} {filtname} {pupilname}\n###')
                    self.imtable.write(indices=ixs_pupil)
                    
                    # fit the distortions
                    (errorflag,coefffilename) = self.fit_distortions(apername, filtname, pupilname,
                                                                     outrootdir=outrootdir, outsubdir=outsubdir,
                                                                     outbasename=outbasename,
                                                                     skip_if_exists=skip_if_exists,
                                                                     ixs_im=ixs_pupil, progIDs=progIDs)
                    # skipped it since already exists and skip_if_exists==True
                    # Don't change anything in the summary table
                    if errorflag<0:
                        print(f'Distortion coefficients for {apername} {filtname} {pupilname} already exist, continuing to next aperture/filter/pupil...')
                        skipcounter += 1
                        continue

                    # remove previous entry for aperture/filter/pupil from summary table if exists
                    # This makes sure that nothing from previous work makes it accidently through
                    self.remove_summary_entry(apername,filtname,pupilname)

                    # add entry to the summary table
                    ix_result = self.summary.newrow({'apername':apername,'filter':filtname,'pupil':pupilname,
                                                     'Nim':len(self.ixs_im),
                                                     'errorflag':errorflag,'coefffilename':os.path.basename(coefffilename)})

                    # update entry in summary table with statistics of residuals if good distortions coeffs
                    if errorflag>0:
                        errorcounter += 1
                        errorlist.append('{apername} {filtname} {pupilname}')
                        print('ERROR! There is an error for {apername} {filtname} {pupilname}!! continuing to next aperture/filter/pupil...')
                    elif errorflag == 0:
                        goodcounter += 1
                        # copy over the statistics of residuals
                        cols=['dx_mean','dx_mean_err','dx_stdev','dy_mean','dy_mean_err','dy_stdev','Ngood','Nclip']
                        self.summary.t.loc[ix_result,cols] = self.Sci2Idl_residualstats.t.loc[0,cols]

                    # We save the summary file after each distortion calculation in order to make sure
                    # that we do not loose this info in case the loop crashes.
                    if (summaryfilename is not None) and self.save_coeffs:
                        ixs_summary_sorted = self.summary.ix_sort_by_cols(['apername','filter','pupil']) 
                        print(f'Saving summary into {summaryfilename}')
                        self.summary.write(summaryfilename,indices=ixs_summary_sorted) 


        # sort and write the summary table
        ixs_summary_sorted = self.summary.ix_sort_by_cols(['apername','filter','pupil']) 
        self.summary.write(indices=ixs_summary_sorted)
        # Only save into summary files if coefficients were saved as well.
        if (summaryfilename is not None) and self.save_coeffs:
            print(f'Saving summary into {summaryfilename}')
            self.summary.write(summaryfilename,indices=ixs_summary_sorted) 
        
        # finished! print some boookkeeping info
        allfilters = sorted(unique(allfilters))
        allpupils = sorted(unique(allpupils))
        print('\n###########\n### SUMMARY\n###########')
        print(f'{goodcounter:3} distortion coefficents were successfully calculated for apertures {" ".join(apertures)}, filters {" ".join(allfilters)}, and pupils {" ".join(allpupils)}')
        if skipcounter>0:
            print(f'{skipcounter:3} distortion coefficents were skipped because they already existed!')
        if nodata_counter>0:
            print('### WARNING: some apertures or filters had no data. This could be because the data was removed with --apertures, --filters, or --pupils constraints. Pls check!')
        if errorcounter>0:
            print(f'### ERROR: There were {errorcounter} errors for the following aperture/filter/pupil tuples:')
            print('\n'.join(errorlist))
        
        return(0)
            
    

if __name__ == '__main__':
    
    distortions = calc_distortions_list_class()

    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('input_filepatterns', nargs='+', type=str, help='list of input file(pattern)s. These get added to input_dir if input_dir is not None')
    parser.add_argument('--apertures', default=None, nargs='+', type=str, help='list of aperture names, e.g. nrca1_full')
    parser.add_argument('--filters', default=None, nargs='+', type=str, help='list of filter names, e.g. f200w.')
    parser.add_argument('--pupils', default=None, nargs='+', type=str, help='list of pupil names, e.g. clear. ')
    parser.add_argument('--date4suffix', default=None, type=str, help='date of the form YYYY-MM-DD is added to the photcat suffix')
    parser.add_argument('--summaryfilename', default='distortion_summary.txt', help='filename that contains a summary of the distortion fits. If filename has not path, it is saved in the output directory (default=%(default)s).')


    parser = distortions.define_options(parser=parser)

    args = parser.parse_args()
    
    distortions.verbose=args.verbose
    distortions.savecoeff=not (args.skip_savecoeff)
    distortions.showplots=args.showplots
    distortions.saveplots=args.saveplots

    if args.xypsf:
        distortions.colnames['x']='x_psf'
        distortions.colnames['y']='y_psf'
        distortions.phot_suffix = '.good.phot_psf.txt'
    if args.xy1pass:
        distortions.colnames['x']='x_1p'
        distortions.colnames['y']='y_1p'
        distortions.phot_suffix = '.good.phot.1pass_v2.txt'
        if args.date4suffix is not None:
            dateobj = Time(args.date4suffix)
            date = re.sub('T.*','',dateobj.to_value('isot'))
            distortions.phot_suffix = f'.{date}{distortions.phot_suffix}'
    print(f'suffix for photcat files: {distortions.phot_suffix}')
    
    # get all the files
    distortions.get_inputfiles_imtable(args.input_filepatterns,
                                       directory=args.input_dir,
                                       progIDs=args.progIDs)
    
    distortions.fit_all_distortions(apertures=args.apertures, filters=args.filters, pupils=args.pupils,
                                    outrootdir=args.outrootdir, outsubdir=args.outsubdir,
                                    outbasename=args.outbasename,
                                    summaryfilename=args.summaryfilename,
                                    skip_if_exists=args.skip_if_exists
                                    )
        
