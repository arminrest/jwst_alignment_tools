#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 20:51:06 2026

@author: arest
"""

import sys,argparse,os,re,glob
from pdastro import unique,AandB,pdastroclass
from calc_fpa2fpa_alignment import fpa2fpa_alignmentclass
#from astropy.time import Time

# filename convention:
#https://jwst-pipeline.readthedocs.io/en/latest/jwst/data_products/file_naming.html
# jw<ppppp><ooo><vvv>_<gg><s><aa>_<eeeee>(-<”seg”NNN>)_<detector>_<prodType>.fits
# ppppp: program ID number
# ooo: observation number
# vvv: visit number
# gg: visit group
# s: parallel sequence ID (1=prime, 2-5=parallel)
# aa: activity number (base 36)
# eeeee: exposure number
# segNNN: the text “seg” followed by a three-digit segment number (optional)

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

    if verbose: print(f'Found {len(filenames)} files  matching filepatterns {filepatterns}')
    return(filenames)


class fpa2fpa_alignment_list_class(fpa2fpa_alignmentclass):
    def __init__(self):
        fpa2fpa_alignmentclass.__init__(self)
        
        self.summary = pdastroclass()

        self.verbose = 0
        self.showplots = 0
        self.saveplots = 0
        
        self.imtable = pdastroclass(columns=['imID','progID','obs','visit','group','parallel','fullimage'])

        # plot style for residual plots
        self.plot_style={}
        self.plot_style['good']={'style':'o','color':'blue', 'ms':5 ,'alpha':0.5}
        self.plot_style['cut']={'style':'o','color':'red', 'ms':5 ,'alpha':0.3}
        self.plot_style['excluded']={'style':'o','color':'gray', 'ms':3 ,'alpha':0.3}
        
    def define_options(self,parser=None,usage=None,conflict_handler='resolve'):
        fpa2fpa_alignmentclass.define_options(self,parser=parser,usage=usage,conflict_handler=conflict_handler)
        parser.add_argument('--input_dir', type=str, default=None, help='input_dir is the directory in which the input images are located located (default=%(default)s)')
        parser.add_argument('--progIDs', type=int, default=None, nargs="+", help='list of progIDs (default=%(default)s)')
        parser.add_argument('--apertures', default=None, nargs='+', type=str, help='list of aperture names, e.g. nrca1_full')
        parser.add_argument('--filters', default=None, nargs='+', type=str, help='list of filter names, e.g. f200w.')
        parser.add_argument('--pupils', default=None, nargs='+', type=str, help='list of pupil names, e.g. clear. ')
        parser.add_argument('--date4suffix', default=None, type=str, help='date of the form YYYY-MM-DD is added to the photcat suffix')
        parser.add_argument('--summaryfilename', default='alignment_summary.txt', help='filename that contains a summary of the distortion fits. If filename has not path, it is saved in the output directory (default=%(default)s).')
        parser.add_argument('-p','--showplots', default=0, action='count')
        parser.add_argument('-s','--saveplots', default=0, action='count')

        return(parser)

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

    def get_inputfiles_info(self):
        """
        Populate the self.imtable with the relevant fits keywords 

        Returns
        -------
        None.

        """
        ixs = self.imtable.getindices()

        self.imtable.fitsheader2table('fullimage',requiredfitskeys=['INSTRUME','DETECTOR','APERNAME','FILTER','PUPIL','DATE-OBS','TIME-OBS'])
        self.imtable.fitsheader2table('fullimage',requiredfitskeys=['V2_REF','V3_REF','V3I_YANG'],ext=1)

        # Convert fits keyword columns to lower case, including column names
        renamedict={}
        for col in ['INSTRUME','DETECTOR','APERNAME','FILTER','PUPIL','DATE-OBS','TIME-OBS']:
            self.imtable.t[col] = self.imtable.t[col].str.lower()
            renamedict[col]=col.lower()
        self.imtable.t = self.imtable.t.rename(columns=renamedict)

        #self.imtable.write()
        #sys.exit(0)
        
        for ix in ixs:   
            """
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
            """
            shortname = os.path.basename(self.imtable.t.loc[ix,'fullimage'])
            
            #m = re.search('^jw(\d\d\d\d\d)()',shortname)
            m = re.search('^jw(\d{5})(\d{3})(\d{3})_(\d{2})(\d{1})(\d{2})',shortname)
            if m is not None:
                info = list(m.groups())
                for i in range(5):
                    info[i]=int(info[i])
            else:
                info = [0,0,0,0,0,0]
            self.imtable.t.loc[ix,['progID','obs','visit','group','parallel']]=info[:5]
            #if m is not None:
            #    [progID,obs,visit,groupnumber,parallel,activity] = int(m.groups()[0])
            #else:
            #    progID = [0,0,0,0,0,0]
            #self.imtable.t.loc[ix,'progID']=progID

        # Make sure the type and formatting are all good
        for col in ['imID','progID','obs','visit','group','parallel']:
            self.imtable.t[col]=self.imtable.t[col].astype('int')

    def get_inputfiles_imtable(self, filepatterns, directory=None, progIDs=None):
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
        
        # check for program IDs
        if progIDs is not None:
            if isinstance(progIDs,str):
                progIDs = [progIDs]
            ixs_im = []
            for progID in progIDs:
               ixs_im.extend(self.imtable.ix_equal('progID', int(progID)))

            if self.verbose>1: print(f'{len(ixs_im)} out of {len(self.imtable.t)} images left with progIDs={progIDs}')
            self.imtable.t=self.imtable.t.loc[ixs_im].copy()
        
        if self.verbose>1:
            self.imtable.write()

        return(0)
    
    def select_images(self, ixs_im = None, apertures=None, filters=None, pupils=None, raiseErrorflag=True):
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
        
        ixs_keep = []
        ### Loop through apertures
        for apername in apertures:
            # get image indices for aperture
            ixs_keep.extend(self.imtable.ix_equal('apername', apername, indices=ixs_im))        
        ixs_selected = unique(ixs_keep)
        
        if filters is not None:
            ixs_keep = []
            ### Loop through apertures
            for filt in filters:
                # get image indices for aperture
                ixs_keep.extend(self.imtable.ix_equal('filter', filt, indices=ixs_selected)) 
            ixs_selected = unique(ixs_keep)
            
        if pupils is not None:
            ixs_keep = []
            ### Loop through apertures
            for pupil in pupils:
                # get image indices for aperture
                ixs_keep.extend(self.imtable.ix_equal('pupil', pupil, indices=ixs_selected)) 
            ixs_selected = unique(ixs_keep)
        
        ixs_return = self.imtable.ix_sort_by_cols('imID',indices=ixs_selected)
        if self.verbose>1:
            print(f'selected {len(ixs_return)} images out of {len(ixs_im)}')
            self.imtable.write(indices=ixs_return)
        return(ixs_return)


        
if __name__ == '__main__':
    
    fpa2fpa_list = fpa2fpa_alignment_list_class()

    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('input_filepatterns', nargs='+', type=str, help='list of input file(pattern)s. These get added to input_dir if input_dir is not None')


    parser = fpa2fpa_list.define_options(parser=parser)

    args = parser.parse_args()
    
    fpa2fpa_list.verbose=args.verbose
    fpa2fpa_list.showplots=args.showplots
    fpa2fpa_list.saveplots=args.saveplots
    
    # get all the files
    fpa2fpa_list.get_inputfiles_imtable(args.input_filepatterns,
                                        directory=args.input_dir,
                                        progIDs=args.progIDs)
    
    ixs_im = fpa2fpa_list.select_images(apertures=args.apertures, filters=args.filters, pupils=args.pupils)



    sys.exit(0)
    
    fpa2fpa_list.fit_all_fpa2fpa_list(apertures=args.apertures, filters=args.filters, pupils=args.pupils,
                                    outrootdir=args.outrootdir, outsubdir=args.outsubdir,
                                    outbasename=args.outbasename,
                                    summaryfilename=args.summaryfilename,
                                    skip_if_exists=args.skip_if_exists
                                    )
        
