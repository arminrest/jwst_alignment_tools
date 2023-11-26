#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri June 24 20:37:27 2022

@author: arest
"""

import sys, argparse,glob,re,os
#import numpy as np
from pdastro import pdastrostatsclass,unique,makepath4file
from astropy.io import fits
from subprocess import Popen, PIPE, STDOUT
from jwst import datamodels
import pysiaf
#from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from pdastro import pdastroclass
from astropy.modeling.rotations import Rotation2D
import numpy as np
from calc_fpa2fpa_alignment import fpa2fpa_alignmentclass

def append2file(filename,lines,verbose=0):
    if type(lines) is str:#types.StringType:
        lines = [lines,]
    if os.path.isfile(filename):
        buff = open(filename, 'a')
    else:
        buff = open(filename, 'w')
    r=re.compile('\n$')
    for line in lines:
        if not r.search(line):
            buff.write(line+'\n')
            if verbose: print((line+'\n'))
        else:
            buff.write(line)
            if verbose: print(line)
    buff.close()

def executecommand(cmd,successword,errorlog=None,cmdlog=None,verbose=1):
    if verbose: print(f'executing: {cmd}')

#    (cmd_in,cmd_out)=os.popen4(cmd)
    p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
    cmd_in,cmd_out = p.stdin,p.stdout

    output = cmd_out.readlines()
    if successword=='':
        successflag = 1
    else:
        m = re.compile(successword)
        successflag = 0
        for line in output:
            if sys.version_info[0] >= 3:
                line = line.decode('utf-8')
            if m.search(line):
                successflag = 1
    errorflag = not successflag
    if errorflag:
        print('error executing:',cmd)
        if errorlog != None:
            append2file(errorlog,['\n error executing: '+cmd+'\n'])
            append2file(errorlog,output)
        if cmdlog != None:
            append2file(cmdlog,['\n error executing:',cmd])
    else:
        if cmdlog != None:
            append2file(cmdlog,[cmd])
    return errorflag, output

def define_options(parser=None,usage=None,conflict_handler='resolve'):
    if parser is None:
        parser = argparse.ArgumentParser(usage=usage,conflict_handler=conflict_handler)

    parser.add_argument('--fgs_filepatterns', nargs='+', type=str, default=['*_guider1_jhat.fits'], help='list of fgs file(pattern)s. these get added to fgs_dir if fgs_dir is not None (default=%(default)s)')
    parser.add_argument('--fgs_dir', type=str, default='/Volumes/Ext1/jhat/v2v3ref/hawki_v1', help='fgs_dir is the directory in which the fgs data is located (default=%(default)s)')
    parser.add_argument('--fgs_apername', type=str, choices=['FGS1_FULL','FGS2_FULL'], default='FGS1_FULL', help='reference FGS aperture (default=%(default)s)')
    parser.add_argument('--nircam_dir', type=str, default='/Volumes/Ext1/jhat/v2v3ref/hawki_v1', help='nircam_dir is the directory in which the nircam data is located (default=%(default)s)')
    parser.add_argument('--savetable', type=str, default=None, help='Save the table to the given filename (default=%(default)s)')

    parser.add_argument('--detectors', type=str, choices = ['nrca1','nrca2','nrca3','nrca4','nrca5','nrcb1','nrcb2','nrcb3','nrcb4','nrcb5'], default=None, nargs="+", help='list of detectors (default=%(default)s)')
    parser.add_argument('--filters', type=str, default=None, nargs="+", help='list of filters (default=%(default)s)')
    parser.add_argument('--pupils', type=str, default=None, nargs="+", help='list of pupils (default=%(default)s)')
    parser.add_argument('--progIDs', type=int, default=None, nargs="+", help='list of progIDs (default=%(default)s)')

    parser.add_argument('--maxin', type=int, default=None, help='constraint the # of images (good for testing/debugging) (default=%(default)s)')
    parser.add_argument('--skip_if_exists', default=False, action='store_true', help='Don\'t reanalyze if outsubdir/focal_plane_calibration already exists')

    parser.add_argument('-v','--verbose', default=0, action='count')

    return(parser)

def calc_averages(v2v3,v2v3params=['V3IdlYAngle','V2Ref','V3Ref'],progIDs0=None, apertures0=None,filts0=None, indices=None, format_params = '{:.8f}',verbose=1):
    mean_results = pdastrostatsclass()
    
    # remove empty rows
    ixs = v2v3.ix_not_null(['progID','V2Ref'],indices=indices)
    
    if apertures0 is None:
        apertures = unique(v2v3.t.loc[ixs,'aperture'])
    else:
        apertures = apertures0
    apertures.sort()
    
    v2v3.write(indices=ixs)
    
    for aperture in apertures:
        if verbose: print(f'### aperture {aperture}')
        # get entries for aperture
        ixs_aper = v2v3.ix_equal('aperture',aperture.lower(),indices=ixs)
        if len(ixs_aper)==0:
            print(f'Warning: skipping aperture {aperture} since there are no entries!')
            continue
    
        if filts0 is None:
            filts = unique(v2v3.t.loc[ixs_aper,'filter'])
        else:
            filts = filts0
        filts.sort()
    
        for filt in filts:
            ixs_filt = v2v3.ix_equal('filter',filt.lower(),indices=ixs_aper)
            v2v3.write(indices=ixs_aper)
            v2v3.write(indices=ixs_filt)
            if len(ixs_filt)==0:
                print(f'Warning: skipping filter {filt} for aperture {aperture} since there are no entries!')
                continue
    
            if progIDs0 is None:
                progIDs = unique(v2v3.t.loc[ixs_filt,'progID'])
                progIDs.append('all')
            else:
                progIDs = progIDs0
            progIDs.sort()
            
            
            ixs_progID_all = []
            for progID in progIDs:
                if progID != 'all':
                    ixs_progID = v2v3.ix_equal('progID',progID,indices=ixs_filt)
                    if len(ixs_progID)==0:
                        print(f'Warning: skipping progID {progID} for filter {filt} and aperture {aperture} since there are no entries!')
                        continue
                    ixs_progID_all.extend(ixs_progID)
                else:
                    # use all indices for this filter for the final fit!!
                    ixs_progID = ixs_progID_all
    
                # for each aperture, filter, and progID there is one row in the results
                ix_results = mean_results.newrow({'aperture':aperture,'filter':filt,'progID':progID})
    
    
                if verbose>2:
                    v2v3.write(indices=ixs_progID,columns=['aperture','filter','V3IdlYAngle','V2Ref','V3Ref'])
    
                for v2v3paramname in v2v3params:
                    v2v3.calcaverage_sigmacutloop(v2v3paramname,indices=ixs_progID,percentile_cut_firstiteration=60)
    
                    mean_results.param2columnmapping = mean_results.intializecols4statparams(
                        setcol2None=False,
                        params=['mean','mean_err','stdev','Ngood','Nclip'], # define the statistic parameters that you want to keep
                        prefix=f'{v2v3paramname}_', # define the prefix for the output statistics columns, e.g. for mean it will be f'{prefix}mean'
                        format4outvals=format_params) # format of statistics columns
    
                    mean_results.statresults2table(v2v3.statparams,
                                                   mean_results.param2columnmapping,
                                                   destindex=ix_results)

    return(mean_results)

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

#####################################################################
#####################################################################
#####################################################################

if __name__ == '__main__':
    
    fpa2fpa = fpa2fpa_alignmentclass()
    
    parser = define_options()
    parser = fpa2fpa.define_options(parser=parser)

    args = parser.parse_args()
    fpa2fpa.verbose=args.verbose
    
    fgstable = pdastrostatsclass()
    nrctable = pdastrostatsclass(columns=['nrc_filename','fgs_filename','detector','aperture','filter','pupil'])
    
    # find the fgs filenames
    fgsfilenames = []
    for fgs_filepattern in args.fgs_filepatterns:
        newfgsfilenames = glob.glob(f'{args.fgs_dir}/{fgs_filepattern}')
        if len(newfgsfilenames)==0:
            raise RuntimeError(f'Could not find any files that match {fgs_filepattern}')
        fgsfilenames.extend(newfgsfilenames)
        
    fgsfilenames=unique(fgsfilenames)
    fgsfilenames.sort()
    
    fgstable.t['fgs_filename'] = fgsfilenames
    
    # fill the fgs table
    for ix in fgstable.getindices():
        basename = os.path.basename(fgstable.t.loc[ix,'fgs_filename'])
        # https://jwst-pipeline.readthedocs.io/en/latest/jwst/data_products/file_naming.html#exposure-file-names
        # substitute s: parallel sequence ID with '?'
        m = re.search('(jw\d+_\d\d)\d(\d+_\d+)_',basename)
        if m is None:
            raise RuntimeError(f'Could not parse {basename}')
        nrc_filepattern = f'{m.groups()[0]}?{m.groups()[1]}*_nrc*_jhat.fits'
        if args.verbose>1: print(f'{basename} -> {nrc_filepattern}')
        fgstable.t.loc[ix,'nrc_filepattern']=nrc_filepattern
        
    if args.verbose:
        fgstable.write()

    ixs_unmatched_fgsfiles=[]

    # get the associated nircam files, and get detector, filter, pupil
    for ix in fgstable.getindices():
        
        
        nrc_filepattern = f'{args.nircam_dir}/{fgstable.t.loc[ix,"nrc_filepattern"]}'
        nrc_filenames = glob.glob(nrc_filepattern)
        if len(nrc_filenames)==0:
            ixs_unmatched_fgsfiles.append(ix)
            continue
            #raise RuntimeError(f'Could not find any files that match {nrc_filepattern}')
        for nrc_filename in nrc_filenames:
            try:
                imhdr = fits.getheader(nrc_filename)
            except:
                print(f'{nrc_filename} corrupt, skipping...')
                continue
            
            m = re.search('^jw(\d\d\d\d\d)',os.path.basename(nrc_filename))
            if m is not None:
                progID = int(m.groups()[0])
            else:
                progID = 0

                
            ix_nrc = nrctable.newrow({'progID':progID,
                                      'nrc_filename':nrc_filename,
                                      'fgs_filename':fgstable.t.loc[ix,"fgs_filename"],
                                      'instrument':imhdr["INSTRUME"].lower(),
                                      'detector':re.sub('long$','5',imhdr['DETECTOR'].lower()),
                                      'aperture':re.sub('long$','5',imhdr['APERNAME'].lower()),
                                      'filter':imhdr['FILTER'].lower(),
                                      'pupil':imhdr['PUPIL'].lower(),
                                      'errorflag':0},
                                      )
            
            #outsubdir = f'{nrctable.t.loc[ix_nrc,"detector"]}_{nrctable.t.loc[ix_nrc,"filter"]}_{nrctable.t.loc[ix_nrc,"pupil"]}_' \
            #    +re.sub('_nrc\w+\_cal\.fits$','',os.path.basename(nrc_filename))
            #nrctable.t.loc[ix_nrc,'outsubdir']=outsubdir
               
    nrctable.write()
            
    ixs_nrc = nrctable.getindices()
    print(f'{len(ixs_nrc)} images associated with the fgs images!')
    
    # cut in detector, filter, and pupil
    if args.detectors is not None:
        ixs_tmp = []
        for detector in args.detectors:
            ixs_tmp.extend(nrctable.ix_equal('detector',detector.lower(),indices=ixs_nrc))
        print(f'Cutting {len(ixs_nrc)-len(ixs_tmp)} from {len(ixs_nrc)} because of detector={args.detectors}, leaving {len(ixs_tmp)}!')
        ixs_nrc = ixs_tmp

    if args.filters is not None:
        ixs_tmp = []
        for filt in args.filters:
            ixs_tmp.extend(nrctable.ix_equal('filter',filt.lower(),indices=ixs_nrc))
        print(f'Cutting {len(ixs_nrc)-len(ixs_tmp)} from {len(ixs_nrc)} because of filter={args.filters}, leaving {len(ixs_tmp)}!')
        ixs_nrc = ixs_tmp

    if args.pupils is not None:
        ixs_tmp = []
        for pupil in args.pupils:
            ixs_tmp.extend(nrctable.ix_equal('pupil',pupil.lower(),indices=ixs_nrc))
        print(f'Cutting {len(ixs_nrc)-len(ixs_tmp)} from {len(ixs_nrc)} because of pupil={args.pupils}, leaving {len(ixs_tmp)}!')
        ixs_nrc = ixs_tmp
            
    if args.progIDs is not None:
        ixs_tmp = []
        for progID in args.progIDs:
            ixs_tmp.extend(nrctable.ix_equal('progID',progID,indices=ixs_nrc))
        print(f'Cutting {len(ixs_nrc)-len(ixs_tmp)} from {len(ixs_nrc)} because of progID={args.progIDs}, leaving {len(ixs_tmp)}!')
        ixs_nrc = ixs_tmp
            
    if len(ixs_nrc)<1:
        print('WARNING: no selected pairs, exiting...')
        sys.exit(0)
        
    if args.maxin is not None:
        print('constraining selected pairs to {args.maxin} entries')
        ixs_nrc = ixs_nrc[:args.maxin]
        
    ixs_nrc = nrctable.ix_sort_by_cols(['detector','aperture','filter','nrc_filename'],indices=ixs_nrc)    

    print('Selected alignment pairs:')
    nrctable.write(indices=ixs_nrc)
 
    if len(ixs_unmatched_fgsfiles)>0:
        print('\n!!!!! WARNING !!!!\nUnmatched FGS files:')
        fgstable.write(indices=ixs_unmatched_fgsfiles)


    do_it = input('Do you want to continue and analyse these alignment pairs [y/n]?  ')
    if do_it.lower() in ['y','yes']:
        pass
    elif do_it.lower() in ['n','no']:
        print('OK, stopping....')
        sys.exit(0)
    else:
        print(f'Hmm, \'{do_it}\' is neither yes or no. Don\'t know what to do, so stopping ....')
        sys.exit(0)


    for ix_nrc in ixs_nrc:
        print(f'\n############################################ {nrctable.t.loc[ix_nrc,"detector"]}\n',nrctable.t.loc[ix_nrc,"nrc_filename"],nrctable.t.loc[ix_nrc,"fgs_filename"])
        m = re.search('^jw(\d\d\d\d\d)',os.path.basename(nrctable.t.loc[ix_nrc,"nrc_filename"]))
        if m is not None:
            progID = m.groups()[0]
        else:
            progID = None
        nrctable.t.loc[ix_nrc,'progID']=progID
        
        fpa2fpa.calc_trg_v2v3info(nrctable.t.loc[ix_nrc,"fgs_filename"],
                                  nrctable.t.loc[ix_nrc,"nrc_filename"],
                                  args.siaf_file,
                                  args.v2v3refvalues)

        sys.exit(0)

        ###### Add the results to the  nircam table
        nrctable.add2row(ix_nrc,{'NRC_V2':v2v3list.t.loc[nrc_ix_0,"v2"],
                                 'NRC_V3':v2v3list.t.loc[nrc_ix_0,"v3"],
                                 'FGS_V2':v2v3list.t.loc[fgs_ix_0,'v2'],
                                 'FGS_V3':v2v3list.t.loc[fgs_ix_0,'v3'],
                                 'FGS_V3IdlYAngle':fgs_V3IdlYAngle,
                                 'V2Ref':v2v3list.t.loc[nrc_ix_0,"v2rot"],
                                 'V3Ref':v2v3list.t.loc[nrc_ix_0,"v3rot"],
                                 'V3IdlYAngle':nrc_V3IdlYAngle
                                 })


        

        
    ################################################
    # save the nrctable
    ################################################
    outcols = ['nrc_filename','progID','aperture','filter','NRC_V2','NRC_V3',
               'FGS_V2','FGS_V3','FGS_V3IdlYAngle',
               'V2Ref','V3Ref','V3IdlYAngle']
    nrctable.write(indices=ixs_nrc,columns=outcols)  
    if args.savetable:
        print(f'Saving results table to {args.savetable}')
        makepath4file(args.savetable)
        nrctable.write(filename=args.savetable,indices=ixs_nrc,columns=outcols)
        
    ################################################
    # calculate the 3-sigma clipped mean of the parameters and save them
    ################################################
    mean_results = calc_averages(nrctable,indices=ixs_nrc)
    #mean_results = calc_averages(nrctable)
    mean_results.write()
    if args.savetable:
        outputfilename = re.sub('\.txt','.mean.txt',args.savetable)
        if outputfilename == args.savetable:
            raise RuntimeError(f'could not determine outputfilename from args.savetable={args.savetable}')
        print(f'Saving averages into {outputfilename}')
        mean_results.write(outputfilename)
