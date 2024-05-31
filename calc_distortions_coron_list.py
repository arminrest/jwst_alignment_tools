#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 13:55:35 2024

@author: arest
"""

import argparse,re,sys,os,random
from calc_distortions import calc_distortions_class
from pdastro import makepath,rmfile,pdastroclass,AnotB,AorB,unique,makepath4file
from calc_distortions_list import calc_distortions_list_class
from calc_distortions_coron import calc_distortions_coron_class

class calc_distortions_coron_list_class(calc_distortions_list_class,calc_distortions_coron_class):
    def __init__(self):
        calc_distortions_list_class.__init__(self)
        calc_distortions_coron_class.__init__(self)

if __name__ == '__main__':
    
    distortions = calc_distortions_coron_list_class()

    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('input_filepatterns', nargs='+', type=str, help='list of input file(pattern)s. These get added to input_dir if input_dir is not None')
    parser.add_argument('--apertures', default=None, nargs='+', type=str, help='list of aperture names, e.g. nrca1_full')
    parser.add_argument('--filters', default=None, nargs='+', type=str, help='list of filter names, e.g. f200w.')
    parser.add_argument('--pupils', default=None, nargs='+', type=str, help='list of pupil names, e.g. clear. ')
    parser.add_argument('--summaryfilename', default='distortion_summary.txt', help='filename that contains a summary of the distortion fits. If filename has not path, it is saved in the output directory (default=%(default)s).')


    parser = distortions.define_optional_arguments(parser=parser)

    args = parser.parse_args()
    
    distortions.verbose=args.verbose
    distortions.savecoeff=not (args.skip_savecoeff)
    distortions.showplots=args.showplots
    distortions.saveplots=args.saveplots

    # define the x/y columns and phot cat suffix based on what photometry is used
    distortions.define_xycols(xypsf=args.xypsf,xy1pass=args.xy1pass,date4suffix=args.date4suffix)

    # get the coron info table!
    distortions.load_coron_info(args.coron_info_filename)
    
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
