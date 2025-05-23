#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 12:19:04 2022

@author: ahilbert, arest

This module contains functions to create NIRCAM reference files, using distortion
information in pysiaf (rather then the previous version, which used excel spreadsheets)

NIRCAM model:
im.meta.instrument.name :'NIRCAM'
im.meta.instrument.channel : 'SHORT'
im.meta.instrument.module : 'B'
im.meta.instrument.detector : 'NRCB1'
im.meta.exposure.type : 'NRC_IMAGE'

???
im.meta.instrument.pupil : 'FLAT' (would be GRISMR or GRISMC for slitless)

Transform Paths for Imaging mode:

science --> ideal --> V2V3
V2V3 --> ideal --> science

Where the "science" frame has units of distorted pixels. The "ideal" frame
is distortion-free and is the distance in arcseconds from 0,0. V2/V3 is also
in units of arcseconds from the refrence pixel.


"""
from asdf import AsdfFile
from astropy.modeling.models import Box1D, Const1D, Polynomial2D, Mapping, Shift
import astropy.units as u
from jwst.datamodels import DistortionModel
from stdatamodels import util
import numpy as np
import pysiaf
import argparse,glob,re,sys,os

# pdastroclass is wrapper around pandas.
from pdastro import pdastroclass,makepath4file,unique,AnotB,AorB,AandB,rmfile
from pandas.core.dtypes.common import is_string_dtype
from v2v3ref import v2v3refclass


history_mainentry='distortion coefficients'

class coeffs2asdf(pdastroclass):
    def __init__(self):
        pdastroclass.__init__(self)

        self.imtable = None
        self.fitsummary = None

        self.verbose=0

        self.aperture_col = 'AperName'
        self.siaf_name_col = 'siaf_index'


        # metadata is used depending on instrument/aperture.
        # Right now only NIRCam LW and SW are implemented.
        self.instrument = None
        self.aperture = None
        # Camera, e.g. 'NRC_SW','NRC_LW'
        self.camera=None
        # channel can be 'SHORT' or 'LONG'
        self.channel=None
        # detector can be NRC1-5
        self.detector=None
        # subarray mode. If the aperture is 'FULL', then this is set to ['GENERIC']
        self.subarr=None
        # module, A or B for NIRCam
        self.module=None

        self.metadata = {}

        # IMAGING metadata-----------------------------------------------
        self.metadata['imaging_pupil']={}
        #self.metadata['imaging_pupil']['NRC_SW']=['CLEAR', 'F162M', 'F164N', 'GDHS0', 'GDHS60', 'WLM8', 'WLP8', 'PINHOLES', 'MASKIPR', 'FLAT']
        #self.metadata['imaging_pupil']['NRC_LW']=['CLEAR', 'F323N', 'F405N', 'F466N', 'F470N', 'PINHOLES', 'MASKIPR', 'GRISMR', 'GRISMC', 'FLAT']
        self.metadata['imaging_pupil']['NRC_SW']=['CLEAR', 'GDHS0', 'GDHS60', 'WLM8', 'WLP8', 'PINHOLES', 'MASKIPR', 'FLAT']
        self.metadata['imaging_pupil']['NRC_LW']=['CLEAR', 'PINHOLES', 'MASKIPR', 'GRISMR', 'GRISMC', 'FLAT']
        # NIRCam mask
        self.metadata['imaging_pupil']['FULL_WEDGE_RND']=['MASKRND']
        self.metadata['imaging_pupil']['FULL_WEDGE_BAR']=['MASKBAR']
        #self.metadata['imaging_pupil']['']=['MASKBAR']

        #self.filters_with_distortions = ['F070W','F150W','F200W','F277W','F356W','F444W','F210M','F335M']
        self.metadata['imaging_filter']={}
        #NIRCam mapping
        self.metadata['imaging_filter']['NIRCAM']={}
        #self.metadata['imaging_filter']['NIRCAM']['F070W']=['F070W','F090W']

        #self.metadata['imaging_filter']['NIRCAM']['F210M']=['F182M','F187N','F210M','F212N']
        #self.metadata['imaging_filter']['NIRCAM']['F150W']=['F150W','F115W','F140M','F150W2']
        #self.metadata['imaging_filter']['NIRCAM']['F335M']=['F335M']
        #self.metadata['imaging_filter']['NIRCAM']['F356W']=['F322W2','F356W','F360M']

        #self.metadata['imaging_filter']['NIRCAM']['F200W']=['F200W']
        #self.metadata['imaging_filter']['NIRCAM']['F277W']=['F250M','F277W','F300M']
        #self.metadata['imaging_filter']['NIRCAM']['F444W']=['F410M','F430M','F444W','F460M','F480M']

        # This applies for images with MASKs in the pupil
        self.metadata['imaging_filter']['NIRCAMMASK']={}
        self.metadata['imaging_filter']['NIRCAMMASK']['F210M']=['F200W','F182M','F187N','F210M','F212N']
        # F405N F466N F470N cannot be added here
        self.metadata['imaging_filter']['NIRCAMMASK']['F335M']=['F250M','F300M','F322W2','F356W','F360M','F335M','F410M','F430M','F444W','F460M','F480M']


        # EXPTYPE metadata-----------------------------------------------
        self.metadata['exptype']={}
        """
        self.metadata['exptype']['NRC_SW']=['NRC_IMAGE', 'NRC_TSIMAGE', 'NRC_FLAT', 'NRC_LED',
                                            'NRC_WFSC', 'NRC_TACQ', 'NRC_TACONFIRM', 'NRC_FOCUS',
                                            'NRC_DARK', 'NRC_WFSS', 'NRC_TSGRISM', 'NRC_GRISM']
        self.metadata['exptype']['NRC_LW']=AandB(self.metadata['exptype']['NRC_SW'],['NRC_WFSS', 'NRC_TSGRISM', 'NRC_GRISM'],keeporder=True)
        """


        self.metadata['exptype']['NRC_SW']=['NRC_DARK','NRC_FLAT','NRC_FOCUS','NRC_GRISM','NRC_IMAGE','NRC_LED','NRC_TACONFIRM','NRC_TACQ','NRC_TSGRISM','NRC_TSIMAGE','NRC_WFSC','NRC_WFSS']
        self.metadata['exptype']['NRC_LW']=self.metadata['exptype']['NRC_SW']

        self.metadata['exptype']['NIS']=['NIS_IMAGE', 'NIS_AMI', 'NIS_WFSS', 'NIS_TACQ', 'NIS_FOCUS', 'NIS_TACONFIRM']

        # subarray metadata-----------------------------------------------
        self.metadata['subarr']={}
        self.metadata['subarr']['FULL']=['GENERIC']
        self.metadata['subarr']['FULL_WEDGE_RND']=['GENERIC']
        self.metadata['subarr']['FULL_WEDGE_BAR']=['GENERIC']
        # NIRISS
        self.metadata['subarr']['FULLP']=['GENERIC']

    def define_options(self,parser=None,usage=None,conflict_handler='resolve'):
        if parser is None:
            parser = argparse.ArgumentParser(usage=usage,conflict_handler=conflict_handler)

        parser.add_argument('coeff_filepatterns', nargs='+', type=str, default=None, help='list of coefficient file(pattern)s')

        parser.add_argument('--siaf_filename',default=None, help='pass the siaf xml file. This can be used to pass new alignments etc (pattern)s')
        parser.add_argument('-v','--verbose', default=0, action='count')

        return(parser)

    def get_instrument_info(self,aperture=None,aperture_col=None):
        """
        This routine parses the aperture string to populate the following info:
        self.instrument (e.g., NIRCAM)
        self.aperture (e.g., NRCB5_FULL)
        self.camera (e.g., NRC_SW,NRC_LW: this is used to obtain the correct list of pupils, exptype etc from self.metadata)
        self.channel (e.g. SHORT, LONG)
        self.detector (e.g., NRCB5)
        self.subarr (e.g. FULL, SUB160, SUB320, SUB640, GRISM_F322W2)
        self.module (e.g. A, B)

        Parameters
        ----------

        aperture: str
            full aperture name, e.g. NRCA5_FULL
            if None, then the unique string in the column aperture_col is used
            if aperture_col is None, then the default self.aperture_col is used
            The aperture is passed to self.


        Returns
        -------
        None.

        """

        self.instrument = None
        self.aperture = None
        self.camera=None
        self.channel=None
        self.detector=None
        self.subarr=None
        self.module=None

        if aperture is None:
            if aperture_col is None:
                aperture_col = self.aperture_col
            AperNameList = unique(self.t[aperture_col])
            if len(AperNameList)!=1:
                raise RuntimeError(f'AperNameList={AperNameList}, only exactly one entry allowed!')
            self.aperture = AperNameList[0].upper()
        else:
            self.aperture = aperture.upper()

        m = re.search('(^[a-zA-Z0-9]+)_(.*)',self.aperture)
        if m is None:
            raise RuntimeError(f'Could not get detector and aperture from {self.aperture}')
        else:
            self.detector, self.subarr = m.groups()
        self.detector = self.detector.upper()
        self.subarr = self.subarr.upper()

        if re.search('^NRC[AB]\d$',self.detector):
            self.instrument = "NIRCAM"
            if int(self.detector[-1]) < 5:
                self.camera = 'NRC_SW'
                self.channel= 'SHORT'
            elif int(self.detector[-1]) == 5:
                self.camera = 'NRC_LW'
                self.channel= 'LONG'
            else:
                raise RuntimeError(f'BUG! detector {self.detector} not in NRC?1-5!')
            self.module = self.detector[-2]
        elif self.detector.upper() == 'NIS':
            self.instrument = "NIRISS"
            self.camera = "NIS"
            if self.subarr == 'CEN': self.subarr = 'FULLP'
        else:
            raise RuntimeError(f'detector {self.detector} not implemented yet!')

#        if subarray=='FULL':
#            self.subarr = ['GENERIC']
#        else:
#            raise RuntimeError('subarray {subarray} not implemented yet!')

        if self.verbose: print(f'APERTURE: {self.aperture}\n CAMERA: {self.camera}\n CHANNEL: {self.channel}\n DETECTOR: {self.detector}\n SUBARR: {self.subarr}\n')

        return(None)

    def load_coeff_file(self,filename):
        if self.verbose: print(f'Loading {filename}')
        self.filename = filename
        # read the file
        #frames[counter] = pd.read_csv(filename,sep=',',skipinitialspace=True,comment='#')

        firstline = open(filename,'r').readline()
        if re.search('^\#',firstline):
            #print('jwst_distortion.py format!')
            self.load(filename,sep=',',skipinitialspace=True,comment='#',delim_whitespace=False)

            # some of the column names have spaces, removed them!!
            mapper={}
            for col in self.t.columns:
                if re.search('\s+',col):
                    mapper[col]=re.sub('\s+','',col)
                if is_string_dtype(self.t[col]):
                    self.t[col] = self.t[col].str.strip()
            self.t = self.t.rename(columns=mapper)
        elif re.search('^\s*AperName\s+siaf_index',firstline):
            #print('pandas table format!')
            self.load(filename,verbose=1)
        else:
            raise RuntimeError(f'Something is wrong, cannot understand input file {filename} format')

        # just some sanity check
        if self.siaf_name_col not in self.t.columns:
            print(f'Something is wrong!! {self.siaf_name_col} not a column table!')
            self.write()
            raise RuntimeError(f'trouble understanding table {filename}')

        if self.verbose>1:
            self.write()

        # add filename to table
        self.t['filename']=filename

        # make sure these columns are integer.
        for col in ['siaf_index','exponent_x','exponent_y']:
            if col in self.t.columns:
                self.t[col]=self.t[col].astype('int')

        if re.search('polycoeff\.txt$',self.filename):
            imtablefilename = re.sub('polycoeff\.txt$','images.txt',self.filename)
            if not os.path.isfile(imtablefilename):
                raise RuntimeError(f'The suffix of the coefficient file is polycoeff.txt, and therefore we expect an image list file of the name {imtablefilename} which does not exist. This file is only used for meta data, so not essential, but still exiting...')
            self.imtable=pdastroclass()
            self.imtable.load(imtablefilename,verbose=2)

            fitsummaryfilename = re.sub('\.txt$','.fitsummary.txt',self.filename)
            if os.path.isfile(fitsummaryfilename):
                self.fitsummary = pdastroclass()
                self.fitsummary.load(fitsummaryfilename,verbose=1)
                self.fitsummary.write()
            else:
                self.fitsummary = None
                

        else:
            self.imtable = None
            self.fitsummary = None

    def get_refpixold(self,siaf_instance, apername,instrument):
        """Return the reference location within the given aperture

        Parameters
        ----------
        siaf_instance : pysiaf.Siaf('nircam')
        """

        siaf_aperture = siaf_instance[apername]
        xref = siaf_aperture.XSciRef
        yref = siaf_aperture.YSciRef
        #return Shift(-xref) & Shift(-yref)
        print('\n'
              '#############################################################################\n'
              '### WARNING: we assume that the lower left corner\'s pixel center is x,y=0,0!\n'
              '### If this is not correct, you need to fix it!!!\n'
              '#############################################################################\n')
        return (Shift(-xref) & Shift(-yref))

    def get_refpix(self,siaf_aperture):
        """Return the reference location within the given aperture

        Parameters
        ----------
        siaf_aperture : aperture siaf instance
        """

        xref = siaf_aperture.XSciRef
        yref = siaf_aperture.YSciRef
        #return Shift(-xref) & Shift(-yref)
        print('\n'
              '#############################################################################\n'
              '### WARNING: we assume that the lower left corner\'s pixel center is x,y=0,0!\n'
              '### If this is not correct, you need to fix it!!!\n'
              '#############################################################################\n')
        return (Shift(-xref) & Shift(-yref))

        # Check to see if we can use coeffs from a subarray aperture
        # and have them apply to all apertures. Need to update the shift
        # in that case by adding the distance from detector (0, 0) to the
        # lower left corner of the aperture
        #siaf = pysiaf.Siaf('nircam')
        #xc, yc = sci_subarray_corners(instrument, apername, siaf=siaf_instance, verbose=False)
        #llx, urx = xc
        #lly, ury = yc
        #print('Lower left corner x and y:', llx, lly)
        #sys.exit(0)

        #return Shift(-xref-llx) & Shift(-yref-lly)

    def get_coeff_dict(self, coeffname, siaf_index_col = 'siaf_index'):
        """Convert the one set of coefficients output into
        the proper dictionary format to be then saved in the reference file.

        Parameters
        ----------
        coeffname : str
           e.g. 'Sci2IdlX'

        siaf_index_col: str
            column name with the siaf index

        Returns
        -------
        coeffs : dict
            Dictionary of one set of coefficients (e.g. Sci2IdlX) from the input table.
            Keys list the polynomial order numbers (e.g. c3_4)
        """
        coeffs = {}
        for ix in self.t.index.values:
            i = int(f'{self.t.loc[ix,siaf_index_col]:02d}'[0])
            j = int(f'{self.t.loc[ix,siaf_index_col]:02d}'[1])
            key = f'c{i-j}_{j}'
            coeffs[key] = self.t.loc[ix,coeffname]

        return coeffs

    def v2v3_model(self,from_sys, to_sys, par, angle):
        """
        Creates an astropy.modeling.Model object
        for the undistorted ("ideal") to V2V3 coordinate translation
        """
        if from_sys != 'v2v3' and to_sys != 'v2v3':
            raise ValueError("This function is designed to generate the transformation either to or from V2V3.")

        # Cast the transform functions as 1st order polynomials
        xc = {}
        yc = {}
        if to_sys == 'v2v3':
            xc['c1_0'] = par * np.cos(angle)
            xc['c0_1'] = np.sin(angle)
            yc['c1_0'] = (0.-par) * np.sin(angle)
            yc['c0_1'] = np.cos(angle)

        if from_sys == 'v2v3':
            xc['c1_0'] = par * np.cos(angle)
            xc['c0_1'] = par * (0. - np.sin(angle))
            yc['c1_0'] = np.sin(angle)
            yc['c0_1'] = np.cos(angle)

        #0,0 coeff should never be used.
        xc['c0_0'] = 0
        yc['c0_0'] = 0

        xmodel = Polynomial2D(1, **xc)
        ymodel = Polynomial2D(1, **yc)

        return xmodel, ymodel

    def get_v2v3ref(self,siaf_instance):
        """Return v2 and v3 at the reference location
        These are arcsec in the SIAF file. Convert to degrees

        Parameters
        ----------
        siaf_instance : pysiaf.Siaf[aperture]

        Returns
        -------
        v2 : astropy.modeling.models.Shift
            Shift between V2 at reference location and V2=0

        v3 : astropy.modeling.models.Shift
            Shift between V3 at reference location and V3=0
        """
        v2ref = siaf_instance.V2Ref
        v3ref = siaf_instance.V3Ref
        return Shift(v2ref) & Shift(v3ref)


    def create_asdf_reference_for_distortion(self,
                                 aperture=None, aperture_col=None,
                                 filt=None,
                                 pupil=None,
                                 degree=None,
                                 exponent_col='exponent_x',
                                 siaf_filename=None,
                                 sci_filter=None,
                                 sci_pupil=None, sci_subarr=None, sci_exptype=None,
                                 history2add=[],
                                 author=None, descrip=None, pedigree=None,
                                 useafter=None):
        """
        Create an asdf reference file with all distortion components.


        Parameters
        ----------

        outname : str
            Name of output file.

        aperture: str
            full aperture name, e.g. NRCA5_FULL
            if None, then the unique string in the column aperture_col is used
            if aperture_col is None, then the default self.aperture_col is used
            The aperture is passed to self.get_instrument_info, which parses the aperture string to populate the following info:
            self.instrument (e.g., NIRCAM)
            self.aperture (e.g., NRCB5_FULL)
            self.camera (e.g., NRC_SW,NRC_LW: this is used to obtain the correct list of pupils, exptype etc from self.metadata)
            self.channel (e.g. SHORT, LONG)
            self.detector (e.g., NRCB5)
            self.subarr (e.g. FULL, SUB160, SUB320, SUB640, GRISM_F322W2)

        aperturecol: str
            if aperture_col is None, then the default self.aperture_col is used

        filt: str
            filter name.

        pupil: str
            pupil name. if 'clear', then sci_pupil is used

        degree: int
            degree of polynomial distortions
            if degree is None, then the max value in exponent_col is used

        exponent_col: str
            column used to determine the degree of polynomial distortions

        siaf_filename : str
            Name of SIAF file to use in place of the default SIAF version from pysiaf (can be txt or xml).
            If None, the default version in pysiaf will be used.

        sci_filter : list
            filter wheel values for which this distortion solution applies
            If None:
                if pupil is clear, then the mapping in self.metadata['imaging_filter']['NIRCAM'][filter]
                is used
                if pupil is a mask, then the mapping in self.metadata['imaging_filter']['NIRCAMMASK'][filter]
                is used

        sci_pupil : list
            Pupil wheel values for which this distortion solution applies
            If None, self.metadata['imaging_pupil'][self.camera] will be used.
            Only used if pupil='clear'

        sci_subarr : list
            List of subarray/aperture names to which this distortion solution applies
            If None,

        sci_exptype : list
            List of exposure types to which this distortion solution applies
            If None, self.metadata['exptype'][self.camera] will be used.

        history2add : list
            list of text to be added as a HISTORY entry in the output reference file

        author : str
            Value to place in the output file's Author metadata entry

        descrip : str
            Text to place in the output file's DECRIP header keyword

        pedgree : str
            Value to place in the output file's PEDIGREE header keyword

        useafter : str
            Value to place in the output file's USEAFTER header keyword (e.g. "2014-10-01T00:00:01")

        Examples
        --------

        """

        # this list will contain text to be added to the history, e.g., V2/V3ref used
        history=[]

        if degree is None:
            degree = int(self.t[exponent_col].max())
        print(f'Distortions are {degree}th degree polynomials')

        # this uses aperture to set self.instrument self.aperture self.camera self.channel self.detector self.subarr
        # if aperture is None, then the string in aperture_col in self.t is used.
        self.get_instrument_info(aperture=aperture,aperture_col=aperture_col)


        # if filter_pupil is None, check if the filter is in self.metadata['imaging_filter']['NIRCAM']
        # or self.metadata['imaging_filter']['NIRCAMMASK'] for corongraphy
        if self.instrument=='NIRISS':
            # NIRISS doesn't have a mapping yet
            pass
        elif self.instrument=='NIRCAM':
            if sci_filter is None:
                #if self.subarr in ['FULL_WEDGE_RND','FULL_WEDGE_BAR']:
                if pupil in ['maskrnd','maskbar']:
                    if filt.upper() in self.metadata['imaging_filter']['NIRCAMMASK']:
                        sci_filter = self.metadata['imaging_filter']['NIRCAMMASK'][filt.upper()]
                else:
                    if filt.upper() in self.metadata['imaging_filter']['NIRCAM']:
                        sci_filter = self.metadata['imaging_filter']['NIRCAM'][filt.upper()]
        else:
            raise RuntimeError(f'instrument {self.instrument} not supported yet')


        # if sci_pupil is None, use the default defined in self.metadata['imaging_pupil']
        # for the given self.camera. self.camera gets defined in self.get_instrument_info
        if self.instrument=='NIRISS':
            # NIRISS doesn't have imaging_pupil
            pass
        elif self.instrument=='NIRCAM':
            if pupil is not None:
                if  pupil.lower() == 'clear':
                    if sci_pupil is None:
                        if self.subarr in self.metadata['imaging_pupil']:
                            sci_pupil = self.metadata['imaging_pupil'][self.subarr]
                        elif self.camera in self.metadata['imaging_pupil']:
                            sci_pupil = self.metadata['imaging_pupil'][self.camera]
                        else:
                            raise RuntimeError('not able to determine sci_pupil')
        #        else:
        #            sci_pupil = [pupil.upper()]

        else:
            raise RuntimeError(f'instrument {self.instrument} not supported yet')

        # if sci_exptype is None, use the default defined in self.metadata['exptype']
        # for the given self.camera. self.camera gets defined in self.get_instrument_info
        if sci_exptype is None:
            if self.camera in self.metadata['exptype']:
                sci_exptype = self.metadata['exptype'][self.camera]
            else:
                raise RuntimeError(f'camera {self.camera} not in {self.metadata["exptype"]}')

        # if sci_subarr is None, use the default defined in self.metadata['subarr']
        # for the given self.subarr. self.subarr gets defined in self.get_instrument_info
        if sci_subarr is None:
            if self.subarr in self.metadata['subarr']:
                sci_subarr = self.metadata['subarr'][self.subarr]
            else:
                raise RuntimeError(f'subarray {self.subarr} not in {self.metadata["subarr"]}')

#        degree = 5  # distotion in pysiaf is a 5th order polynomial
#        numdet = detector[-1]
#        module = detector[-2]
#        channel = 'SHORT'
#        if numdet == '5':
#            channel = 'LONG'

#        full_aperture = detector + '_' + self.subarr

        # Get Siaf instance for detector/aperture
        inst_siaf = pysiaf.Siaf(self.instrument.lower())
        aper_siaf = inst_siaf[self.aperture]

        # needed for ideal -> v2v3 or v2v3 -> ideal model
        parity = aper_siaf.VIdlParity

        if siaf_filename is None:

            if re.search('^mask',pupil) is not None:
                raise RuntimeError(f'This image has pupil={pupil}, therefore cannot just use siaf V2/V3ref since it has a wedge in it!')

            V3IdlYAngle = aper_siaf.V3IdlYAngle
            V2Ref = aper_siaf.V2Ref
            V3Ref = aper_siaf.V3Ref
            s = f'Getting (V2Ref, V3Ref, V3IdlYAngle) = ({V2Ref},{V3Ref},{V3IdlYAngle}) from default SIAF version in pysiaf.'
            print(s)
            history.append(s)
        else:
            v2v3ref = v2v3refclass()
            v2v3ref.load_v2v3ref(siaf_filename)
            v2v3ref.write()
            print('fff',v2v3ref.t.columns)
            print(pupil)
            print(self.aperture)
            (V2Ref, V3Ref, V3IdlYAngle,ix) = v2v3ref.get_v2v3info(self.aperture,pupilname=pupil) #,filtername=self.src_filter,pupilname=self.src_pupil)
            s = f'Getting (V2Ref, V3Ref, V3IdlYAngle) = ({V2Ref},{V3Ref},{V3IdlYAngle}) from {os.path.basename(siaf_filename)} for aperture {self.aperture.lower()}.'
            print(s)
            history.append(s)
            history.append(f'Using siaf={os.path.basename(siaf_filename)}')

        #siaf = inst_siaf[self.aperture]


        # Find the distance between (0,0) and the reference location
        xshift, yshift = self.get_refpix(aper_siaf)

        # convert the coefficients into dictionaries
        xcoeffs = self.get_coeff_dict('Sci2IdlX')
        ycoeffs = self.get_coeff_dict('Sci2IdlY')
        inv_xcoeffs = self.get_coeff_dict('Idl2SciX')
        inv_ycoeffs = self.get_coeff_dict('Idl2SciY')

        #print(xcoeffs,ycoeffs)

        # V3IdlYAngle and V2Ref, V3Ref should always be taken from the latest version
        # of SIAF, rather than the output of jwst_fpa. Separate FGS/NIRISS analyses must
        # be done in order to modify these values.
        v3_ideal_y_angle_radian = V3IdlYAngle * np.pi / 180.

        # *****************************************************
        # "Forward' transformations. science --> ideal --> V2V3
        #label = 'Sci2Idl'
        ##from_units = 'distorted pixels'
        ##to_units = 'arcsec'

        #xcoeffs, ycoeffs = get_distortion_coeffs(label, siaf)

        sci2idlx = Polynomial2D(degree, **xcoeffs)
        sci2idly = Polynomial2D(degree, **ycoeffs)

        # Get info for ideal -> v2v3 or v2v3 -> ideal model

        idl2v2v3x, idl2v2v3y = self.v2v3_model('ideal', 'v2v3', parity, v3_ideal_y_angle_radian)

        # Finally, we need to shift by the v2,v3 value of the reference
        # location in order to get to absolute v2,v3 coordinates
        #v2shift, v3shift = self.get_v2v3ref(siaf)
        v2shift, v3shift = Shift(V2Ref) & Shift(V3Ref)

        # *****************************************************
        # 'Reverse' transformations. V2V3 --> ideal --> science
        #label = 'Idl2Sci'
        ##from_units = 'arcsec'
        ##to_units = 'distorted pixels'

        #xcoeffs, ycoeffs = get_distortion_coeffs(label, siaf)

        idl2scix = Polynomial2D(degree, **inv_xcoeffs)
        idl2sciy = Polynomial2D(degree, **inv_ycoeffs)

        # Get info for ideal -> v2v3 or v2v3 -> ideal model
        #parity = siaf.VIdlParity
        #v3_ideal_y_angle_radian = siaf.V3IdlYAngle * np.pi / 180.
        v2v32idlx, v2v32idly = self.v2v3_model('v2v3', 'ideal', parity, v3_ideal_y_angle_radian)

        ##"Forward' transformations. science --> ideal --> V2V3
        #sci2idlx, sci2idly, sciunit, idlunit = read_siaf_table.get_siaf_transform(coefffile,self.aperture,'science','ideal', 5)
        #idl2v2v3x, idl2v2v3y = read_siaf_table.get_siaf_v2v3_transform(coefffile,self.aperture,from_system='ideal')

        ##'Reverse' transformations. V2V3 --> ideal --> science
        #v2v32idlx, v2v32idly = read_siaf_table.get_siaf_v2v3_transform(coefffile,self.aperture,to_system='ideal')
        #idl2scix, idl2sciy, idlunit, sciunit = read_siaf_table.get_siaf_transform(coefffile,self.aperture,'ideal','science', 5)

        # Now create a compound model for each with the appropriate inverse
        sci2idl = Mapping([0, 1, 0, 1]) | sci2idlx & sci2idly
        sci2idl.inverse = Mapping([0, 1, 0, 1]) | idl2scix & idl2sciy

        idl2v2v3 = Mapping([0, 1, 0, 1]) | idl2v2v3x & idl2v2v3y
        idl2v2v3.inverse = Mapping([0, 1, 0, 1]) | v2v32idlx & v2v32idly

        # Coronographic step
        if (self.fitsummary is not None) and ('y_transition' in self.fitsummary.t.columns) and isinstance(self.fitsummary.t.loc[0,'y_transition'],float):
            ### CHANGE BACK!!!!!!
            y_transition = self.fitsummary.t.loc[0,'y_transition']
            #y_transition = self.fitsummary.t.loc[0,'y_transition']-1023.5
            y_step_pixels = self.fitsummary.t.loc[0,'y_step_pixels']
            print(f'#######\ny_transition: {y_transition}\ny_step_pixels:{y_step_pixels} ')
            # Insert a Box1D model here to account for the offset introduced by
            # the COM along the top of the detector. Shift the pixels in the COM region upwards
            # in y. No shift in x.
            # Set the "center" of the box at the top of the detector, and make the width
            # equal to twice 2047 minus the y coordinate of the transition.
            
            ysize = aper_siaf.YSciSize-1.0

            box_width = (ysize + 1 - y_transition) * 2
            com_y = yshift + Box1D(amplitude=y_step_pixels, x_0=ysize, width=box_width)
            com_y.inverse = Box1D(amplitude=0. - y_step_pixels, x_0=ysize + 1 + yshift(0), width=box_width) + yshift.inverse 

            yshift = com_y


            core_model = sci2idl | idl2v2v3


            #yshift = Shift(-1023.5)

        else:
            core_model = sci2idl | idl2v2v3



        # Now add in the shifts to create the full model
        # including the shift to go from 0-indexed python coords to
        # 1-indexed

        # SIAF coords
        index_shift = Shift(1)

        model = index_shift & index_shift | (xshift & yshift) | core_model | v2shift & v3shift




        # Since the inverse of all model components are now defined,
        # the total model inverse is also defined automatically

        # Save using the DistortionModel datamodel
        d = DistortionModel(model=model, input_units=u.pix,
                            output_units=u.arcsec)


        #Populate metadata

        # Keyword values in science data to which this file should
        # be applied

        if sci_filter is not None:
            p_filter = ''
            for p in sci_filter:
                p_filter = p_filter + p + '|'
            d.meta.instrument.p_filter = p_filter

        if sci_pupil is not None:
            p_pupil = ''
            for p in sci_pupil:
                p_pupil = p_pupil + p + '|'
            d.meta.instrument.p_pupil = p_pupil


        if sci_subarr is not None:
            p_subarr = ''
            for p in sci_subarr:
                p_subarr = p_subarr + p + '|'
            d.meta.subarray.p_subarray = p_subarr

        if sci_exptype is not None:
            p_exptype = ''
            for p in sci_exptype:
                p_exptype = p_exptype + p + '|'
            d.meta.exposure.p_exptype = p_exptype

        # metadata describing the reference file itself
        d.meta.title = f'{self.instrument} Distortion'
        d.meta.instrument.name = self.instrument.upper()
        d.meta.instrument.module = self.module
        d.meta.instrument.channel = self.channel

        # In the reference file headers, we need to switch NRCA5 to
        # NRCALONG, and same for module B.
        detector=self.detector
        if self.instrument.lower() == 'nircam' and detector[-1] == '5':
            detector = detector[0:4] + 'LONG'
        d.meta.instrument.detector = detector
        d.meta.telescope = 'JWST'
        if self.subarr in ['FULL_WEDGE_RND','FULL_WEDGE_BAR']:
            d.meta.subarray.name = 'FULL'
        else:
            d.meta.subarray.name = self.subarr


        if pedigree is None:
            d.meta.pedigree = 'INFLIGHT'
            if self.imtable is not None:
                if 'date-obs' in self.imtable.t.columns:
                    dates = sorted(unique(self.imtable.t['date-obs']))
                    if len(dates)==1:
                        d.meta.pedigree += f' {dates[0]} {dates[0]}'
                    elif len(dates)>1:
                        d.meta.pedigree += f' {dates[0]} {dates[-1]}'
                    else:
                        raise RuntimeError(f'Bug? The image table {self.imtable.filename} exists, but no entries in column "date-obs"?')
                else:
                    print('####   !!!!! WARNING !!!! Did not find column "date-obs" in the image table!')
        else:
            if re.search('^DUMMY|^GROUND|^INFLIGHT',pedigree.upper()) is None:
#            if pedigree.upper() not in ['DUMMY', 'GROUND', 'INFLIGHT']:
                raise ValueError("Bad PEDIGREE value.")
            d.meta.pedigree = pedigree.upper()

        d.meta.reftype = 'DISTORTION'

        if author is None:
            author = os.path.basename(os.path.expanduser('~'))
        d.meta.author = author

        d.meta.litref = "https://github.com/arminrest/jwst_alignment_tools/blob/main/distortion2asdf.py"

        if descrip is None:
            d.meta.description = "This is a distortion correction reference file."
            if self.imtable is not None:
                if 'progID' in self.imtable.t.columns:
                    progIDs = sorted(unique(self.imtable.t['progID'].astype(str)))
                    d.meta.description += ' The following PIDs were use for the creation of this reference file: '+' '.join(progIDs)
                    #d.meta.description.append(' The following PIDs were use for the creation of this reference file: '+' '.join(progIDs))
                else:
                    print('####   !!!!! WARNING !!!! Did not find column "progID" in the image table!')
                if 'fullimage' in self.imtable.t.columns:
                    pass
                    #d.meta.description += 'Files used to create this reference file:'

        else:
            d.meta.description = descrip

       #d.meta.exp_type = exp_type
        if useafter is None:
            d.meta.useafter = "2022-01-01T00:00:01"
        else:
            d.meta.useafter = useafter

        # To be ready for the future where we will have filter-dependent solutions
        if filt is None:
            d.meta.instrument.filter = 'N/A'
        else:
            d.meta.instrument.filter = filt.upper()

        if pupil is None:
            d.meta.instrument.pupil = 'N/A'
        else:
            d.meta.instrument.pupil = pupil.upper()

        # Create initial HISTORY ENTRY
        sdict = {'name': 'distortion2asdf.py',
                 'author': author,
                 'homepage': 'https://github.com/arminrest/jwst_alignment_tools',
                 'version': '2.0'}

        #print('meta data: ',d.meta.instance)

        entry = util.create_history_entry(history_mainentry, software=sdict)
        d.history = [entry]
        if (history is not None):
            for entry in history:
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!',entry)
                #histentry = util.create_history_entry(history_2)
                d.history.append(util.create_history_entry(entry))
        if self.imtable is not None:
            if 'fullimage' in self.imtable.t.columns:
                d.history.append(util.create_history_entry('Files used to create this reference file:'))
                for ix in self.imtable.getindices():
                    filename = os.path.basename(self.imtable.t.loc[ix,'fullimage'])
                    d.history.append(util.create_history_entry(f'{filename}'))
            else:
                print('####   !!!!! WARNING !!!! Did not find column "fullimage" in the image table!')
        if self.fitsummary is not None:
            d.history.extend(self.fitsummary.t.to_string().split('\n'))

        print(d.history)
        #sys.exit(0)

        #entry = util.create_history_entry(history_entry, software=sdict)
        #d.history = [entry]


        #Create additional HISTORY entries
        #entry2 = util.create_history_entry(history_2)
        #d.history.append(entry2)


        return(d)


    def coefffile2adfs(self, filename,
                       filt=None,
                       pupil=None,
                       outname=None,
                       siaf_filename=None,
                       savemeta=True,
                       history2add=[],
                       author=None,
                       descrip=None,
                       pedigree=None,
                       useafter=None):
        # load the file
        self.load_coeff_file(filename)

        distcoeff = self.create_asdf_reference_for_distortion(filt=filt,
                                                              pupil=pupil,
                                                              siaf_filename=siaf_filename,
                                                              history2add=history2add,
                                                              author=author,
                                                              descrip=descrip,
                                                              pedigree=pedigree,
                                                              useafter=useafter)

        if outname is not None:
            print('Saving',outname)
            distcoeff.save(outname)
            #distcoeff.save('~/delme/vvv.asdf')
            print(f'Distortion coefficients saved to {outname}')
            if savemeta:
                metaoutname = re.sub('\.asdf$','',outname)
                metaoutname +='.meta.txt'
                rmfile(metaoutname)
                print('meta data: \n',distcoeff.meta.instance)
                metalist = '\n'.join([f'{k}:{distcoeff.meta.instance[k]}' for k in distcoeff.meta.instance])
                histlist = '\n'.join([f'HISTORY:{k}' for k in distcoeff.history])
                #print(metalist)
                #print(histlist)
                metalist+='\n'+histlist
                open(metaoutname,'w').writelines(metalist)
                print(f'Distortion coefficients meta data saved to {metaoutname}')
            #    open()
        return(distcoeff)


if __name__ == '__main__':

    coeffs = coeffs2asdf()
    parser = coeffs.define_options()
    args = parser.parse_args()

    coeffs.verbose=args.verbose

    filenames=[]
    for filepattern in args.coeff_filepatterns:
        filenames.extend(glob.glob(filepattern))
    filenames.sort()

    for filename in filenames:
        if re.search('\.txt$',filename) is not None:
            outname = re.sub('\.txt$','.asdf',filename)
        else:
            outname = filename +'' + '.asdf'

        # get the filter!
        m = re.search('_(f\d+[wmn2]+)_(\w+)\.polycoeff\.txt',filename.lower())
        #if m is None:
        #    m = re.search('_(f\d\d\dw2)_',filename.lower())
        if m is not None:
            (filt,pupil) = m.groups()
        else:
            filt=pupil=None
            raise RuntimeError(f'####### !!!!! WARNING!!! Could not determine the filter from filename {filename.lower()}!')


        coeffs.coefffile2adfs(filename,outname=outname,filt=filt,pupil=pupil,siaf_filename=args.siaf_filename)
