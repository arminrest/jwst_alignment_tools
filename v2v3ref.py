#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 12:41:05 2023

@author: arest
"""

import argparse,glob,re,sys,os
from pdastro import pdastroclass,makepath4file,unique,AnotB,AorB,AandB,rmfile
import pandas as pd

class v2v3ref(pdastroclass):
    def __init__(self):
        pdastroclass.__init__(self)
        
    def load_xml(self,filename):
        
    
    def load_v2v3ref(self,filename,**kwargs):
        