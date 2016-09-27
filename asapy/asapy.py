import argparse
import logging
import sys
from collections import namedtuple


from autofolio.data.aslib_scenario import ASlibScenario 
from dask.dataframe.io import csv_defaults

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "MIT"
__email__ = "lindauer@cs.uni-freiburg.de"

class ASAPy(object):
    
    def __init__(self,
                 output_dn:str=".",):
        '''
        Constructor
        
        Arguments
        ---------
        output_dn:str
            output directory name
        '''
        
        self.scenario = None
        self.output_dn = output_dn
        
    def read_scenario_ASlib(self, scenario_dn:str):
        '''
        Read scenario from ASlib format
        
        Arguments
        ---------
        scenario_dn: str
            Scenario directory name 
        '''
        
        self.scenario = ASlibScenario()
        self.scenario.read_scenario(dn=scenario_dn)
        
    def read_scenario_CSV(self, csv_data: namedtuple):
        '''
        Read scenario from ASlib format
        
        Arguments
        ---------
        csv_data: namedtuple
            namedtuple with the following fields: "perf_csv", "feat_csv", "obj", "cutoff", "maximize"
        '''
        self.scenario = ASlibScenario()
        self.scenario.read_from_csv(perf_fn=csv_data.perf_csv, 
                               feat_fn=csv_data.feat_csv,
                               objective=csv_data.obj,
                               runtime_cutoff=csv_data.cutoff,
                               maximize=csv_data.maximize)
    def main(self):
        '''
            main method
        '''
        
        if self.scenario is None:
            raise ValueError("Please first read in Scenario data")
        
        
        
        
        
        
        
        
        
        
        
        
        
        