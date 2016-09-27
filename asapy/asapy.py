import argparse
import logging
import sys
import os
from collections import namedtuple

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plottingscripts.plotting.scatter import plot_scatter_plot

from autofolio.data.aslib_scenario import ASlibScenario 

from asapy.perf_analysis.perf_analysis import PerformanceAnalysis

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
        self.logger = logging.getLogger("ASAPy")
        
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
            raise ValueError("Please first read in Scenario data; use scenario input or csv input")
        
        pa = PerformanceAnalysis(output_dn=self.output_dn,
                            scenario=self.scenario)
        # generate scatter plots
        scatter_plots = pa.scatter_plots()
        
        # generate correlation plot
        correlation_plot = pa.correlation_plot()
        
        # get shapley values
        df_contributions = pa.get_contribution_values()
        
