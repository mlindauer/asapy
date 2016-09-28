import argparse
import logging
import sys
import os
from collections import namedtuple, OrderedDict

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plottingscripts.plotting.scatter import plot_scatter_plot

from autofolio.data.aslib_scenario import ASlibScenario 

from asapy.perf_analysis.perf_analysis import PerformanceAnalysis
from asapy.out_builder.html_builder import HTMLBuilder

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
        
        data = OrderedDict()
        
        # performance analysis
        data["Performance Analysis"] = OrderedDict()
        
        # generate scatter plots
        scatter_plots = pa.scatter_plots()
        data["Performance Analysis"]["Scatter Plots"] = {"tooltip": "Scatter plot to compare the performance of two algorithms on all instances -- each dot represents one instance."}
        for plot_tuple in scatter_plots:
            key = "%s vs %s" %(plot_tuple[0], plot_tuple[1])
            data["Performance Analysis"]["Scatter Plots"][key] = {"figure": plot_tuple[2]} 
        
        # generate correlation plot
        correlation_plot = pa.correlation_plot()
        data["Performance Analysis"]["Correlation Plot"] = {"tooltip": "Correlation based on Spearman Correlation Coefficient between all algorithms and clustered with Wards hierarchical clustering approach. Darker fields corresponds to a larger correlation between the algorithms.",
                                                             "figure":correlation_plot}
        
        
        # get shapley values
        df_contributions = pa.get_contribution_values()
        data["Performance Analysis"]["Contribution of Algorithms"] = {"tooltip": "Contribution of each algorithm wrt to its average performance across all instances, the marginal contribution to the virtual best solver (VBS, aka oracle) (i.e., how much decreases the VBS performance by removing the algorithm; higher value correspond to more importance), and Shapley values (marginal contribution across all possible subsets of portfolios; again higher values corresponds to more importance).",
                                                             "table":df_contributions.to_html()}
        
        # get cdf plot
        cdf_plot = pa.get_cdf_plots()
        data["Performance Analysis"]["CDF Plot"] = {"tooltip": "Cumulative Distribution function (CDF) plots. At each point x (e.g., running time cutoff), how many of the instances (in percentage) can be solved. Better algorithms have a higher curve.",
                                                             "figure":cdf_plot}
        
        
        # get violin plot
        violion_plot = pa.get_violin_plots()
        data["Performance Analysis"]["Violin Plot"] = {"tooltip": "Violin plots to show the performance distribution of each algorithm",
                                                             "figure":violion_plot}
        
        # get box plot
        box_plot = pa.get_box_plots()
        data["Performance Analysis"]["Box Plot"] = {"tooltip": "Box plots to show the performance distribution of each algorithm",
                                                             "figure":box_plot}
        
        self.create_html(data=data)
        
    def create_html(self, data:OrderedDict):
        '''
        create html report
        '''
        
        html_builder = HTMLBuilder(output_dn=self.output_dn,
                 scenario_name=self.scenario.scenario)
        html_builder.generate_html(data)
