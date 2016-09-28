import argparse
import logging
import sys
import os
from collections import namedtuple, OrderedDict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plottingscripts.plotting.scatter import plot_scatter_plot

from autofolio.data.aslib_scenario import ASlibScenario

from asapy.perf_analysis.perf_analysis import PerformanceAnalysis
from asapy.feature_analysis.feature_analysis import FeatureAnalysis
from asapy.out_builder.html_builder import HTMLBuilder

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "MIT"
__email__ = "lindauer@cs.uni-freiburg.de"


class ASAPy(object):

    def __init__(self,
                 output_dn: str=".",):
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

    def read_scenario_ASlib(self, scenario_dn: str):
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
            raise ValueError(
                "Please first read in Scenario data; use scenario input or csv input")

        data = OrderedDict()

        # meta data
        meta_data_df = self.get_meta_data()
        data["Meta Data"] = {
            "table": meta_data_df.to_html()
        }

        # performance analysis
        pa = PerformanceAnalysis(output_dn=self.output_dn,
                                 scenario=self.scenario)
        data["Performance Analysis"] = OrderedDict()
 
        # get box plot
        box_plot = pa.get_box_plots()
        data["Performance Analysis"]["Box Plot"] = {"tooltip": "Box plots to show the performance distribution of each algorithm",
                                                    "figure": box_plot}
 
        # get violin plot
        violion_plot = pa.get_violin_plots()
        data["Performance Analysis"]["Violin Plot"] = {"tooltip": "Violin plots to show the performance distribution of each algorithm",
                                                       "figure": violion_plot}
 
        # get cdf plot
        cdf_plot = pa.get_cdf_plots()
        data["Performance Analysis"]["CDF Plot"] = {"tooltip": "Cumulative Distribution function (CDF) plots. At each point x (e.g., running time cutoff), how many of the instances (in percentage) can be solved. Better algorithms have a higher curve.",
                                                    "figure": cdf_plot}
 
        # generate scatter plots
        scatter_plots = pa.scatter_plots()
        data["Performance Analysis"]["Scatter Plots"] = {
            "tooltip": "Scatter plot to compare the performance of two algorithms on all instances -- each dot represents one instance."}
        for plot_tuple in scatter_plots:
            key = "%s vs %s" % (plot_tuple[0], plot_tuple[1])
            data["Performance Analysis"]["Scatter Plots"][
                key] = {"figure": plot_tuple[2]}
 
        # generate correlation plot
        correlation_plot = pa.correlation_plot()
        data["Performance Analysis"]["Correlation Plot"] = {"tooltip": "Correlation based on Spearman Correlation Coefficient between all algorithms and clustered with Wards hierarchical clustering approach. Darker fields corresponds to a larger correlation between the algorithms.",
                                                            "figure": correlation_plot}
 
        # get shapley values
        df_contributions = pa.get_contribution_values()
        data["Performance Analysis"]["Contribution of Algorithms"] = {"tooltip": "Contribution of each algorithm wrt to its average performance across all instances, the marginal contribution to the virtual best solver (VBS, aka oracle) (i.e., how much decreases the VBS performance by removing the algorithm; higher value correspond to more importance), and Shapley values (marginal contribution across all possible subsets of portfolios; again higher values corresponds to more importance).",
                                                                      "table": df_contributions.to_html()}

        # feature analysis
        data["Feature Analysis"] = OrderedDict()
        fa = FeatureAnalysis(output_dn=self.output_dn,
                             scenario=self.scenario)

        #box and violin plots
        name_plots = fa.get_box_violin_plots()
        data["Feature Analysis"]["Violin and Box Plots"] = OrderedDict({
            "tooltip": "Violin and Box plots to show the distribution of each instance feature. We removed NaN from the data."})
        for plot_tuple in name_plots:
            key = "%s" % (plot_tuple[0])
            data["Feature Analysis"]["Violin and Box Plots"][
                key] = {"figure": plot_tuple[1]}

        # correlation plot
        correlation_plot = fa.correlation_plot()
        data["Feature Analysis"]["Correlation Plot"] = {"tooltip": "Correlation based on Pearson Correlation Coefficient between all features and clustered with Wards hierarchical clustering approach. Darker fields corresponds to a larger correlation between the features.",
                                                        "figure": correlation_plot}
 
        # feature importance
        importance_plot = fa.feature_importance()
        data["Feature Analysis"]["Feature Importance"] = {"tooltip": "Using the approach of SATZilla'11, we train a cost-sensitive random forest for each pair of algorithms and average the feature importance (using gini as splitting criterion) across all forests. We show only the 15 most important features",
                                                        "figure": importance_plot}

        # cluster instances in feature space
        cluster_plot = fa.cluster_instances()
        data["Feature Analysis"]["Clustering"] = {"tooltip": "Similar to ISAC, we use a k-means to cluster the instances in the feature space. As pre-processing, we use standard scaling and pca to 2 dimensions. To guess the number of clusters, we use the silhouette score on the range of 2 to 12 in the number of clusters",
                                                        "figure": cluster_plot}
        


        self.create_html(data=data)

    def create_html(self, data: OrderedDict):
        '''
        create html report
        '''

        html_builder = HTMLBuilder(output_dn=self.output_dn,
                                   scenario_name=self.scenario.scenario)
        html_builder.generate_html(data)

    def get_meta_data(self):
        '''
            read meta data from self.scenario and generate a pandas.Dataframe with it
        '''
        data = []

        data.append(
            ("Performance measure", self.scenario.performance_measure[0]))
        data.append(("Performance type", self.scenario.performance_type[0]))
        data.append(("Maximize?", str(self.scenario.maximize[0])))
        if self.scenario.algorithm_cutoff_time:
            data.append(
                ("Running time cutoff (algorithm)", str(self.scenario.algorithm_cutoff_time)))
        if self.scenario.algorithm_cutoff_memory:
            data.append(
                ("Memory cutoff (algorithm)", str(self.scenario.algorithm_cutoff_memory)))
        if self.scenario.features_cutoff_time:
            data.append(
                ("Running time cutoff (features)", str(self.scenario.features_cutoff_time)))
        if self.scenario.features_cutoff_memory:
            data.append(
                ("Memory cutoff (Features)", str(self.scenario.features_cutoff_memory)))
        data.append(
            ("# Deterministic features", len(self.scenario.features_deterministic)))
        data.append(
            ("# Stochastic features", len(self.scenario.features_stochastic)))
        data.append(("# Feature groups", len(self.scenario.feature_steps)))
        data.append(
            ("# Deterministic algorithms", len(self.scenario.algortihms_deterministics)))
        data.append(
            ("# Stochastic algorithms", len(self.scenario.algorithms_stochastic)))
        if self.scenario.feature_cost_data is not None:
            data.append(("Feature costs provided?", "True"))
        else:
            data.append(("Feature costs provided?", "False"))

        meta_data = pd.DataFrame(data=list(map(lambda x: x[1], data)), index=list(
            map(lambda x: x[0], data)), columns=[""])

        return meta_data
