import logging
import os

import numpy as np

from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage
from scipy.misc import comb

from pandas import DataFrame

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plottingscripts.plotting.scatter import plot_scatter_plot

from autofolio.data.aslib_scenario import ASlibScenario

from asapy.utils.util_funcs import get_cdf_x_y

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "MIT"
__email__ = "lindauer@cs.uni-freiburg.de"


class PerformanceAnalysis(object):

    def __init__(self,
                 output_dn: str,
                 scenario: ASlibScenario):
        '''
        Constructor

        Arguments
        ---------
        output_dn:str
            output directory name
        '''
        self.logger = logging.getLogger("Performance Analysis")
        self.scenario = scenario
        self.output_dn = os.path.join(output_dn, "performance_plots")
        if not os.path.isdir(self.output_dn):
            os.mkdir(self.output_dn)

    def scatter_plots(self):
        '''
            generate scatter plots of all pairs of algorithms in the performance data of the scenario
            and save them in the output directory

            Returns
            -------
            list of all generated file names of plots
        '''
        matplotlib.pyplot.close()
        self.logger.info("Plotting scatter plots........")

        plots = []
        self.algorithms = self.scenario.algorithms
        n_algos = len(self.scenario.algorithms)

        if self.scenario.performance_type[0] == "runtime":
            max_val = self.scenario.algorithm_cutoff_time
        else:
            max_val = self.scenario.performance_data.max().max()

        for i in range(n_algos):
            for j in range(i + 1, n_algos):
                algo_1 = self.scenario.algorithms[i]
                algo_2 = self.scenario.algorithms[j]
                y_i = self.scenario.performance_data[algo_1].values
                y_j = self.scenario.performance_data[algo_2].values

                matplotlib.pyplot.close()

                fig = plot_scatter_plot(x_data=y_i, y_data=y_j, max_val=max_val,
                                        labels=[algo_1, algo_2],
                                        metric=self.scenario.performance_type[0])
                out_name = os.path.join(self.output_dn,
                                        "scatter_%s_%s.png" % (algo_1.replace("/", "_"), algo_2.replace("/", "_")))
                fig.savefig(out_name)
                plots.append((algo_1, algo_2, out_name))

        return plots

    def correlation_plot(self):
        '''
            generate correlation plot using spearman correlation coefficient and ward clustering

            Returns
            -------
            file name of saved plot
        '''
        matplotlib.pyplot.close()
        self.logger.info("Plotting correlation plots........")

        perf_data = self.scenario.performance_data.values
        algos = list(self.scenario.performance_data.columns)

        n_algos = len(algos)

        data = np.zeros((n_algos, n_algos)) + 1  # similarity
        for i in range(n_algos):
            for j in range(i + 1, n_algos):
                rho, p = spearmanr(perf_data[:, i], perf_data[:, j])
                data[i, j] = rho
                data[j, i] = rho

        link = linkage(data * -1, 'ward')  # input is distance -> * -1

        sorted_algos = [[a] for a in algos]
        for l in link:
            new_cluster = sorted_algos[int(l[0])][:]
            new_cluster.extend(sorted_algos[int(l[1])][:])
            sorted_algos.append(new_cluster)

        sorted_algos = sorted_algos[-1]

        # resort data
        indx_list = []
        for a in algos:
            indx_list.append(sorted_algos.index(a))
        indx_list = np.argsort(indx_list)
        data = data[indx_list, :]
        data = data[:, indx_list]

        fig, ax = plt.subplots()
        heatmap = ax.pcolor(data, cmap=plt.cm.Blues)

        # put the major ticks at the middle of each cell
        ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
        ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)

        plt.xlim(0, data.shape[0])
        plt.ylim(0, data.shape[0])

        # want a more natural, table-like display
        ax.invert_yaxis()
        ax.xaxis.tick_top()

        ax.set_xticklabels(sorted_algos, minor=False)
        ax.set_yticklabels(sorted_algos, minor=False)
        labels = ax.get_xticklabels()
        plt.setp(labels, rotation=90, fontsize=12, ha="left")
        labels = ax.get_yticklabels()
        plt.setp(labels, rotation=0, fontsize=12)

        fig.colorbar(heatmap)

        plt.tight_layout()

        out_plot = os.path.join(self.output_dn, "correlation_plot.png")
        plt.savefig(out_plot, format="png")

        return out_plot

    def get_contribution_values(self):
        '''
            contribution value computation

            Returns
            ------
            pandas.DataFrame() with columns being "Average Performance", "Marginal Performance", "Shapley Values" and indexes being the algorithms
        '''
        
        self.logger.info("Get contribution scores........")

        algorithms = self.scenario.algorithms
        instances = self.scenario.instances
        scenario = self.scenario
        
        performance_data = scenario.performance_data


        if self.scenario.maximize[0] == False:
            # Shapley code assumes higher is better
            performance_data = performance_data * -1 

        is_time_scenario = self.scenario.performance_type[0] == "runtime"

        def metric(algo, inst):
            if is_time_scenario:
                perf = scenario.algorithm_cutoff_time - \
                    min(scenario.algorithm_cutoff_time,
                        performance_data[algo][inst])
                return perf
            else:
                return performance_data[algo][inst]

        shapleys = self._get_VBS_Shap(instances, algorithms, metric)
        if self.scenario.maximize[0] == False:
            performance_data = performance_data * -1
        
        if self.scenario.maximize[0] == True:
            # marginal contribution code assumes: smaller is better
            self.scenario.performance_data = self.scenario.performance_data * -1
            
        marginales = self._get_marginal_contribution() 
        if self.scenario.maximize[0] == True:
            self.scenario.performance_data = self.scenario.performance_data * -1
            
        averages = self._get_average_perf()

        data = np.zeros((len(algorithms), 3))
        for indx, algo in enumerate(algorithms):
            data[indx] = np.array(
                [averages[algo], marginales[algo], shapleys[algo]])
        df = DataFrame(data=data, index=algorithms, columns=[
                       "Average Performance", "Marginal Contribution", "Shapley Values"])

        return df

    def _get_average_perf(self):
        '''
            compute average performance of algorithms
        '''
        averages = {}
        for algorithm in self.scenario.algorithms:
            averages[algorithm] = self.scenario.performance_data[
                algorithm].sum()
        return averages

    def _get_marginal_contribution(self):
        '''
            compute marginal contribution of each algorithm
        '''
        marginales = {}
        all_vbs = self.__get_vbs(self.scenario.performance_data)
        for algorithm in self.scenario.algorithms:
            remaining_algos = list(
                set(self.scenario.algorithms).difference([algorithm]))
            perf_data = self.scenario.performance_data[remaining_algos]
            rem_vbs = self.__get_vbs(perf_data)
            marginales[algorithm] = rem_vbs - all_vbs 

        return marginales

    def __get_vbs(self, performance_data):
        return np.sum(performance_data.min(axis=1))

    def _get_VBS_Shap(self, instances, algorithms, metric):
        '''
            instances - the instances to solve.
            algorithms - the set of available algorithms.
            metric - the performance metric from (algorithm,instance) to real number. Higher is better.

            Returns a dictionary from algorithms to their Shapley value, where the coalitional function is
            $$ v(S) = \frac{1}{|X|} \sum_{x\in X} \max_{s\in S} metric(s,x),$$
            where X is the set of instances. This is the "average VBS game" with respect to the given instances,
            algorithms and metric.

            __author__ = "Alexandre Frechett et al"
            slight modification by me
        '''
        n = len(algorithms)
        m = len(instances)

        shapleys = {}

        # For each instance
        for instance in instances:
            instance_algorithms = sorted(
                algorithms, key=lambda a: metric(a, instance))

            # For each algorithm, from worst to best.
            for i in range(len(instance_algorithms)):
                ialgorithm = instance_algorithms[i]

                #print >> sys.stderr, 'Processing the rule corresponding to %d-th algorithm "%s" being the best in the coalition.' % (i,ialgorithm)

                # The MC-rule is that you must have the algorithm and none of
                # the better ones.
                '''
                If x is the instance and s_1^x,...,s_n^x are sorted from worst, then the rule's pattern is:
                $$ p_i^x = s_i^x \wedge \bigwedge_{j=i+1}^n \overline{s}_j^x $$
                and its weight is 
                $$ w_i^x = metric(s_i^x,x).$$
                '''
                pos = 1
                neg = n - i - 1

                metricvalue = metric(ialgorithm, instance)
                # normalised as fraction of instances
                #value = 1/float(m)*metricvalue
                value = metricvalue
                #print >> sys.stderr, 'Value of this rule : 1/%d * %.4f = %.4f' % (m,metricvalue,value)

                # Calculate the rule Shapley values, and add them to the global
                # Shapley values.

                # Shapley value for positive literals in the rule.
                pos_shap = value / \
                    float(pos * comb(pos + neg, neg, exact=True))
                # Shapley value for negative literals in the rule.
                if neg > 0:
                    neg_shap = -value / \
                        float(neg * comb(pos + neg, pos, exact=True))
                else:
                    neg_shap = None

                # Update the Shapley value for the literals appearing in the
                # rule.
                for j in range(i, len(instance_algorithms)):
                    jalgorithm = instance_algorithms[j]

                    if jalgorithm not in shapleys:
                        shapleys[jalgorithm] = 0

                    if j == i:
                        shapleys[jalgorithm] += pos_shap
                    else:
                        shapleys[jalgorithm] += neg_shap

        return shapleys

    def get_cdf_plots(self):
        '''
            plot the cummulative distribution function of each algorithm

            Returns
            -------
            file name of saved plot
        '''
        matplotlib.pyplot.close()
        self.logger.info("Plotting CDF plots........")

        from cycler import cycler

        gs = matplotlib.gridspec.GridSpec(1, 1)

        fig = plt.figure()
        ax1 = plt.subplot(gs[0:1, :])

        colormap = plt.cm.gist_ncar
        fig.gca().set_prop_cycle(cycler('color', [
            colormap(i) for i in np.linspace(0, 0.9, len(self.scenario.algorithms))]))

        if self.scenario.performance_type[0] == "runtime":
            max_val = self.scenario.algorithm_cutoff_time
            min_val = max(0.0005, self.scenario.performance_data.min().min())
        else:
            max_val = self.scenario.performance_data.max().max()
            min_val = self.scenario.performance_data.min().min()

        for algorithm in self.scenario.algorithms:
            x, y = get_cdf_x_y(
                self.scenario.performance_data[algorithm], max_val)
            x = np.array(x)
            y = np.array(y)
            x[x < min_val] = min_val
            ax1.step(x, y, label=algorithm)

        ax1.grid(
            True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
        ax1.set_xlabel(self.scenario.performance_measure[0])
        ax1.set_ylabel("P(x<X)")
        ax1.set_xlim([min_val, max_val])
        if self.scenario.performance_type[0] == "runtime":
            ax1.set_xscale('log')

        #ax1.legend(loc='lower right')
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        out_fn = os.path.join(self.output_dn, "cdf_plot.png")

        plt.savefig(out_fn, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, pad_inches=0.02, bbox_inches='tight')

        return out_fn

    def get_violin_plots(self):
        '''
            compute violin plots (fancy box plots) for each algorithm
        '''
        matplotlib.pyplot.close()
        self.logger.info("Plotting vilion plots........")

        cutoff = self.scenario.algorithm_cutoff_time
        fig, ax = plt.subplots(nrows=1, ncols=1)
        all_data = self.scenario.performance_data.values
        all_data[all_data > cutoff] = cutoff

        if self.scenario.performance_type[0] == "runtime":
            all_data = np.log10(all_data)
            y_label = "log(%s)" % (self.scenario.performance_type[0])
        else:
            y_label = "%s" % (self.scenario.performance_type[0])
        n_points = all_data.shape[0]
        all_data = [all_data[:, i] for i in range(all_data.shape[1])]
        ax.violinplot(
            all_data, showmeans=False, showmedians=True, points=n_points)

        ax.yaxis.grid(True)
        ax.set_ylabel(y_label)

        plt.setp(ax, xticks=[y + 1 for y in range(len(all_data))],
                 xticklabels=self.scenario.performance_data.columns.values)

        labels = ax.get_xticklabels()
        plt.setp(labels, rotation=45, fontsize=12, ha="right")

        plt.tight_layout()

        out_fn = os.path.join(self.output_dn, "violin_plot.png")
        plt.savefig(out_fn)

        return out_fn

    def get_box_plots(self):
        '''
            compute violin plots (fancy box plots) for each algorithm
        '''
        matplotlib.pyplot.close()
        self.logger.info("Plotting box plots........")

        cutoff = self.scenario.algorithm_cutoff_time
        fig, ax = plt.subplots(nrows=1, ncols=1)
        all_data = self.scenario.performance_data.values
        all_data[all_data > cutoff] = cutoff

        n_points = all_data.shape[0]
        all_data = [all_data[:, i] for i in range(all_data.shape[1])]

        ax.boxplot(all_data)

        ax.yaxis.grid(True)
        y_label = "%s" % (self.scenario.performance_type[0])
        ax.set_ylabel(y_label)

        if self.scenario.performance_type[0] == "runtime":
            ax.set_yscale('log')

        plt.setp(ax, xticks=[y + 1 for y in range(len(all_data))],
                 xticklabels=self.scenario.performance_data.columns.values)

        labels = ax.get_xticklabels()
        plt.setp(labels, rotation=45, fontsize=12, ha="right")

        plt.tight_layout()

        out_fn = os.path.join(self.output_dn, "box_plot.png")
        plt.savefig(out_fn)

        return out_fn

    def get_bar_status_plot(self):
        '''
            get status distribution as stacked bar plot
        '''
        matplotlib.pyplot.close()
        self.logger.info("Plotting bar plots........")
        runstatus_data = self.scenario.runstatus_data

        width = 0.5
        stati = ["ok", "timeout", "memout", "not_applicable", "crash", "other"]

        count_stats = np.array(
            [runstatus_data[runstatus_data == status].count().values for status in stati])
        count_stats = count_stats / len(self.scenario.instances)

        colormap = plt.cm.gist_ncar
        cc = [colormap(i) for i in np.linspace(0, 0.9, len(stati))]

        bottom = np.zeros((len(self.scenario.algorithms)))
        ind = np.arange(len(self.scenario.algorithms)) + 0.5
        plots = []
        for id, status in enumerate(stati):
            plots.append(
                plt.bar(ind, count_stats[id, :], width, color=cc[id], bottom=bottom))
            bottom += count_stats[id, :]

        plt.ylabel('Frequency of runstatus')
        plt.xticks(
            ind + width / 2., list(runstatus_data.columns), rotation=45, ha="right")
        lgd = plt.legend(list(map(lambda x: x[0], plots)), stati, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                         ncol=3, mode="expand", borderaxespad=0.)

        plt.tight_layout()
        out_fn = os.path.join(self.output_dn, "status_bar_plot.png")
        plt.savefig(out_fn, bbox_extra_artists=(lgd,), bbox_inches='tight')

        return out_fn
