from itertools import tee
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
import numpy as np

class OrdinationResult(object):
    """
    Stores results from an ordination.
    
    """

    def __init__(self, axes, method, eigvals=None, explained=None, metric=None, stress=None):

        self.axes = axes
        self.method = method

        # conditionally add attributes
        if eigvals is not None:
            self.eigvals = eigvals

        if explained is not None:
            self.explained = explained

        if metric is not None:
            self.metric = metric

        if stress is not None:
           self.stress = stress

    def __repr__(self):
        return self.axes

    def __str__(self):

        result_str = '{method} ordination of {dist} distances/dissimilarities.\n\n'.format(
                method=self.method,
                dist=self.metric
        )

        # conditionally format string
        if self.stress is not None:
            result_str += 'Stress: {stress}\n\n'.format(stress=self.stress)

        result_str += 'Axes:\n{axes}'.format(axes=self.axes)

        return result_str

    def plot(self, seq_exp=None, axes=None, color=None, **kwargs):
        """
        Plots the ordination results        
        
        :param seq_exp: the SeqExp object containing the sample metadata, or the sample metadata from a SeqExp object
        :type seq_exp: SeqExp or SampleDataTable
        :param axes: which axes to plot
        :type axes: tuple(int, int)
        :param color: which column of the sample_data_table to color the samples by
        :type color: str or None
        :param kwargs: 
        :return: 
        
        """

        if axes is None:
            axes=(1,2)

        # define default plotting arguments and append to kwargs if not already there
        # this allows for pretty default graph settings the user can easily overwrite
        default_args = {
            'figsize': (10,10),
            'cmap': 'Set1',
            'alpha': 0.8,
            'marker': 'o'
        }

        for k, v in default_args.items():
            if k not in kwargs.keys():
                kwargs[k] = v

        # check that none of the axes to plot is greater than the number of available axes
        if any(axes) > len(self.axes.columns):
            raise(ValueError('can not plot more axes then exist in the ordintion results.'))

        # check that the samples in the ordination match those in the supplied SeqExp
        if seq_exp is not None:
            if not self.axes.index.equals(seq_exp.feature_table.columns):
                raise(ValueError('ordination axes and seq_exp axes do not match.'))

        # get groups information for plotting and append to axes data frame
        if color is None:
            # just color each sample the same
            # We get this from the feature table to prevent enforcing sample metadata if not needed
            groups = seq_exp.feature_table.columns.tolist()
        else:
            try:
                # get groups from the sample data table column specified by `color` if it exists
                groups = list(seq_exp.sample_data_table[color])
            except KeyError:
                raise

        data = self.axes.copy()
        data['group'] = groups

        # create subset dataframes split by groups
        sub_dfs = (data[data['group'] == group] for group in set(groups))

        # setup plotting area and colors
        colors, leg_colors = tee(iter(getattr(cm, kwargs['cmap'])(np.linspace(0, 1, len(groups)))), 2)
        _, ax = plt.subplots(kwargs['figsize'])

        # add data to plot
        for sub_df in sub_dfs:
            self._sub_scatter(ax, axes, sub_df, color=next(colors))

        # add formatting to plot
        ax.set_xlabel('\n{}'.format(data.columns[axes[0] - 1]), fontsize=18)
        ax.set_ylabel('{}\n'.format(data.columns[axes[1] - 1]), fontsize=18)
        ax.set_title('Non-Metric Multidimensional Scalaing', fontsize=20, y=1.05)
        ax.tick_params(axis='both', which='major', labelsize=14)

        # customize legend
        leg = ax.legend(set(groups), fontsize=13.2, frameon=True, loc='best')
        for l in range(len(set(groups))):
            leg.legendHandles[l].set_color(next(leg_colors))

        return ax

    @staticmethod
    def _sub_scatter(ax, axes, s, color, marker='o', alpha=0.8):
        for row in s.iterrows():
            index = row[0]
            data = row[1]
            ax.scatter(x=data[axes[0] - 1], y=data[axes[1] - 1], c=color, marker=marker, s=300, label=index, alpha=alpha)
            # axes.annotate(index, xy=(data.axis1, data.axis2),  xytext = (0, -23),
            #               textcoords = 'offset points', ha = 'center', va = 'center')
