from itertools import tee
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
import numpy as np

class OrdinationResult(object):
    """
    
    
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

    def __str__(self):

        return('{method} ordination\n{axes}'.format(self.method, self.axes))

    def plot(self, seq_exp, axes=None, color=None, **kwargs):
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

        seq_exp = seq_exp
        axes = axes

        # define default plotting arguments and append to kwargs if not already there
        # this allows for pretty default graph settings the user can easily overwrite
        default_args = {
            'cmap': 'Set1',
            'alpha': 0.8,
            'marker': 'o'
        }
        for k, v in default_args:
            if k not in kwargs.keys():
                kwargs[k] = v

        # check that the number of axes to plot is not more than the number of available axes
        if axes > len(self.axes.columns):
            raise(ValueError('can not plot more axes then exist in the ordintion results.'))

        # check that the samples in the ordination match those in the supplied SeqExp
        if not self.axes.index.equals(axes.index):
            raise(ValueError('ordination axes and seq_exp axes do not match.'))

        # get groups information for plotting and append to axes data frame
        if color is None:
            # just color each sample separately
            # We get this from the feature table to prevent enforcing sample metadata if not needed
            groups = seq_exp.columns.tolist()
        else:
            try:
                # get groups from the sample data table column specified by `color` if it exists
                groups = list(set(seq_exp.sample_data_table[color]))
            except KeyError:
                raise
        axes['group'] = groups

        # create subset dataframes split by groups
        sub_dfs = (axes[axes['group'] == group] for group in groups)

        # setup plotting area and colors
        colors, leg_colors = tee(iter(getattr(cm, kwargs['cmap'])(np.linspace(0, 1, len(groups)))), 2)
        fig, ax = plt.subplots(figsize=(8, 8))

        # add data to plot
        for sub_df in sub_dfs:
            self._sub_scatter(ax, sub_df, color=next(colors))

        # add formatting to plot
        ax.set_xlabel('\nAxis 1', fontsize=18)
        ax.set_ylabel('Axis 2\n', fontsize=18)
        ax.set_title('Non-Metric Multidimensional Scalaing', fontsize=20, y=1.05)
        ax.tick_params(axis='both', which='major', labelsize=14)

        # customize legend
        leg = ax.legend(groups, fontsize=13.2, frameon=True, loc='best')
        for l in range(len(groups)):
            leg.legendHandles[l].set_color(next(leg_colors))

        return fig, ax, leg

    @staticmethod
    def _sub_scatter(axes, s, color, marker='o', alpha=0.8):
        for row in s.iterrows():
            index = row[0]
            data = row[1]
            axes.scatter(x=data.axis1, y=data.axis2, c=color, marker=marker, s=300, label=index, alpha=alpha)
            # axes.annotate(index, xy=(data.axis1, data.axis2),  xytext = (0, -23),
            #               textcoords = 'offset points', ha = 'center', va = 'center')
