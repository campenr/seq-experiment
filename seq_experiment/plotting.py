"""
Copyright (c) 2018, Richard Campen
All rights reserved.
Licensed under the Modified BSD License.
For full license terms see LICENSE.txt

"""

from itertools import repeat
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.cm import get_cmap
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches


def _make_segmented_cmap(cmap):
    """
    Creates a segmented color map.

    Segmented color maps are more useful for displaying abundance data than non-segmented as they provide
    more dramatic shifts in color between each color.

    """

    cmap_ = get_cmap(cmap)

    if not isinstance(cmap_, LinearSegmentedColormap):
        cmap_colors = [[rgb for rgb in color] for color in cmap_.colors]
        return LinearSegmentedColormap.from_list(cmap, cmap_colors)
    else:
        # cmap is already a LinearSegmentedColor map so we don't need to convert it
        return cmap_


# TODO: fix color-by; disabled for now
# def plot_abundance(sxp, axis=0, facet_by=None, color_by=None, cmap='Paired', figsize=None, **kwargs):
def plot_abundance(sxp, axis=0, facet_by=None, cmap='Paired', gridspec_kw=None, **kwargs):

    # deal with kwargs
    # kwargs contains kwargs for the call to pyplot.subplots, which itself contains kwargs for gridspec, subplot, and Figure
    # by the time we call plot_bar we only want kwargs lefts that are valid for matplotlib.axes.Axes.bar.
    fig_kws = {
        # 'dpi': kwargs.pop('dpi', None),
        # 'edgecolor': edgecolor,
        # 'linewidth': linewidth,
    }
    if gridspec_kw is None:
        gridspec_kw = {}

    # need to check that facet_by and color_by arguments are valid, check axis argument simultaneously
    facet_attr = None
    # color_attr = None
    if axis == 0:
        # feature abundances grouped by row (features)
        # can facet by metadata and color by classifications
        if facet_by is not None and facet_by not in sxp.metadata.columns:
            raise IndexError('%s not in metadata.columns' % facet_by)
        # if color_by is not None and color_by not in sxp.classifications.columns:
        #     raise IndexError('%s not in classifications.columns' % color_by)
        facet_attr = 'metadata'
        # color_attr = 'classifications'
    elif axis == 1:
        # feature abundances grouped by columns (class/sample)
        # can facet by classification and color by metadata
        if facet_by is not None and facet_by not in sxp.classifications.columns:
            raise IndexError('%s not in classifications.columns' % facet_by)
        # if color_by is not None and color_by not in sxp.metadata.columns:
        #     raise IndexError('%s not in metadata.columns' % color_by)
        facet_attr = 'classifications'
        # color_attr = 'metadata'
    else:
        raise ValueError('axis should be either 0 or 1')

    # get data for plotting, conditionally split into separate DataFrames
    data_array = []
    if facet_by is None:
        # no need to subset, append abundance data intact as tuple along with name of x axis
        if axis == 0:
            data_array.append(('features', sxp.features))
        elif axis == 1:
            data_array.append(('classes', sxp.features))
    else:
        # if facet_by was specified we need to split the data by the values in the matching attribute column
        # get column values and determine the set; we do this manually rather than use `set` to preserve order
        facet_col = getattr(sxp, facet_attr)[facet_by]
        facet_values = list()
        for value in facet_col:
            if value not in facet_values:
                facet_values.append(value)

        # subset data use loc indexer
        for value in facet_values:
            if axis == 0:
                data_array.append((value, getattr(sxp, 'loc')[:, facet_col == value].features))
            elif axis == 1:
                data_array.append((value, getattr(sxp, 'loc')[facet_col == value].features))

    # conditionally transpose the data depending what axis we are plotting along the x axis
    y_col = sxp.features.columns
    if axis == 1:
        data_array = [(data[0], data[1].transpose()) for data in data_array]
        y_col = sxp.features.index

    # calculate colors for plotting
    # if color_by is None:
    if True:
        # color each item in the main axis separately
        if axis == 0:
            # color each feature separately
            color_col = sxp.features.index
        elif axis == 1:
            # color each class separately
            color_col = sxp.features.columns
    # else:
    #     # get user specific column to color by
    #     color_col = getattr(sxp, color_attr)[color_by]

    # get column values and determine the set; we do this manually rather than use `set` to preserve order
    color_values = list()
    for value in color_col:
        if value not in color_values:
            color_values.append(value)
    num_colors = len(color_values)

    # convert cmap to one suited for the kind of bar chart we're using
    cmap = _make_segmented_cmap(cmap)

    # format cmap based on the number of separate color values in the column to color by
    colors_set = list(iter(cmap(np.linspace(0, 1, num_colors))))
    colors = colors_set
    if len(colors) != len(color_col):
        # not a 1-1 ratio between colors and items to color so need to expand the colors array to match
        color_counts = Counter(color_col)
        colors = list()
        for index, color in enumerate(color_values):
            colors.extend(repeat(colors_set[index], color_counts[color]))

    # setup plotting environment and set custom plotting defaults if user has not specified them
    default_args = {
        'linewidth': 1,
        'edgecolor': 'black',
    }
    for arg, value in default_args.items():
        if arg not in kwargs.keys():
            kwargs[arg] = value

    # calculate default figure size
    scale = 5
    height = 1 * scale
    width = len(y_col)

    if kwargs.pop('figsize', None) is None:
        figsize = (width, height)

    fig_kws['figsize'] = figsize

    # get title
    # TODO: should move this to the figure_kw's dictionary
    title = kwargs.pop('title', '')

    # create plotting area
    if len(data_array) == 0:
        raise ValueError('no data for plotting')
    elif len(data_array) == 1:
        # basic plotting with single axis
        data = data_array[0]
        fig, ax = plt.subplots(**fig_kws)

        # set title
        if not title:
            title = data[0]

        ax = plot_bars(data=data[1], ax=ax, colors=colors, title=title, **kwargs)
    else:
        # faceted plotting with multiple axis
        fig, ax = plt.subplots(nrows=1, ncols=len(data_array), sharey=True, figsize=figsize,
                               gridspec_kw={'width_ratios':[len(data[1].columns) for data in data_array],
                                            'wspace': 0.05})
        for i in range(len(data_array)):
            data = data_array[i]
            ax[i] = plot_bars(data=data[1], ax=ax[i], colors=colors, title=data[0], **kwargs)

    # add axis labels
    if facet_by is None:
        ax.set_ylabel('abundance\n')
    else:
        ax[0].set_ylabel('abundance\n')

    # add legend
    handles = list()
    labels = list()

    patch_linewidth = kwargs.get('linewidth', None)
    patch_edgecolor = kwargs.get('edgecolor', None)

    for i in range(len(color_values)):

        patch_color = colors_set[i]
        patch_label = color_values[i]

        handles.append(mpatches.Patch(
            facecolor=patch_color,
            label=patch_label,
            linewidth=patch_linewidth,
            edgecolor=patch_edgecolor
        ))
        labels.append(patch_label)

    # to help with alignment we place the legend on the axes object
    if facet_by is None:
        legend = ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5))
    else:
        legend = ax[-1].legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5))

    # TODO: fix sizing of figure
    ## -- FROM SEABORN/AXISGRID.PY -- ##
    # Calculate and set the new width of the figure so the legend fits
    legend_width = legend.get_window_extent().width / fig.dpi
    figure_width = fig.get_figwidth()
    fig.set_figwidth(figure_width + legend_width)

    # Draw the plot again to get the new transformations
    fig.draw(fig.canvas.get_renderer())

    # Now calculate how much space we need on the right side
    legend_width = legend.get_window_extent().width / fig.dpi
    space_needed = legend_width / (figure_width + legend_width)
    margin = .04
    space_needed = (margin + space_needed) * 1.1
    right = 1 - space_needed

    # Place the subplot axes to give space for the legend
    fig.subplots_adjust(right=right)

    return ax


def plot_bars(data, stacked=True, ax=None, colors=None, title='', **kwargs):

    x = data.columns
    y = data.index

    len_x = len(x)
    len_y = len(y)

    # setup plotting constants
    x_indexes = np.arange(len_x)
    width = kwargs.pop('width', 0.8)
    if not stacked:
        width = width / len_y
    bottom = None

    # iterate over the x columns
    for index, col_name in enumerate(y):

        # get the column data and the position where to plot that data
        col_data = data.loc[col_name]

        # conditionally bump the location of where to plot the bars
        # needed if plotting multiple bars per x item when plotting stacked bars
        if stacked:
            x_pos = x_indexes
        else:
            x_pos = x_indexes + (width * index)

        # generate bars
        bars = ax.bar(x_pos, height=col_data, width=width, bottom=bottom, color=colors[index], **kwargs)

        # if creating stacked bar charts we need to increment the bottom for the next bars by the height of the current
        if stacked:
            if bottom is None:
                bottom = col_data.copy(deep=True).values
            else:
                # on first iteration we need to set the bottom values for the first time
                bottom += col_data.values

    # calculate margins at ends of axis. Default is 5% of the width of the axes, but this makes it
    # bigger on wider bar charts need to adjust the margin to account for different figure width.
    # We base the margin size off the bar width so the gaps are consistent.
    margin = ((1 - width) / 1) / len_x
    ax.margins(margin)

    # tidy axes
    if not stacked:
        x_tick_loc = (x_indexes - width) + (len_y * width / 2)
    else:
        x_tick_loc = x_indexes
    ax.set_xticks(x_tick_loc)
    # expect long labels so rotate them 90 degrees
    ax.set_xticklabels(x, rotation=90)
    ax.set_title(title)

    return ax
