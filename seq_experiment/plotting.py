"""
Copyright (c) 2017, Richard Campen
All rights reserved.
Licensed under the Modified BSD License.
For full license terms see LICENSE.txt

"""

import numpy as np
import matplotlib.pyplot as plt


def plot_abundance(sxp, axis=0, stacked=True, **kwargs):

    # set custom plotting parameters if user has not specified them
    default_args = {
        'linewidth': 1,
        'edgecolor': 'black',
    }
    for arg, value in default_args.items():
        if arg not in kwargs.keys():
            kwargs[arg] = value

    # get data for plotting, and the axis's
    if axis == 0:
        data = sxp.features
    elif axis == 1:
        data = sxp.features.transpose()
    else:
        raise ValueError('axis should be either 0 or 1')

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

    # create plotting area
    fig, ax = plt.subplots()

    # iterate over the x columns
    for index, col_name in enumerate(y):

        # get the column data and the position where to plot that data
        col_data = data.loc[col_name]

        # conditionally bump the location of where to plot the bars
        # needed if plotting multiple bars per x item
        if stacked:
            x_pos = x_indexes
        else:
            x_pos = x_indexes + (width * index)

        # generate bars
        bars = ax.bar(x_pos, height=col_data, width=width, bottom=bottom, **kwargs)

        # if creating stacked bar charts we need to increment the bottom for the next bars by the height of the current
        if stacked:
            if bottom is None:
                bottom = col_data.copy(deep=True).values
            else:
                # on first iteration we need to set the bottom values for the first time
                bottom += col_data.values

    # tidy axes
    if not stacked:
        x_tick_loc = (x_indexes - width) + (len_y * width / 2)
    else:
        x_tick_loc = x_indexes
    ax.set_xticks(x_tick_loc)
    ax.set_xticklabels(x, rotation=90)

    # add legend
    legend = ax.legend(y, frameon=True, loc='center left', bbox_to_anchor=(1.0, 0.5))

    return ax
