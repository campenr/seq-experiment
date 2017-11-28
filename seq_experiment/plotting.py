"""
Copyright (c) 2017, Richard Campen
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


def _make_segmented_cmap(cmap):

    cmap_colors = [[rgb for rgb in color] for color in get_cmap(cmap).colors]

    return LinearSegmentedColormap.from_list(cmap, cmap_colors)


def plot_abundance(sxp, axis=0, facet_by=None, **kwargs):

    # get data for plotting
    data_array = []

    if facet_by is None:
        data_array.append(sxp.features)
    else:
        # if facet_by was specified we need to split the data by the values in the matching metadata column
        facet_col = sxp.metadata[facet_by]
        for value in set(facet_col):
            data_array.append(sxp.mx[getattr(sxp.metadata, facet_by) == value].features)

    # conditionally transpose the data depending what axis we are plotting
    if axis == 0:
        pass
    elif axis == 1:
        for data_ in data_array:
            data_ = data_.transpose()
    else:
        raise ValueError('axis should be either 0 or 1')

    # setup plotting environment
    # set custom plotting parameters if user has not specified them
    default_args = {
        'linewidth': 1,
        'edgecolor': 'black',
    }
    for arg, value in default_args.items():
        if arg not in kwargs.keys():
            kwargs[arg] = value

    # create plotting area
    if len(data_array) == 0:
        raise ValueError('no data for plotting')
    elif len(data_array) == 1:
        # basic plotting with single axis
        fig, ax = plt.subplots()
        ax = plot_bars(data=data_array[0], ax=ax, **kwargs)
    else:
        # faceted plotting with multiple axis
        fig, ax = plt.subplots(nrows=1, ncols=len(data_array))
        for i in range(len(data_array)):
            ax[i] = plot_bars(data=data_array[i], ax=ax[i], **kwargs)

    return ax


def plot_bars(data, stacked=True, ax=None, color_by=None, cmap='Paired', **kwargs):

    x = data.columns
    y = data.index

    len_x = len(x)
    len_y = len(y)

    print('####')
    print('len_x: ', len_x)
    print('len_y: ', len_y)

#     # sort colors
#     # convert cmap to one suited for the kind of bar chart we're using
#     cmap = _make_segmented_cmap(cmap)
#
#     # check what kind of colors argument was passed
#     # if colors is None plot using different color per feature
#     if color_by is None:
#         color_groups = x
#         len_color_groups = len_x
#     else:
#         color_groups = set(color_by)
#         len_color_groups = len(list(set(color_groups)))
#
#     # format cmap based on the number of separate color groups specified in `color_by`
#     _colors = list(iter(cmap(np.linspace(0, 1, len_color_groups))))
#     if len(_colors) == len_x:
#         pass
#     elif len(_colors) < len_x and len_x % len(_colors) == 0:
#         color_counts = Counter(color_by)
#         new_colors = []
#         for index, color in enumerate(color_groups):
#             new_colors.extend(repeat(_colors[index], color_counts[color]))
#         _colors = new_colors
#     else:
#         raise IndexError('length of colors must be equal to the number of features, or a multiple thereof')
#
#     # setup plotting constants
#     x_indexes = np.arange(len_x)
#     width = kwargs.pop('width', 0.8)
#     if not stacked:
#         width = width / len_y
#     bottom = None
#
#     # iterate over the x columns
#     for index, col_name in enumerate(y):
#
#         # get the column data and the position where to plot that data
#         col_data = data.loc[col_name]
#
#         # conditionally bump the location of where to plot the bars
#         # needed if plotting multiple bars per x item when plotting stacked bars
#         if stacked:
#             x_pos = x_indexes
#         else:
#             x_pos = x_indexes + (width * index)
#
#         color_ = _colors[index]
#
#         # generate bars
#         bars = ax.bar(x_pos, height=col_data, width=width, bottom=bottom, color=color_, **kwargs)
#
#         # if creating stacked bar charts we need to increment the bottom for the next bars by the height of the current
#         if stacked:
#             if bottom is None:
#                 bottom = col_data.copy(deep=True).values
#             else:
#                 # on first iteration we need to set the bottom values for the first time
#                 bottom += col_data.values
#
#     # tidy axes
#     if not stacked:
#         x_tick_loc = (x_indexes - width) + (len_y * width / 2)
#     else:
#         x_tick_loc = x_indexes
#     ax.set_xticks(x_tick_loc)
#     # expect long labels so rotate them 90 degrees
#     ax.set_xticklabels(x, rotation=90)
#
#     # add legend
#     legend = ax.legend(y, frameon=True, loc='center left', bbox_to_anchor=(1.0, 0.5))
#
#     # TODO: fix sizing of figure
#     ## -- FROM SEABORN/AXISGRID.PY -- ##
# #     # Calculate and set the new width of the figure so the legend fits
# #     legend_width = figlegend.get_window_extent().width / self.fig.dpi
# #     figure_width = self.fig.get_figwidth()
# #     self.fig.set_figwidth(figure_width + legend_width)
# #
# #     # Draw the plot again to get the new transformations
# #     self.fig.draw(self.fig.canvas.get_renderer())
# #
# #     # Now calculate how much space we need on the right side
# #     legend_width = figlegend.get_window_extent().width / self.fig.dpi
# #     space_needed = legend_width / (figure_width + legend_width)
# #     margin = .04 if self._margin_titles else .01
# #     self._space_needed = margin + space_needed
# #     right = 1 - self._space_needed
# #
# #     # Place the subplot axes to give space for the legend
# #     self.fig.subplots_adjust(right=right)
#
#     return ax
    return