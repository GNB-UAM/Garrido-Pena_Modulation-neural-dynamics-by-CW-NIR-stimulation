# -*- coding: utf-8 -*-
# @File name: plot_utils.py
# @Author: Alicia Garrido-Peña
# @Date:   2023-08-18 14:09:53
# @Contact: alicia.garrido@uam.es

# "Copyright (c) 2023 Alicia Garrido-Peña. All Rights Reserved."
# Use of this source code is govern by GPL-3.0 license that 
# can be found in the LICENSE file

# Code used for manuscript submitted. 
# If you use any of this code, please cite:
# Garrido-Peña, Alicia, Sánchez-Martín, Pablo, Reyes-Sanchez, Manuel, Levi, Rafael, Rodriguez, Francisco B., Castilla, Javier, Tornero, Jesús, Varona, Pablo. Modulation of neuronal dynamics by sustained and activity-dependent continuous-wave near-infrared laser stimulation. Submitted.
#
#
# To import from child dir
# script_path = sys.argv[0][:sys.argv[0].rfind('/')]
# sys.path.append(script_path+'/..')
# import plot_utils as pu


import matplotlib.pyplot as plt

import numpy as np
def small_x_axe(ax, label,offset=0):
    # Set the tick positions and labels
    ax.tick_params(axis='x', length=2, width=0,left='True')
    range = ax.get_xticks()[np.where(ax.get_xticks()==0)[0]+1]

    ax.set_xticks(range/2)
    ax.spines['bottom'].set_bounds(0, range)
    ax.spines['bottom'].set_position(('axes', 0 ))

    ax.set_xticklabels(['%.0f %s'%(range,label)])


def small_y_axe(ax, label, offset=0):
    # Set the tick positions and labels
    ax.tick_params(axis='y', length=2, width=0,left='True')
    range = ax.get_yticks()[np.where(ax.get_yticks()==0)[0]+1]
    
    ax.spines['left'].set_bounds(0, range)
    # ax.spines['bottom'].set_position(('axes', 0 ))

    ax.set_yticks([])
    # ax.set_yticklabels(['%.0f %s'%(range[0],label)], rotation=90, fontsize=fontsize)
    ax.text(-0.04, 0.25, '%.0f %s'%(range,label), transform=ax.transAxes,
            rotation=90)

def remove_axes(ax, positions=['top','right']):
    for p in positions:
        ax.spines[p].set_visible(False)

def color_axes(ax, color='r'):
    for spine in ax.spines:
        ax.spines[spine].set_color(color)


def plot_cmap(fig, ax, colors, nValues=None, location=[0.7, 0.2, 0.02, 1], alpha=1, cmap=None): # [x, y, width, height]

    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Create the color map and color normalization
    cmap = mcolors.LinearSegmentedColormap.from_list('my_cmap', colors) if cmap is None else cmap
    norm = mcolors.Normalize(min(nValues), max(nValues)) if nValues is not None else None

    # Create the scalar mappable
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Create the color bar
    cb_ax = fig.add_axes(location) # [x, y, width, height]
    cbar = fig.colorbar(sm, cax=cb_ax, alpha=alpha)
    return cbar

# https://matplotlib.org/stable/gallery/text_labels_and_annotations/custom_legends.html
def create_line(line_t, color, label):
    from matplotlib.lines import Line2D

    if line_t == 'Line':
        return Line2D([0], [0], color=color, lw=4, label=label)
    elif line_t == 'Scatter':
        return Line2D([0], [0], marker='o', color='w', label=label,
                          markerfacecolor=color, markersize=5)
    elif line_t == 'Patch':
        return Patch(facecolor=color, edgecolor='k',
                         label=label)
    else:
        return Line2D([0], [0], color='k', lw=4, label='')



def custom_legend(labels, types): # types = [(Scatter, 'b'), ('Line', 'g'), ('Patch', 'r')]
    legend_elements = []
    for label, (line_t, color) in zip(labels,types):
        legend_elements.append(create_line(line_t, color, label))

    return legend_elements


def xticks_vertical_adjust(ax, labels, fontsize=17):

    ax.set_xticklabels('')

    # Adjust the vertical position of each label
    for i, label in enumerate(labels):
        # even top, odds bottom
        if not i%2:
            ax.text(i, -0.02, label, ha='center', va='top', fontsize=fontsize)
        else:
            ax.text(i, -0.13, label, ha='center', fontsize=fontsize)

def modify_cmap(cmap, desat=1, alpha=1, hue=0.9):
    import matplotlib.colors as mcolors
    colors_rgb = cmap(np.linspace(0, 1, 256))[:, :3]
    colors_hsv = mcolors.rgb_to_hsv(colors_rgb)
    colors_hsv[:, 0] = (colors_hsv[:, 0] + hue) % 1.0
    colors_rgb_mod = mcolors.hsv_to_rgb(colors_hsv)

    # colors_rgb_mod = colors_rgb.copy()
    colors_rgb_mod *= desat # reduce the saturation by 30%
    colors_rgb_mod = np.concatenate((colors_rgb_mod, np.ones((256, 1)) * alpha), axis=1) # add alpha values

    return mcolors.ListedColormap(colors_rgb_mod)