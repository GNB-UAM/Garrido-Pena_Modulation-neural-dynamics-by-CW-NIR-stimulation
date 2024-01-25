# -*- coding: utf-8 -*-
# @File name: generate_table_stats_model.py
# @Author: Alicia Garrido-Peña
# @Date:   2023-08-11 01:54:37
# @Contact: alicia.garrido@uam.es

# "Copyright (c) 2023 Alicia Garrido-Peña. All Rights Reserved."
# Use of this source code is govern by GPL-3.0 license that 
# can be found in the LICENSE file

# Code used for manuscript submitted. 
# If you use any of this code, please cite:
# Garrido-Peña, Alicia, Sánchez-Martín, Pablo, Reyes-Sanchez, Manuel, Levi, Rafael,
# Rodriguez, Francisco B., Castilla, Javier, Tornero, Jesús, Varona, Pablo. 
# Modulation of neuronal dynamics by sustained and activity-dependent
# continuous-wave near-infrared laser stimulation. Submitted.
#
#
# Author: Alicia Garrido-Peña
# Date: 30-01-2023

import pickle as pkl
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

script_path = sys.argv[0][:sys.argv[0].rfind('/')]
sys.path.append(script_path+'/../..')
import superpos_functions as laser_utils
import plot_utils as pu


def get_first_line(file):
    try:
        if 'pkl' in file:
            with open(file[:-3]+'info','r') as fs:
                first_line = fs.readline()

            # print(first_line)
        else:
            # fs=open(file)
            with open(file,'r') as fs:
                first_line = fs.readline()
            # fs.close()

        if first_line == '':
            raise Exception
    except Exception as e:
        print(e)
        print("Skiped",file)
        first_line = ''
        # raise Exception
    # index =fs.find(f_format)
    # ini = fs.rfind("_")

    return first_line

def create_custom_palette(colors, n_colors=100):
    # Define the color points for the custom palette
    # color_positions = [0, 0.5, 1]
    # from matplotlib.colors import LinearSegmentedColormap
    # # Create a LinearSegmentedColormap
    # cmap = LinearSegmentedColormap.from_list('custom_palette', list(zip(color_positions, colors)), N=n_colors)
    
    # return cmap

    import seaborn as sns

    colors = sns.color_palette(colors)
    cmap = sns.blend_palette(colors, as_cmap=True)

    return cmap

def generate_table(df, plainbar=False, sufix=''):

    # Create style object
    styler = df.style

    # if not plainbar:
    shoulder_values = [0.43,0.28,0.86,0.015]
    symmetric_values = [0.24,0.11,0.26,0.028]
    mins = [min(a,b) for a,b in zip(shoulder_values,symmetric_values)]
    maxs = [max(a,b) for a,b in zip(shoulder_values,symmetric_values)]
    print(mins, maxs)
    ranges = [(min_*0.8,max_*1.3) for min_,max_ in zip(mins,maxs)]
    print(ranges)
    print()

    # Create a custom colormap that transitions from blue to white and back to blue
    # colors = ['royalblue','white', 'white','lightsteelblue', 'royalblue']
    # colors = ['seagreen','mediumseagreen', 'white','mediumseagreen', 'seagreen']
    colors = ['rebeccapurple','mediumpurple','white','mediumpurple','rebeccapurple']
    cm1 = create_custom_palette(colors)

    # save color bar reference
    fig,ax = plt.subplots()
    pu.plot_cmap(fig, ax, [], location=[0.5,0,0.08,1], cmap=cm1) # [x, y, width, height]
    fig.delaxes(ax) 
    plt.savefig('color_bar_1'+'.pdf', format='pdf', bbox_inches='tight')

    # Create map for amplitude
    # colors = ['white', 'lightsteelblue', 'royalblue']
    # colors = ['white', 'mediumseagreen', 'seagreen']
    colors = ['white','mediumpurple','rebeccapurple']
    cm2 = create_custom_palette(colors)      

    # cm3 = plt.get_cmap('Purples')
    cm3 = cm1
    
    # save color bar reference
    fig,ax = plt.subplots()
    pu.plot_cmap(fig, ax, [], location=[0.5,0,0.08,1], cmap=cm2) # [x, y, width, height]
    fig.delaxes(ax) 
    plt.savefig('color_bar_2'+'.pdf', format='pdf', bbox_inches='tight')


    # for metric in metrics:
    styler = styler.background_gradient(cmap=cm3, vmin=-100, vmax=100)

    if any('ref' in s for s in df.columns):
        # Apply background color gradient to each column based on min-max values.
        styler = styler.background_gradient(cmap=cm1, subset=['duration_ref'],
                                            low=mins[0], high=maxs[0],
                                            vmin=ranges[0][0], vmax=ranges[0][1])

        styler = styler.background_gradient(cmap=cm1, subset=['depol._ref'],
                                            low=mins[1], high=maxs[1],
                                            vmin=ranges[1][0], vmax=ranges[1][1])
        
        styler = styler.background_gradient(cmap=cm1, subset=['repol._ref'],
                                            low=mins[2], high=maxs[2],
                                            vmin=ranges[2][0], vmax=ranges[2][1])

        styler = styler.background_gradient(cmap=cm2, subset=['amplitude_ref'],
                                            low=mins[3], high=maxs[3])
                                         #  vmin=0, vmax=ranges[3][1])

    # text style
    styler = styler.set_properties(**{'text-align': 'center', 'font-family': 'garuda','width': '180px'})
    styler = styler.format(precision=3)

    # Convert style object to HTML
    html = styler.to_html()

    import pdfkit
    pdfkit.from_string(html, 'styled_table-%s.pdf'%(name.replace('.','')+sufix))

    print('Saving styled_table-%s.pdf'%(name.replace('.','')+sufix))
    

import argparse

#     print("Example: python3 superpos_from_model.py ../../laser_model/HH/data/gna/ gna \"Gna simulation\" 0.001 8 20")
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="Path to the file to show stats from")
ap.add_argument("-dt", "--time_step", required=False, default=0.01, help="Sampling freq of -fs")

args = vars(ap.parse_args())


path = args['path']

dt = float(args['time_step'])

dirs = sorted(glob.glob(path+"/*/*"))
# dirs.sort(key=os.path.getmtime)

print(dirs)


if(dirs==[]):
    print("Error: No dirs found. Check the extension provided")


path = path.replace('*','')
name = path.replace('/','')

all_df = {}

df = pd.DataFrame()

# for c in candidate:
#     print(c)
for i,dir_ in enumerate(dirs):
    files = sorted(glob.glob(dir_+"/*waveform.pkl"))

    param = dir_[dir_.find('/')+2:]

    param = ''.join([p for p in param if not p.isnumeric()])
    param = param.replace('/','_')
    print(param)

    # if 'cgc' in param:
    #     continue
    plt.figure()
    plt.title(dir_)

    for file in files:
        # print(file)
        with open(file,'rb') as f:
            waveform = pkl.load(f)

        # print(waveform.shape)
        # plt.plot(waveform.T)
        # plt.show()

        ref = file.find("Euler")
        if(ref!=-1):
            f_events = file[:ref]+"spikes_"+file[ref:file.rfind('waveform')-1]+'.pkl'
        else:
            f_events = file[:file.rfind('waveform')]+"spikes.pkl"#+f_format


        try:
            first_line = get_first_line(f_events)
        except Exception as e:
            print("************************",e)
            continue
        if first_line == '':
            continue

        # print(first_line)

        if 'HH' in file or 'Vav' in file:
            dt = 0.001
            n_points=3
            repol_points=10
        else:
            dt = 0.01
            n_points=10
            repol_points=60
        # print(dt)

        dur_refs,th = laser_utils.get_spike_duration(waveform, dt, plot=True)
        duration = dur_refs[1]-dur_refs[0]
        # print("Duration value:", duration)

        amplitude = laser_utils.get_spike_amplitude(waveform, dt)
        # print("Amplitude value:", amplitude)

        slope_dep, slope_rep = laser_utils.get_slope(waveform, dt, n_points=n_points, plot=True)
        slopes_dep2, slopes_rep2 = laser_utils.get_slope2(waveform, dt, n_points=n_points, repol_points=repol_points, plot=True)


        if param == 'Cm':
            param = 'Cm\n'+dir_[:dir_.find('/')]

        new_row = pd.DataFrame([{'param':float(first_line), 'duration':duration, 'amplitude':amplitude, 'depol.':slope_dep, 'repol.':slope_rep}])

        try:
            all_df[param] = pd.concat([all_df[param], new_row])
        except:
            all_df[param] = pd.DataFrame()
            all_df[param] = pd.concat([all_df[param], new_row])
    
    # if show:
    #     plt.show()
    plt.close()


plt.rcParams.update({'font.size': 20})

metrics_labels = ['duration','depol.','repol.','amplitude']

# Get normalized differences
df_diffs = {}
for rc,(key,value) in enumerate(all_df.items()):
    #Value is the dataframe for each Candidate/Neuron (directory in the path given)
    max_ = value.max()
    min_ = value.min()

    df_diffs[key] = abs((max_ - min_))/abs(max_)

# plot table 
# # get dataframe from dict
# DF_diffs = pd.concat(df_diffs, axis=1).T
# # generate table colored by metric
# generate_table(DF_diffs[metrics_labels])

# generate_table with change percentages:

# Get DF with percentages
shoulder_values = {'duration': 0.43, 'depol.': 0.28, 'repol.': 0.86, 'amplitude': 0.015}
symmetric_values = {'duration': 0.24, 'depol.': 0.11, 'repol.': 0.26, 'amplitude': 0.028}

exp_mean_change = {'duration': 0.30, 'depol.': 0.21, 'repol.': 0.29, 'amplitude': 0.016}

# Get normalized differences
dict_percent = {}

dict_percent['.ref'] = {}
# For each model
for rc,(model,metrics) in enumerate(df_diffs.items()):
    dict_percent[model] = {}

    # Iterate all metrics in model
    
    for i, (label, metric) in enumerate(metrics.iteritems()):
        if 'param' in label:
            continue
        #Value is the dataframe for each Candidate/Neuron (directory in the path given)
        # average_reference = (shoulder_values[label] - symmetric_values[label])/2 
        # average_reference = ((shoulder_values[label]))
        average_reference = exp_mean_change[label]

        dict_percent[model][label] = (average_reference - metric)/average_reference * 100
        dict_percent['.ref'][label+'_ref'] = average_reference
        dict_percent[model][label+'_ref'] = metric
        # dict_percent[model][label] = abs(shoulder_values[label] - metric) * 100
        # dict_percent[model][label] = (symmetric_values[label] - metric) * 100
        # dict_percent[model][label] = (shoulder_values[i-1] - metric) * 100
        # dict_percent[model][label] = (symmetric_values[i-1] - metric) * 100


df_percent = pd.DataFrame(dict_percent)
df_percent = df_percent.T

generate_table(df_percent[['duration', 'depol.', 'repol.', 'amplitude']], plainbar=True, sufix='percent')


#antigua y nueva

generate_table(df_percent[['duration', 'duration_ref', 'depol.', 'depol._ref', 'repol.', 'repol._ref', 'amplitude', 'amplitude_ref']], plainbar=False, sufix='percent_n_refs')
