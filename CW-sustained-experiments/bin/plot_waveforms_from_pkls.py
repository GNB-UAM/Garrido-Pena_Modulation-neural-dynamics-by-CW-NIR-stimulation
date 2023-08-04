
from math import ceil
import pickle as pkl
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
import glob
import pandas as pd

import matplotlib

from matplotlib.patches import PathPatch
from matplotlib.ticker import AutoLocator, AutoMinorLocator


# custom library
script_path = sys.argv[0][:sys.argv[0].rfind('/')]
sys.path.append(script_path+'/../..')
import superpos_functions as laser_utils
import plot_utils as pu

plt.rcParams.update({'font.size': 25})

def plot_metrics(group, metrics, verb=False, ext=''):
    the_min = ('', np.inf)
    for g_name in group.groups.keys():

        fig, ax = plt.subplots(figsize=(1.7*len(metrics),12))
        plt.title(g_name)

        mini_df = pd.DataFrame(group.get_group(g_name)[metrics])
        # print(mini_df)
        # mini_df = (mini_df-mini_df.min()) / (mini_df.max()-mini_df.min())

        mini_df = abs(mini_df) / mini_df.abs().max() #abs for repolarization
    
        if verb:        
            min_duration = mini_df.min()['duration']
            # if min_duration < the_min[1]:
            the_min = (g_name, min_duration)
            print('%s, %f, %d'%(g_name, min_duration, mini_df.count()['duration']))


        mini_df.boxplot(grid=False)

        labels = [l.get_text().replace(' ', '\n') for l in ax.get_xticklabels()]
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylim(0,1.1)

        g_name = str(g_name)
        g_name = g_name.replace('(','')
        g_name = g_name.replace(', ','_')
        g_name = g_name.replace(')','')
        g_name = g_name.replace('\'','')

        save_name = _dir + g_name + ext
        plt.savefig(save_name+'.'+format, format=format, bbox_inches='tight')
        fig.clear()

def normalize_by(df, type):
    control_means = df[df['type']==type].groupby('file').mean()
    control_means = control_means[metrics]
    df_norm = df.copy()
    for group in control_means.index:
        df_norm.loc[df['file']==group, metrics] = df.loc[df['file']==group, metrics].apply(lambda x: abs(control_means.loc[group]-x), axis=1)

    return df_norm


def plot_metrics_bars(df, metrics, verb=False, ext=''):
    # plt.rcParams.update({'font.size': 25})
    the_min = ('', np.inf)

    df_norm = normalize_by(df, 'control')
    # df_norm = df
    # print(df_norm)

    group = df_norm.groupby('file')
    group_df = df.groupby('file')

    for g_name in group.groups.keys():
        metrics_pp = [metric.replace(' ', '\n') for metric in metrics.copy()]

        fig, ax = plt.subplots(figsize=(1.9*len(metrics),12))
        plt.title(g_name)
        # print(group.get_group(g_name))
        mini_df = pd.DataFrame(group.get_group(g_name).loc[df_norm.type=='laser',metrics])
        mini_df_control = pd.DataFrame(group.get_group(g_name).loc[df_norm.type=='control',metrics])
        mini_df_raw = pd.DataFrame(group_df.get_group(g_name).loc[df.type=='control',metrics])
        mini_df_raw_laser = pd.DataFrame(group_df.get_group(g_name).loc[df.type=='laser',metrics])

        mini_df = (mini_df) / (mini_df_raw.abs().mean())

        ax.bar(metrics_pp, abs(mini_df.mean()), color='firebrick')

        print('\n\n\n BARS VALUES')
        print(g_name)
        print(mini_df.mean())

        ax.set_ylim(0,1)
        ax.set_xticklabels(metrics_pp,rotation=45, ha='right')
        ax.set_ylabel("Normalized difference from control to laser", fontsize=25)
        pu.remove_axes(ax, ['top','right'])
        if verb:        
            min_duration = mini_df.min()['duration']
            # if min_duration < the_min[1]:
            the_min = (g_name, min_duration)
            print('%s, %f, %d'%(g_name, min_duration, mini_df.count()['duration']))

        g_name = str(g_name)
        g_name = g_name.replace('(','')
        g_name = g_name.replace(', ','_')
        g_name = g_name.replace(')','')
        g_name = g_name.replace('\'','')

        save_name = _dir + g_name + ext
        plt.savefig(save_name+'_bars.'+format, format=format, bbox_inches='tight')
        fig.clear()


_dir = sys.argv[1]

df = pd.read_pickle(_dir+'/df_all_waveforms.pkl')


colors = {'control':'cornflowerblue', 'laser':'firebrick','recovery':'olivedrab'}
dt = 0.1

lw = 60
rw = 30

# format = 'png'
format = 'pdf'

group = df.groupby(['file'])

for groups_names in group.groups.keys():
    fig = plt.figure(figsize=(10,10))
    plt.title(str(groups_names))

    plot = False
    for a_type, a_waveform in zip(group.get_group(groups_names)['type'],group.get_group(groups_names)['waveform']):


        dur = laser_utils.get_spike_duration_value(a_waveform, dt, plot=plot)
        amplitude = laser_utils.get_spike_amplitude(a_waveform, dt)
        depolarization_slope, repolarization_slope = laser_utils.get_slope(a_waveform, dt, plot=plot)
        slopes_dep2, slopes_rep2 = laser_utils.get_slope2(a_waveform, dt, slope_position=0.2, plot=plot)
        

        a_waveform = a_waveform[int(lw/dt):-int(rw/dt)]
        a_waveform = a_waveform-a_waveform[0]
        time = np.arange(a_waveform.size) * dt
        
        plt.plot(time,a_waveform,color=colors[a_type],alpha=0.1)
    
    type_group = group.get_group(groups_names).groupby('type')
    
    for group_name in type_group.groups.keys():
        # print(type_group.get_group(group_name)['waveform'].to_numpy())
        mean_waveforms = type_group.get_group(group_name)['waveform'].values
        all_w = mean_waveforms[0]
        for w in mean_waveforms[1:]:
            all_w = np.append(all_w, w, axis=0) 

        all_w = all_w.reshape((mean_waveforms.shape[0],mean_waveforms[0].shape[0]))
        all_w = np.array([w-w[0] for w in all_w])

        print(groups_names,"Number of spikes",group_name,all_w.shape)

        mean = np.mean(all_w, axis=0)
        mean = mean[int(lw/dt):-int(rw/dt)]
        time = np.arange(mean.size) *dt

        plt.plot(time, mean, color=colors[group_name], linewidth=2)

    plt.xlabel('ms')
    plt.ylabel('mV')
    plt.savefig(_dir+groups_names+'_superpos.png', dpi=200, bbox_inches='tight')
    fig.clear()


df = pd.read_pickle(_dir+'/df_all_waveforms_metrics.pkl')


print(df.columns)

metrics = ['duration','amplitude','depolarization slope','repolarization slope','depolarization slope2','repolarization slope2']


group = df.groupby(['file'])


plot_metrics(group, metrics, verb=True, ext='all_metrics')

metrics = ['duration','depolarization slope','repolarization slope','amplitude']

plot_metrics(group, metrics, verb=True)

# group = df.groupby(['file','type'])


# plot_metrics(group, metrics)
plot_metrics_bars(df, metrics, verb=True)

# plt.show()