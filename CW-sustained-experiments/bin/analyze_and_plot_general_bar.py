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


def clean_by(df, column, max_value, min_value):
    df = df[df.index.isin(df.loc[df[column] > 20].index)].copy()
    discard = df[df.index.isin(df.loc[df[column] <= 25].index)].copy()
    for file in discard.groupby('file').groups.keys():
        df.drop(df.index[df['file']==file], inplace=True)
    return df

def add_metrics(df):
    df['duration'] = df['waveform'].apply(lambda x: laser_utils.get_spike_duration_value(x.T,dt))
    df = df[df.index.isin(df.loc[df['duration'] > 0].index)].copy()

    df['amplitude'] = df['waveform'].apply(lambda x: laser_utils.get_spike_amplitude(x.T,dt))
    df['slopes'] = df['waveform'].apply(lambda x: laser_utils.get_slope(x.T,dt))
    df[['depolarization slope', 'repolarization slope']] = pd.DataFrame(df['slopes'].tolist(), index=df.index)
    df.drop(columns=['slopes'], inplace=True)

    df['slopes2'] = df['waveform'].apply(lambda x: laser_utils.get_slope2(x.T,dt))
    df[['depolarization slope2', 'repolarization slope2']] = pd.DataFrame(df['slopes2'].tolist(), index=df.index)
    df.drop(columns=['slopes2'], inplace=True)

    return df

def plot_boxplot(df):

    # Plot general boxplot
    plt.rcParams.update({'font.size': 20})

    n_elements = len(df.groupby('file').groups.keys())

    boxprops=dict(color='k')
    meanprops=dict(markeredgecolor='k')
    medianprops=dict(color='k')
    meanpointprops = dict(marker='D', markeredgecolor='black',
                          markerfacecolor='black')

    axes = df.boxplot(by=['type'], column=metrics, grid=False,
                    showmeans=True, patch_artist=True, boxprops=boxprops,
                    medianprops=medianprops,meanprops=meanpointprops,
                     # meanprops=meanprops,
                    layout=(1,len(metrics)),figsize=(25,10), fontsize=25,
                    showfliers=False)

    types = df.groupby('type').sum().index.values

    axes = axes.flatten()
    for ax in axes:
        handles = []
        for i,color in enumerate(colors):
            # box face color
            ax.findobj(matplotlib.patches.Patch)[i].set_facecolor(color)
            # handle for legend colors
            handles.append(ax.findobj(matplotlib.patches.Patch)[i])
        ax.set_xticklabels('')
        ax.set_xlabel('')
        pu.remove_axes(ax,['top','bottom'])
        ax.set_xticks([])
        ax.set_xlabel(ax.get_title())
        ax.set_title('')

    plt.legend(handles, types, loc='upper right', fontsize=17)
    plt.suptitle('')
    axes[0].set_ylabel('Normalized variation from initial control (N=%d)'%n_elements)
    axes[0].set_ylim(0.2,2.2)

import seaborn as sns

def plot_violin(df):
    import numpy as np

    # Calculate mean for each category
    mean_values = df.groupby('type')[metrics].mean()

    # Plot general boxplot
    plt.rcParams.update({'font.size': 20})

    n_elements = len(df.groupby('file').groups.keys())

    fig, axes = plt.subplots(nrows=1, ncols=len(metrics), figsize=(30,10), sharey=True)
    for i, metric in enumerate(metrics):
        ax = axes[i]
        sns.violinplot(data=df, ax=ax, y=metric, x='type', column=metrics, grid=False,
                       palette=colors, fontsize=25, cut=0, inner=None, width=1)
        sns.boxenplot(data=df, ax=ax, y=metric, x='type', color="grey", saturation=0.5, width=0.05, showfliers=False)

        # Plot mean as a point with a different color
        for j, label in enumerate(ax.get_xticklabels()):
            category = label.get_text()
            mean = mean_values.loc[category, metric]
            ax.scatter(j, mean, marker='.', color='black', s=300, alpha=0.7)

    types = df.groupby('type').sum().index.values

    axes = axes.flatten()
    for ax in axes:
        ax.set_xticklabels('')
        pu.remove_axes(ax,['top','bottom'])
        ax.set_xticks([])
        ax.set_xlabel(ax.get_ylabel())
        ax.set_ylabel('')

    handles = [matplotlib.patches.Patch(facecolor=color) for color in colors]
    plt.legend(handles, types, loc='upper right', fontsize=17)
    plt.suptitle('')
    axes[0].set_ylabel('Normalized variation from initial control (N=%d)' % n_elements)
    axes[0].set_ylim(0.2, 2.2)
    plt.tight_layout()





#Only big changes:
def greater_than(value, error):
    return value >= error

def smaller_than(value, error):
    return value <= error

def not_dur_recovery(df, col1, col2, error=0.1):
    return abs(df[col1]-df[col2]) < (error*df[col1])

def get_drop_list(group, metrics, condition, error=0.6):
    the_min = ('', np.inf)
    to_drop_list = []
    for g_name in group.groups.keys():
        mini_df = pd.DataFrame(group.get_group(g_name)[metrics])
        mini_df = abs(mini_df) / mini_df.abs().max()
    
        min_duration = mini_df.min()['duration']
        # if min_duration < the_min[1]:
        the_min = (g_name, min_duration)
        print('%s, %f, %d'%(g_name, min_duration, mini_df.count()['duration']))
        if condition(min_duration, error):
            to_drop_list.append(g_name)

    return to_drop_list


def drop_by_file(df, to_drop_list):

    for file in to_drop_list:
        df.drop(df.index[df['file']==file], inplace=True)



import argparse


ap = argparse.ArgumentParser()

ap.add_argument("-p", "--path", required=True, help="Path to the file to analyze")
ap.add_argument("-en", "--exp_name", required=True, help="Experiment Name")
ap.add_argument("-pc", "--classifier", required=False, default=None, help="Path to the file with neural classification")
ap.add_argument("-pf", "--filter", required=False, default=None, help="Path to the file with file seleccion to filter data")
ap.add_argument("-mf", "--metric_filter", required=False, default=None, help="Metric to filter by. options: 'duration' 'amplitude', 'repolarization slope', 'depolarization slope'")

args = vars(ap.parse_args())

_dir = args['path']
classifier = args['classifier']
filter_path = args['filter']
exp_name = args['exp_name']
filter_metric = args['metric_filter']

try:
    df_classifier = pd.read_csv(classifier, sep='\t', names=['file','neu_type','comments'])
    print(df_classifier)
except:
    df_classifier = None
    pass


files = glob.glob(_dir+"/*/*.pkl")

if filter_path is not None:  
    print("Filtering data based on", filter_path)
    with open(filter_path, 'r') as file:
        filters = list(file.read().split('\n'))[:-1]
    files = [file for file in files if any(filter in file for filter in filters)]

print("Number of files after filter",len(files))


from pathlib import Path
output_path = Path(_dir+'/%s/results/'%exp_name)
img_path = Path(_dir+'/%s/results/images/'%exp_name)
log_path = Path(_dir+'/%s/results/log/'%exp_name)
img_path.mkdir(exist_ok=True, parents=True)
log_path.mkdir(exist_ok=True, parents=True)
output_path = str(output_path)


dt = 0.1

width = 100 / dt 

types = ['control', 'laser', 'recovery']

plot = False

colors = ['cornflowerblue','firebrick','olivedrab']

df = pd.DataFrame()
signals = {}

read_from_file = False


try:
    df = pd.read_pickle(output_path+'/df_all_waveforms.pkl')
except:
    read_from_file = True


for rc, pkl_name in enumerate(files):
    if not read_from_file:
        break
    print(pkl_name)

    # if 'exp30-03-2022' not in pkl_name:
    #     continue

    with open(pkl_name,'rb') as file:
        data = pkl.load(file)

    print(data.shape)
    name = pkl_name[pkl_name.rfind('/', 0, pkl_name.rfind('/')):-4]

    signals[name] = data

    if plot:
        fig, ax = plt.subplots(nrows=4, figsize=(40,30))

    for i,signal in enumerate(data.T):
        if i> 2:
            break
        amp = np.max(signal) - np.min(signal)
        max_height = amp - amp*0.3
        max_height = np.max(signal) - amp*0.3

        spikes_t, spikes_v = find_peaks(signal, height=max_height, distance=1000)

        results_half = peak_widths(signal, spikes_t, rel_height=0.5)

        print(results_half[0].shape)

        waveforms = np.array([signal[int(peak-width):int(peak+width)] for peak in spikes_t[1:-1]])

        if abs(waveforms.mean()) < 0.1:
            print(pkl_name)
            print(waveforms.mean(), waveforms.min(), waveforms.max())
            waveforms *=100
            # input()
        else:
            waveforms *=10

        for waveform, peak in zip(waveforms, spikes_t):
            new_row = pd.DataFrame([{'file':name, 'type':types[i], 'waveform':waveform, 'peak':peak}])
            df = pd.concat([df,new_row])


        if plot:

            time = np.arange(signal.shape[0])*dt 
            ax[i].plot(time, signal, color=colors[i])
            ax[i].plot(spikes_t*dt, signal[spikes_t], 'x')

            try:
                time = np.arange(waveforms.shape[1])*dt
                ax[3].plot(time, waveforms.T, color=colors[i])
            except Exception as e:
                print("Error ploting ", name,types[i])
                print("Waveform", waveforms.shape)
                print(e)
                pass


    if plot:
        plt.suptitle(name, fontsize=10)
        plt.xlabel("Time (ms)")
        plt.ylabel("Voltage")


        plt.savefig(pkl_name[:-3]+'png', dpi=120)
        # plt.show()
        plt.close(fig)

df.to_pickle(output_path+'/df_all_waveforms.pkl')


print(output_path+'/df_all_waveforms.pkl')

df = df.reset_index()

metrics = ['duration','amplitude','depolarization slope','repolarization slope','depolarization slope2','repolarization slope2']
metrics = ['duration','depolarization slope','repolarization slope','amplitude']


# Load/generate metrics

try:
    df = pd.read_pickle(output_path+'/df_all_waveforms_metrics.pkl')
except:
    df = add_metrics(df)
    df.to_pickle(output_path+'/df_all_waveforms_metrics.pkl')

print(len(df.groupby('file').groups.keys()))


df_absolute = df.copy()

# Normalize dataframe to the mean control
df = laser_utils.normalize_by(df,'control')

# Clean spikes with different shapes
org_std = df['duration'].std()

from scipy.stats import zscore
for col in metrics:
    df[col+'_std'] = df.groupby(['file','type'])[[col]].transform('std')
    df[col+'_z'] = df.groupby(['file','type'])[[col]].transform(zscore)

print("Number of spikes discarded:", df.groupby(['file','type']).count())

z_error = 0.1

df_outliers = df[df.index.isin(df.loc[(df['duration_z'] >= -z_error) & (df['duration_std'] >= z_error)].index)].copy()
df = df[df.index.isin(df.loc[(df['duration_z'] < -z_error) | (df['duration_std'] < z_error)].index)].copy()

print(df_outliers.file)
print(df.file)

print("Number of spikes discarded:", df_outliers.groupby(['file','type']).count())

#######################################################################################

# Get same number of spikes per file
df.groupby(['file','type']).count().to_csv(_dir+exp_name+'/results/log/files_info_original.log')

# n_spikes = 20
n_spikes = -1

df, discard = laser_utils.get_sample(df, n_spikes)
print("Number of files discarded:", len(discard.groupby('file').groups.keys()))
print("Number of files discarded:", discard.groupby('file').groups.keys())
with open(_dir+exp_name+'/results/log/discarded_files_less_%d.log'%n_spikes,'w') as f:
    f.write("Number of files discarded:" + str(list(discard.groupby('file').groups.keys())))
print(df.groupby(['file','type']).count())

df.groupby(['file','type']).count().to_csv(_dir+exp_name+'/results/log/files_info_reduced.log')
##############################################################################################

# Plot boxplot/violin

# plot_boxplot(df)
# plt.savefig(output_path+'/images'+'/continuous_laser_results_%d-spikes.pdf'%n_spikes, format='pdf')
plot_violin(df)
plt.savefig(output_path+'/images'+'/continuous_laser_results_%d-spikes_violin.pdf'%n_spikes, format='pdf')
# plt.show()

# df = df[df.index.isin(df.loc[~(df['file'] == '/exp08-09-2022/exp8_laser')].index)].copy()



# plt.show()

# Plot results in bars

plt.rcParams.update({'font.size':11})
fig, ax = plt.subplots(nrows=4, figsize=(30,10))
groups = df.groupby(['file','type']).mean()

for i, metric in enumerate(metrics):
    groups[metric].plot(kind='bar', ax=ax[i], color=colors*len(groups[metric].keys()))
    if i < len(metrics)-1:
        ax[i].set_xticklabels('')
    else:
        labels = ['']*len(groups[metric])
        labels = [label if n%3!=1 else groups[metric].keys()[n][0] for n,label in enumerate(labels)]
        ax[i].set_xticklabels(labels)

    ax[i].set_xlabel('')

    ax[i].set_title(metric)
plt.tight_layout()


plt.savefig(output_path+'/images'+'/continuous_laser_results_bars_%d-spikes.pdf'%n_spikes, format='pdf')

###################################################################################################
def plot_reduced(df, condition, error, title):
    print(condition.__name__)

    group = df.groupby('file')
    to_drop_list = get_drop_list(group, metrics, condition, error)
    drop_by_file(df, to_drop_list)

    print("Values %s than 0.6\n"%condition.__name__, set(to_drop_list))

    plot_boxplot(df)
    plt.savefig(_dir+title+'.pdf', format='pdf')
    plot_violin(df)
    plt.savefig(_dir+title+'_violin.pdf', format='pdf')

# # # plot reduced files
error=0.6
df_org = df.copy()
# title = '/continuous_laser_results_%d-spikes_filtered_small'%n_spikes
# plot_reduced(df, greater_than, error, title)

# df = df_org.copy()
# title = '/continuous_laser_results_%d-spikes_filtered_big'%n_spikes

# plot_reduced(df, smaller_than, error, title)

# plt.figure()
# plt.hist(df_absolute.loc[df_absolute['type']=='control','duration'])

##################################################################################################
# Plot results differenciating shoulder/no shoulder
if df_classifier is not None:

    # get only shoulder
    # dur_lim = 20
    df = df_org.copy()
    # no_shoulder_files = df_absolute.loc[(df_absolute['duration'] < dur_lim) & (df_absolute['type'] == 'control'), 'file']
    # no_shoulder_files = df_absolute.loc[(df_absolute['duration'] < dur_lim) & (df_absolute['type'] == 'control'), 'file']
    no_shoulder_files = df_classifier[df_classifier['neu_type']=='symmetrical']['file']
    no_shoulder_files = [file for file in df['file'] if any(substring in file for substring in no_shoulder_files)]
    no_shoulder_files = set(no_shoulder_files)
    print(no_shoulder_files)
    drop_by_file(df, no_shoulder_files)

    unique_files = df['file'].unique()
    np.savetxt(output_path + '/log/shoulder.log', unique_files, fmt='%s')
    df.to_pickle(output_path+'/df_shoulder_waveforms_metrics.pkl')


    # plot_boxplot(df)
    # plt.suptitle('Shoulder neurons')
    # plt.savefig(output_path+'/images'+'/continuous_laser_results_%d-spikes_shoulders.pdf'%n_spikes, format='pdf')
    plot_violin(df)
    plt.suptitle('Shoulder neurons')
    plt.savefig(output_path+'/images'+'/continuous_laser_results_%d-spikes_shoulders_violin.pdf'%n_spikes, format='pdf')


    # get only durations

    df = df_org.copy()
    # no_symmetrical_files = df_absolute.loc[(df_absolute['duration'] > dur_lim) & (df_absolute['type'] == 'control'), 'file']
    no_symmetrical_files = df_classifier[df_classifier['neu_type']=='shoulder']['file']
    print(no_symmetrical_files)
    no_symmetrical_files = [file for file in df['file'] if any(substring in file for substring in no_symmetrical_files)]
    no_symmetrical_files = set(no_symmetrical_files)

    print(no_symmetrical_files)
    drop_by_file(df, no_symmetrical_files)

    unique_files = df['file'].unique()
    np.savetxt(output_path + '/log/no_shoulder.log', unique_files, fmt='%s')
    df.to_pickle(output_path+'/df_symmetrical_waveforms_metrics.pkl')

    # plot_boxplot(df)
    # plt.suptitle('Symmetrical neurons')
    # plt.savefig(output_path+'/images'+'/continuous_laser_results_%d-spikes_symmetrical.pdf'%n_spikes, format='pdf')
    plot_violin(df)
    plt.suptitle('Symmetrical neurons')
    plt.savefig(output_path+'/images'+'/continuous_laser_results_%d-spikes_symmetrical_violin.pdf'%n_spikes, format='pdf')

    # title = '/continuous_laser_results_%d-spikes_symmetrical_big'
    # print("\n\n\nsymmetrical with big change")
    # plot_reduced(df, smaller_than, error, title)


    # plt.show()

#########################################################################################################

# Filter by metric

if filter_metric is not None:
    df = df_org.copy()
    metric = filter_metric
    group_means = df.groupby(['type','file'])[metric].mean()
    print(group_means)

    # Step 2: Determine the difference and the threshold
    threshold = 0.1 *group_means['control'] # 10%
    mean_diff = abs(group_means['recovery'] - group_means['control'])
    # print(mean_diff)

    # print()
    # print("THE THRESHOLD")
    # print(threshold)

    files_to_filter = mean_diff[mean_diff > threshold].index
    # print("THE FILES")
    np.savetxt(output_path + '/log/files_to_filter_by_%s.log'%metric, files_to_filter, fmt='%s')

    print("Number of files to filter:",len(files_to_filter))
    drop_by_file(df, files_to_filter)


    df.to_pickle(output_path+'/df_recovery_%s_waveforms_metrics.pkl'%metric)


#########################################################################################################
# Check outliers filtering:

# outliers_group = df_outliers.groupby(['file','type'])
# group = df.groupby(['file','type'])

# for groups_names in group.groups.keys():
#     fig = plt.figure(figsize=(10,10))
#     plt.title(str(groups_names))
#     if '27' in str(groups_names):
#         print(groups_names)
#     print(groups_names)

#     try:
#         for a_waveform in outliers_group.get_group(groups_names)['waveform']:
#             time = np.arange(a_waveform.size) *dt
#             plt.plot(time,a_waveform,color='r',alpha=0.2)

#             dur = laser_utils.get_spike_duration_value(a_waveform, dt, plot=True)
#             amplitude = laser_utils.get_spike_amplitude(a_waveform, dt)
#             depolarization_slope, repolarization_slope = laser_utils.get_slope(a_waveform, dt, plot=True)
#             slopes_dep2, slopes_rep2 = laser_utils.get_slope2(a_waveform, dt, slope_position=0.2, plot=True)
#     except:
#         pass

#     for a_waveform in group.get_group(groups_names)['waveform']:
#         time = np.arange(a_waveform.size) *dt
#         plt.plot(time,a_waveform,color='k',alpha=0.4)

#         # print(a_waveform)
#         dur = laser_utils.get_spike_duration_value(a_waveform, dt, plot=True)
#         amplitude = laser_utils.get_spike_amplitude(a_waveform, dt)
#         depolarization_slope, repolarization_slope = laser_utils.get_slope(a_waveform, dt, plot=True)
#         slopes_dep2, slopes_rep2 = laser_utils.get_slope2(a_waveform, dt, slope_position=0.2, plot=True)
   


#     plt.savefig(_dir+groups_names[0]+'_'+groups_names[1]+'_outliers')
#     plt.show()
#     plt.close(fig)
# # n_elements = len(df.groupby('file').groups.keys())

# # print("Number of files plotted:", n_elements)
