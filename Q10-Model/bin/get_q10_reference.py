import numpy as np
import pandas as pd
import seaborn as sns
import glob
import sys
import matplotlib.pyplot as plt




path = sys.argv[1]
title = sys.argv[2]

# get different_files from folder
files = sorted(glob.glob(path + "/*/differences-df.pkl"))

# Custom sorting function to move names containing "Gd" to the end
def custom_sort(name):
    return name if ("all" not in name and "Gd" not in name and "Gnap" not in name) else "z" + name  # Prefix with "z" to move to the end

def custom_sort2(name):
    return name if ("Gnat" in name) else "a" + name  # Prefix with "z" to move to the end

print(files)

files = sorted(files, key=custom_sort2)
files = sorted(files, key=custom_sort)

print(files)

metrics = ['duration (ms)', 'depolarization slope (mV/ms)', 'repolarization slope (mV/ms)', 'amplitude (mV)']
# exp_mean_change = [0.30,  0.21,  0.29, 0.016]
# exp_mean_change = [0.3266, 0.2344, 0.4207, 0.0133]
exp_mean_change = [ 31.97,  22.55, 37.14, 1.99]
# exp_mean_change = {'duration': 31.97, 'depol.': 22.55, 'repol.': 37.14, 'amplitude': 1.99}

std_low = [12.912403, 0, 0.203289, 0]
std_up = [51.038393, 46.619157, 74.093582, 6.041610]


positions = [1, 2, 3, 4]

# colors = {'Ga': (0.4, 0.7607843137254902, 0.6470588235294118), 'Gd': (0.9882352941176471, 0.5529411764705883, 0.3843137254901961), 'Ghva': (0.5529411764705883, 0.6274509803921569, 0.796078431372549), 'Glva': (0.9058823529411765, 0.5411764705882353, 0.7647058823529411), 'Gnat': (0.6509803921568628, 0.8470588235294118, 0.32941176470588235), 'Gnap': (1.0, 0.8509803921568627, 0.1843137254901961)}

if any('all' in file for file in files):
    extra = 4
else:
    extra = 2

print(len(files)+extra)

colors = plt.cm.terrain(np.linspace(0, 1, 8))
from cycler import cycler
# plt.gca().set_prop_cycle(cycler('color', colors))
color_cycle = cycler('color', colors)
color_iter = iter(color_cycle)
names = []


pretty_labels = {'Ga': r'$I_A$', 'Gd': r'$I_D$', 'Ghva': r'$I_{HVA}$', 'Glva': r'$I_{LVA}$', 'Gnat': r'$I_{NaT}$', 'Gnap': r'$I_{NaP}$', 'al': r'All with $\Delta T=5$', 'all3': r'All with $\Delta T=3$', 'all': 'All3'}


plt.figure(figsize=(9, 4))

for i, file in enumerate(files):
    # Find the last occurrence of '/'
    last_slash_index = file.rfind('/')

    # Find the second-to-last occurrence of '/'
    second_last_slash_index = file.rfind('/', 0, last_slash_index)

    candidate_name = file[second_last_slash_index + 1:last_slash_index]
    simple_candidate_name = candidate_name[candidate_name.find('_')+1: candidate_name.rfind('_')]

    # if candidate_name == 'dT' or candidate_name == 'all':
    #     continue

    names.append(candidate_name[candidate_name.find('_')+1: candidate_name.rfind('_')])

    # Load DataFrame from .pkl file
    df = pd.read_pickle(file)

    # df[metrics]*=100

    # Display DataFrame contents
    print()
    print(candidate_name)
    print(simple_candidate_name)
    print()

    print(df)
    print()

    # print(i/len(files))
    i_pos = [pos+0.1*i for pos in positions]
    # print(i_pos)
    # try:
    #     # plt.bar(i_pos, df[metrics], label=candidate_name, width=0.1, alpha=0.6, color=colors[simple_candidate_name])
    #     plt.bar(i_pos, df[metrics], label=candidate_name, width=0.1, alpha=0.8, color=colors[i])
    # except:

    if simple_candidate_name == 'Gnap' or simple_candidate_name == 'Gnat': # skip color to differentiate from Gd  
        next(color_iter)

    color = next(color_iter)['color'] if simple_candidate_name != 'al' else 'indianred'

    if simple_candidate_name == 'Gnap':
        color = 'sandybrown'
    
    if simple_candidate_name == 'all3':
        color = 'darkred'

    plt.bar(i_pos, df[metrics], label=pretty_labels[simple_candidate_name], width=0.1, alpha=1, color=color, clip_on=True)
    plt.suptitle(title)

    print(df[metrics])

    print("\nPercentage of change for ", candidate_name)
    for i, metric in enumerate(metrics):
        print(metric)
        print("\tOriginal value", df[metric]*100)
        print("\tPercentage ", (df[metric]*100 - exp_mean_change[i])/exp_mean_change[i] * 100)
        min_ = std_low[i]
        max_ = std_up[i]

        # dict_percent[model][label] = (metric - average_reference)/average_reference * 100
        print("\tGradient value ",  (df[metric]*100 - min_)/(max_-min_))



ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


# duration                 9.531498
# amplitude                2.023198
# repolarization slope    18.472573
# depolarization slope    12.029966

std = [9.53, 12.02, 18.47, 2.02]

i_pos = [pos+0.1*(i+extra) for pos in positions]
print(i_pos)

print(exp_mean_change)
exp_mean_change = [value/100 for value in exp_mean_change]

std_low = [value/100 for value in std]
std_up = [value/100 for value in std]

print(exp_mean_change)

plt.bar(i_pos, exp_mean_change, label='Exp. change', width=0.1, alpha=0.9, color='black')

# Añadir barras de error
plt.errorbar(i_pos, exp_mean_change, yerr=[std_low, std_up], fmt='none', color='grey', capsize=5)

# names.append("Gnap")
# # Set3 color map with 12 distinct colors
# set3_colors = sns.color_palette("Set2", n_colors=len(names))

# # Create a dictionary mapping names to colors
# name_color_map = dict(zip(names, set3_colors))
# print(name_color_map)

center = 0.25

plt.xticks([1+center, 2+center, 3+center, 4+center], metrics)
plt.ylim(0, 1)
plt.ylabel("Normalized change in the waveform model")

plt.legend()
plt.tight_layout()

plt.savefig("comparative_barplot.pdf",format='pdf')
# plt.show()
