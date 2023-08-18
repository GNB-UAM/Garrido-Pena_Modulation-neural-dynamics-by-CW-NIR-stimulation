# -*- coding: utf-8 -*-
# @File name: t-test_continuous-laser.py
# @Author: Alicia Garrido-Peña
# @Date:   2023-08-04 12:57:08
# @Contact: alicia.garrido@uam.es

# "Copyright (c) 2023 Alicia Garrido-Peña. All Rights Reserved."
# Use of this source code is govern by GPL-3.0 license that 
# can be found in the LICENSE file

# Code used for manuscript submitted. 
# If you use any of this code, please cite:
# Garrido-Peña, Alicia, Sánchez-Martín, Pablo, Reyes-Sanchez, Manuel, Levi, Rafael, Rodriguez, Francisco B., Castilla, Javier, Tornero, Jesús, Varona, Pablo. Modulation of neuronal dynamics by sustained and activity-dependent continuous-wave near-infrared laser stimulation. Submitted.
#
#
import sys

import pandas as pd
import numpy as np

from scipy.stats import ttest_ind, ttest_rel, f_oneway, mannwhitneyu

script_path = sys.argv[0][:sys.argv[0].rfind('/')]
sys.path.append(script_path+'/../..')
import superpos_functions as laser_utils

# get test value.
def t_test_by_group(data, metrics, group1,group2,title, outpath, test=ttest_rel):
    groups = df.groupby(['file'])
    file_pvalues = []
    for group_name, df_group in groups:
        pvalues = [group_name]
        for metric in metrics:
            pvalue = test(df_group[metric][df_group['type'] == group1],
                                      df_group[metric][df_group['type'] == group2]).pvalue
            pvalues.append(format(pvalue,'e'))
        
        file_pvalues.append(pvalues)

    from tabulate import tabulate   
    # print(tabulate(pvalues, headers=metrics, tablefmt='latex'))
    print(tabulate(file_pvalues, headers=metrics, tablefmt='plain'))

    output_name = outpath[:-4]+ "_t-test_%s_by_file.txt"%(test.__name__)
    print("Saving t-test to ", output_name)
    with open(output_name, 'a') as file:
        # Write the output to the file
        file.write(title)
        np.savetxt(file, [metrics], delimiter=',', fmt='%s')
        for pvalues in file_pvalues:
            np.savetxt(file, [pvalues], delimiter=',', fmt='%s')

def t_test(group1, group2, metrics, title, outpath, test=ttest_rel):
    pvalues = []
    for metric in metrics:
        tresult = test(group1[metric], group2[metric])

        # print(metric, tresult)

        pvalues.append(format(tresult.pvalue,'e'))

    from tabulate import tabulate   
    print(tabulate([pvalues], headers=metrics, tablefmt='plain'))

    output_name = outpath[:-4]+ "_t-test_%s.txt"%(test.__name__)
    print("Saving t-test to ", output_name)
    with open(output_name, 'a') as file:
        # Write the output to the file
        file.write(title)
        np.savetxt(file, [metrics], delimiter=',', fmt='%s')
        np.savetxt(file, [pvalues], delimiter=',', fmt='%s')
        # file.write(tabulate([pvalues], headers=metrics, tablefmt='plain'))

# read dataframe (generated in analyze_and_plot_general_bar.py)
_dir = sys.argv[1]

df = pd.read_pickle(_dir)

# Normalize data
metrics = ['duration','depolarization slope','repolarization slope','amplitude']

# get group
df_controls = df[df['type']=='control']
df_lasers = df[df['type']=='laser']
df_recovery = df[df['type']=='recovery']

df_controls = df_controls.groupby('file').mean()
df_lasers = df_lasers.groupby('file').mean()
df_recovery = df_recovery.groupby('file').mean()



# test = mannwhitneyu
test = ttest_rel

print("\n\tTest ", test.__name__)

title="Control vs lasers\n"
print("\n",title)
t_test(df_controls, df_lasers, metrics, title, _dir, test=test)
title="Control vs recovery\n"
print("\n",title)
t_test(df_controls, df_recovery, metrics, title, _dir, test=test)
title="Recovery vs lasers\n"
print("\n",title)
t_test(df_recovery, df_lasers, metrics, title, _dir, test=test)



df,_ = laser_utils.get_sample(df, 20)

t_test_by_group(df, metrics, 'control', 'laser', "Control vs lasers\n", _dir, test=test)

t_test_by_group(df, metrics, 'control', 'recovery', "Recovery vs lasers\n", _dir, test=test)
