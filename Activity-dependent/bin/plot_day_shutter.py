# -*- coding: utf-8 -*-
# @File name: plot_day_shutter.py
# @Author: Alicia Garrido-Peña
# @Date:   2023-08-09 18:26:14
# @Contact: alicia.garrido@uam.es

# "Copyright (c) 2023 Alicia Garrido-Peña. All Rights Reserved."
# Use of this source code is govern by GPL-3.0 license that 
# can be found in the LICENSE file

# Code used for manuscript submitted. 
# If you use any of this code, please cite:
# Garrido-Peña, Alicia, Sánchez-Martín, Pablo, Reyes-Sanchez, Manuel, Levi, Rafael, Rodriguez, Francisco B., Castilla, Javier, Tornero, Jesús, Varona, Pablo. Modulation of neuronal dynamics by sustained and activity-dependent continuous-wave near-infrared laser stimulation. Submitted.
#
#
import re
import os
import argparse
import matplotlib.pyplot as plt

import glob

from shutter_functions import *

all_metrics = ['to_on','to_off','duration','repol_slope','depol_slope','repol_slope2','depol_slope2', 'file']


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="Path to the file to show stats from")
ap.add_argument("-ex", "--extension", required=False, default='', help="Extension in the file, such as 'depol' in 'exp1_depol_30.asc'")
ap.add_argument("-bin", "--bin-size", required=False, default=50, help="Bin size range")
ap.add_argument("-sa", "--save", required=False, default='y', help="Option to save plot file")
ap.add_argument("-sh", "--show", required=False, default='y', help="Option to show plot file")
ap.add_argument("-pkl", "--pkl", required=False, default='y', help="Read from pkl if existing")
ap.add_argument("-rang","--range", required=False, default=None, help="Cut range for boxplot")
ap.add_argument("-rastep","--range_step", required=False, default=50, help="Step for distance chunks")
ap.add_argument("-lim","--limit", required=False, default=np.inf, help="Time limit to beginning of shutter event")
args = vars(ap.parse_args())


path = args['path']
extension = args['extension'] #subpath where 

show= True if args['show']=='y' else False 
save= True if args['save']=='y' else False 

step_range = int(args['range_step'])

lim = float(args['limit'])

read = args['pkl'] != 'y' or (not len(glob.glob(path+path[path[:-1].rfind('/'):-1]+"%s_shutter*.pkl"%extension)) > 0)

day_dict = {m:[] for m in all_metrics}
controls_dict = {'control_'+m if m!='file' else m:[]  for m in all_metrics}
recovery_dict = {'recovery_'+m if m!='file' else m:[] for m in all_metrics}
laser_dict = {'laser_'+m if m!='file' else m:[] for m in all_metrics}

if read:
	print("Reading from asc")
	files = glob.glob(path+"*exp*%s*.asc"%extension)
	files.sort(key=os.path.getmtime)

	if len(files) == 0:
		print("Files not found for path:", path) 
	# 	exit() 

	files_laser = glob.glob(path+"*exp*laser*.asc")

	# print(files_laser)
	files += files_laser

	files = list(set(files))

	path_images = path+"events/images/"
	os.system("mkdir -p %s"%path_images)

	min_dur = 1
	max_dur = np.inf

	for file in files:
		file_ext = file[file.rfind('/'):-4]
		file = path + '/events/' + file_ext
		print(file)

		plt.figure()

		# if extension == '' and ('depol' not in file and 'repol' not in file and 'slope' not in file):
		if ('depol' not in file and 'repol' not in file and 'slope' not in file):
			# continue
			bunched_data = get_metrics_from_file(file, 'laser', slope_position=0.99, dict_=laser_dict, ext = 'laser_',max_dur=max_dur)
		else:
			bunched_data = get_metrics_from_file(file, 'laser', slope_position=0.99, dict_=day_dict,max_dur=max_dur)

		plt.title('laser\n'+file_ext)
		plt.tight_layout()
		print(path+"events/images/charact_"+file_ext[1:-4]+'_laser')
		if save:
			plt.savefig(path+"events/images/charact_"+file_ext[1:-4]+'_laser')
		# plt.show()

		plt.figure()
		bunched_data_control = get_metrics_from_file(file, 'control', slope_position=0.99, dict_=controls_dict, ext='control_',max_dur=max_dur)
		if ('depol' not in file and 'repol' not in file and 'slope' not in file):
			print(bunched_data_control)
		plt.title('control\n'+file_ext)
		plt.tight_layout()
		if save:
			plt.savefig(path+"events/images/charact_"+file_ext[1:-4]+'_control')
		
		plt.figure()
		bunched_data_recovery = get_metrics_from_file(file, 'recovery', slope_position=0.99, dict_=recovery_dict, ext='recovery_',max_dur=max_dur)
		plt.title('recovery\n'+file_ext)
		plt.tight_layout()
		if save:
			plt.savefig(path+"events/images/charact_"+file_ext[1:-4]+'_recovery')
		# plt.show()

	df_controls = pd.DataFrame.from_dict(controls_dict, orient='index')
	df_controls = df_controls.transpose()

	df_recovery = pd.DataFrame.from_dict(recovery_dict, orient='index')
	df_recovery = df_recovery.transpose()

	df = pd.DataFrame.from_dict(day_dict, orient='index')
	df = df.transpose()

	df_laser = pd.DataFrame.from_dict(laser_dict, orient='index')
	df_laser = df_laser.transpose()


	print(df.describe())

	save_path = path + path[path[:-1].rfind('/'):-1]
	df_controls.to_pickle(save_path +"%s_shutter_controls.pkl"%extension)
	df_recovery.to_pickle(save_path +"%s_shutter_recovery.pkl"%extension)
	df_laser.to_pickle(save_path +"%s_shutter_laser_continuous.pkl"%extension)
	df.to_pickle(save_path +"%s_shutter_laser.pkl"%extension)

else:
	save_path = path + path[path[:-1].rfind('/'):-1]
	df = pd.read_pickle(save_path +"%s_shutter_laser.pkl"%extension)
	df_controls = pd.read_pickle(save_path +"%s_shutter_controls.pkl"%extension)
	df_recovery = pd.read_pickle(save_path +"%s_shutter_recovery.pkl"%extension)
	df_laser = pd.read_pickle(save_path +"%s_shutter_laser_continuous.pkl"%extension)


if df.empty:
	raise Exception("No data found, check file path",save_path+"%s_shutter_laser.pkl"%extension)

df = df.dropna()

metrics = ["duration", "depol_slope", "repol_slope"]

def clean_zeros():
	remove_zeros(df, metrics)
	remove_zeros(df_controls, ['control_'+m for m in metrics])
	remove_zeros(df_recovery, ['recovery_'+m for m in metrics])
	remove_zeros(df_laser, ['laser_'+m for m in metrics])

def clean_data(lim=2, g_by='file', metrics=['duration', 'depol_slope', 'repol_slope']):
    clean_zeros()

    # print(df.loc[df['depol_slope']<8])
    # print(df.loc[df['depol_slope']<8].mean())
    # print(df.loc[df['depol_slope']<8].std())
    df_clean = remove_outliers(df, metrics, lim, g_by=g_by)

    if g_by == 'range':
        return 
    df_controls_clean = remove_outliers(df_controls,
            ['control_'+m for m in metrics], lim, g_by=g_by)
    df_laser_clean = remove_outliers(df_laser,
            ['laser_'+m for m in metrics], lim, g_by=g_by)
    df_recovery_clean = remove_outliers(df_recovery,
            ['recovery_'+m for m in metrics], lim, g_by=g_by)

clean_data(1.5, metrics=['duration'])


if lim != np.inf:
	df.drop(df[df.to_off > lim].index, inplace=True)


df.loc[:,"pulse"] = df["to_on"]-df["to_off"]
df.loc[:,"pulse"] = pd.to_numeric(df["pulse"])

if args['range'] is not None:
	cut_range = args['range'].replace('\\','').split(',')

	df.drop(df[df.to_off < int(cut_range[0])].index, inplace=True)
	df.drop(df[df.to_off > int(cut_range[1])].index, inplace=True)
 
ext = "ext"+extension+"_"

path_own = ext + str(args['range_step']) + str(args['range'].replace('\\','') if args['range'] is not None else '' + "_" + str(step_range))
path_images = path + '/events/images/shutter/' +path_own +'/'
os.system("mkdir -p %s"%path_images)

metrics = ["duration", "depol_slope", "repol_slope"]


clean_zeros()


cut_range = args['range']    

df = set_range(df, cut_range, 'to_off', step_range) # does not work for to_on

print(df)

clean_data(1, g_by='range')

print(df)

plt.rcParams.update({'font.size': 50})

for metric in metrics:

	add_norm_diff(df, metric, df_controls=df_controls, df_recovery=df_recovery, df_laser=df_laser, norm = False)

	cut_range = args['range']

	plot_boxplot_mean(df,"to_off",metric, cut_range, step_range, df_controls=df_controls, df_recovery=df_recovery, df_laser=df_laser)
	plt.suptitle(path_images)
	
	if save:
		savefig(path, path_images, "_%s_mean_boxplot_to_off"%metric,format='.pdf')
		savefig(path, path_images, "_%s_mean_boxplot_to_off"%metric,format='.svg')


if show:
	plt.show()
########################

