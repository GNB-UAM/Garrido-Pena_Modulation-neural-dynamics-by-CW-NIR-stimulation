# -*- coding: utf-8 -*-
# @File name: metrics_model_Q10.py
# @Author: Alicia Garrido-Peña
# @Date:   2023-08-04 13:54:09
# @Contact: alicia.garrido@uam.es

# "Copyright (c) 2023 Alicia Garrido-Peña. All Rights Reserved."
# Use of this source code is govern by GPL-3.0 license that 
# can be found in the LICENSE file

# Code used for manuscript submitted. 
# If you use any of this code, please cite:
# Garrido-Peña, Alicia, Sánchez-Martín, Pablo, Reyes-Sanchez, Manuel, Levi, Rafael, Rodriguez, Francisco B., Castilla, Javier, Tornero, Jesús, Varona, Pablo. Modulation of neuronal dynamics by sustained and activity-dependent continuous-wave near-infrared laser stimulation. Submitted.
#
#
import os
import math 
import pickle as pkl
import glob
import json
import h5py
import yaml
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.preprocessing import minmax_scale
import sys
import argparse 
from scipy.signal import find_peaks
import pandas as pd


script_path = sys.argv[0][:sys.argv[0].rfind('/')]
sys.path.append(script_path+'/../..')
import superpos_functions as laser_utils
import plot_utils as pu

plt.rcParams.update({'font.size': 21})

dt = 0.001
def get_events(f_data, f_events, ms_r, ms_l, dt=0.001):
	#read data
	data = pd.read_csv(f_data, delimiter = " ",skiprows=2,header=None)
	data = data.values

	#read events
	try:
		events = pd.read_csv(f_events, delimiter = " ",skiprows=2,header=None)
		events = events.values
	except Exception as e:
		print("Error in file: ",f_events)
		print(e.args)
		return np.array([])

	points_r = int(ms_r /dt)
	points_l = int(ms_l /dt)

	waveforms = np.empty((events.shape[0],points_l+points_r),float)

	time = data[:,0]

	count =0
	for i,event in enumerate(events[:,0]):
		indx = np.where(time == event)[0][0] #finds spike time reference

		try:
			waveforms[i] =data[indx-points_l:indx+points_r,1]
		except:
			count +=1
			# print(i)

	print(count, "events ignored")
	# print(waveforms)
	return waveforms[2:-2] #Ignore 2 first events, usally artefacts

def get_waveforms(f,f_events,ms_l=100, ms_r=100, n_waveforms=1):
	waveforms = []

	waveforms_file = f[:-4] + '_waveforms.pkl'

	try:
		print("Trying to load",waveforms_file)
		with open(waveforms_file,'rb') as f_waveforms:
			waveforms = pkl.load(f_waveforms)

	except Exception as e:
		print("Error loading", e)
		print("Calculating waveforms")

		waveforms = get_events(f,f_events,ms_r, ms_l)
		print(waveforms.shape)

		if waveforms.shape[0]==0:
			waveforms = []
			print('rm %s'%f)
			os.system('rm %s'%f)
		else:
			waveforms = waveforms[len(waveforms)//2]
			print(waveforms.shape)
		# exit()

		print("Writing",waveforms_file)

		with open(waveforms_file,'wb') as f_waveforms:
			pkl.dump(waveforms, f_waveforms)

	return waveforms

def plot_df_by(qs_df, temps, ref_label):
	nrows = 4
	ncols = math.ceil(len(qs_df.columns)/nrows)
	fig, axes = plt.subplots(ncols=ncols,nrows=nrows,figsize=(ncols*4,nrows*3))
	axes = axes.flat
	norm_temps = minmax_scale(np.array(temps))
	# print(norm_temps)
	for i,column in enumerate(qs_df):
		# print(qs_df[column].index)
		# print("\n\n\n")
		# print(qs_df.loc[temps, column])
		axes[i].scatter(temps, qs_df[column], 
						s=1/norm_temps*plt.rcParams['lines.markersize'],color=colors)
		axes[i].set_xlabel(ref_label)
		axes[i].set_ylabel(column)

		# for x, y in zip(temps, qs_df.loc[temps, column]):
		#     axes[i].text(x, y, str(y), color="red", fontsize=12)
		# axes[i].margins(0.1)

def q10_formula(const, q10, dT):
	return const * (q10 ** (dT/10)) if q10 > 0 else 1*const

def get_q10_value(params, key_ref='Q10_'):
	static_items = dict(params).items()

	# print("name original q10 new")
	for key, value in static_items:   # iter on both keys and values
		if key.startswith(key_ref):
			param = key[len(key_ref):]
			if param not in params.keys():
				params[param]=1
			qs[param] = q10_formula(params[param], params[key], params['diff_T'])
			# if param == 'Q10_Gd':
			# 	print(param, qs[param])
			# print(params['diff_T'], key, params[param], params[key], qs[param])
	return qs

def get_logs(params_path):	
	with open(params_path) as log:
	    data = log.read()
	return json.loads(data)

def save_as_yaml(dict_file, path):
	print("Saving parameters at ",path)
	with open(path, 'w') as file:
		documents = yaml.dump(dict_file, file)

import matplotlib.colors as mcolors
import matplotlib.cm as cm

def plot_shape(waveforms, title, fig_format, all_labels, cmaplocation=[0.8, 0.25, 0.03, 0.6], figsize=(9,8.5)):
	time = np.arange(waveforms.shape[1])*dt
	# tam = (15,10)
	fig, ax = plt.subplots(figsize=figsize,constrained_layout=True)
	plt.gca().set_prop_cycle(cycler('color', colors))
	# plt.legend(all_labels,prop={'size': 6})

	cbar = pu.plot_cmap(fig, ax, colors, all_labels, location=cmaplocation)
	# cbar = pu.plot_cmap(fig, ax, colors, all_labels)
	# cbar.set_label(u"Δ Temperature ºC")
	cbar.set_label(u"Δ Temperature ºC", labelpad=-120, y=0.5, rotation=90)
	cbar.formatter.set_useOffset(False)
	cbar.ax.yaxis.set_major_formatter('{x:.1f}')

	ax.plot(time, waveforms.T)
	# ax.set_xlabel("Time (ms)")
	# ax.set_ylabel("Voltage (mV)")

	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	if 'dT/' in title or 'dT5' in title:
		ax.set_title(title)
	# ax.set_title('Aligned spikes')

	pu.small_x_axe(ax, label='ms')
	pu.small_y_axe(ax, label='mV')
	# plt.tight_layout()
	fig.subplots_adjust(right=1)  # Adjust the right margin of the figure

	# plt.tight_layout()


def plot_scatter(qs_df, param, metric, title, fig_format, save_name):	
	if 'duration' in metric: # WARNING: Remove this to remove y_axis in duration
		qs_df.plot.scatter( metric,param, color=colors, figsize=(10,8.5), s=200)
	else:
		qs_df.plot.scatter( metric,param, color=colors, figsize=(9,8.5), s=200)
	ax = plt.gca()
	if 'duration' in metric: # WARNING: Move this if after diff_T elif to remove y_axis in duration
		# plt.ylabel(u"Δ"+param)
		plt.ylabel(u"ΔTemperature ºC")
		# plt.ylabel("Spike "+metric)
		pu.remove_axes(ax)
	
	elif 'diff_T' in param:
		pu.remove_axes(ax, ['right','top','left'])
		ax.set_yticks([])
		# ax.set_yticklabels([''])
		plt.ylabel("")

		# plt.title(title)

	else: 
		pu.remove_axes(ax, ['right','top'])
		ax.set_yticklabels([''])
		# ax.set_yticks([])
		plt.ylabel("")
		plt.title(title)

	plt.xlabel(metric)
	plt.tight_layout()
	if save:
		plt.savefig(path+save_name+fig_format, format=fig_format[1:],dpi=200)


# colors = ['teal', 'lightsalmon', 'darkseagreen','maroon','teal', 'brown', 'blue', 'green','maroon']
colors=['b', 'r', 'g', 'brown', 'teal', 'maroon', 'lightsalmon', 'darkseagreen', 'k']

ap = argparse.ArgumentParser()

ap.add_argument("-p", "--path", required=True, help="Path to the file to analyze")
ap.add_argument("-c","--cols",required=False, default=None,help="Index to the elements in the trial separated by space")
ap.add_argument("-t","--trials",required=False, default=None,help="Selection of trials to plot")
ap.add_argument("-nw", "--n_waveforms", required=False, default=1, help="Number of waveforms to save per file")
ap.add_argument("-sf", "--save_format", required=False, default='.png', help="Format to save figure (png, pdf, etc)'")
ap.add_argument("-sa", "--save", required=False, default='y', help="y save figure")
ap.add_argument("-sh", "--show", required=False, default='n', help="y show figure")

args = vars(ap.parse_args())

path = args['path']
fig_format = args['save_format']

save = True if args['save']=='y' else False
show = True if args['show']=='y' else False

n_waveforms = int(args['n_waveforms'])


if '.' not in fig_format:
	fig_format = '.' + fig_format

files = sorted(glob.glob(path+"*[!_spikes].asc"))

if len(files) == 0:
	print("No files to plot")
	exit()

# files = files[:5]
# print(files)

# plt.gca().set_color_cycle(colors)

# durations = {}
all_qs = {}
all_waveforms = []
all_labels = []
# plt.figure()

for i,f in enumerate(files):

	qs = {}
	params_path = f[:-4] + '_params.log'
	f_events = f[:-4]+"_spikes.asc"
	file_name = f[f.rfind("/")+1:]

	print(f)
	print(f_events)
	print(params_path)

	params = get_logs(params_path)

	save_as_yaml(params, params_path[:-4]+'.yaml')

	waveforms = get_waveforms(f,f_events,ms_l=100, ms_r=100, n_waveforms=n_waveforms)
	if len(waveforms) == 0:
		print("Skiping",f)
		continue

	if len(waveforms.shape)>1:
		#align to the peak
		waveforms = np.array([w-w[0] for w in waveforms])
		# waveforms = np.array([w-np.max(w) for w in waveforms])
		a_waveform = waveforms[len(waveforms)//2]
	else:
		a_waveform = waveforms

	all_waveforms.append(a_waveform )
	all_labels.append(params['diff_T'])

	# for spike in waveforms:
	# All spikes the same in the model, get middle one
	dur_refs,th = laser_utils.get_spike_duration(a_waveform, 0.001)
	# plt.plot(a_waveform)
	# plt.show()
	duration = dur_refs[1]-dur_refs[0]
	print("Duration value:", duration)


	amplitude = laser_utils.get_spike_amplitude(a_waveform, 0.001)
	print("Amplitude value:", amplitude)


	slope_dep, slope_rep = laser_utils.get_slope(a_waveform, dt, plot=False)
	# slopes_dep2, slopes_rep2 = get_slopes(sf.get_slope2, waveforms, slope_position=slope_position)
	slopes_dep2, slopes_rep2 = laser_utils.get_slope2(a_waveform, dt, plot=False)



	# durations[params['diff_T']] = duration

	qs['amplitude (mV)'] = amplitude
	qs['duration (ms)'] = duration
	qs['depolarization slope (mV/ms)'] = slope_dep
	qs['repolarization slope (mV/ms)'] = slope_rep
	qs['depolarization slope2 (mV/ms)'] = slopes_dep2
	qs['repolarization slope2 (mV/ms)'] = slopes_rep2


	# try:
	# 	print(params)
	# 	print( params['general_Q10'])
	# 	qs['Q10'] = params['general_Q10']
	# except:
	qs['Q10'] = params['Q10_f']

	qs['diff_T'] = params['diff_T']

	qs = get_q10_value(params)
	qs['cm'] = params['cm'] + params['cm'] * params['gamma_T'] * params['diff_T']

	# all_qs[params['diff_T']] = qs
	all_qs[file_name] = qs
	# print(all_qs)
# plt.show()
# print("\nTHE DataFrame\n")


# print(all_qs)

qs_df = pd.DataFrame.from_dict(all_qs, orient='index')
# print(qs_df.n_rows)
print(qs_df)
# if (qs_df['diff_T'][0]==qs_df['diff_T']).all():
# 	print(path, all_labels)
# 	print(qs_df)
# 	all_labels = qs_df['Q10'].values
# 	print(all_labels)
# 	ref_value = ('diff_T',qs_df['diff_T'][0])
# else:
# 	ref_value = ('Q10',qs_df['Q10'][0])

ref_value = [('Q10',qs_df['Q10'][0]), ('diff_T',qs_df['diff_T'][1])]


if qs_df.empty or len(qs_df.index)==1:
	print("No data to plot")
	plt.figure()
	plt.plot([])
	plt.title(path[path[:-1].rfind('/')+1:-1])
	plt.savefig(path+'shape_zoom'+fig_format, format=fig_format[1:],dpi=200)
	exit()



colors = plt.cm.coolwarm(np.linspace(0,1,qs_df['diff_T'].size))
# colors = plt.cm.get_cmap('Oranges')
from cycler import cycler
plt.gca().set_prop_cycle(cycler('color', colors))


all_waveforms = np.array([w-w[0] for w in all_waveforms])
# all_waveforms = np.array(all_waveforms)

title = path[path[:-1].rfind('/')+1:-1]

plot_shape(all_waveforms, title, fig_format, all_labels)
if save:
	plt.savefig(path+'shape'+fig_format, format=fig_format[1:],dpi=200)


small_waveforms = all_waveforms[:,int(70/dt):-int(60/dt)]
# time = np.arange(small_waveforms.shape[1])*dt

plot_shape(small_waveforms, title, fig_format, all_labels)
if save:
	plt.savefig(path+'shape_zoom'+fig_format, format=fig_format[1:],dpi=200)
	plt.savefig(path+'shape_zoom'+'.pdf', format='pdf',dpi=200)

plot_shape(small_waveforms, title, fig_format, all_labels, cmaplocation=[0,0.1,0.04,0.8])
if save:
	plt.savefig(path+'shape_zoom_left_bar'+fig_format, format=fig_format[1:],dpi=200, bbox_inches='tight')
	plt.savefig(path+'shape_zoom_left_bar'+'.pdf', format='pdf', bbox_inches='tight')


metrics = set(qs_df.keys())-set(params.keys()).intersection(qs_df.columns)

print(metrics)
# params_names = path[path.rfind('Q10_')+4:path.find('_-1')].split('-')
# for param in qs_df.keys():
# 	# if param == path[path.rfind('Q10_')+4:path.find('_-1')]:
# 	if param in params_names:
# 		for metric in metrics:
# 			plot_scatter(qs_df, metric, param, title, fig_format, save_name=metric.replace('/','_')+'_'+param)
plt.rcParams.update({'font.size': 45})

for metric in metrics:
	# plot_scatter(qs_df, 'diff_T', metric, title, fig_format, save_name=metric.replace('/','_')+'_dt')
	plot_scatter(qs_df,'diff_T', metric, title, fig_format='.pdf', save_name=metric.replace('/','_')+'_dt')
	# plot_scatter(qs_df, 'Q10', metric, title, fig_format, save_name=metric.replace('/','_')+'_q10')

temps = qs_df['diff_T'].values
# norm_temps = minmax_scale(np.array(temps))
plot_df_by(qs_df, temps,u"ΔT")

plt.suptitle(title)
plt.tight_layout()
if save:
	plt.savefig(path+"parameters_dt"+fig_format, format=fig_format[1:],dpi=200)

temps = qs_df['Q10'].values

plot_df_by(qs_df, temps,u"ΔQ10")

plt.suptitle(title)
plt.tight_layout()
if save:
	plt.savefig(path+"parameters_Q10"+fig_format, format=fig_format[1:],dpi=200)

# for metric in metrics:

# 	temps = qs_df[metric].values

# 	plot_df_by(qs_df, temps,u"Δ"+metric)

# 	plt.suptitle(title)
# 	plt.tight_layout()
# 	if save:
# 		plt.savefig(path+"parameters_"+metric[:metric.find(' (')]+fig_format, format=fig_format[1:],dpi=200)



# close_all_figs()
# Plot difference in range for each parameter

def get_range(qs_df, column):
	min_T = qs_df['diff_T'].min()
	max_T = qs_df['diff_T'].max()

	v1 = qs_df.loc[qs_df['diff_T']==min_T, column].values[0]
	v2 = qs_df.loc[qs_df['diff_T']==max_T, column].values[0]

	return ((v2-v1)/abs(v1))*100



plt.figure()
variables = set(params.keys()).intersection(qs_df.columns) 
print(variables)

ranges = {}
for i, column in enumerate(variables):
	ranges[column] = get_range(qs_df, column)

plt.bar(ranges.keys(),ranges.values())



print(metrics)
list_metrics = list(metrics-{'Q10'}-{'depolarization slope2 (mV/ms)'}-{'repolarization slope2 (mV/ms)'})
# list_metrics = list(metrics-{'Q10'})
list_metrics.sort()
print(list_metrics)
list_metrics = [list_metrics[2],list_metrics[1], list_metrics[3], list_metrics[0]]

# fig, ax = plt.subplots(figsize=(10,10))
fig, ax = plt.subplots(figsize=(1.7*len(metrics),12))
# qs_df_norm = qs_df[list_metrics].apply(lambda x: ((x)-x.min())/(x.max()-x.min()))
qs_df_norm = abs(qs_df[list_metrics]) / qs_df[list_metrics].abs().max()


qs_df_norm.boxplot(column=list_metrics, fontsize=25, grid=False, ax=ax)
ax.set_xticklabels(['duration', 'depolarization\nslope', 'repolarization\nslope','amplitude'], rotation=45, ha='right')
ax.set_ylabel("Variability normalization to the maximum", fontsize=25)
list_metrics_copy = list_metrics.copy()
list_metrics = [metric.replace(' ','\n') for metric in list_metrics]


# print(list_metrics)

ax.set_xticklabels(list_metrics)

plt.tight_layout()
if save:
	plt.savefig(path+'metrics_boxplot'+fig_format, format=fig_format[1:])
	plt.savefig(path+'metrics_boxplot.pdf', format='pdf')


plt.rcParams.update({'font.size': 30})

qs_df_norm = abs(qs_df[list_metrics_copy].max()-qs_df[list_metrics_copy].min())/abs(qs_df[list_metrics_copy].max())
plt.figure(figsize=(2*len(list_metrics),17))
plt.bar(['duration', 'depolarization\nslope', 'repolarization\nslope','amplitude'], qs_df_norm, color='firebrick')
plt.xticks(['duration', 'depolarization\nslope', 'repolarization\nslope','amplitude'], ha='right', rotation=45, fontsize=30)
plt.suptitle('Spike waveform\n change', fontsize=50)

plt.tight_layout()

plt.ylim(0,0.8)

pu.remove_axes(plt.gca())

for ref in ref_value: 
	qs_df_norm = qs_df_norm.append(pd.Series([ref[1]], index=[ref[0]]))

print(qs_df_norm)
qs_df_norm.to_pickle(path+'differences-df.pkl')

if save:
	plt.savefig(path+'metrics_bars.pdf', format='pdf')

if show:
	plt.show()
