# -*- coding: utf-8 -*-
# @File name: plot_firing_rate.py
# @Author: Alicia Garrido-Peña
# @Date:   2023-08-10 21:29:51
# @Contact: alicia.garrido@uam.es

# "Copyright (c) 2023 Alicia Garrido-Peña. All Rights Reserved."
# Use of this source code is govern by GPL-3.0 license that 
# can be found in the LICENSE file

# Code used for manuscript submitted. 
# If you use any of this code, please cite:
# Garrido-Peña, Alicia, Sánchez-Martín, Pablo, Reyes-Sanchez, Manuel, Levi, Rafael, Rodriguez, Francisco B., Castilla, Javier, Tornero, Jesús, Varona, Pablo. Modulation of neuronal dynamics by sustained and activity-dependent continuous-wave near-infrared laser stimulation. Submitted.
#
#
from math import ceil
import os
import pickle as pkl
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import sem
import glob


script_path = sys.argv[0][:sys.argv[0].rfind('/')]
sys.path.append(script_path+'/../..')

import plot_utils as pu
colors = ['cornflowerblue','firebrick','olivedrab']


def color_axes(ax, color='r'):
	for spine in ax.spines:
		ax.spines[spine].set_color(color)
		ax.spines[spine].set_linewidth(2)

def is_excitation(f_rates, error=0.1):
	return abs(f_rates[0]-f_rates[2]) < (error*f_rates[0]) and abs(f_rates[1]-f_rates[0]) > (error*f_rates[0]) and f_rates[1]-f_rates[0] > 0

def is_inhibition(f_rates, error=0.1):
	return abs(f_rates[0]-f_rates[2]) < (error*f_rates[0]) and abs(f_rates[1]-f_rates[0]) > (error*f_rates[0]) and f_rates[1]-f_rates[0] < 0

def is_unchanged(f_rates, error=0.1):
	return abs(f_rates[0]-f_rates[2]) < (error*f_rates[0]) and abs(f_rates[1]-f_rates[0]) < (error*f_rates[0])

def plot_grid(all_f_rates,all_isis,all_names, title, no_titles=False):
	colors = ['C0','C1','C2']
	n_rows = 4
	n_cols = ceil(all_f_rates.shape[0]/ n_rows)

	# print(len(files))
	# print(n_rows, n_cols)
	fig_bar, ax_bar = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(4*n_cols,3*n_rows))
	fig_hist, ax_hist = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(4*n_cols,3*n_rows))

	for rc,(f_rates, isis,name) in enumerate(zip(all_f_rates,all_isis,all_names)):

		#I = Row * LineLenght + Column
		r = rc // n_rows
		c = rc % n_rows

		ax_bar[c,r].set_ylabel('Hz')
		ax_bar[c,r].bar(labels,f_rates, color=colors,alpha=0.3)

		if not no_titles:
			ax_bar[c,r].set_title(name, fontsize=10)
			ax_hist[c,r].set_title(name, fontsize=10)

		for i in range(3):
			if rc==1:
				ax_hist[c,r].hist(isis[i], alpha=0.3,color=colors[i], label=labels[i])
			else:
				ax_hist[c,r].hist(isis[i], alpha=0.3,color=colors[i])

		if c == n_rows-1:
			ax_hist[c,r].set_xlabel('ISI (ms)')

		if is_excitation(f_rates):
			color_axes(ax_hist[c,r])
			color_axes(ax_bar[c,r])

		if rc==1:
			fig_hist.legend()
	
	fig_hist.delaxes(ax_hist[-1,-1]) 
	fig_bar.delaxes(ax_bar[-1,-1]) 

	fig_hist.suptitle(title)
	fig_bar.suptitle(title)

	fig_hist.tight_layout()
	fig_bar.tight_layout()

	if save:
		title += '_no_titles_' if no_titles else ''
		fig_hist.savefig(_dir+'/FR_images/'+title+'isis.pdf',format='pdf')
		fig_bar.savefig(_dir+'/FR_images/'+title+'bar.pdf',format='pdf')


def plot_fr_errors(all_f_rates, title, linewidth=1):
	# plt.figure()
	plt.ylabel('Mean of Absolute Firing rate (Hz)')
	plt.title(title)
	# linestyle = {"linestyle":"--", "linewidth":linewidth, "color":'k'}
	plt.errorbar(['control','laser','recovery'], np.mean(all_f_rates,axis=0), yerr=sem(all_f_rates,axis=0),
					ecolor=colors, elinewidth=linewidth*2.2, linewidth=linewidth*2, color='k')

	plt.ylim(0.45,0.8)

	pu.remove_axes(plt.gca(),['top','right','bottom'])
	plt.xticks([])

	plt.tight_layout()
	if save:
		plt.savefig(_dir+'/FR_images/'+title+'.pdf',format='pdf')
	# plt.show()
	# plt.show()

def plot_diff_scatter(all_f_rates, title, all_names):
	fig, ax = plt.subplots()

	diffs = [abs(control-laser) for control,laser, rec in all_f_rates]
	controls = [control for control,_,_ in all_f_rates]

	colors = ['g' if is_excitation(triplet) else 'k' for triplet in all_f_rates ]

	ax.scatter(diffs, controls, color=colors)
	ax.set_xlabel('Controls FR')
	ax.set_ylabel('Control-laser FR')
	
	legend_elements = pu.custom_legend(['Excitation', 'Inhibition'],[('Scatter','g'),('Scatter','k')])

	ax.legend(handles=legend_elements)
	ax.set_title(title)

	# for i, txt in enumerate(all_names):
	# 	ax.annotate(txt[txt.find('/'):txt.rfind('/')], (diffs[i], controls[i]),fontsize='8')
	# plt.show()
	if save:
		print("Saving at: ", _dir+'/FR_images/'+title+'_diff_bar.pdf')
		plt.savefig(_dir+'/FR_images/'+title+'_diff_bar.pdf',format='pdf')
		plt.savefig(_dir+'/FR_images/'+title+'_diff_bar.png',format='png')

def get_shortest_trial(data):
	min_len = np.inf
	for i,signal in enumerate(data.T):
		if i> 2:
			break
			
		amp = np.max(signal) - np.min(signal)
		max_height = np.max(signal) - amp*0.3

		spikes_t, spikes_v = find_peaks(signal, height=max_height, distance=1000)

		signal_points = len(signal[:spikes_t[-1]+1])
		signal = signal[:signal_points]
		n_spikes = len(spikes_t)

		if signal_points < min_len:
			min_len = signal_points

	return min_len



_dir = sys.argv[1]

try:
	save = False if sys.argv[2]=='n' else True
except:
	save = True


files = glob.glob(_dir+"/*[!FR_data]/*.pkl")

# files = glob.glob(_dir+".pkl")
# print(files)

plt.rcParams.update({'font.size': 14})
# print(_dir+".pkl")

# print(files)

# files = ['/home/agarpe/Workspace/data/laser/single_neuron/best_sorted/exp20/exp1_5475_50f.pkl']
dict_data = {}
labels = ['control', 'laser', 'recovery']

all_f_rates=[]
all_isis=[]
all_names=[]

data_dict = {'files':{}}


try: # to open pkl files with the data
	with open(_dir+"/all_f_rates.pkl", "rb") as fp:   # Unpickling
		all_f_rates = pkl.load(fp)
	with open(_dir+"/all_isis.pkl", "rb") as fp:   # Unpickling
		all_isis = pkl.load(fp)
	with open(_dir+"/all_names.pkl", "rb") as fp:   # Unpickling
		all_names = pkl.load(fp)
	
	load_n_compute = False

except Exception as e:
	load_n_compute = True # load files and compute FR
	print(e)
	pass

# Read and calculate FR and ISIs
for rc, pkl_name in enumerate(files):
	if not load_n_compute:
		break

	print(pkl_name)

	with open(pkl_name,'rb') as file:
		data = pkl.load(file)

	print(data.shape)
	name = pkl_name[pkl_name.rfind('/', 0, pkl_name.rfind('/')):-4]
	# if 'exp20' not in name:
	# 	continue
	dt = 0.1
	f_rates=[]
	isis = {}
	zscores = {}
	min_len = 0

	fig, ax = plt.subplots(nrows=3)

	min_len = get_shortest_trial(data) 

	for i,signal in enumerate(data.T):
		if i> 2:
			break

		amp = np.max(signal) - np.min(signal)
		max_height = amp - amp*0.3
		max_height = np.max(signal) - amp*0.3
		# max_height = 0.04
		# print(amp, max_height)

		signal = signal[:min_len]

		spikes_t, spikes_v = find_peaks(signal, height=max_height, distance=1000)


		# signal_points = len(signal[:spikes_t[-1]+1])
		# signal = signal[:signal_points]
		signal_points = len(signal)
		n_spikes = len(spikes_t)


		f_rate = (n_spikes *1000 )/ ((signal_points)*dt)
		# f_rate = 1/(len(spikes_t) / (len(signal)*dt))
		f_rates.append(f_rate)
		print("Firing rate:", f_rate)
		print("\t%d spikes / %d ms"%(len(spikes_t),(len(signal)*dt)	))

		time = np.arange(signal.shape[0])*dt 
		ax[i].plot(time, signal)
		ax[i].plot(spikes_t*dt, signal[spikes_t], 'x')

		isis[i] = np.array([spike2-spike1 for spike1,spike2 in zip(spikes_t[:-1],spikes_t[1:])])

		#remove IBIs
		import scipy.stats as stats
		zscores[i] = stats.zscore(isis[i])
		isis[i] = isis[i][np.where(abs(zscores[i]) < 2)]

	plt.suptitle(name, fontsize=10)

	files_plot = []
	# files_plot=['/exp10-05-2022/exp1_laser','/exp08-11-2022/exp0_laser','exp08-09-2022/exp8']
	# files_plot=['exp11Feb-22-2021']
	# files_plot=['exp13Feb-10-2021', 'exp08-11-2022']

	for file in files_plot:
		if file in pkl_name:
			plt.show()
			# exit()

	if save:
		plt.savefig(pkl_name[:-4]+'_activity.png')
	plt.close(fig)

	all_names.append(name)
	all_f_rates.append(f_rates)
	all_isis.append(isis)


if load_n_compute:
	with open(_dir+"/FR_data/all_f_rates.pkl", "wb") as fp:   #Pickling
		pkl.dump(all_f_rates, fp)
	with open(_dir+"/FR_data/all_isis.pkl", "wb") as fp:   #Pickling
		pkl.dump(all_isis, fp)
	with open(_dir+"/FR_data/all_names.pkl", "wb") as fp:   #Pickling
		pkl.dump(all_names, fp)


# Create images and logs directories

os.makedirs(_dir+'FR_images', exist_ok=True)
os.makedirs(_dir+'FR_log', exist_ok=True)



all_isis = np.array(all_isis)
all_f_rates = np.array(all_f_rates)
all_names = np.array(all_names)


org_isis = np.array(all_isis)
org_f_rates = np.array(all_f_rates)
org_names = np.array(all_names)

title = 'All (N=%d)'%all_f_rates.shape[0]
plot_fr_errors(all_f_rates, title)
print("All",all_f_rates.shape)

plot_grid(all_f_rates,all_isis,all_names, title)


##############################################################################################
# Filter by recovery - control difference.

error = 0.1

indexes = [abs(f_rates[0]-f_rates[2]) < (error*f_rates[0]) for f_rates in all_f_rates]
all_f_rates = all_f_rates[indexes]
all_isis = all_isis[indexes]
all_names = all_names[indexes]

# # Remove by lenght
# indexes = [all(len(row) >= 20 for i,row in enumerate(isis.values())) for isis in all_isis]
# print(indexes)
# all_f_rates = all_f_rates[indexes]
# all_isis = all_isis[indexes]
# all_names = all_names[indexes]


print("Filtered by controls", all_f_rates.shape)
plt.figure()

title = u"εControl-Recovery < 0.1 (N=%d)"%all_f_rates.shape[0]

plot_fr_errors(all_f_rates, title)

# with open(_dir+'/FR_log/'+'recovery_files.log','w') as f:
	# f.write(str(all_names))
	# f.write(str(all_names))

np.savetxt( _dir+'/FR_log/'+'recovery_files.log', all_names, fmt='%s')

plot_grid(all_f_rates,all_isis,all_names, title)
plot_grid(all_f_rates,all_isis,all_names, title, True)

N_all = all_f_rates.shape[0]

plot_diff_scatter(all_f_rates, title, all_names)


##############################################################################################
# Filter by recovery - control difference. excitation
indexes = [is_excitation(f_rates, error) for f_rates in all_f_rates]
all_f_rates = all_f_rates[indexes]
all_isis = all_isis[indexes]
all_names = all_names[indexes]

print("Show excitation", all_f_rates.shape)
print(all_f_rates.shape[0]%N_all)

title = 'Excitation (N=%d)'%all_f_rates.shape[0]
plt.figure()
plot_fr_errors(all_f_rates, title, linewidth=(all_f_rates.shape[0]%N_all)/10)

# with open(_dir+'/FR_log/'+'increase_files.log','w') as f:
# 	f.write(str(all_names))
np.savetxt( _dir+'/FR_log/'+'increase_files.log', all_names, fmt='%s')

# plot_grid(all_f_rates,all_isis,all_names)

##############################################################################################
# Filter by recovery - control difference. inhibition
indexes = [is_inhibition(f_rates, error) for f_rates in org_f_rates]
all_f_rates = org_f_rates[indexes]
all_isis = org_isis[indexes]
all_names = org_names[indexes]

print("Show inhibition", all_f_rates.shape)
print(all_f_rates.shape[0]%N_all)

title = 'Inhibition (N=%d)'%all_f_rates.shape[0]
plt.figure()
plot_fr_errors(all_f_rates, title, linewidth=(all_f_rates.shape[0]%N_all)/10)

with open(_dir+'/FR_log/'+'inhibition_files.log','w') as f:
	f.write(str(all_names))


# plot_grid(all_f_rates,all_isis,all_names)

##############################################################################################
# Filter by recovery - control difference. Unchanged
indexes = [is_unchanged(f_rates, error) for f_rates in org_f_rates]
all_f_rates = org_f_rates[indexes]
all_isis = org_isis[indexes]
all_names = org_names[indexes]

print("No change", all_f_rates.shape)
print(all_f_rates.shape[0]%N_all)

title = 'No change (N=%d)'%all_f_rates.shape[0]
plt.figure()
plot_fr_errors(all_f_rates, title, linewidth=(all_f_rates.shape[0]%N_all)/10)

with open(_dir+'/FR_log/'+'no-change_files.log','w') as f:
	f.write(str(all_names))


# plot_grid(all_f_rates,all_isis,all_names)



# plt.show()