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

def plot_grid_by_metric():

	# print(all_df['Cm'])
	n_rows = 2
	n_cols = 2
	print(n_cols)
	fig,axis = plt.subplots(figsize=(25,8),nrows = n_rows, ncols=n_cols)

	key_colors = plt.cm.tab20(np.linspace(0,1,len(all_df.keys())))
	key_colors = np.flip(key_colors, axis=0)

	for rc,(key,value) in enumerate(all_df.items()):
		boxprops = dict(linestyle='-', linewidth=2, color=key_colors[rc])
		
		for i,metric in enumerate(metrics):
			value.boxplot(column=metric, ax=axis[i//2,i%2], positions=[rc], showfliers=False, showmeans=True, grid=False, fontsize=20, boxprops=boxprops)
		
			axis[i//2,i%2].set_title(metric)


	for ai in axis:
		for a in ai:
			ax=a.axes
			ax.set_xticklabels([list(all_df.keys())[n] for n in range(len(ax.get_xticklabels()))])

	plt.tight_layout()	

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

def generate_table(df):

	# Create style object
	styler = df.style

	shoulder_values = [0.43,0.28,0.86,0.015]
	symmetric_values = [0.24,0.11,0.26,0.028]
	mins = [min(a,b) for a,b in zip(shoulder_values,symmetric_values)]
	maxs = [max(a,b) for a,b in zip(shoulder_values,symmetric_values)]
	print(mins, maxs)
	ranges = [(min_*0.8,max_*1.3) for min_,max_ in zip(mins,maxs)]
	print(ranges)
	print()


	# Create a custom colormap that transitions from blue to white and back to blue
	colors = ['royalblue','lightsteelblue', 'white','lightsteelblue', 'royalblue']
	cm1 = create_custom_palette(colors)

	# save color bar reference
	fig,ax = plt.subplots()
	pu.plot_cmap(fig, ax, [], location=[0.5,0,0.08,1], cmap=cm1) # [x, y, width, height]
	fig.delaxes(ax) 
	plt.savefig('color_bar_1'+'.pdf', format='pdf', bbox_inches='tight')

	# Create map for amplitude
	colors = ['white', 'lightsteelblue', 'royalblue']
	cm2 = create_custom_palette(colors)      

	# save color bar reference
	fig,ax = plt.subplots()
	pu.plot_cmap(fig, ax, [], location=[0.5,0,0.08,1], cmap=cm2) # [x, y, width, height]
	fig.delaxes(ax) 
	plt.savefig('color_bar_2'+'.pdf', format='pdf', bbox_inches='tight')



	# Apply background color gradient to each column based on min-max values.
	styler = styler.background_gradient(cmap=cm1, subset=['duration'],
										 low=mins[0], high=maxs[0],
										 vmin=ranges[0][0], vmax=ranges[0][1])

	styler = styler.background_gradient(cmap=cm1, subset=['depol.'],
										 low=mins[1], high=maxs[1],
										 vmin=ranges[1][0], vmax=ranges[1][1])
	
	styler = styler.background_gradient(cmap=cm1, subset=['repol.'],
										 low=mins[2], high=maxs[2],
										 vmin=ranges[2][0], vmax=ranges[2][1])

	styler = styler.background_gradient(cmap=cm2, subset=['amplitude'],
										 low=mins[3], high=maxs[3])
										 # vmin=0, vmax=ranges[3][1])

	# text style
	styler = styler.set_properties(**{'text-align': 'center', 'font-family': 'garuda','width': '120px'})
	styler = styler.format(precision=3)

	# Convert style object to HTML
	html = styler.to_html()

	# Save pdf with the table
	import pdfkit
	pdfkit.from_string(html, 'styled_table-%s.pdf'%name)
	

import argparse

# 	print("Example: python3 superpos_from_model.py ../../laser_model/HH/data/gna/ gna \"Gna simulation\" 0.001 8 20")
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="Path to the file to show stats from")
ap.add_argument("-sa", "--save", required=False, default='y', help="Option to save plot file")
ap.add_argument("-sh", "--show", required=False, default='y', help="Option to show plot file")
ap.add_argument("-dt", "--time_step", required=False, default=0.01, help="Sampling freq of -fs")

args = vars(ap.parse_args())


path = args['path']
show= True if args['show']=='y' else False 
save= True if args['save']=='y' else False 

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
# 	print(c)
for i,dir_ in enumerate(dirs):
	files = sorted(glob.glob(dir_+"/*waveform.pkl"))

	param = dir_[dir_.find('/')+2:]

	param = ''.join([p for p in param if not p.isnumeric()])
	param = param.replace('/','_')
	print(param)

	# if 'cgc' in param:
	# 	continue
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
	# 	plt.show()
	plt.close()


plt.rcParams.update({'font.size': 20})

metrics = ['duration','depol.','repol.','amplitude']

# Get normalized differences
df_diffs = {}
for rc,(key,value) in enumerate(all_df.items()):
	#Value is the dataframe for each Candidate/Neuron (directory in the path given)
	max_ = value.max()
	min_ = value.min()

	df_diffs[key] = abs((max_ - min_))/abs(max_)

# plot table 
# get dataframe from dict
DF_diffs = pd.concat(df_diffs, axis=1).T
generate_table(DF_diffs[metrics])
# generate table colored by metric
