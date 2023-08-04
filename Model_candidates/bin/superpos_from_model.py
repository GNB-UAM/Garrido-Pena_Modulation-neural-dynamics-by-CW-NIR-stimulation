import matplotlib.colors as mcolors
import matplotlib.cm as cm
from colour import Color
import glob
import pickle as pkl
from cycler import cycler

script_path = sys.argv[0][:sys.argv[0].rfind('/')]
sys.path.append(script_path+'/../..')
import plot_utils as pu
from superpos_functions import *

# rcParams["legend.markerscale"]=10
plt.rcParams.update({'legend.markerscale': 40})
plt.rcParams.update({'font.size': 50})

show='n'
stats='y'

		
def get_events(f_data, f_events, ms_r, ms_l, dt=0.001):
	#read data
	if '.asc' in f_data:
		data = pd.read_csv(f_data, delimiter = " ",skiprows=2,header=None)
		data = data.values
	elif '.pkl' in f_data:
		with open(f_data,'rb') as f:
			data = pkl.load(f)

	#read events
	if '.asc' in f_events:
		events = pd.read_csv(f_events, delimiter = " ",skiprows=2,header=None)
		events = events.values
	elif '.pkl' in f_events:
		with open(f_events,'rb') as f:
			events = pkl.load(f)

	points_r = int(ms_r /dt)
	points_l = int(ms_l /dt)

	# waveforms = np.empty((events.shape[0],(points*2)),float)
	waveforms = np.empty((events.shape[0],points_l+points_r),float)
	if verb :
		print("Waveform shape:",waveforms.shape)
		print("Events shape:",events.shape)

	# print(points)
	print(data.shape)
	# print(events)
	time = data[:,0]

	count =0
	for i,event in enumerate(events[:,0]):

		indx = np.where(time == event)[0][0] #finds spike time reference
		# plt.plot(data[indx-points_l:indx+points_r,1])
		# plt.show()
		try:
			waveforms[i] = data[indx-points_l:indx+points_r,1]
		except Exception as e:
			count +=1
			print(e)
			# print(i)

	print(count)
	if waveforms.shape[0] > 4:
		return waveforms[2:-2] #Ignore 2 first events, usally artefacts
	else:	
		return waveforms #Ignore 2 first events, usally artefacts


def get_first_line(file):
	try:
		if 'pkl' in file:
			with open(file[:-3]+'info','r') as fs:
				first_line = fs.readline()
		else:
			with open(file,'r') as fs:
				first_line = fs.readline()

		if first_line == '':
			raise Exception
	except Exception as e:
		print(e)
		print("Skiped",file)
		first_line = ''

	return first_line



import argparse

# 	print("Example: python3 superpos_from_model.py ../../laser_model/HH/data/gna/ gna \"Gna simulation\" 0.001 8 20")
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="Path to the file to plot")
ap.add_argument("-rp", "--ref_param", required=True, help="Parameter varied during simulations")
ap.add_argument("-ti", "--title", required=False,default="", help="Title of the resulting plot")
ap.add_argument("-dt", "--time_step", required=False, default=0.001, help="Sampling freq of -fs")
ap.add_argument("-c", "--color", required=False, default="blue", help="Color for plot")
ap.add_argument("-wt", "--wind_t", required=False, default=10, help="Half window size in ms")
ap.add_argument("-wt_r", "--wind_t_r", required=False, default=None, help="Right Half window size in ms. Ignored if wt specified")
ap.add_argument("-wt_l", "--wind_t_l", required=False, default=None, help="Left Half window size in ms. Ignored if wt specified")
ap.add_argument("-xl", "--x_lim", required=False, default='', help="Plot xlim values separated by white space")
ap.add_argument("-ali","--align",required=False, default='ini',help="Choose alignment mode. 'peak', 'min', 'max', 'ini', 'end'")
ap.add_argument("-nn", "--neu_name", required=False, default='', help="Extension of the events file p.e. for '*_v1_spikes.asc' -nn _v1")
ap.add_argument("-fl", "--file_limit", required=False, default=-1, help="Limit of files to load")
ap.add_argument("-sa", "--save", required=False, default='y', help="Option to save plot file")
ap.add_argument("-sh", "--show", required=False, default='y', help="Option to show plot file")
ap.add_argument("-re", "--reduced", required=False, default='n', help="Option to plot only biggest and smallest")
ap.add_argument("-st", "--stats", required=False, default='y', help="Option to save stats pkl file")
ap.add_argument("-v", "--verbrose", required=False, default='n', help="Option to verbrose process")
ap.add_argument("-ff", "--file_format", required=False, default='.asc', help="File format, default = asc")
ap.add_argument("-oe", "--one_event", required=False, default='n', help="Option to plot only one event per file")
ap.add_argument("-ra", "--range", required=False, default='-inf,inf', help="Range for the reference parameter, separated by comma")

args = vars(ap.parse_args())


path = args['path']
ref_param = args['ref_param'] #name of the parameter varied during simulations
title = args['title']
dt = float(args['time_step'])
t = float(args['wind_t'])

if args['wind_t_r'] is not None and args['wind_t_l'] is not None:
	t_r = float(args['wind_t_r'])
	t_l = float(args['wind_t_l'])
else: 
	t_r = t
	t_l = t

print(t_r,t_l)
lim = int(args['file_limit']) #number of files limit

neu_name = args['neu_name']

mode = args['align']

show = True if args['show']=='y' else False 
save = True if args['save']=='y' else False 
stats = True if args['stats']=='y' else False 
verb= True if args['verbrose']=='y' else False 

color = args['color']

xlim = args['x_lim']
if xlim != '':
	xlim = [float(l) for l in xlim.split()]

f_format = args['file_format']

# lim= 25
one_event = True if args['one_event']=='y' else False

rang = [float(a) for a in args['range'].split(',')]



if verb:
	print(path)
# files = sorted(os.listdir(path))
# files = sorted(glob.glob(path+"*[!_spikes]"+f_format))
files = sorted(glob.glob(path+"*"+f_format))
# files.sort(key=os.path.getmtime)
if lim > 0:
	files = files[:lim]


if args['reduced'] == 'y':
	files = [files[3]] + [files[-1]]


print(files)
# dt = 0.001
# t = 10 #hh 10ms #vav 10ms

# exit()


if len(files) == 0:
	print("No asc files found")
	exit()


axs = []
labels = []

if color == 'palette':
	file_ref={}
	for f in files:
		if 'spikes' in f:
			line = get_first_line(f)
			file_ref[f] = float(line)

	ref_values = {k: v for k, v in sorted(file_ref.items(), key=lambda item: item[1]) if (rang[0] <= v <= rang[1])}
	color_indexes = {k: i for i,(k, v) in enumerate(sorted(ref_values.items(), key=lambda item: item[1])) if(rang[0] <= v <= rang[1])}

	# print(color_indexes)
	# print(len(ref_values))
	n_colors = len(ref_values)

	# colors = plt.cm.coolwarm(np.linspace(0,1,n_colors)  
	colors = plt.cm.Blues(np.linspace(0.2, 1, n_colors)) 
	plt.gca().set_prop_cycle(cycler('color', colors))
	p_color = color
	
	file_ref={}
else:
	p_color = Color(color)
	# colors = list(red.range_to(Color("green"),len(files)//2))
	luminances = np.arange(0.9,0.2,-0.7/len(files))
	colors=[]
logs = []
logs_cols = []
log ={}

nValues =[]
# normalize = mcolors.Normalize(vmin=nValues.min(), vmax=nValues.max())
# colormap = cm.jet

legend = []

# fig = plt.figure(figsize=(18,23))
fig, ax0 = plt.subplots(figsize=(15,19))
plt.gca().set_prop_cycle(cycler('color', colors))



for i,f in enumerate(files):
	legend = ''

	if("spikes" not in f and f_format in f and 'waveform' not in f):
	# if("spikes" not in f and f_format in f):
		if verb:
			print(f)

		ref = f.find("Euler")
		if(ref!=-1):
			f_events = f[:ref]+neu_name+"spikes_"+f[ref:]
		else:
			f_events = f[:-4]+neu_name+"_spikes"+f[-4:]#+f_format


		print(f_events)
		f_name = f_events[f_events.rfind('/'):]
		print(f_name)

		try:
			first_line = get_first_line(f_events)
		except Exception as e:
			print("************************",e)
			continue

		label = ref_param+"="+first_line
		print(label)
		labels.append(label)
		print(first_line)
		try:
			nValues.append(float(first_line))
		except Exception as e:
			print("************************",e)
			print(first_line)
			print(f_events)
			nValues.append(float(first_line.split(',')[0]))

		if not(rang[0] <= nValues[-1] <= rang[1]):
			print(nValues)
			nValues = nValues[:-1]
			continue

		# try:
		trial = get_events(f,f_events,ms_l=t_l, ms_r=t_r, dt=dt)

		print(trial.shape)

		# except Exception as ex:
		# 	print("Error reading events from ",f)
		# 	print(type(ex).__name__, ex.args)
		# 	print("skiping and removing corrupt file")
		# 	print(f)
		# 	# os.system("rm "+f+" "+f_events)
		# 	continue

		if verb:
			print(trial.shape)
		# if(trial.shape[0] <=8):
		# 	print("skiping and removing corrupt file")
		# 	print(f)
		# 	os.system("rm "+f+" "+f_events)
		# 	continue

		# color=colors[i%(len(files)//2)].hex_l
		lw = 0.5
		lw = 3
		if f_name.find('normal') > 0:
			print("##########################normal")
			color = Color('blue')
			legend = 'Control %.2f'%float(first_line)
			# continue
		elif f_name.find('laser') > 0:
			print("##########################laser")
			color = Color('red')
			legend = 'NI continuous'
			legend = 'Laser %.2f'%float(first_line)
			# continue
		elif f_name.find('rdepol') > 0:
			print("##########################rdepol")
			legend = 'NI depol.'
			if(p_color == Color('red')):
				color = Color('firebrick')
			else:
				color = Color('yellowgreen')
			# continue
		elif f_name.find('rrepol') > 0:
			print("##########################rrepol")
			if(p_color == Color('red')):
				color = Color('salmon')
			else:
				color = Color('darkgreen')
			# continue
			legend = 'NI repol.'
		elif f_name.find('n1') > 0:
			print("##########################n1")
			color = Color('green')
			legend = 'n1'
			# lw = 3
		# elif f_name.find('n2') > 0:
		# 	print("##########################n2")
		# 	color = Color('green')
		# 	legend = 'n2'
		elif p_color == 'palette':
			color = colors[color_indexes[f_events]]
		else:
			print("#######################LAST CASE COLOR")
			print(color)
			print(p_color)
			color = p_color
			color.luminance = luminances[i%(len(files))]
			colors.append(color.hex_l)
			# legend = ''
			legend = first_line

		try:
			color = color.hex_l
		except Exception as e:
			# print("************************",e)
			color=mcolors.to_hex(color)

		trial =trial[:-1]


		print(trial.shape)
		if one_event:
			try:
				trial = trial[trial.shape[0]//2]
				trial = trial.reshape(1,trial.shape[0])
				print(trial.shape)
			except Exception as e:
				print(e)
				pass



		ax,ax1,ax2=plot_events(trial,color,tit=title,width_ms_l=t_l,width_ms_r=t_r,dt=dt,df_log=log,show_durations=False, lw=lw, mode=mode)
		try:
			ax.set_label(legend)
		except Exception as e:
			print("************************",e)
			pass

		logs_cols.append(label)
		logs.append(log)
		if stats:
			try:
				log_df = pd.DataFrame(log)
				log_df.to_pickle(path+ref_param+"-"+first_line[:-3]+"_info.pkl")

			except ValueError as e:
				print("Passing data ",i,"with length",[len(log[x]) for x in log if isinstance(log[x], list)])
				print("Exception:",e)
				pass
			except Exception as e:
				print("Passing data",i,e)
				pass

		axs.append(ax)


plt.title(title)

pu.small_x_axe(plt.gca(), label='ms')
pu.small_y_axe(plt.gca(), label='mV')
pu.remove_axes(plt.gca())


# lgnd = plt.legend(axs,labels)
# for i,line in enumerate(lgnd.get_lines()):
#     line.set_linewidth(2)
#     line.set_color(colors[i])

#TODO WARNING UNCOMMENT FOR BAR!!!!!
try:
	pu.plot_cmap(plt.gcf(), plt.gca(), colors, nValues, location=[0.2, 0.3, 0.025, 0.55]) # [x, y, width, height])

except Exception as e:
	print("************************",e)
	plt.legend(fontsize=40)
	# pass
#######################
# plt.legend()


if xlim != '':
# plt.xlim([20,90])
	plt.xlim(xlim)


plt.tight_layout()


if save:

	print( path+title+neu_name+".png")
	# plt.savefig(path+title+".png")
	# save_name = path+title+neu_name+"_"+args['color']
	save_name = path+title+neu_name+"_"+mode
	print( save_name+".svg")
	plt.savefig(save_name+".svg",format="svg")
	print( save_name+".png")
	plt.savefig(save_name+".png",format="png", dpi=200)

if show:
	plt.show()

df = create_dataframe(logs,logs_cols)
print(df.describe())

os.system("mkdir -p %s/general"%path)
df.to_pickle(path+"/general/"+ref_param+"_info.pkl")

