#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 16:57:38 2022

@author: rlevi
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
script_path = sys.argv[0][:sys.argv[0].rfind('/')]
sys.path.append(script_path+'/../..')
import plot_utils as pu

path='./data/17h33m37s_Trial8_pipette_temperature_thermistor_14-09-22.asc'
#path='/home/agarpe/Workspace/data/laser/pipette_temperature/16h39m28s_Trial4_pipette temperature_thermistor.asc'
# slope = 0.5

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="Path to the file to analyze")
# ap.add_argument("-l", "--labels", required=True, help="Columns of waves")
ap.add_argument("-slope", "--slope", required=True, help="Slope for Voltage-temp relation")
ap.add_argument("-scx", "--scale_x", required=False, default=1000, help="Scale for Volts. Signal*scale")
ap.add_argument("-scy", "--scale_y", required=False, default=1000, help="Scale for Volts. Signal*scale")
# ap.add_argument("-dw", "--down", required=False,default='n', help="Downsample trace")
ap.add_argument("-sh", "--show", required=False,default='n', help="Show plot")
ap.add_argument("-sv", "--save", required=False,default='y', help="Save events")
ap.add_argument("-sf", "--save_format", required=False,default='png', help="Save format")
# ap.add_argument("-sf", "--save_format", required=False,default='pdf', help="Save format")
args = vars(ap.parse_args())

path = args['path']

scale_x = int(args['scale_x'])
scale_y = int(args['scale_y'])
slope = float(args['slope'])

show= True if args['show']=='y' else False 
save= True if args['save']=='y' else False 
f_format = args['save_format']



# slope = 2.326

# f_format = 'png'

plt.rcParams.update({'font.size': 17})

df = pd.read_csv(path, delimiter = " ",skiprows=2,header=None)

V=df[0].values
pulse=df[1].values


t = [x / 10 for x in range(0, len(V))]
# t=numpy.arange(0, len(V), step)

wn =  np.arange(20000, 190000, dtype=int)
t_wn = np.array(t)[wn]
V_wn = np.array(V)[wn]
pulse_wn =np.array(pulse)[wn]


# print(p_off[0]-p_on[0])
# print(p_on, p_off)


fig, ax1 = plt.subplots()

ax1.plot(t_wn, V_wn)
ax1.set_title('laser and pipet')

ax2 = ax1.twinx()
ax2.plot(t_wn, pulse_wn, 'C1')
ax2.set_ylim([0, 20])

output_path = path[:path.rfind('/')] + '/images/'
os.system('mkdir -p %s'%output_path)
plt.savefig(output_path + 'pipette_pulse_example.'+f_format, format=f_format)
#plt.show()


# Plot all pulses aligned. 
#p_df=np.diff(pulse_wn)
#%%
plt.figure()

p_on=np.where(np.diff(pulse)>2)
p_off=np.where(np.diff(pulse)<-2)
V_avg=np.empty([51500])
V_sm=np.empty( [51500])
for i in p_on[0][0:-1]:
    wn=np.arange(i-1000, i+50500)
    t_wn = np.array(t)[wn]
    pulse_wn =np.array(pulse)[wn]
    V_wn = np.array(V)[wn]
    V_ofs=np.mean(V_wn[:1000])
    V_wn=V_wn-V_ofs
    #V_sm=np.array([V_sm, V_wn], dtype=float)
    V_sm=np.vstack((V_sm, V_wn))
    #V_sm=np.append(V_sm, V_wn, axis=1)
    #np.concatenate(V_sm, V_wn)
    #V_sm=[V_sm, V_wn ]
    
    plt.plot(scale_y*V_wn)
    plt.xlabel("Time (time steps)")
    plt.ylabel("Vm (mV)")
    
    #print(wn)
#V_sm=np.asarray(V_wn)
plt.savefig(output_path + 'pipette_pulse_alinged.'+f_format, format=f_format)
#print(np.mean(V_sm, axis=0, dtype=float))


# Plot mean of pulses

print(scale_y*np.max(np.mean(V_sm, axis=0, dtype=float)[:10000]))
plt.figure()
plt.plot(t_wn/1000, scale_y*np.mean(V_sm, axis=0, dtype=float))
plt.xlabel("Time (s)")
plt.ylabel("Vm (mV)")

# plt.show()
plt.savefig(output_path + 'pipette_pulse_mean_v.'+f_format, format=f_format)


# Plot mean of pulses with Voltage-Temperature conversion. 

V_sm *=slope

# Smooth signal
# Number of data points to include in the moving average window
window_size = 15  # Adjust this based on your desired level of smoothing
# Smooth the data using a moving average filter
smoothed_data = np.convolve(np.mean(V_sm, axis=0, dtype=float), np.ones(window_size)/window_size, mode='same')

# Plot the original and smoothed data
plt.figure(figsize=(10, 6))
plt.plot(t_wn, scale_y * np.mean(V_sm, axis=0, dtype=float), label='Original Data')
plt.plot(t_wn, scale_y * smoothed_data, label=f'Smoothed Data (window_size={window_size})')
plt.legend()
plt.xlabel("Time (ms)")
plt.ylabel("Temperature (ºC)\nestimated with %.2f"%slope)
plt.tight_layout()
pu.remove_axes(plt.gca())
# pu.small_x_axe(plt.gca(), 'Time (ms)')
# pu.small_y_axe(plt.gca(), "Temperature (ºC)")
# plt.show()
plt.savefig(output_path + 'pipette_pulse_mean_temp.'+f_format, format=f_format)


# Plot the smoothed data
print(V_sm.shape)
plt.figure(figsize=(6, 5))
plt.plot(t_wn, scale_y * smoothed_data, label=f'Smoothed Data\n (window_size={window_size})',color='slategray', alpha=1)

plt.xlabel("Time (ms)")
plt.ylabel("Temperature (ºC)")
plt.tight_layout()
pu.remove_axes(plt.gca())
# pu.small_x_axe(plt.gca(), label='ms')
# pu.small_y_axe(plt.gca(), label="ºC")
plt.savefig(output_path + 'pipette_pulse_mean_temp_smooth.'+f_format, format=f_format)

with open(path[:-4] + '_pulse.log', 'w') as f:
    f.write("Pipette pulse estimation with %.2f"%slope+ "\n")
    f.write("Slope: "+str(slope)+ "\n")

# plt.show()