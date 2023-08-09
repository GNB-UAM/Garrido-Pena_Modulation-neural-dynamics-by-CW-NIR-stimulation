import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# To import from child dir
import sys
script_path = sys.argv[0][:sys.argv[0].rfind('/')]
sys.path.append(script_path+'/../../')
sys.path.append('../..')

import superpos_functions as sf


def savefig(path, path_images, name, format='.png'):
    savepath = path_images + path[path[:-1].rfind('/'):-1] + name
    print("Saving fig at",savepath)
    plt.savefig(savepath + format, dpi=100, format=format[1:])


def get_slopes(f_slope, waveforms,dt=0.1, width_r=60, width_l=60, slope_position=0.5):


    slope_dep =[]
    slope_rep =[]
    for i,spike in enumerate(waveforms):
        spike = sf.align_spike(spike, width_ms_r=width_r,width_ms_l=width_l, dt=dt,id_=i )

        if len(spike) ==0:
            slope_dep.append(0)
            slope_rep.append(0)
            continue

        # vals,th = sf.get_spike_duration(spike,dt=0.1,tol=1)
        slope_inc,slope_dec = f_slope(spike,dt,slope_position=slope_position, plot=True)

        # plt.plot(spike)
        # vals = np.array(vals)
        # plt.plot(vals/0.1,(th,th),'x')
        # plt.title(str(vals[1]-vals[0]))
        # plt.show()
        # try:
        slope_dep.append(slope_inc)
        slope_rep.append(slope_dec)   


        # except:
            # print(vals)
            # continue
    # plt.show()
    return np.array(slope_dep), np.array(slope_rep)

def get_durs(waveforms, width_r=60, width_l=60, max_dur=25):

    durations =[]
    for i,spike in enumerate(waveforms):
        spike = sf.align_spike(spike, width_ms_r=width_r,width_ms_l=width_l, dt=0.1,id_=i )

        if len(spike) ==0:
            durations.append(0)
            continue


        vals,th = sf.get_spike_duration(spike,dt=0.1,tol=1)
        # plt.plot(spike)
        vals = np.array(vals)
        if (vals[1]-vals[0]) < max_dur:

            plt.plot(vals,(th,th),'.',markersize=20,color='r')
            plt.title(str(vals[1]-vals[0]))
        # plt.show()
        try:
            durations.append(vals[1]-vals[0])
        except:
            print(vals)
            continue

    return np.array(durations)

def read_data(path, ctype):
    try:
        waveforms =  sf.read_from_events(path+'_%s_waveform_single.txt'%ctype,max_cols=80/0.1,dt=0.1)
        waveforms = waveforms.values
    except:
        waveforms = np.array([])

    if ctype == 'control':
        print(path+'_%s_waveform_single.txt'%ctype)
    print(waveforms.shape)

    try:
        stim_path = path + '_%s_shutter_time_references.txt'%ctype # name from v1
        # stim_path = path[:path.rfind('/')] + 'laser_shutter_time_references.txt'
        print(stim_path)
        stim = np.loadtxt(stim_path)
        print("Stim ",stim.shape)
    except Exception as e:
        stim = []
        if ctype == 'laser':
            print("EXCEPTION:",e.args)
        pass

    return waveforms, stim

def remove_zeros(df, metrics):
    for metric in metrics:
        # print(metric)
        # print('\n',df[metric])
        if 'repol' in metric:
            # print(df.loc[df['file'] == '30-03-2023//events//exp9_depol_50ON_130bck'])
            df.drop(df[df[metric] >= 0].index, inplace=True)

            # print(df)
        else:
            df.drop(df[df[metric] <= 0].index, inplace=True)


def remove_outliers(df,columns,n_std, g_by='file'):
    for file in set(df[g_by]):
        for col in columns:
            # print('Working on column: {}'.format(col))
            data = df.loc[df[g_by] == file, col]
            mean = data.mean()
            sd = data.std()

            # print(col,file, data.mean(),data.std())

            df.loc[df[g_by] == file, col] = df.loc[df[g_by] == file,
                     col].apply(lambda x: x if x <= mean+(n_std*sd) else 0)

    prev = len(df.index)
    remove_zeros(df,columns)
    print("Spikes removed as outliers: ",prev-len(df.index),"/",prev)

    return df


def get_metrics_from_file(path, ctype, slope_position=0.98, dict_={}, min_dur=2, ext='', max_dur=25):

    waveforms, stim = read_data(path, ctype)
    durations = get_durs(waveforms, max_dur =max_dur)
    slopes_dep, slopes_rep = get_slopes(sf.get_slope, waveforms)
    # slopes_dep2, slopes_rep2 = get_slopes(sf.get_slope2, waveforms, slope_position=slope_position)
    slopes_dep2, slopes_rep2 = get_slopes(sf.get_slope2, waveforms)

    try:
        stim = stim[np.where(durations>min_dur)]
        stim[:,:] *= -1
        dict_[ext+'to_on'].extend(stim[:,0].tolist())
        dict_[ext+'to_off'].extend(stim[:,1].tolist())
    except Exception as e:
        print(e)
        pass

    #TODO: do this in dataframe
    durations = durations[np.where(durations>min_dur)]
    slopes_dep = slopes_dep[np.where(durations>min_dur)]
    slopes_rep = slopes_rep[np.where(durations>min_dur)]
    slopes_dep2 = slopes_dep2[np.where(durations>min_dur)]
    slopes_rep2 = slopes_rep2[np.where(durations>min_dur)]


    durations = durations[np.where(durations<max_dur)]
    slopes_dep = slopes_dep[np.where(durations<max_dur)]
    slopes_rep = slopes_rep[np.where(durations<max_dur)]
    slopes_dep2 = slopes_dep2[np.where(durations<max_dur)]
    slopes_rep2 = slopes_rep2[np.where(durations<max_dur)]

    dict_[ext+'duration'].extend(durations.tolist())
    dict_[ext+'depol_slope'].extend(slopes_dep.tolist())
    dict_[ext+'repol_slope'].extend(slopes_rep.tolist())
    dict_[ext+'depol_slope2'].extend(slopes_dep2.tolist())
    dict_[ext+'repol_slope2'].extend(slopes_rep2.tolist())
    dict_['file'].extend([path]*durations.size)

    return durations, slopes_dep, slopes_rep, slopes_dep2, slopes_rep2, stim


def plot_boxplot_mean(df,column, metric, cut_range, step_range, df_controls=None, df_laser=None, df_recovery=None):
    
    # set_range(df, cut_range, column, step_range)

    boxprops = dict(linestyle='-', linewidth=3, color='C0')
    ax = df.boxplot(column='diff_'+metric,by='range',figsize=(40,20), showmeans=True, showfliers=False,boxprops=boxprops)
    ax.set_ylabel(u"Δ Spike %s"%metric)
    ax.set_xlabel("Time %s event (ms)"%column)
    plt.tight_layout()

    if df_recovery is not None and df_laser is not None:
        ticks = ax.get_xticklabels()
        # try:

        ax.boxplot(df_recovery["diff_%s"%metric], boxprops=dict(linestyle='-', linewidth=3, color='olivedrab'),
         positions = [0], showmeans=True, showfliers=False)
        ax.boxplot(df_laser["diff_%s"%metric], boxprops=dict(linestyle='-', linewidth=3, color='firebrick'),
         positions = [len(ticks)+1], showmeans=True, showfliers=False)
        # except:
            # return ax
        ticks = ["%s"%t.get_text() for t in ticks] +["recovery"] + ["sustained \nCW laser"]

        n_ticks = ax.get_xticks()

        # print(n_ticks)
        # print(ticks)
        ax.set_xticks(n_ticks,ticks)
        ax.set_xticklabels(ticks)

        plt.tight_layout()
    
    return ax

def set_range(df, cut_range, column, step_range):

    if cut_range is not None:
        cut_range = [int(r.replace('\\','')) for r in cut_range.split(',')]
    else:
        cut_range = [df[column].min(), df[column].max()]
    # print(cut_range)

    # https://stackoverflow.com/questions/21441259/pandas-groupby-range-of-values
    # print(df[column])
    df.loc[:,"range"] = pd.cut(df[column], np.arange(cut_range[0], cut_range[1], step_range))

    return df



def plot_boxplot(df,column, metric, cut_range, step_range, df_controls=None, df_laser=None, df_recovery=None):
   
    # set_range(df, cut_range, column, step_range)

    boxprops = dict(linestyle='-', linewidth=3, color='C0')
    ax = df.boxplot(column=metric,by='range',figsize=(50,30), showmeans=True, showfliers=False,boxprops=boxprops)
    ax.set_ylabel("Spike %s"%metric)
    ax.set_xlabel("Time %s event (ms)"%column)

    plt.tight_layout()

    if df_controls is not None and df_laser is not None:
        ticks = ax.get_xticklabels()
        # try:
        ax.boxplot(df_controls["control_%s"%metric][df_controls["control_%s"%metric].notnull()], positions = [-1], showmeans=True, showfliers=False)
        ax.boxplot(df_recovery["recovery_%s"%metric][df_recovery["recovery_%s"%metric].notnull()], positions = [0], showmeans=True, showfliers=False)
        ax.boxplot(df_laser["laser_%s"%metric][df_laser["laser_%s"%metric].notnull()], positions = [len(ticks)+1], showmeans=True, showfliers=False)

        ticks = ["%s"%t.get_text() for t in ticks] +["control"]+["recovery"] + ["continuous laser"]

        n_ticks = ax.get_xticks()

        ax.set_xticks(n_ticks,ticks)
        ax.set_xticklabels(ticks)

        plt.tight_layout()
    
    # plt.xticks(rotation=45, ha='right')

    return ax

#df dataframe
# column e.g. to_off
# metric e.g. duration
def plot_scatter(df, column, metric, ylabel, xlabel, df_controls=None, df_recovery=None, median = True):
    plt.figure(figsize=(20,15))

    colors = plt.cm.tab20(np.linspace(0,1,len(df.groupby("file"))))
    # colors = plt.cm.get_cmap('Oranges')
    from cycler import cycler
    plt.gca().set_prop_cycle(cycler('color', colors))

    for name, group in df.groupby("file"):
        fig = plt.plot(group[column], group[metric], marker="o", linestyle="", label=name)
        # print(df_controls.loc[df_controls['file'] == name]["control_duration"])
        if median:
            plt.plot(group[column].median(), group[metric].median(), marker="o", markersize=15, color=fig[-1].get_color(), label= name+' median')
        
        if df_controls is not None:
            control = df_controls.loc[df_controls['file'] == name]["control_"+metric]
            recovery = df_recovery.loc[df_recovery['file'] == name]["recovery_"+metric]
            plt.plot(np.zeros(control.size), control, marker="o", linestyle="", color='k')
            plt.plot(np.zeros(recovery.size)+5, recovery, marker="o", linestyle="", color='k')
        
            if median:
                plt.plot(0, control.median(), marker="o", markersize=15, color=fig[-1].get_color())
                plt.plot(0.5, recovery.median(), marker="o", markersize=15, color=fig[-1].get_color())

    plt.legend(fontsize=10)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.tight_layout()


def plot_all_scatter(df, metric, save, df_controls, df_recovery, path, path_images):

    ## Plot scatter
    plot_scatter(df, "to_off", metric, "Spike %s "%metric, "Time from spike to off (ms)", df_controls=df_controls, df_recovery=df_recovery)

    if save:
        savefig(path, path_images, "_%s_general_scatter_to_off_control"%metric)

    ## Plot scatter on
    plot_scatter(df, "to_on", metric, "Spike %s "%metric, "Time from spike to on (ms)", df_controls=df_controls, df_recovery=df_recovery)

    if save:
        savefig(path, path_images, "_%s_general_scatter_to_on_control"%metric)



def add_norm_diff(df, metric, df_controls=None, df_laser=None, df_recovery=None, norm=True, general=False):

    df_controls.loc[:,'control_'+metric] = pd.to_numeric(df_controls['control_'+metric])

    metric_control = df_controls[['control_'+metric, 'file']].dropna()
    df_laser.loc[:,'laser_'+metric]= pd.to_numeric(df_laser['laser_'+metric])
    metric_laser = df_laser[['laser_'+metric, 'file']].dropna()

    control_means = metric_control[['control_'+metric,'file']].groupby('file').mean()
    laser_means = metric_laser[['laser_'+metric,'file']].groupby('file').mean()


    df.loc[:, "diff_"+metric] = np.nan
    df_laser.loc[:, "diff_"+metric] = np.nan
    df_recovery.loc[:, "diff_"+metric] = np.nan

    for i,file in enumerate(control_means.index):
        day = file[file.rfind('/')+1:file.find('-202')+5]
        file_control_mean = control_means.loc[file]
        
        laser_mean = metric_laser.loc[df_laser['file'].str.contains(day),['laser_'+metric]].mean().values[0]
        # all files
        general_control = metric_control.loc[df_controls['file'].str.contains(day),['control_'+metric]].mean().values[0]

        _max = laser_mean
        _min = general_control

        if not norm:
            f = lambda x: x - file_control_mean
        else:
            f = lambda x: (abs(x) - abs(_min))/(abs(_max)-abs(_min))

        df.loc[df['file'] == file, "diff_"+metric] = df.loc[df['file'] == file, metric].apply(f).values
        
        if general: 
            f = lambda x: abs(x) - abs(general_control)
        df_recovery.loc[df_recovery['file'] == file, "diff_"+metric] = df_recovery.loc[df_recovery['file'] == file, 'recovery_'+metric].apply(f).values
        df_laser.loc[df_laser['file'] == file, "diff_"+metric] = df_laser.loc[df_laser['file'] == file, 'laser_'+metric].apply(f).values
    

def remove_axes(ax):
   ax.spines['top'].set_visible(False)
   ax.spines['right'].set_visible(False)


def plot_mean_error(df, column, metric, cut_range, step_range, df_laser=None, ax=None):

    if 'diff_'+metric not in df:
        raise Exception("Diff metrics for diff_%s not generated in dataframe"%metric)

    df.loc[:,'diff_'+metric] = pd.to_numeric(df['diff_'+metric])
    group_of_ranges = df.groupby(['range'])['diff_'+metric]

    groups = [str(name) for name,unused_df in group_of_ranges]

    ax.errorbar(groups, group_of_ranges.mean(), yerr=group_of_ranges.sem().values.flatten(), marker='o', markersize=25, linewidth=10)

    if df_laser is not None:
        ax.errorbar([-1], df_laser['diff_'+metric].mean(), fmt=' ', marker='o', markersize=20, linewidth=1)

    # plt.ylabel(u"Δ Spike %s"%metric)
    ax.set_xlabel("Time %s event (ms)"%column)
    ax.set_ylabel("Normalized change")
    remove_axes(ax)
