import numpy as np
import sys
import pandas as pd
import glob
# To import from child dir
script_path = sys.argv[0][:sys.argv[0].rfind('/')]
sys.path.append(script_path+'/..')
sys.path.append(script_path+'/../..')
import plot_utils as pu
import matplotlib.pyplot as plt

path = sys.argv[1]

try:
    title = sys.argv[2]
except:
    title = ''

# get different_files from folder
files = sorted(glob.glob(path+"/*/differences-df.pkl"))

print(files)

for i,file in enumerate(files):
    # load series
    temp_df = pd.read_pickle(file)
    # convert to dataframe
    temp_df = temp_df.to_frame()

    if i == 0:
        df = temp_df.copy()
        continue

    print(file)
    print(temp_df)
    # concat data
    df = pd.concat([df, temp_df],axis=1)

df = df.T

print(df)

# get metrics names
metrics = list(df.columns)
if 'Q10' in metrics:
    metrics.remove('Q10')
if 'diff_T' in metrics:
    metrics.remove('diff_T')

def plot_barchart(df, y_col, title):
    subset = df[metrics+[y_col]].copy()

    subset = subset.dropna()
    print(subset)

    subset.set_index(y_col, inplace=True)

    subset = subset[metrics].T
    # Sort the columns based on numeric values
    subset = subset[subset.columns.sort_values().astype(float)]

    fig, ax = plt.subplots()

    if y_col=='Q10':
        colors = plt.cm.OrRd(np.linspace(0.2,1,len(subset.columns)))
    else:
        colors = plt.cm.coolwarm(np.linspace(0,1,len(subset.columns)))
    # colors = plt.cm.get_cmap('Oranges')
    from cycler import cycler
    plt.gca().set_prop_cycle(cycler('color', colors))

    subset.plot.bar(rot=0, alpha=0.6, figsize=(9,7), ax=ax, color=colors)

    # cbar=pu.plot_cmap(fig, ax, colors, subset.columns, location=[0.9, 0.4, 0.02, 0.5], alpha=0.6)# [x, y, width, height]
    
    # if y_col == 'diff_T':
    #     cbar.set_label(u"ΔTemperature ºC")
    # else:
    #     cbar.set_label(u"Q10")
    # ax.legend('')

    ax.set_ylim(0,2)
    pu.remove_axes(ax) 

    plt.suptitle(title, fontsize=20)
    plt.tight_layout()



try:
    plot_barchart(df,'diff_T',title)
    print("saving fig at ", path+'temp_comparation'+'.pdf')
    plt.savefig(path+'temp_comparation'+'.pdf', format='pdf')

except:
    pass
try:
    plot_barchart(df, 'Q10',title)
    print("saving fig at ", path+'q10_comparation'+'.pdf')
    plt.savefig(path+'q10_comparation'+'.pdf', format='pdf')
except:
    pass
# plt.show()

