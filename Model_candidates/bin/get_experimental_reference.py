
import pandas as pd

# Load DataFrame from .pkl file
file_path = '../CW-sustained-experiments/no_filter/results/df_all_waveforms_metrics.pkl'
df = pd.read_pickle(file_path)

# Display DataFrame contents
print("DataFrame Contents:")
print(df)

print(df.columns)

df_laser = df[df['type'] == 'laser']
df_control = df[df['type'] == 'control']

metrics = ['duration', 'amplitude', 'repolarization slope', 'depolarization slope']


# # Extract and calculate mean of specific columns
# norm_change = (df_control.groupby('file')[metrics].mean().mean() - df_laser.groupby('file')[metrics].mean().mean()) / df_control.groupby('file')[metrics].mean().mean()

# # Display mean values
# print("\nMean Values:")
# print(norm_change*100)


# Extract and calculate mean of specific columns
norm_changes = (df_control.groupby('file')[metrics].mean() - df_laser.groupby('file')[metrics].mean()) / df_control.groupby('file')[metrics].mean()

norm_changes*=100

# Display mean values
print("\nMean Values:")
mean_change = norm_changes.abs().mean()
print(mean_change)


print("\nMaximum change")

# norm_changes = (df_control.groupby('file')[metrics].mean() - df_laser.groupby('file')[metrics].mean()) / df_control.groupby('file')[metrics].mean()
max_change = norm_changes.abs().max()
print(max_change)


print("\nMinimum change")
min_change = norm_changes.abs().min()

# min_change = (df_control.groupby('file')[metrics].mean().abs().min() - df_laser.groupby('file')[metrics].mean().abs().min()) / df_control.groupby('file')[metrics].mean().min()
print(min_change)


Q1 = norm_changes.abs().quantile(0.25)
Q3 = norm_changes.abs().quantile(0.75)

RIC = Q3 - Q1

upper_thres = Q3 + 1.5 * RIC
lower_thres = Q1 - 1.5 * RIC

print("\nUpper thres")

print(upper_thres)

print("\nLower thres")

print(lower_thres)


# norm_changes *=100


print("\n Standard deviation")
stds = norm_changes.std()
print(stds)


print("\n Norm upper")
norm_upper = norm_changes.abs().mean() + stds*2
print(norm_upper)
print("\n Norm lower")
norm_lower = norm_changes.abs().mean() - stds*2
print(norm_lower)

df = pd.DataFrame({
    'Mean': mean_change,
    'Maximum': max_change,
    'RIC-upper': upper_thres,
    'STD-upper': norm_upper,
    'Minimum': min_change,
    'RIC-lower': lower_thres,
    'STD-lower': norm_lower,
})

print(df)
