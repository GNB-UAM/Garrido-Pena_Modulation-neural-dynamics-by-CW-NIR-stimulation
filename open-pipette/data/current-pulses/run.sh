python ./bin/plot_correlation.py -p ./data/current-pulses/current_pulses_data.csv -c "1 2" -l "V(mV) TemperatureºC" -sv y -sh n -scx -1000 -sf eps
python ./bin/laser_pulse.py -p './data/current-pulses/pipette_pulse_recording.asc' -slope 1.1931 -sf eps

python ./bin/plot_correlation.py -p ./data/current-pulses/current_pulses_data.csv -c "1 2" -l "V(mV) TemperatureºC" -sv y -sh n -scx -1000 
python ./bin/laser_pulse.py -p './data/current-pulses/pipette_pulse_recording.asc' -slope 1.1931 

