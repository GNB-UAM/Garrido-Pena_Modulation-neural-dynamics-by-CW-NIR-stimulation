# python ./bin/plot_correlation.py -p ./data/current-pulses/correlation_data.csv -c "1 2" -l "V(mV) TemperatureºC" -sv y -sh n -scx -1000 -sf pdf
# python ./bin/laser_pulse.py -p './data/current-pulses/pipette_pulse_recording.asc' -slope 1.1931 -sf pdf

python ./bin/plot_correlation.py -p ./data/current-pulses/correlation_data.asc -c "1 2" -l "V(mV) Temperature(ºC)" -sv y -sh n -scx -1000 
python ./bin/laser_pulse.py -p './data/current-pulses/pipette_pulse_recording.asc' -slope 1.1931 

