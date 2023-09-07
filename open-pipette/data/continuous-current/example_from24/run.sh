# python ./bin/plot_correlation.py -p ./data/continuous-current/example_from24/correlation_data.asc -c "0 2" -l "V(mV) Temperature ºC" -sv y -sh n -scx 1000 -my 24 -sf pdf -dw y
# python ./bin/laser_pulse.py -p './data/continuous-current/example_from24/pipette_pulse_recording.asc' -slope 2.073 -sf pdf

python ./bin/plot_correlation.py -p ./data/continuous-current/example_from24/correlation_data.asc -c "0 2" -l "V(mV) Temperature(ºC)" -sv y -sh n -scx 1000 -my 24 -dw y
python ./bin/laser_pulse.py -p './data/continuous-current/example_from24/pipette_pulse_recording.asc' -slope 2.073 