python ./bin/plot_correlation.py -p ./data/continuous-current/example_from24/17h14m41s_Trial7_pipette_temperature_thermistor_14-09-22.asc -c "0 2" -l "V(mV) Temperature ºC" -sv y -sh n -scx 1000 -my 24 -sf eps
python ./bin/laser_pulse.py -p './data/continuous-current/example_from24/17h33m37s_Trial8_pipette_temperature_thermistor_14-09-22.asc' -slope 2.073 -sf eps

python ./bin/plot_correlation.py -p ./data/continuous-current/example_from24/17h14m41s_Trial7_pipette_temperature_thermistor_14-09-22.asc -c "0 2" -l "V(mV) Temperature ºC" -sv y -sh n -scx 1000 -my 24
python ./bin/laser_pulse.py -p './data/continuous-current/example_from24/17h33m37s_Trial8_pipette_temperature_thermistor_14-09-22.asc' -slope 2.073