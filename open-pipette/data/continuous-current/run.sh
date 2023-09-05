python ./bin/plot_correlation.py -p ./data/continuous-current/17h14m41s_Trial7_pipette_temperature_thermistor_14-09-22.asc -c "0 2" -l "V(mV) Temperature ºC" -sv y -sh n -scx 1000 -sf eps
python ./bin/laser_pulse.py -p './data/continuous-current/17h33m37s_Trial8_pipette_temperature_thermistor_14-09-22.asc' -slope 2.326 -sf eps
python ./bin/plot_correlation.py -p ./data/continuous-current/17h14m41s_Trial7_pipette_temperature_thermistor_14-09-22.asc -c "0 2" -l "V(mV) Temperature ºC" -sv y -sh n -scx 1000
python ./bin/laser_pulse.py -p './data/continuous-current/17h33m37s_Trial8_pipette_temperature_thermistor_14-09-22.asc' -slope 2.326
