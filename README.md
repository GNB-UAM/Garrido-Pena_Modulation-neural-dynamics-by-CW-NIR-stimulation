# Garrido-Pena_Modulation-neural-dynamics-by-NIR-stimulation
Code and data used in Garrido-Peña et al. publication. Computational models and code for data analysis and figures plotting.

Data used with this code and for the figures in the publication is available in each "data" directory and in the following link: 

https://osf.io/nbvps/?view_only=e3f4246dc3204982b07684d517fe74f8

Also raw data can be provided under request: alicia.garrido@uam.es; pablo.varona@uam.es

If you use any of this code or material, please cite:

Alicia Garrido-Peña, Pablo Sanchez-Martin, Manuel Reyes-Sanchez, Rafael Levi, Francisco B. Rodriguez, Javier Castilla, Jesus Tornero, Pablo Varona, Modulation of neuronal dynamics by sustained and activity-dependent continuous-wave near-infrared laser stimulation, Neurophoton. 11(2), 024308 (2024), doi: 10.1117/1.NPh.11.2.024308. 

Please check the repositories related to this work. 

https://github.com/GNB-UAM/spike_predictor

https://github.com/GNB-UAM/Neun

And our Research's group website: www.ii.uam.es/~gnb


## How to use it
Load and activate conda environment from environment.yml file with the following commands:

	conda env create -f environment.yml
	conda activate Modulation-CW-NIR
	
### Figure 2
To plot the figures in panels A and B (the examples of the spike types) run:

	python bin/plot_waveforms_from_pkls.py data/shoulder-example/
	python bin/plot_waveforms_from_pkls.py data/symmetric-example/
This files read the info in df_all_waveforms.pkl and df_all_waveforms_metrics.pkl from their respective directories and generate the figures. In those dataframes the only data stored is these examples. 

To plot the figure in panel C, run the following:

	python bin/analyze_and_plot_general_bar.py -p ./data/ -en violins-data -pc data/violins-data/shoulder_classification.info

Note that the df_all_waveforms.pkl must be stored in data/violins-data/. This script automatically generates a directory unless it already exists.

Figure 2 in the supporting material will be generated also after running the script if "-pc" option is included. 

To run the t-test from the violin experiments run:

	python bin/t-test_continuous-laser.py data/violins-data/results/df_all_waveforms_metrics.pkl

The results from the statistical analysis will be displayed by terminal.

### Figure 3
In CW-sustained-experiments directory run:
	
	python bin/plot_firing_rate.py data/FR-analysis/
It will automatically save in FR_images the analysis output in PDF format. It will also create automatically a FR_log directory with the information of the files used. 

### Figure 4
For this figure, it is necessary to simulate the models. The equations of the model are specified in the manuscript. 

An implementation of the models can be found in: https://github.com/angellareo/NEUN

Also code for the simulations can be found at:
https://bitbucket.org/aligarpe/cgc-neuron/
https://bitbucket.org/aligarpe/laser_model/src/master/

To display the alignment of waveforms from Model_candidates directory run bin/superpos_from_model.py. This script requires parameters for the plot, for details run 

	python bin/superpos_from_model.py  --help
and follow this example:	
	 
	python bin/superpos_from_model.py -p ./1CGC-gHVA/ -rp "ghva" -ti "CGC Calcium gHVA" -sa y -sh n -st n -ff .pkl -c palette -oe y  -wt 50 -dt 0.01
	
where -p indicates the path where the traces are and the rest of parameters correspond to specifications for the plots. This script requires one file with the trace and another file with the events of the spikes in time. Both files must have the same name except of the spikes events file that will have "\_spikes" before the extension.
Example:
cgc_ghva.pkl; cgc_ghva_spikes.pkl 

To generate the table in panel E run from Model-candidates path:

	python3 bin/generate_table_stats_model.py -p ./;

with all waveforms located in subdirectories in Model-candidates path.

It might require to install the following tool 
	
	sudo apt install -y wkhtmltopdf

### Figure 5
To generate the data from the simulations, run the yaml files using the exe file *Q10-Model/bin/CGCNeuron-Q10*. The source and complete data can be found at https://bitbucket.org/aligarpe/cgc-neuron/

To generate the superposition and barchart with the metrics run:

	python bin/metrics_model_Q10.py -p data/general/dT5/

To generate panel B with the temperature comparative run:

	python bin/model_Q10_comparation.py ./data/general/q103_t10/ "Relation of dT with Q10=3";
	
### Figure 6
Run
	
	python bin/plot_day_shutter.py -sa y -sh n -pkl y -rastep 20 -rang "\-60,100" -ex "" -p ./data/activity-dependent-example/
### Figure 7
Run
	
	python bin/plot_general_shutter.py -sa y -sh n -pkl y -rastep 20 -rang "\-100,100" -ex "" -p ./data/activity-dependent-all-experiments/

### Supplementary Figure 2

To generate the superposition and metric/temperature relation dataframe for excluding and not excluding temperature, run:

	python bin/metrics_model_Q10.py -p data/[exclude_one|only_one]/[parameter name]

To generate the barchart comparative for both cases, run:
 	
  	cd Q10-Model/data/[exclude_one|only_one]/Q103/T5
	python ../../../../bin/get_q10_reference.py ./ "Only one parameter WITHOUT temperature dependency"
	

	
