Repository for testing anomaly detection models and benchmark datasets

You can run and test the Convolutional Autoencoder on 2 datasets CWRU and PU bearing dataset.
Due to the size of PU dataset, only a fraction of normal data can be used for training. To edit how many files in the training data should be used, edit line 24 in ConvAE.py currently its set to use first 5 files, but if you have the  memory, simply remove "[0:5]"
CWRU uses all data available for training

~~PROCESS PU dataset~~

1) For PU dataset download all files from https://mb.uni-paderborn.de/en/kat/main-research/datacenter/bearing-datacenter/data-sets-and-download
2) Extract all folders to the PU folder in datasets_unprocessed
3) Open up the PU_label_list_maker.py in datasets_process_scripts folder and specify the absolute file path to the folder created in line 44 and then run the script. This is to create a csv with all the data labels
4) Open up "PU_normal_splitter.py" in datasets_process_scripts folder and specify the absolute path to the orion_anomaly_pipeline folder
5) Run "PU_normal_splitter.py" which will process the data and move it to a new folder called datasets

~~PROCESS CWRU dataset~~

1) For CWRU dataset download all files from https://engineering.case.edu/bearingdatacenter/download-data-file
2) Place all .mat files into th CWRU folder in datasets_unprocessed
3) Open up in datasets_process_scripts folder dataset_proccesor_CWRU.ipynb specify the path to the orion_anomaly_pipeline folder and run the script

~~TRAINING AND TESTING MODELS~~
1) open up run_anomaly_pipeline.sh and edit the parameters then once done in the command line type "bash run_anomaly_pipeline.sh"


~~PACKAGE REQUIREMENTS~~
cudatoolkit=11.8.0
nvidia-cudnn-cu11==8.6.0.163
tensorflow==2.12.0
numpy==1.23.5
matplotlib==3.7.1