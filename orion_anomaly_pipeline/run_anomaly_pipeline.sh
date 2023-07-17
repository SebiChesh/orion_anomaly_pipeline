#!/bin/bash

#specify the absolute path to the orion_anomaly_pipeline folder
cd /home/lunet/cosoc/Desktop/orion_anomaly_pipeline

#specify how many times you want to repeat the experiment
iterations=1

#sensors specifies which sensor data should be used for training, CWRU dataset can use either "DE" or "FE" for drive-end or fan end bearing. PU dataset has only one sensor so specify "sensor"
sensors="DE"
#dataset specifies the dataset to use, can specify either "PU" for Paderborn University Bearing dataset or "CWRU" for Case Western University bearing dataset.
dataset="CWRU"
#experiment specify the name you would like to give to the experiment you will run, it will save all experiment specific files to that name
experiment="handover_CWRU"
#for CWRU and PU there are 4 different operating conditions specify "0" or "1" or "2" or "3" to choose which operating condition to train on
train_task="0"
#test_task specify "0" or "1" or "2" or "3" or "all" to choose which operating condition data you want to use to evaluate the model. If you specify "all", all operating conditions will be evaluated.
test_task="1"
#run_type specify either "train" to train the model or "test" to test the model.`
run_type="test"
#fault_location specify which sensor data to use during test for PU there is only one so specify "nan" for CWRU specify either "FE" or "DE"
fault_location="DE"
#sample_freq some datasets have data sampled at different sampling frequencies, for "PU" only 64KHz so specify "64" for CWRU specify either "12" or "48"
sample_freq="12"
#model specify the model name, currently only "ConvAE" for the convolutional autoencoder.
model="ConvAE"

for sensor in $sensors
do

  for i in $(seq 1 $iterations)
  do

    python run_anomaly_pipeline.py $sensor $dataset $i $iterations $experiment $train_task $test_task $run_type $fault_location $sample_freq $model

  done
done
