#!/bin/bash

cd /home/lunet/cosoc/Desktop/orion_anomaly_pipeline

iterations=10
#sensors="M-6 M-1 M-2 S-2 P-10 T-4 T-5 F-7 M-3 M-4 M-5 P-15 C-1 C-2 T-12 T-13 F-4 F-5 D-14 T-9 P-14 T-8 P-11 D-15 D-16 M-7 F-8"
#dataset="MSL"

#sensors="P-1 S-1 E-1 E-2 E-3 E-4 E-5 E-6 E-7 E-8 E-9 E-10 E-11 E-12 E-13 A-1 D-1 P-2 P-3 D-2 D-3 D-4 A-2 A-3 A-4 G-1 G-2 D-5 D-6 D-7 F-1 P-4 G-3 T-1 T-2 D-8 D-9 F-2 G-4 T-3 D-11 D-12 B-1 G-6 G-7 P-7 R-1 A-5 A-6 A-7 D-13 P-2 A-8 A-9 F-3"
#dataset="SMAP"

sensors="sensor"
dataset="PU"
experiment="PU_0_full"
train_task="0"
test_task="all"
run_type="train"
fault_location="nan"
sample_freq="64"
model="ConvAE"

for sensor in $sensors
do

  for i in $(seq 1 $iterations)
  do

  python run_anomaly_pipeline.py $sensor $dataset $i $iterations $experiment $train_task $test_task $run_type $fault_location $sample_freq $model

  done
done
