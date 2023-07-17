
import model_pipelines.ConvAE
import sys

if __name__ == "__main__":

    sensor = sys.argv[1]
    dataset = sys.argv[2]
    iteration = int(sys.argv[3]) - 1
    iterations = sys.argv[4]
    experiment = sys.argv[5]
    train_task = sys.argv[6]
    test_task = sys.argv[7]
    run_type = sys.argv[8]
    fault_location = sys.argv[9]
    sample_freq = sys.argv[10]
    model = sys.argv[11]

# sensor = "sensor"
# dataset = "PU"
# iteration = "1"
# iterations = "1"
# experiment = "PU_test"
# model = "ConvAE"
# train_task = "3"
# test_task = "all"
# run_type = "train"
# fault_location = "DE"
# sample_freq = "64"

if model == "ConvAE":

    model_pipelines.ConvAE.run(sensor=sensor, dataset=dataset, iteration=iteration, iterations=iterations, experiment=experiment, train_task=train_task, test_task=test_task, run_type=run_type, fault_location=fault_location, sample_freq=sample_freq)









