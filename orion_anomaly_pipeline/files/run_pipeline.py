from orion import Orion
import pandas as pd
import os
import csv
import sys

# gets sensor names for dataset specified
def get_dataset_sensors():

    if dataset == "MSL":
        sensors = ["M-6", "M-1", "M-2", "S-2", "P-10", "T-4", "T-5", "F-7", "M-3", "M-4", "M-5", "P-15", "C-1", "C-2", "T-12", "T-13", "F-4", "F-5", "D-14", "T-9", "P-14", "T-8", "P-11", "D-15", "D-16", "M-7", "F-8"]
        return sensors

    elif dataset == "SMAP":
        sensors = ["P-1", "S-1", "E-1", "E-2", "E-3", "E-4", "E-5", "E-6", "E-7", "E-8", "E-9", "E-10", "E-11", "E-12", "E-13", "A-1", "D-1", "P-2", "P-3", "D-2", "D-3", "D-4", "A-2", "A-3", "A-4", "G-1", "G-2", "D-5", "D-6", "D-7", "F-1", "P-4", "G-3", "T-1", "T-2", "D-8", "D-9", "F-2", "G-4", "T-3", "D-11", "D-12", "B-1", "G-6", "G-7", "P-7", "R-1", "A-5", "A-6", "A-7", "D-13", "P-2", "A-8", "A-9", "F-3"]
        return sensors
    else:
        raise Exception("Dataset name not recognised")

# gets the training, test or labels based on mode
def get_data(sensor, mode):

    path = os.path.join(base_path, "datasets", dataset, mode, sensor) + ".csv"
    data = pd.read_csv(path)

    return data

# grabs the directory of the model and creates a directory if it doesnt exist
def get_model_dir(sensor, model_name):

    path = os.path.join(base_path, "models", experiment, dataset, sensor)
    os.makedirs(path, exist_ok = True)
    model_path = os.path.join(path, model_name)

    return model_path

# saves the trained model
def save_model(model_path, model):
    Orion.save(model, path=model_path)

# saves the training by taking the temp log file outputted from the library hard coded in tadgan.py
def save_training_log(model_name, sensor):
    path = os.path.join(base_path, "training_logs")
    old_path = os.path.join(path, "temp/loss.csv")
    new_path = os.path.join(path, experiment, dataset, sensor)
    os.makedirs(new_path, exist_ok = True)
    new_name = os.path.join(new_path, model_name) + ".csv"
    os.replace(old_path, new_name)

# evaluates the model on the test set and saves results
def evaluate_model(model, model_name, test_data, ground_truth):
    import orion.evaluation

    detected_anomalies = model.detect(test_data)

    results = {'model_name': model_name}


    confus_matrix = orion.evaluation.contextual_confusion_matrix(expected=ground_truth, observed=detected_anomalies, data=test_data, weighted=False)
    results['ovrlap_T_N'] = confus_matrix[0]
    results['ovrlap_F_P'] = confus_matrix[1]
    results['ovrlap_F_N'] = confus_matrix[2]
    results['ovrlap_T_P'] = confus_matrix[3]
    results['ovrlap_precision'] = orion.evaluation.contextual_precision(expected=ground_truth, observed=detected_anomalies, data=test_data, weighted=False)
    results['ovrlap_recall'] = orion.evaluation.contextual_recall(expected=ground_truth, observed=detected_anomalies, data=test_data, weighted=False)
    results['ovrlap_f1'] = orion.evaluation.contextual_f1_score(expected=ground_truth, observed=detected_anomalies, data=test_data, weighted=False)

    for i in range(len(ground_truth.index)):
        start = ground_truth['start'][i]
        end = ground_truth['end'][i]
        results["fault_" + str(i) + "_acc_weighted"] = orion.evaluation.contextual_accuracy(expected=ground_truth, observed=detected_anomalies, start=start, end=end, weighted=True)
        results["fault_" + str(i) + "_prec_weighted"] = orion.evaluation.contextual_precision(expected=ground_truth, observed=detected_anomalies, start=start, end=end, weighted=True)
        results["fault_" + str(i) + "_recall_weighted"] = orion.evaluation.contextual_recall(expected=ground_truth, observed=detected_anomalies, start=start, end=end, weighted=True)
        results["fault_" + str(i) + "_f1_weighted"] = orion.evaluation.contextual_f1_score(expected=ground_truth, observed=detected_anomalies, start=start, end=end, weighted=True)

    return results

# saves results to either new results file or appends to existing one
def save_results(results, sensor):
    path = os.path.join(base_path, "results", experiment, dataset)
    os.makedirs(path, exist_ok = True)
    results_path = path + ".csv"

    if not os.path.exists(results_path):
        with open(file=results_path, mode="a") as file:
            writer_object = csv.writer(file)
            writer_object.writerow(list(results.keys()))
            file.close()

    with open(file=results_path, mode="a") as file:
        writer_object = csv.writer(file)
        writer_object.writerow(results.values())
        file.close()

sensor = sys.argv[1]
dataset = sys.argv[2]
iteration = int(sys.argv[3])
iterations = sys.argv[4]
experiment = sys.argv[5]

# set path where all folders to run code are
base_path = "/home/lunet/cosoc/Desktop/orion_anomaly_pipeline"

# set the hyperparameters for training
hyperparameters = {
    "mlprimitives.custom.timeseries_preprocessing.time_segments_aggregate#1": {
        "time_column": "timestamp",
        "interval": 1,
        "method": "mean"
    },
    "orion.primitives.tadgan.TadGAN#1": {
        "epochs": 70,
        "batch_size": 64
    },
    "orion.primitives.tadgan.score_anomalies#1": {
        "rec_error_type": "dtw",
        "comb": "mult"
    }
}

sensors = get_dataset_sensors()

train_data = get_data(sensor, mode="train")
test_data = get_data(sensor, mode="test")
ground_truth = get_data(sensor, mode="labels")

orion = Orion(pipeline='tadgan', hyperparameters=hyperparameters)
print("Begun training " + str(iteration) + "/" + str(iterations) + " iterations for: " + dataset + "-" + sensor + " | sensor: " + str(sensors.index(sensor) + 1) + "/" + str(len(sensors)))
orion.fit(train_data)
model_name = sensor + "_" + str(iteration)
model_path = get_model_dir(sensor=sensor, model_name=model_name)
save_model(model_path=model_path, model=orion)
save_training_log(model_name=model_name, sensor=sensor)
results = evaluate_model(model=orion, model_name=model_name, test_data=test_data, ground_truth=ground_truth)
save_results(results=results, sensor=sensor)


