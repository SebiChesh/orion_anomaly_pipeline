import os
import csv
import numpy as np
import pandas as pd

def get_data(mode, dataset, sensor="", task_id="", fault_location="", fault_type="",  sample_freq="", file_name=""):
    if dataset == "CWRU":
        if mode == "train":
            path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "datasets", dataset, mode, task_id, sensor) + ".csv")
            data = pd.read_csv(path, index_col="timestamp")

        elif mode == "test":
            if fault_type == "normal":
                path = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..", "datasets", dataset, mode, task_id, sensor)) + ".csv"
                data = pd.read_csv(path, index_col="timestamp")
            else:
                freq = sample_freq + "k"
                path = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..", "datasets", dataset, mode, freq, fault_location,
                                 task_id, sensor)) + ".csv"
                data = pd.read_csv(path, index_col="timestamp")
    else:
        path_to_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets", dataset, mode,
                                                    file_name))

        data = pd.read_csv(path_to_file, index_col="timestamp")


    return data

def create_sequences(values, time_steps=400):
    # Generates a widows of length time_steps of a sequence

    output = []
    i = 0
    while i <= len(values) - time_steps:
        output.append(values[i: (i + time_steps)])
        i += 1
    return np.stack(output)

def save_training_log(training_log, model_name, experiment, dataset, sensor=""):
    # Takes training logs as a dict and saves it to a csv
    if dataset == "CWRU":
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training_logs", experiment,
                                            dataset, sensor))
    else:
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training_logs", experiment,
                                            dataset))

    os.makedirs(path, exist_ok = True)
    file_name = model_name + ".csv"
    log_df = pd.DataFrame(training_log)
    log_df.to_csv(path_or_buf=os.path.join(path, file_name))

def get_threshold(x_train, model):
    # Calculates the threshold by getting the max MSE from reconstructing the training data

    x_train_pred = model.predict(x_train)
    train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)
    threshold = np.max(train_mae_loss)

    return threshold

def save_model(model, experiment, dataset,  model_name, threshold, mean, std, sensor=""):
    if dataset == "CWRU":
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", experiment, dataset, sensor,
                                            model_name))
    else:
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", experiment, dataset, model_name))

    model.save(path)
    threshold_path = os.path.join(path, "train_info.csv")
    with open(threshold_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["threshold", "train_data_mean", "train_data_std"])
        writer.writerow([str(threshold), str(mean), str(std)])


def evaluate_model(model, model_name, train_task, test_task,  dataset, experiment, fault_location="", sensor="", sample_freq=""):
    labels = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets_unprocessed", dataset,
                                                      'labels.csv')))
    results = []

    if test_task == "all":
        tasks = list(range(0, 4))

    else:
        tasks = [test_task]

    for i in tasks:
        #gets list of testing tasks based on the dataset selected
        if dataset == "CWRU":
            test_list = labels.loc[(labels["motor_load"]==int(i)) & ((labels["fault_location"]==fault_location) | (labels["fault"]=="normal")) & (labels["sample_freq"]==int(sample_freq))]
        elif dataset == "PU":
            test_list = labels.loc[labels["motor_load"]==i]
        task_number = 0

        for row in test_list.iterrows():
            print("Test task: " + str(i) + " | subtask: " + str(task_number) + "/" + str(len(test_list)))
            task_number += 1
            if dataset == "CWRU":
                data = get_data(mode="test", dataset=dataset, sensor=sensor, task_id=row[1].label,
                                fault_type=row[1].fault, fault_location=fault_location, sample_freq=sample_freq)

                path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", experiment, dataset,
                                                sensor, model_name, "train_info.csv"))
            else:
                file_name = row[1].file_name + ".csv"
                data = get_data(mode="test", dataset=dataset, file_name=file_name)
                path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", experiment, dataset,
                                                    model_name, "train_info.csv"))

            train_info = pd.read_csv(path)
            test_data = normalise_data(data, mean=train_info.train_data_mean[0],
                                       std=train_info.train_data_std[0])
            n = 0
            window = 3000
            correct_id, incorrect_id = 0,0

            # Converts the test task signal into windows of length window. converts the window into segments
            # which are then used to determine which are fed into the model to reconstruct and the output is
            # used to determine which points in the window are anomalous

            while n <= len(test_data) - window:
                test_sequence = create_sequences(test_data.values[n:n+window])

                # Get test MAE loss for that window
                test_pred = model.predict(test_sequence)
                test_mae_loss = np.mean(np.abs(test_pred - test_sequence), axis=1)
                test_mae_loss = test_mae_loss.reshape((-1))

                # Determines which points are anomalous based on if that points window is above threshold
                anomalous_points = test_mae_loss > train_info.threshold[0]

                # data i is an anomaly if samples [(i - time_steps + 1) to (i)] are anomalies
                anomalous_data_indices = []
                for data_idx in range(test_pred.shape[1] - 1, window - test_pred.shape[1] + 1):
                    if np.all(anomalous_points[data_idx - test_pred.shape[1] + 1: data_idx + 1]):
                        anomalous_data_indices.append(data_idx)
                # Label the window as anomalous if at least 1 point in the segment is identified as anomalous
                if np.count_nonzero(anomalous_data_indices) > 0:
                    anomalous_segment = True
                else:
                    anomalous_segment = False

                # determines whether model correctly identified an anomalous or normal window
                if anomalous_segment != (row[1].fault == "normal"):
                        correct_id +=1

                else:
                    incorrect_id +=1

                n += window

            # Adds test result to results list
            result = {"file_name": row[1].file_name, "model_name": model_name, "train_task": train_task, "test_task": row[1].motor_load,
                      "fault": row[1].fault, "correct_id": correct_id, "incorrect_id": incorrect_id}



            result_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", experiment))
            file_name = "results.csv"
            dicts_to_csv(path=result_path, file_name=file_name, list_dicts=[result])



# Function to normalise data
def normalise_data(data, std, mean):
    normalised_data = (data - mean) / std
    return normalised_data

# Takes a list of dicts and either creates a csv at the path if one doesn't exist or it writes to the csv existing csv
def dicts_to_csv(path, file_name, list_dicts):
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, file_name)

    if not os.path.isfile(file_path):
        df = pd.DataFrame(list_dicts)
        df.to_csv(file_path, index=False)

    else:
        with open(file_path, 'a') as file:
            writer = csv.writer(file)
            for i in list_dicts:
                writer.writerow(i.values())



