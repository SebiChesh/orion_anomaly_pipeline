import os

import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import model_pipelines.utils as ut


def train_model(dataset, train_task, sensor=""):
    # loads training data, converts it to sequences to feed model during training,
    # trains model, calculates threshold for detecting anomalies

    task_id = "normal_" + str(train_task)
    base_path = "/home/lunet/cosoc/Desktop/orion_anomaly_pipeline/datasets/"

    # gets data for training depending on dataset
    if dataset == "CWRU":
        data = ut.get_data(mode="train", dataset=dataset, sensor=sensor, task_id=task_id)
    else:
        labels = pd.read_csv(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets_unprocessed", dataset,
                                         'labels.csv')))

        train_list = labels.loc[labels["label"] == task_id][0:5]

        data = pd.DataFrame({"value": []})
        for row in train_list.iterrows():
            file_name = row[1].file_name + ".csv"
            df = ut.get_data(mode="train", dataset=dataset, file_name=file_name)
            # concatinates multple normal files into one dataframe
            data = pd.concat([data, df], ignore_index=True)

        # for row in train_list:
        #     file_name = row + ".csv"
        #     df = ut.get_data(mode="train", dataset=dataset, file_name=file_name)
        #     # concatinates multple normal files into one dataframe
        #     data = pd.concat([data, df], ignore_index=True)



    train_mean = data.mean()
    train_std = data.std()
    train_data = ut.normalise_data(data=data, mean=train_mean, std=train_std)

    window = 400
    x_train = ut.create_sequences(values=train_data, time_steps=window)

    model = keras.Sequential(
        [
            layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
            layers.Conv1D(
                filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Dropout(rate=0.2),
            layers.Conv1D(
                filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Conv1DTranspose(
                filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Dropout(rate=0.2),
            layers.Conv1DTranspose(
                filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")

    history = model.fit(
        x_train,
        x_train,
        epochs=50,
        batch_size=128,
        validation_split=0.1,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")],
    )

    threshold = ut.get_threshold(x_train, model)

    return model, history.history, threshold, train_mean.value, train_std.value


def run(sensor, dataset, iteration, iterations, experiment, train_task, test_task, run_type, fault_location,
        sample_freq="12"):
    if dataset == "CWRU":
        model_name = sensor + "_tr_" + str(train_task) + "_it_" + str(iteration)

        if run_type == "train":
            model, history, threshold, train_mean, train_std = train_model(dataset=dataset, sensor=sensor,
                                                                           train_task=train_task)
            ut.save_training_log(training_log=history, model_name=model_name, sensor=sensor, experiment=experiment,
                                 dataset=dataset)
            ut.save_model(model=model, model_name=model_name, experiment=experiment, dataset=dataset, sensor=sensor,
                          threshold=threshold, mean=train_mean, std=train_std)

        else:
            path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", experiment, dataset, sensor,
                                                model_name))

            model = keras.models.load_model(path)
            ut.evaluate_model(model=model, model_name=model_name, train_task=train_task, test_task=test_task,
                              sample_freq=sample_freq, sensor=sensor, dataset=dataset, experiment=experiment,
                              fault_location=fault_location)

    elif dataset == "PU":
        model_name = "model_tr_" + str(train_task) + "_it_" + str(iteration)

        if run_type == "train":
            model, history, threshold, train_mean, train_std = train_model(dataset=dataset, train_task=train_task)
            ut.save_training_log(training_log=history, model_name=model_name, experiment=experiment,
                                 dataset=dataset)

            ut.save_model(model=model, model_name=model_name, experiment=experiment, dataset=dataset,
                          threshold=threshold, mean=train_mean, std=train_std)


        else:
            path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", experiment, dataset,
                                                model_name))
            model = keras.models.load_model(path)

            ut.evaluate_model(model=model, model_name=model_name, train_task=train_task, test_task=test_task,
                              dataset=dataset, experiment=experiment)


