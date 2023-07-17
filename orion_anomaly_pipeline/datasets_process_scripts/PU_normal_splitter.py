from scipy.io import loadmat
import numpy as np
import pandas as pd
import os
import math

base_path = "/home/lunet/cosoc/Desktop/orion_anomaly_pipeline/"

bearing_list = [{"bearing": "K001", "fault": "normal"}, {"bearing": "K002", "fault": "normal"},
                {"bearing": "K003", "fault": "normal"}, {"bearing": "K004", "fault": "normal"},
                {"bearing": "K005", "fault": "normal"}, {"bearing": "K006", "fault": "normal"},
                {"bearing": "KA01", "fault": "OR"}, {"bearing": "KA03", "fault": "OR"},
                {"bearing": "KA05", "fault": "OR"}, {"bearing": "KA06", "fault": "OR"},
                {"bearing": "KA07", "fault": "OR"}, {"bearing": "KA08", "fault": "OR"},
                {"bearing": "KA09", "fault": "OR"}, {"bearing": "KI01", "fault": "IR"},
                {"bearing": "KI03", "fault": "IR"}, {"bearing": "KI05", "fault": "IR"},
                {"bearing": "KI07", "fault": "IR"}, {"bearing": "KI08", "fault": "IR"},
                {"bearing": "KA04", "fault": "OR"}, {"bearing": "KA15", "fault": "OR"},
                {"bearing": "KA16", "fault": "OR"}, {"bearing": "KA22", "fault": "OR"},
                {"bearing": "KA30", "fault": "OR"}, {"bearing": "KB23", "fault": "COMBO"},
                {"bearing": "KB24", "fault": "COMBO"}, {"bearing": "KI04", "fault": "IR"},
                {"bearing": "KI14", "fault": "IR"}, {"bearing": "KI16", "fault": "IR"},
                {"bearing": "KI17", "fault": "IR"}, {"bearing": "KI18", "fault": "IR"},
                {"bearing": "KI21", "fault": "IR"},
                ]

state_list = [{"state_name": "N15_M07_F10", "state_number": 0},
              {"state_name": "N09_M07_F10", "state_number": 1},
              {"state_name": "N15_M01_F10", "state_number": 2},
              {"state_name": "N15_M07_F04", "state_number": 3}]

split = 0.6

def mk_dir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        # directory already exists
        pass

new_dir_train = os.path.join(base_path, "datasets", "PU", "train")
new_dir_test = os.path.join(base_path, "datasets", "PU", "test")

mk_dir(new_dir_train)
mk_dir(new_dir_test)

for code in bearing_list:
    for state in state_list:
        for i in range(1,21):
            file_name = "_".join([state["state_name"], code["bearing"], str(i)]) + ".mat"
            if file_name != "N15_M01_F10_KA08_2.mat":
                matfile = loadmat(os.path.join(base_path, "datasets_unprocessed", "PU", code["bearing"], file_name))
                data = matfile[file_name[:-4]][0][0]["Y"]["Data"][0][-1][0]

                if code["fault"] == "normal":
                    index_split = math.ceil((data.size) * split)

                    train_data = pd.DataFrame(
                        {"timestamp": list(range(0, index_split)), "value": list(data[0:index_split].flatten())})
                    test_data = pd.DataFrame(
                        {"timestamp": list(range(index_split, data.size)), "value": list(data[index_split:].flatten())})

                    file_name_csv = file_name[:-4] + ".csv"
                    train_data.to_csv(path_or_buf=os.path.join(new_dir_train, file_name_csv), index=False)
                    test_data.to_csv(path_or_buf=os.path.join(new_dir_test, file_name_csv), index=False)

                    print("saved: " + file_name_csv)

                else:
                    test_data = pd.DataFrame(
                        {"timestamp": list(range(0, data.size)), "value": list(data[:].flatten())})
                    file_name_csv = file_name[:-4] + ".csv"
                    test_data.to_csv(path_or_buf=os.path.join(new_dir_test, file_name_csv), index=False)

                    print("saved: " + file_name_csv)
