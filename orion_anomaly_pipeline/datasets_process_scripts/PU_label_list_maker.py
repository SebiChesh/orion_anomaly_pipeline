import model_pipelines.utils as ut

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

label_list =[]

for code in bearing_list:
    for state in state_list:
        for i in range(1, 21):
            if state["state_number"] == 1:
                rpm = "900"
            else:
                rpm = "1500"
            entry = {"file_name": "_".join([state["state_name"], code["bearing"], str(i)]),
                     "label": "_".join([code["fault"], str(state["state_number"])]),
                     "fault_location": "nan", "fault_diameter": "nan", "motor_load": str(state["state_number"]),
                     "rpm": rpm, "sample_freq": "64", "fault": code["fault"]
                     }

            label_list.append(entry)

#specify absolute file path to the PU folder in datasets_unprocessed
path = "/home/lunet/cosoc/Desktop/orion_anomaly_pipeline/datasets_unprocessed/PU/"
ut.dicts_to_csv(path=path, file_name="labels.csv", list_dicts=label_list)
