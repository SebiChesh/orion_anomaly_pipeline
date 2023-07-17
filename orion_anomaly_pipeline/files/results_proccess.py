import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import csv

path = "/home/lunet/cosoc/Desktop/orion_anomaly_pipeline/results/TR_3_W_400_FL_DE_full_/"
path_res = os.path.join(path, "results.csv")
results = pd.read_csv(path_res)
eval_results = []
train_domain = 3

for row in results.iterrows():
    if row[1].fault[:6] != "normal":
        TP = row[1].correct_id
        FP = row[1].incorrect_id

        norm_name = "normal_" + str(row[1].test_task)
        norm = results.loc[(results["model_name"] == row[1].model_name) & (results["fault"] == norm_name)]

        TN = int(norm["correct_id"])
        FN = int(norm["incorrect_id"])

        acc = round(((TP + TN) / (TP + TN + FP + FN)), 2)
        prec = round(((TP) / (TP + FP)), 2)
        recall = round(((TP) / (TP + FN)), 2)
        f1 = round(((2 * prec * recall) / (prec + recall)), 2)

        result = {"model_name": row[1].model_name, "train_task": row[1].train_task, "test_task": row[1].test_task,
                  "fault": row[1].fault, "acc": acc, "prec": prec, "recall": recall, "f1": f1}
        eval_results.append(result)

eval_df = pd.DataFrame.from_dict(eval_results)
fault_type = ["IR007_", "IR014_", "IR021_", "IR028_", "OR007@6_", "OR014@6_", "OR021@6_", "OR007@3_", "OR021@3_",
                  "OR007@12_", "OR021@12_", "B007_", "B007_", "B014_", "B021_", "B028_"]

final_results = []

for i in range(0,4):

    for fault in fault_type:
            fault_name = fault + str(i)
            faulty = eval_df.loc[eval_df["fault"] == fault_name]
            acc = round(faulty["acc"].mean(), 2)
            acc_std = round(faulty["acc"].std(), 2)
            prec = round(faulty["prec"].mean(), 2)
            prec_std = round(faulty["prec"].std(), 2)
            recall = round(faulty["recall"].mean(), 2)
            recall_std = round(faulty["recall"].std(), 2)
            f1 = round(faulty["f1"].mean(), 2)
            f1_std = round(faulty["f1"].std(), 2)

            task_res = {"train_domain": train_domain, "test_domain": i, "fault": fault_name, "acc": acc, "acc_std": acc_std, "prec": prec, "prec_std": prec_std, "recall": recall, "recall_std": recall_std, "f1": f1, "f1_std": f1_std}
            final_results.append(task_res)


final_results_df = pd.DataFrame.from_dict(final_results)

final_results_df.to_csv(path_or_buf=os.path.join(path, "results_summary.csv"))
# # for i in fault_type:
# #     try:
# #         fault_name =