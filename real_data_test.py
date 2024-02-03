import os
from glob import glob

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from kfall_classification import plot_confusion, cal_tp_tn_fp_fn
from process import sliding_window
from utils.tcn import TCN


class Tester:
    def __init__(self, model, model_path, test_data_path, window_size, step_size, columns):
        self.model = model
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.window_size = window_size
        self.step_size = step_size
        self.columns = columns

    def load_model(self):
        model_weights = torch.load(self.model_path)
        self.model.load_state_dict(model_weights)
        self.model.eval()

    def get_windows_from_df(self, file: str) -> torch.Tensor:
        df = pd.read_parquet(file)
        arr = df[self.columns].to_numpy()
        windows = sliding_window(arr, window_size=self.window_size, step_size=self.step_size)
        windows = torch.from_numpy(windows).float()
        return windows

    def get_label_from_file_path(self, path: str) -> int:
        label = os.path.basename(os.path.dirname(path))
        label = int(label == 'fall')
        return label

    def test(self):
        list_data_files = glob(self.test_data_path)
        device = torch.device("cpu")
        self.model.to(device)

        predictions = []
        actuals = []

        with torch.no_grad():
            for file in tqdm(list_data_files, ncols=0):
                data = self.get_windows_from_df(file).to(device)
                label = self.get_label_from_file_path(file)

                pred = self.model(data)
                pred = pred.squeeze(1)
                binary_outputs = torch.round(torch.sigmoid(pred)).cpu().numpy()
                binary_outputs = (binary_outputs == 1).any().item()
                predictions.append(binary_outputs)
                actuals.append(label)

        accuracy = accuracy_score(actuals, predictions)
        confusion = plot_confusion(actuals, predictions, [0, 1])

        FP, FN, TP, TN = cal_tp_tn_fp_fn(confusion)
        TPR = TP / (TP + FN)  # Sensitivity
        FPR = FP / (FP + TN)  # false positive rate
        F1 = float(f1_score(actuals, predictions, average='weighted'))

        return accuracy, F1, TPR, FPR, FP, FN, TP, TN

    def run(self, n_times=1):
        self.load_model()

        accuracies = []
        f1_scores = []
        TPRs = []
        FPRs = []
        FPs = []
        FNs = []
        TPs = []
        TNs = []

        for _ in range(n_times):
            accuracy, F1, TPR, FPR, FP, FN, TP, TN = self.test()
            accuracies.append(accuracy)
            f1_scores.append(F1)
            TPRs.append(TPR)
            FPRs.append(FPR)
            FPs.append(FP)
            FNs.append(FN)
            TPs.append(TP)
            TNs.append(TN)

        print(f'Average Test Accuracy: {np.mean(accuracies):.4f}, Average F1 score: {np.mean(f1_scores):.4f},'
              f' Average TPR: {np.mean(TPRs):.4f}, Average FPR: {np.mean(FPRs):.4f},'
              f' Average FP: {np.mean(FPs):.2f}, Average FN: {np.mean(FNs):.2f},'
              f' Average TP: {np.mean(TPs):.2f}, Average TN: {np.mean(TNs):.2f}')


if __name__ == "__main__":


    model = TCN(3, 1, (64,) * 5 + (128,) * 2, 2, 0.5, "mean")
    columns_to_use = ['acc_x', 'acc_y', 'acc_z']

    path = 'D:/Repository/master/Processed_Dataset/RealData/*/*.parquet'

    Tester(
        model=model,
        model_path=
        "C:/Users/46270/PycharmProjects/pythonProject3/fallAllD_cla_model_new_aug_waist_acc/model_size4_epoch12.pth",
        test_data_path=path,
        window_size=200,
        step_size=100,
        columns=columns_to_use
    ).run()
