import re

import pandas as pd
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from tcn import TemporalConvNet
import os
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from torch.optim import Adam
from tcn import TCN
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
from data_load_and_EarlyStop import EarlyStopping, BasicArrayDataset, ResampleArrayDataset
import utils.augmentation as aug
from trajectory_cal import whole_trajectory, cal_acc_norm, cal_gyr_norm


def load_dataset(root_dir, preprocess_fn):
    train_dict = {0: [], 1: []}
    valid_dict = {0: [], 1: []}
    test_dict = {0: [], 1: []}

    windows = []
    labels = []

    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            arr = np.load(file_path)[:, :, :]
            processed_arr = preprocess_fn(arr)
            if folder.endswith('_fall'):
                file_label = 1
            else:
                file_label = 0
            windows.append(processed_arr)
            labels += [file_label] * len(arr)

    windows = np.concatenate(windows)
    labels = np.array(labels)

    np.random.seed(123)
    fall_idx = np.where(labels == 1)[0]
    not_fall_idx = np.where(labels == 0)[0]

    np.random.shuffle(fall_idx)
    np.random.shuffle(not_fall_idx)

    train_fall_idx = fall_idx[:int(0.7 * len(fall_idx))]
    valid_fall_idx = fall_idx[int(0.7 * len(fall_idx)):int(0.9 * len(fall_idx))]
    test_fall_idx = fall_idx[int(0.9 * len(fall_idx)):]

    train_not_fall_idx = not_fall_idx[:int(0.7 * len(not_fall_idx))]
    valid_not_fall_idx = not_fall_idx[int(0.7 * len(not_fall_idx)):int(0.9 * len(not_fall_idx))]
    test_not_fall_idx = not_fall_idx[int(0.9 * len(not_fall_idx)):]

    train_idx = np.concatenate([train_fall_idx, train_not_fall_idx])
    valid_idx = np.concatenate([valid_fall_idx, valid_not_fall_idx])
    test_idx = np.concatenate([test_fall_idx, test_not_fall_idx])

    # split dataset into train, valid and test
    windows_train = windows[train_idx]
    labels_train = labels[train_idx]
    windows_valid = windows[valid_idx]
    labels_valid = labels[valid_idx]
    windows_test = windows[test_idx]
    labels_test = labels[test_idx]

    # append train into train_dict(s)
    train_dict[0].append(windows_train[labels_train == 0])
    train_dict[1].append(windows_train[labels_train == 1])

    # append valid into valid dict
    valid_dict[0].append(windows_valid[labels_valid == 0])
    valid_dict[1].append(windows_valid[labels_valid == 1])

    # append test into test dict
    test_dict[0].append(windows_test[labels_test == 0])
    test_dict[1].append(windows_test[labels_test == 1])

    # return result
    train_dict = {key: np.concatenate(value) for key, value in train_dict.items()}
    valid_dict = {key: np.concatenate(value) for key, value in valid_dict.items()}
    test_dict = {key: np.concatenate(value) for key, value in test_dict.items()}

    return train_dict, valid_dict, test_dict


def load_dataset_with_raw(root_dir):
    return load_dataset(root_dir, lambda arr: arr[:, :, 1:])


def load_dataset_with_acc(root_dir):
    return load_dataset(root_dir, lambda arr: arr[:, :, 1:4])


def load_dataset_with_acc_euler_gyr_norm(root_dir):
    def preprocess_fn(arr):
        gyr_norms = cal_gyr_norm(arr)
        norm_array = np.expand_dims(gyr_norms, axis=-1)
        euler = arr[:, :, 7:].copy()
        acc = arr[:, :, 1:4].copy()
        combined = np.concatenate((acc, norm_array), axis=2)
        combined = np.concatenate((combined, euler), axis=2)
        return combined

    return load_dataset(root_dir, preprocess_fn)


def load_dataset_with_trajectory_acc(root_dir):
    def preprocess_fn(arr):
        trajectories = whole_trajectory(arr)
        acc = arr[:, :, 1:4].copy()
        combined = np.concatenate((trajectories, acc), axis=2)
        return combined

    return load_dataset(root_dir, preprocess_fn)


def load_dataset_with_euler_acc(root_dir):
    def preprocess_fn(arr):
        acc = arr[:, :, 1:4].copy()
        euler = arr[:, :, 7:].copy()
        combined = np.concatenate((acc, euler), axis=2)
        return combined
    return load_dataset(root_dir, preprocess_fn)


def load_dataset_with_trajectory_acc_norm(root_dir):
    def preprocess_fn(arr):
        acc_norms = cal_acc_norm(arr)
        trajectory = whole_trajectory(arr)
        norm_array = np.expand_dims(acc_norms, axis=-1)
        combined = np.concatenate((trajectory, norm_array), axis=2)
        return combined

    return load_dataset(root_dir, preprocess_fn)


def load_dataset_with_trajectory_raw(root_dir):
    def preprocess_fn(arr):
        raw = arr[:, :, 1:].copy()
        trajectory = whole_trajectory(arr)
        combined = np.concatenate((raw, trajectory), axis=2)
        return combined

    return load_dataset(root_dir, preprocess_fn)


def cal_tp_tn_fp_fn(confusionmatrix):
    tn, fp, fn, tp = confusionmatrix.ravel()
    return tn, fp, fn, tp


def plot_confusion(y_true, y_pred, labels):
    sns.set()
    f, ax = plt.subplots()
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(matrix, annot=True, ax=ax)
    ax.set_xlabel('predict')
    ax.set_ylabel('true')
    plt.show()
    return matrix


class ClassificationModel:
    def __init__(self, dataset_path, batch_size_train, batch_size_valid, batch_size_test, input_size, output_size,
                 num_channels, flatten_method, kernel_size, dropout, learning_rate, num_epochs, model_save_path,
                 augmenter, aug_name, load_method):
        self.load_method = load_method
        self.augmenter = augmenter
        self.dataset_path = dataset_path
        self.batch_size_train = batch_size_train
        self.batch_size_valid = batch_size_valid
        self.batch_size_test = batch_size_test
        self.input_size = input_size
        self.output_size = output_size
        self.num_channels = num_channels
        self.flatten_method = flatten_method
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.model_save_path = model_save_path
        self.aug_name = aug_name

    def run(self):
        # Load data
        if self.load_method == 'raw':
            train, valid, test = load_dataset_with_raw(self.dataset_path)
        elif self.load_method == 'acc':
            train, valid, test = load_dataset_with_acc(self.dataset_path)
        elif self.load_method == 'trajectory_acc':
            train, valid, test = load_dataset_with_trajectory_acc(self.dataset_path)
        elif self.load_method == 'euler_acc':
            train, valid, test = load_dataset_with_euler_acc(self.dataset_path)
        elif self.load_method == 'trajectory_acc_norm':
            train, valid, test = load_dataset_with_trajectory_acc_norm(self.dataset_path)
        elif self.load_method == 'trajectory_raw':
            train, valid, test = load_dataset_with_trajectory_raw(self.dataset_path)
        elif self.load_method == 'acc_euler_gyr_norm':
            train, valid, test = load_dataset_with_acc_euler_gyr_norm(self.dataset_path)
        train_set = ResampleArrayDataset(train, augmenter=self.augmenter)
        train_loader = DataLoader(train_set, batch_size=self.batch_size_train, shuffle=True)
        valid_set = BasicArrayDataset(valid)
        valid_loader = DataLoader(valid_set, batch_size=self.batch_size_valid, shuffle=False)
        test_set = BasicArrayDataset(test)
        test_loader = DataLoader(test_set, batch_size=self.batch_size_test, shuffle=False)

        window_size = self.dataset_path.split('window_sec')[1]

        if not os.path.exists(self.model_save_path):
            # If the directory doesn't exist, create it
            os.makedirs(self.model_save_path)
        early_stopping = EarlyStopping(self.model_save_path, window_size)

        # Instantiate the TCN model
        model = TCN(self.input_size, self.output_size, self.num_channels, self.kernel_size, self.dropout,
                    self.flatten_method)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)  # 每30个epoch，学习率乘以0.1

        # Move the model to the GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Define the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        # Create a dictionary to store the metrics during training
        metrics = {
            'epoch': [],
            'train_loss': [],
            'valid_loss': [],
            'valid_accuracy': [],
            'valid_f1': []
        }

        # Training loop
        for epoch in range(self.num_epochs):
            # Training loop
            model.train()  # Set the model in training mode
            train_loss = 0
            pbar = tqdm(total=len(train_loader), ncols=0)
            for x, y in train_loader:
                # Move the inputs and labels to the GPU if available
                x = x.to(device)
                y = y.to(device)

                # Clear the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(x)
                outputs = outputs.squeeze(1)  # shape[batch, channel]
                # Calculate the loss
                t_loss = F.binary_cross_entropy_with_logits(outputs, y.float())  # use the new function here

                train_loss += t_loss.item()
                # Backward pass
                t_loss.backward()

                # Update the parameters
                optimizer.step()

                scheduler.step()

                pbar.update(1)

            # Track the training progress
            pbar.close()

            model.eval()
            valid_loss = 0
            valid_outputs, valid_labels = [], []

            with torch.no_grad():
                for x, y in valid_loader:
                    x = x.to(device)
                    y = y.to(device)

                    pred = model(x)
                    pred = pred.squeeze(1)
                    v_loss = F.binary_cross_entropy_with_logits(pred, y.float())
                    valid_loss += v_loss.item()

                    # Append the model predictions and true labels to their respective lists
                    binary_outputs = torch.round(torch.sigmoid(pred)).cpu().numpy()
                    valid_outputs.extend(binary_outputs)
                    valid_labels.extend(y.cpu().numpy())

            valid_loss /= len(valid_loader)

            valid_accuracy = accuracy_score(valid_labels, valid_outputs)
            valid_f1 = f1_score(valid_labels, valid_outputs, average='weighted')

            print('Epoch [{}/{}], train_Loss: {:.4f}, valid_Loss: {:.4f}, accuracy: {:.4f}, f1 score: {:.4f}'.format(
                epoch + 1, self.num_epochs, train_loss / len(train_loader), valid_loss, valid_accuracy, valid_f1))

            # Save metrics for this epoch
            metrics['epoch'].append(epoch + 1)
            metrics['train_loss'].append(train_loss / len(train_loader))
            metrics['valid_loss'].append(valid_loss)
            metrics['valid_accuracy'].append(valid_accuracy)
            metrics['valid_f1'].append(valid_f1)

            # 早停止
            early_stopping(epoch, valid_loss, model)
            # 达到早停止条件时，early_stop会被置为True
            if early_stopping.early_stop:
                print("Early stopping")
                break  # 跳出迭代，结束训练

        # Convert metrics dictionary to DataFrame and save as CSV
        df_metrics = pd.DataFrame(metrics)

        csv_file_name = f'./KFall_train_records/{window_size}_training_metrics_{self.aug_name}.csv'

        if not os.path.exists('./KFall_train_records'):
            # If the directory doesn't exist, create it
            os.makedirs('./KFall_train_records')

        df_metrics.to_csv(csv_file_name, index=False)

        # Test loop
        model.eval()  # Set the model in evaluation mode
        test_loss = 0
        predictions = []
        actuals = []

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)

                pred = model(x)
                pred = pred.squeeze(1)

                # Optionally compute test loss
                t_loss = F.binary_cross_entropy_with_logits(pred, y.float())
                test_loss += t_loss.item()

                # Convert predicted probabilities to binary outputs
                binary_outputs = torch.round(torch.sigmoid(pred)).cpu().numpy()

                # Append the model predictions and true labels to their respective lists
                predictions.extend(binary_outputs)
                actuals.extend(y.cpu().numpy())
                print(binary_outputs)
                print(y.cpu().numpy())

        accuracy = accuracy_score(actuals, predictions).mean()
        confusion = plot_confusion(actuals, predictions, [0, 1])

        FP, FN, TP, TN = cal_tp_tn_fp_fn(confusion)
        TPR = TP / (TP + FN)  # Sensitivity
        FPR = FP / (FP + TN)  # false positive rate
        F1 = float(f1_score(actuals, predictions, average='weighted'))

        print('Test Accuracy: {:.4f}, F1 score: {:.4f}, TPR: {:.4f}, FPR: {:.4f}'.format(accuracy, F1, TPR, FPR))
        return window_size, accuracy, F1, TPR, FPR, FP, FN, TP, TN


if __name__ == "__main__":
    # root_dir = 'C:/Repository/master/Processed_Dataset/KFall'
    #
    # augmentations = {
    #     'timewarp': aug.Timewarp(sigma=0.2, knot=4, p=0.5),
    #     'jitter': aug.Jitter(sigma=0.05, p=0.5),
    #     'scale': aug.Scale(sigma=0.1, p=0.5),
    #     'magnitudeWarp': aug.MagnitudeWarp(sigma=0.2, knot=4, p=0.5),
    #     'rotation': aug.Rotation(angle_range=180, p=0.5),
    #     'permutation': aug.Permutation(n_perm=4, min_seg_length=10, p=0.5),
    #     'randSample': aug.RandSample(n_sample=150, p=0.5),
    # }
    #
    # for name, augmentation in augmentations.items():
    #     augmenter = aug.Augmenter([augmentation])
    #     test_metrics = {
    #         'window_size': [],
    #         'test_accuracy': [],
    #         'test_F1': [],
    #         'test_TPR': [],
    #         'test_FPR': [],
    #         'FP': [],
    #         'FN': [],
    #         'TP': [],
    #         'TN': []
    #     }
    #     for folder in os.listdir(root_dir):
    #         folder_path = os.path.join(root_dir, folder)
    #         # Create a dictionary to store the metrics during testing
    #         window_size, accuracy, F1, TPR, FPR, FP, FN, TP, TN = ClassificationModel(
    #             dataset_path=folder_path,
    #             batch_size_train=8,
    #             batch_size_valid=16,
    #             batch_size_test=16,
    #             input_size=9,
    #             output_size=1,
    #             flatten_method="mean",
    #             num_channels=(64,) * 3 + (128,) * 2,
    #             kernel_size=2,
    #             dropout=0.5,
    #             load_method='mean',
    #             learning_rate=0.01,
    #             num_epochs=30,
    #             model_save_path=f"./seg_model_{name}",
    #             augmenter=augmenter,
    #             aug_name=name
    #         ).run()
    #
    #         test_metrics['window_size'].append(window_size)
    #         test_metrics['test_accuracy'].append(accuracy)
    #         test_metrics['test_F1'].append(F1)
    #         test_metrics['test_TPR'].append(TPR)
    #         test_metrics['test_FPR'].append(FPR)
    #         test_metrics['FP'].append(FP)
    #         test_metrics['FN'].append(FN)
    #         test_metrics['TP'].append(FP)
    #         test_metrics['TN'].append(FP)
    #     df_metrics = pd.DataFrame(test_metrics)
    #     df_metrics.to_csv(f'./cla_test_metrics_{name}.csv', index=False)

    # flatten_methods = ["last", "mean", "max"]
    #
    # for method in flatten_methods:
    #     test_metrics = {
    #         'window_size': [],
    #         'test_accuracy': [],
    #         'test_F1': [],
    #         'test_TPR': [],
    #         'test_FPR': [],
    #     }
    #
    #     for folder in os.listdir(root_dir):
    #         # Extract the last number from the folder name
    #         last_number = int(re.findall(r'\d+', folder)[-1])
    #
    #         # Skip this iteration if the last number is greater or equal to 7
    #         if last_number > 8:
    #             continue
    #
    #         folder_path = os.path.join(root_dir, folder)
    #
    #         # Create a dictionary to store the metrics during testing
    #         window_size, accuracy, F1, TPR, FPR = ClassificationModel(
    #             dataset_path=folder_path,
    #             batch_size_train=8,
    #             batch_size_valid=16,
    #             batch_size_test=16,
    #             input_size=9,
    #             output_size=1,
    #             flatten_method=method,  # Changed to the current flatten method
    #             num_channels=(64,) * 3 + (128,) * 2,
    #             kernel_size=2,
    #             dropout=0.5,
    #             load_method = 'mean',
    #             learning_rate=0.01,
    #             num_epochs=30,
    #             model_save_path=f"./cla_model_{method}",  # Changed to reflect the flatten method
    #             augmenter=None,
    #             aug_name=method
    #         ).run()
    #
    #         test_metrics['window_size'].append(window_size)
    #         test_metrics['test_accuracy'].append(accuracy)
    #         test_metrics['test_F1'].append(F1)
    #         test_metrics['test_TPR'].append(TPR)
    #         test_metrics['test_FPR'].append(FPR)

    # df_metrics = pd.DataFrame(test_metrics)
    # df_metrics.to_csv(f'./cla_test_metrics_{method}.csv', index=False)  # Changed to reflect the flatten method

    root_dir = 'C:/Repository/master/Processed_Dataset/KFall'

    load_methods = ['raw', 'acc', 'trajectory_acc', 'euler_acc', 'trajectory_acc_norm', 'trajectory_raw', 'acc_euler_gyr_norm']

    augmenter = None
    test_metrics = {
        'load_method': [],
        'window_size': [],
        'test_accuracy': [],
        'test_F1': [],
        'test_TPR': [],
        'test_FPR': [],
        'FP': [],
        'FN': [],
        'TP': [],
        'TN': []
    }

    for load in load_methods:

        if load == 'raw':
            input_size = 9
        elif load == 'acc':
            input_size = 3
        elif load == 'trajectory_acc':
            input_size = 6
        elif load == 'euler_acc':
            input_size = 6
        elif load == 'trajectory_acc_norm':
            input_size = 4
        elif load == 'trajectory_raw':
            input_size = 12
        elif load == 'acc_euler_gyr_norm':
            input_size = 7

        folder_path = os.path.join(root_dir, 'KFall_window_sec4')
        # Create a dictionary to store the metrics during testing
        window_size, accuracy, F1, TPR, FPR, FP, FN, TP, TN = ClassificationModel(
            dataset_path=folder_path,
            batch_size_train=8,
            batch_size_valid=16,
            batch_size_test=16,
            input_size=input_size,
            output_size=1,
            flatten_method="mean",
            num_channels=(64,) * 3 + (128,) * 2,
            kernel_size=2,
            dropout=0.5,
            load_method=load,
            learning_rate=0.01,
            num_epochs=30,
            model_save_path=f"./seg_model_{load}",
            augmenter=augmenter,
            aug_name=load
        ).run()
        test_metrics['load_method'].append(load)
        test_metrics['window_size'].append(window_size)
        test_metrics['test_accuracy'].append(accuracy)
        test_metrics['test_F1'].append(F1)
        test_metrics['test_TPR'].append(TPR)
        test_metrics['test_FPR'].append(FPR)
        test_metrics['FP'].append(FP)
        test_metrics['FN'].append(FN)
        test_metrics['TP'].append(FP)
        test_metrics['TN'].append(FP)
        print(test_metrics)
    df_metrics = pd.DataFrame(test_metrics)
    df_metrics.to_csv(f'./cla_test_metrics_load_methods.csv', index=False)
