import pickle
import re

import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from kfall_classification_no_testset import compute_mean_std
from utils.cnn_gru import CnnGru
from utils.multihead_tcn import MultiHeadTCN
from utils.tcn import TCN
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
from utils.data_load_and_EarlyStop import EarlyStopping, BasicArrayDataset, ResampleArrayDataset
from utils.feature_engineering import whole_trajectory, cal_acc_norm, cal_gyr_norm
from utils import augmentation as aug
from utils.transformer import Transformer
from utils.transformer_2021 import IMUTransformerEncoder


def load_dataset_new(root_dir, device):
    with open(root_dir, 'rb') as F:
        dataset = pickle.load(F)
    train = dataset['train'][device]
    valid = dataset['valid'][device]
    test = dataset['test'][device]
    return train, valid, test


def load_dataset_cmdfall(root_dir, device):
    with open(root_dir, 'rb') as F:
        dataset = pickle.load(F)
    new_data_dict = {}
    for set_name in ['train', 'valid', 'test']:
        new_data_dict[set_name] = {}
        for sensor_name in dataset[set_name]:
            # Initialize with existing class 0 data, or an empty array
            combined_class_1 = np.empty((0, 200, 3))
            combined_class_0 = np.empty((0, 200, 3))
            # Combine data for class IDs 1, 2, 3, 4 into class ID 0
            for class_id in dataset[set_name][sensor_name].keys():
                if class_id in range(8):
                    combined_class_1 = np.concatenate((combined_class_1, dataset[set_name][sensor_name][class_id]))

                else:
                    combined_class_0 = np.concatenate((combined_class_0, dataset[set_name][sensor_name][class_id]))
            # Assign the combined data back to class ID 0
            new_data_dict[set_name][sensor_name] = {0: combined_class_0, 1: combined_class_1}
    train = new_data_dict['train'][device]
    valid = new_data_dict['valid'][device]
    test = new_data_dict['test'][device]
    return train, valid, test


def load_dataset_upfall(root_dir, device):
    with open(root_dir, 'rb') as F:
        dataset = pickle.load(F)
    new_data_dict = {}
    for set_name in ['train', 'valid', 'test']:
        new_data_dict[set_name] = {}
        for sensor_name in dataset[set_name]:
            # Initialize with existing class 0 data, or an empty array
            combined_class_1 = np.empty((0, 200, 3))
            combined_class_0 = np.empty((0, 200, 3))
            # Combine data for class IDs 1, 2, 3, 4 into class ID 0
            for class_id in dataset[set_name][sensor_name].keys():
                if class_id in range(5):
                    combined_class_1 = np.concatenate((combined_class_1, dataset[set_name][sensor_name][class_id]))

                else:
                    combined_class_0 = np.concatenate((combined_class_0, dataset[set_name][sensor_name][class_id]))
            # Assign the combined data back to class ID 0
            new_data_dict[set_name][sensor_name] = {0: combined_class_0, 1: combined_class_1}
    train = new_data_dict['train'][device]
    valid = new_data_dict['valid'][device]
    test = new_data_dict['test'][device]
    return train, valid, test


def load_dataset(root_dir, preprocess_fn):
    """
    Load and preprocess the dataset.

    Args:
        root_dir (str): Root directory containing the dataset.
        preprocess_fn (function): Preprocessing function for the data.

    Returns:
        dict: Dictionaries containing train, validation, and test data.
    """
    data_dict = {0: [], 1: []}
    windows = []
    labels = []
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            arr = np.load(file_path)[:, :, :]
            processed_arr = preprocess_fn(arr)
            label = 1 if folder.endswith('_fall') else 0
            windows.append(processed_arr)
            labels += [label] * len(arr)
    windows = np.concatenate(windows)
    labels = np.array(labels)
    data_dict[0].append(windows[labels == 0])
    data_dict[1].append(windows[labels == 1])
    data_dict = {key: np.concatenate(value) for key, value in data_dict.items()}
    return data_dict


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
    """
    Calculate true positives, true negatives, false positives, and false negatives from a confusion matrix.

    Args:
        confusionmatrix (numpy.array): Confusion matrix.

    Returns:
        tuple: True negatives, false positives, false negatives, true positives.
    """
    tn, fp, fn, tp = confusionmatrix.ravel()
    return tn, fp, fn, tp


def plot_confusion(y_true, y_pred, labels):
    """
    Plot a confusion matrix.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        labels (array-like): List of label names.

    Returns:
        numpy.array: Confusion matrix.
    """
    sns.set()
    f, ax = plt.subplots()
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(matrix, annot=True, ax=ax)
    ax.set_xlabel('predict')
    ax.set_ylabel('true')
    plt.show()
    return matrix


class ClassificationModel:
    def __init__(self, dataset_path, batch_size_train, batch_size_valid, batch_size_test, position, model_type,
                 input_size, output_size,
                 num_channels, flatten_method, kernel_size, kernel_size1, kernel_size2, dropout, learning_rate,
                 num_epochs, model_save_path, augmenter, aug_name):
        """
                Initialize the ClassificationModel class for KFall dataset.

                Args:
                    dataset_path (str): Path to the dataset.
                    batch_size_train (int): Batch size for training.
                    batch_size_valid (int): Batch size for validation.
                    batch_size_test (int): Batch size for testing.
                    input_size (int): Input size for the model.
                    output_size (int): Output size for the model.
                    num_channels (int): Number of channels in the model.
                    flatten_method (str): Flattening method for the model.
                    kernel_size (int): Kernel size for the model.
                    dropout (float): Dropout rate for the model.
                    learning_rate (float): Learning rate for optimization.
                    num_epochs (int): Number of epochs for training.
                    model_save_path (str): Path to save the trained model.
                    augmenter (object): Data augmenter.
                    aug_name (str): Name of the augmentation.
                    load_method (str): Data loading method.
                """
        # self.load_method = load_method
        self.position = position
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2
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
        self.model_type = model_type
        self.model = self.initialize_model()

    def initialize_model(self):
        if self.model_type == 'TCN':
            return TCN(self.input_size, self.output_size, self.num_channels, self.kernel_size, self.dropout,
                       self.flatten_method)
        elif self.model_type == 'MultiHeadTCN':
            return MultiHeadTCN(self.input_size, self.output_size, self.num_channels, self.kernel_size1,
                                self.kernel_size2, self.dropout, self.flatten_method)
        elif self.model_type == 'CnnGru':
            self.num_channels = None
            return CnnGru(self.input_size, self.output_size, self.num_channels, self.dropout, self.flatten_method)
        elif self.model_type == 'Transformer':
            self.learning_rate = 0.0001
            return Transformer(self.input_size, 6, 128, 256, 8, self.output_size, self.dropout, self.dropout,
                               self.flatten_method)
        elif self.model_type == 'Transformer_2021':
            self.learning_rate = 0.0001
            return IMUTransformerEncoder(self.input_size, self.output_size, 200, "true", 64, 0.1)

    def run(self):
        # # Load data
        # if self.load_method == 'raw':
        #     train, valid, test = load_dataset_with_raw(self.dataset_path)
        # elif self.load_method == 'acc':
        #     train, valid, test = load_dataset_with_acc(self.dataset_path)
        # elif self.load_method == 'trajectory_acc':
        #     train, valid, test = load_dataset_with_trajectory_acc(self.dataset_path)
        # elif self.load_method == 'euler_acc':
        #     train, valid, test = load_dataset_with_euler_acc(self.dataset_path)
        # elif self.load_method == 'trajectory_acc_norm':
        #     train, valid, test = load_dataset_with_trajectory_acc_norm(self.dataset_path)
        # elif self.load_method == 'trajectory_raw':
        #     train, valid, test = load_dataset_with_trajectory_raw(self.dataset_path)
        # elif self.load_method == 'acc_euler_gyr_norm':
        #     train, valid, test = load_dataset_with_acc_euler_gyr_norm(self.dataset_path)

        train, valid, test = load_dataset_upfall(self.dataset_path, self.position)
        # train = load_dataset_with_raw(self.dataset_path + "/train")
        # valid = load_dataset_with_raw(self.dataset_path + "/valid")
        # test = load_dataset_with_raw(self.dataset_path + "/test")
        generator = torch.Generator()
        generator.manual_seed(123)

        # Create data loaders
        train_set = ResampleArrayDataset(train, augmenter=self.augmenter)
        train_loader = DataLoader(train_set, batch_size=self.batch_size_train, shuffle=True)
        valid_set = BasicArrayDataset(valid)
        valid_loader = DataLoader(valid_set, batch_size=self.batch_size_valid, shuffle=False)
        test_set = BasicArrayDataset(test)
        test_loader = DataLoader(test_set, batch_size=self.batch_size_test, shuffle=False)

        # 计算训练数据的均值和标准差
        mean, std = compute_mean_std(train_loader)

        # 应用标准化（如果您的模型需要这一步）
        # 注意：这假设您的模型有一个名为 'input_norm' 的 BatchNorm 层
        # 用于在模型的前向传播中对输入数据进行标准化
        if hasattr(self.model, 'input_norm'):
            self.model.input_norm.running_mean = mean
            self.model.input_norm.running_var = std * std
        window_size = 4

        if not os.path.exists(self.model_save_path):
            # If the directory doesn't exist, create it
            os.makedirs(self.model_save_path)
        early_stopping = EarlyStopping(self.model_save_path, window_size)

        # Move the model to the GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # Define the optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Create a dictionary to store the metrics during training
        metrics = {
            'epoch': [],
            'train_loss': [],
            'valid_loss': [],
            'valid_accuracy': [],
            'valid_f1': [],
            'valid_TPR': [],
            'valid_FPR': [],
            'valid_FP': [],
            'valid_FN': [],
            'valid_TP': [],
            'valid_TN': []
        }

        number = 0
        # Training loop
        for epoch in range(self.num_epochs):
            # Training loop
            self.model.train()  # Set the model in training mode
            train_loss = 0
            pbar = tqdm(total=len(train_loader), ncols=0)
            for x, y in train_loader:
                # Move the inputs and labels to the GPU if available
                x = x.to(device)
                y = y.to(device)

                # Clear the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(x)
                outputs = outputs.squeeze(1)  # shape[batch, channel]
                # Calculate the loss
                t_loss = F.binary_cross_entropy_with_logits(outputs, y.float())  # use the new function here

                train_loss += t_loss.item()
                # Backward pass
                t_loss.backward()

                # Update the parameters
                optimizer.step()

                pbar.update(1)

            # Track the training progress
            pbar.close()

            self.model.eval()
            valid_loss = 0
            valid_outputs, valid_labels = [], []

            with torch.no_grad():
                for x, y in valid_loader:
                    x = x.to(device)
                    y = y.to(device)

                    pred = self.model(x)
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

            # Compute confusion matrix and extract metrics
            valid_confusion = confusion_matrix(valid_labels, valid_outputs)
            valid_TN, valid_FP, valid_FN, valid_TP = cal_tp_tn_fp_fn(valid_confusion)
            valid_TPR = valid_TP / (valid_TP + valid_FN)  # Sensitivity
            valid_FPR = valid_FP / (valid_FP + valid_TN)  # false positive rate

            print('Epoch [{}/{}], train_Loss: {:.4f}, valid_Loss: {:.4f}, accuracy: {:.4f}, f1 score: {:.4f}'.format(
                epoch + 1, self.num_epochs, train_loss / len(train_loader), valid_loss, valid_accuracy, valid_f1))

            # Save metrics for this epoch
            metrics['epoch'].append(epoch + 1)
            metrics['train_loss'].append(train_loss / len(train_loader))
            metrics['valid_loss'].append(valid_loss)
            metrics['valid_accuracy'].append(valid_accuracy)
            metrics['valid_f1'].append(valid_f1)
            metrics['valid_TPR'].append(valid_TPR)  # Added metric
            metrics['valid_FPR'].append(valid_FPR)  # Added metric
            metrics['valid_FP'].append(valid_FP)  # Added metric
            metrics['valid_FN'].append(valid_FN)  # Added metric
            metrics['valid_TP'].append(valid_TP)  # Added metric
            metrics['valid_TN'].append(valid_TN)  # Added metric
            # Early stopping
            number2 = early_stopping(epoch, valid_loss, self.model)
            if number < number2:
                number = number2
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # Convert metrics dictionary to DataFrame and save as CSV
        df_metrics = pd.DataFrame(metrics)

        csv_file_name = f'./upfall_train_records/{self.model_type}_training_metrics_{self.aug_name}.csv'

        if not os.path.exists('./upfall_train_records'):
            # If the directory doesn't exist, create it
            os.makedirs('./upfall_train_records')
        df_metrics.to_csv(csv_file_name)

        model = self.initialize_model()
        state_dict = torch.load(f'{self.model_save_path}/model_size4_epoch{number + 1}.pth')
        model.load_state_dict(state_dict)
        model.to(device)
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

        test_accuracy = accuracy_score(actuals, predictions).mean()
        test_confusion = confusion_matrix(actuals, predictions)

        test_TN, test_FP, test_FN, test_TP = cal_tp_tn_fp_fn(test_confusion)
        test_TPR = test_TP / (test_TP + test_FN)  # Sensitivity
        test_FPR = test_FP / (test_FP + test_TN)  # false positive rate
        test_F1 = float(f1_score(actuals, predictions, average='weighted'))

        test_metrics = {
            'test_Accuracy': [test_accuracy],
            'test_TN': [test_TN],
            'test_FP': [test_FP],
            'test_FN': [test_FN],
            'test_TP': [test_TP],
            'test_TPR': [test_TPR],
            'test_FPR': [test_FPR],
            'test_F1': [test_F1]
        }

        df_test_metrics = pd.DataFrame(test_metrics)
        selected_row = df_metrics.iloc[number:number + 1, :]
        combined_df = pd.concat([selected_row.reset_index(drop=True), df_test_metrics.reset_index(drop=True)], axis=1)
        series = combined_df.squeeze()
        return series


if __name__ == "__main__":
    root_dir = './cmdfall.pkl'
    model_types = ["TCN", "MultiHeadTCN", "CnnGru", "Transformer"]

    '''
        Train model with different augmentations
    '''

    # augmentations = {
    # 'None': None
    # 'timewarp': aug.Timewarp(sigma=0.2, knot=4, p=0.5),
    # 'jitter': aug.Jitter(sigma=0.05, p=0.5),
    # 'scale': aug.Scale(sigma=0.1, p=0.5)
    # 'magnitudeWarp': aug.MagnitudeWarp(sigma=0.2, knot=4, p=0.5),
    # 'rotation': aug.Rotation(angle_range=180, p=0.5),
    # 'permutation': aug.Permutation(n_perm=4, min_seg_length=10, p=0.5),
    # 'randSample': aug.RandSample(n_sample=150, p=0.5),
    # }

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
    #
    #
    #     # Create a dictionary to store the metrics during testing
    #     window_size, accuracy, F1, TPR, FPR, FP, FN, TP, TN = ClassificationModel(
    #         dataset_path=root_dir,
    #         batch_size_train=16,
    #         batch_size_valid=32,
    #         batch_size_test=32,
    #         input_size=9,
    #         output_size=1,
    #         flatten_method="mean",
    #         num_channels=(64,) * 5 + (128,) * 2,
    #         load_method='raw',
    #         kernel_size=2,
    #         dropout=0.5,
    #         learning_rate=0.01,
    #         num_epochs=30,
    #         model_save_path=f"./new_cla_model_{name}",
    #         augmenter=augmenter,
    #         aug_name=name
    #     ).run()
    #
    #     test_metrics['window_size'].append(window_size)
    #     test_metrics['test_accuracy'].append(accuracy)
    #     test_metrics['test_F1'].append(F1)
    #     test_metrics['test_TPR'].append(TPR)
    #     test_metrics['test_FPR'].append(FPR)
    #     test_metrics['FP'].append(FP)
    #     test_metrics['FN'].append(FN)
    #     test_metrics['TP'].append(FP)
    #     test_metrics['TN'].append(FP)
    #     df_metrics = pd.DataFrame(test_metrics)
    #     df_metrics.to_csv(f'./cla_test_metrics_{name}.csv', index=False)

    '''
        Train model with different flatten methods
    '''
    flatten_methods = ["mean", "max"]
    records = pd.DataFrame(columns=['model', 'flatten', 'train_loss', 'valid_loss', 'valid_accuracy', 'valid_f1',
                                    'valid_TPR', 'valid_FPR', 'valid_FP', 'valid_FN', 'valid_TP', 'valid_TN',
                                    'test_Accuracy', 'test_TN', 'test_FP', 'test_FN', 'test_TP', 'test_TPR',
                                    'test_FPR', 'test_F1'])
    row = 0
    for model_type in model_types:
        for method in flatten_methods:
            # Create a dictionary to store the metrics during testing
            record = ClassificationModel(
                dataset_path=root_dir,
                model_type=model_type,
                batch_size_train=16,
                batch_size_valid=32,
                batch_size_test=32,
                position='waist',
                input_size=3,
                output_size=1,
                flatten_method=method,  # Changed to the current flatten method
                num_channels=(64,) * 5 + (128,) * 2,
                kernel_size=2,  # 2
                kernel_size1=2,  # 2
                kernel_size2=3,  # 3
                dropout=0.5,
                learning_rate=0.001,
                num_epochs=10,
                model_save_path=f"./models/upfall/flatten/{model_type}/{method}",
                # Changed to reflect the flatten method
                augmenter=None,
                aug_name=method
            ).run()

            records = pd.concat([records, pd.DataFrame([record])], ignore_index=True)
            records.iloc[row, 0] = model_type
            records.iloc[row, 1] = method

            row += 1

    records.to_csv(f'./experiment_results/flatten/upfall.csv', index=False)
    # '''
    #     Train model with different feature combinations
    # '''
    # root_dir = 'D:/Repository/master/Processed_Dataset/KFall'
    # #
    # # load_methods = ['raw', 'acc', 'trajectory_acc', 'euler_acc', 'trajectory_acc_norm','trajectory_raw', 'acc_euler_gyr_norm']
    # load_methods = ['acc']
    # augmenter = aug.Augmenter([aug.Timewarp(sigma=0.2, knot=4, p=0.5)])
    # test_metrics = {
    #     'load_method': [],
    #     'window_size': [],
    #     'test_accuracy': [],
    #     'test_F1': [],
    #     'test_TPR': [],
    #     'test_FPR': [],
    #     'FP': [],
    #     'FN': [],
    #     'TP': [],
    #     'TN': []
    # }
    #
    # for load in load_methods:
    #
    #     if load == 'raw':
    #         input_size = 9
    #     elif load == 'acc':
    #         input_size = 3
    #     elif load == 'trajectory_acc':
    #         input_size = 6
    #     elif load == 'euler_acc':
    #         input_size = 6
    #     elif load == 'trajectory_acc_norm':
    #         input_size = 4
    #     elif load == 'trajectory_raw':
    #         input_size = 12
    #     elif load == 'acc_euler_gyr_norm':
    #         input_size = 7
    #
    #     folder_path = os.path.join(root_dir, 'KFall_window_sec4')
    #     # Create a dictionary to store the metrics during testing
    #     window_size, accuracy, F1, TPR, FPR, FP, FN, TP, TN = ClassificationModel(
    #         dataset_path=folder_path,
    #         batch_size_train=16,
    #         batch_size_valid=32,
    #         batch_size_test=32,
    #         input_size=input_size,
    #         output_size=1,
    #         flatten_method="mean",
    #         num_channels=(64,) * 5 + (128,) * 2,
    #         kernel_size=2,
    #         dropout=0.5,
    #         load_method=load,
    #         learning_rate=0.01,
    #         num_epochs=20,
    #         model_save_path=f"./new_cla_model_{load}",
    #         augmenter=None,
    #         aug_name=load
    #     ).run()
    #     test_metrics['load_method'].append(load)
    #     test_metrics['window_size'].append(window_size)
    #     test_metrics['test_accuracy'].append(accuracy)
    #     test_metrics['test_F1'].append(F1)
    #     test_metrics['test_TPR'].append(TPR)
    #     test_metrics['test_FPR'].append(FPR)
    #     test_metrics['FP'].append(FP)
    #     test_metrics['FN'].append(FN)
    #     test_metrics['TP'].append(FP)
    #     test_metrics['TN'].append(FP)
    #     print(test_metrics)
    # df_metrics = pd.DataFrame(test_metrics)
    # df_metrics.to_csv(f'./cla_test_metrics_load_methods.csv', index=False)
