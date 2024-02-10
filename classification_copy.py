import pickle
import re

import pandas as pd
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.multibracnch_resnet1d import MSResNet1D, ResNet1D2
from utils.multichannel_cnn_gru import CnnGru
from utils.multihead_tcn import MultiHeadTCN
from utils.net1d import Net1D
from utils.resnet1d import ResNet1D
from utils.singlechannel_cnn_gru import SingleCnnGru
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


def load_dataset_norm(root_dir, device, output_size):
    with open(root_dir, 'rb') as F:
        dataset = pickle.load(F)
    for set_name in ['train', 'valid', 'test']:
        for sensor_name in dataset[set_name]:
            for num in range(output_size):
                acc_x = dataset[set_name][sensor_name][num][:, :, 0]
                acc_y = dataset[set_name][sensor_name][num][:, :, 1]
                acc_z = dataset[set_name][sensor_name][num][:, :, 2]

                norm = np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)
                dataset[set_name][sensor_name][num] = norm

    train = dataset['train'][device]
    valid = dataset['valid'][device]
    test = dataset['test'][device]
    return train, valid, test


def load_dataset_raw_norm(root_dir, device, output_size):
    with open(root_dir, 'rb') as F:
        dataset = pickle.load(F)
    for set_name in ['train', 'valid', 'test']:
        for sensor_name in dataset[set_name]:
            for num in range(output_size):
                data = dataset[set_name][sensor_name][num]

                acc_x = data[:, :, 0]
                acc_y = data[:, :, 1]
                acc_z = data[:, :, 2]

                norm = np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)
                norm = norm.reshape(norm.shape[0], norm.shape[1], 1)
                expanded_data = np.concatenate((data, norm), axis=2)
                dataset[set_name][sensor_name][num] = expanded_data

    train = dataset['train'][device]
    valid = dataset['valid'][device]
    test = dataset['test'][device]
    return train, valid, test


def load_dataset_cmdfall_raw(root_dir, device):
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


def load_dataset_cmdfall_norm(root_dir, device):
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

            for num in [0, 1]:
                acc_x = new_data_dict[set_name][sensor_name][num][:, :, 0]
                acc_y = new_data_dict[set_name][sensor_name][num][:, :, 1]
                acc_z = new_data_dict[set_name][sensor_name][num][:, :, 2]

                norm = np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)

                new_data_dict[set_name][sensor_name][num] = norm

    train = new_data_dict['train'][device]
    valid = new_data_dict['valid'][device]
    test = new_data_dict['test'][device]
    return train, valid, test


def load_dataset_cmdfall_raw_norm(root_dir, device):
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

            for num in [0, 1]:
                data = new_data_dict[set_name][sensor_name][num]

                acc_x = data[:, :, 0]
                acc_y = data[:, :, 1]
                acc_z = data[:, :, 2]

                norm = np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)
                norm = norm.reshape(norm.shape[0], norm.shape[1], 1)
                expanded_data = np.concatenate((data, norm), axis=2)
                new_data_dict[set_name][sensor_name][num] = expanded_data

    train = new_data_dict['train'][device]
    valid = new_data_dict['valid'][device]
    test = new_data_dict['test'][device]
    return train, valid, test


def load_dataset_upfall_small(root_dir, device, ratio):
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
            if (set_name == 'train') or (set_name == 'valid'):
                for id in [0, 1]:
                    data = new_data_dict[set_name][sensor_name][id]
                    new_window_count = int(data.shape[0] * ratio)
                    random_indices = np.random.choice(np.arange(data.shape[0]), size=new_window_count, replace=False)
                    data_subset = data[random_indices]
                    new_data_dict[set_name][sensor_name][id] = data_subset


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


def load_dataset_upfall_multi_class(root_dir, device):
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


def compute_mean_std(train_loader):
    mean = 0.0
    var = 0.0
    total_samples = 0
    for data, _ in train_loader:
        batch_samples = data.size(0)
        mean += data.mean([0, 1])
        var += data.var([0, 1], unbiased=False)
        total_samples += batch_samples

    mean /= total_samples
    var /= total_samples
    std = torch.sqrt(var)

    return mean, std


class ClassificationModel:
    def __init__(self, dataset_path, batch_size_train, batch_size_valid, batch_size_test, position, model_type,
                 input_size, output_size, dataset_name, feature_set,
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
        self.dataset_name = dataset_name
        self.feature_set = feature_set

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
        elif self.model_type == 'SingleCnnGru':
            self.num_channels = None
            return SingleCnnGru(self.input_size, self.output_size, self.num_channels, self.dropout, 3,
                                self.flatten_method)
        elif self.model_type == 'Transformer':
            self.learning_rate = 0.0001
            return Transformer(self.input_size, 6, 128, 256, 8, self.output_size, self.dropout, self.dropout,
                               self.flatten_method)
        elif self.model_type == 'Transformer_2021':
            self.learning_rate = 0.0001
            return IMUTransformerEncoder(self.input_size, self.output_size, 200, "true", 64, 0.1)
        elif self.model_type == 'Resnet1D':
            return ResNet1D(in_channels=self.input_size, base_filters=128, kernel_size=9, n_block=6, stride=4,
                            flatten_method=self.flatten_method, n_classes=self.output_size)
        elif self.model_type == 'Net1D':
            return Net1D(in_channels=self.input_size, base_filters=64, ratio=1.0, kernel_size=9, stride=2,
                         flatten_method=self.flatten_method, n_classes=self.output_size)
        elif self.model_type == 'MultibranchResnet1D':
            return MSResNet1D(
                resnets=nn.ModuleDict({
                    'resnet3': ResNet1D2(in_channels=64, base_filters=64, kernel_size=3, n_block=6, stride=2),
                    'resnet5': ResNet1D2(in_channels=64, base_filters=64, kernel_size=5, n_block=6, stride=2),
                    'resnet7': ResNet1D2(in_channels=64, base_filters=64, kernel_size=7, n_block=6, stride=2),
                }),
                in_channels=self.input_size,
                in_resnet_channels=64,
                first_kernel=7,
                n_outputs=self.output_size,
                flatten_method=self.flatten_method,
            )

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
        if self.feature_set == "acc":
            train, valid, test = load_dataset_new(self.dataset_path, self.position)
        elif self.feature_set == "norm":
            train, valid, test = load_dataset_norm(self.dataset_path, self.position, self.output_size)
        elif self.feature_set == "acc_norm":
            train, valid, test = load_dataset_raw_norm(self.dataset_path, self.position, self.output_size)
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

        mean, std = compute_mean_std(train_loader)

        if hasattr(self.model, 'input_norm'):
            self.model.input_norm.running_mean = mean
            self.model.input_norm.running_var = std * std
        window_size = 4

        if not os.path.exists(self.model_save_path):
            # If the directory doesn't exist, create it
            os.makedirs(self.model_save_path)
        early_stopping = EarlyStopping(self.model_save_path, window_size, patience=15)

        # Move the model to the GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # Define the optimizer
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        # Create a dictionary to store the metrics during training
        metrics = {
            'epoch': [],
            'train_loss': [],
            'valid_loss': [],
            'valid_accuracy': [],
            'valid_f1': []
        }

        number = 0
        # Training loop
        for epoch in range(self.num_epochs):
            # Training loop
            self.model.train()  # Set the model in training mode
            train_loss = 0
            pbar = tqdm(total=len(train_loader), ncols=0)

            for x, y in train_loader:
                if self.feature_set == 'norm':
                    x = x.unsqueeze(-1)
                # Move the inputs and labels to the GPU if available
                x = x.to(device)
                y = y.to(device)

                # Clear the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(x)

                # Calculate the loss
                t_loss = criterion(outputs, y)  # use the new function here

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
                    if self.feature_set == 'norm':
                        x = x.unsqueeze(-1)
                    x = x.to(device)
                    y = y.to(device).long()

                    pred = self.model(x)

                    v_loss = criterion(pred, y)
                    valid_loss += v_loss.item()

                    _, predicted_labels = torch.max(pred, 1)
                    valid_outputs.extend(predicted_labels.cpu().numpy())
                    valid_labels.extend(y.cpu().numpy())

            valid_loss /= len(valid_loader)
            valid_accuracy = accuracy_score(valid_labels, valid_outputs)
            valid_f1 = f1_score(valid_labels, valid_outputs, average='macro')


            print('Epoch [{}/{}], train_Loss: {:.4f}, valid_Loss: {:.4f}, accuracy: {:.4f}, f1 score: {:.4f}'.format(
                epoch + 1, self.num_epochs, train_loss / len(train_loader), valid_loss, valid_accuracy, valid_f1))

            # Save metrics for this epoch
            metrics['epoch'].append(epoch + 1)
            metrics['train_loss'].append(train_loss / len(train_loader))
            metrics['valid_loss'].append(valid_loss)
            metrics['valid_accuracy'].append(valid_accuracy)
            metrics['valid_f1'].append(valid_f1)

            # Early stopping
            number2 = early_stopping(epoch, valid_f1, self.model)
            if number < number2:
                number = number2
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # Convert metrics dictionary to DataFrame and save as CSV
        df_metrics = pd.DataFrame(metrics)

        csv_file_name = f'./{self.dataset_name}_train_records/{self.model_type}_multiclass_training_metrics_{self.aug_name}.csv'

        if not os.path.exists(f'./{self.dataset_name}_train_records'):
            # If the directory doesn't exist, create it
            os.makedirs(f'./{self.dataset_name}_train_records')
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
                if self.feature_set == 'norm':
                    x = x.unsqueeze(-1)
                x = x.to(device)
                y = y.to(device)

                pred = model(x)

                # Optionally compute test loss
                t_loss = criterion(pred, y)
                test_loss += t_loss.item()

                _, predicted_labels = torch.max(pred, 1)
                predictions.extend(predicted_labels.cpu().numpy())
                actuals.extend(y.cpu().numpy())

        test_accuracy = accuracy_score(actuals, predictions).mean()
        test_F1 = f1_score(actuals, predictions, average='macro')

        test_confusion = confusion_matrix(actuals, predictions, labels=range(11))
        cm = test_confusion.astype(np.float32)
        FP = cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)
        test_FPR = FP.sum() / (FP.sum() + TN.sum())
        # weights = [np.sum(valid_labels == i) for i in range(len(fpr_list))]
        # if np.sum(weights) == 0:
        #     test_FPR = 0  # Avoid division by zero if weights sum to zero
        # else:
        #     test_FPR = np.average(fpr_list, weights=weights)

        test_metrics = {
            'test_Accuracy': [test_accuracy],
            'test_FPR': [test_FPR],
            'test_F1': [test_F1]
        }

        df_test_metrics = pd.DataFrame(test_metrics)
        selected_row = df_metrics.iloc[number:number + 1, :]
        combined_df = pd.concat([selected_row.reset_index(drop=True), df_test_metrics.reset_index(drop=True)], axis=1)
        series = combined_df.squeeze()
        return series


if __name__ == "__main__":
    model_types = ["Net1D", "Resnet1D", "TCN", "MultiHeadTCN", "CnnGru", "Transformer", "SingleCnnGru"]
    single_branch_model_types = ["Net1D", "Resnet1D", "TCN", "Transformer", "SingleCnnGru"]
    multi_branch_model_types = ["MultiHeadTCN", "CnnGru"]
    datasets = ["cmdfall", "upfall"]

    '''
        Train single-branch models with different flatten methods
    '''
    #
    # for dataset in datasets:
    #     if dataset == 'upfall':
    #         output_size = 11
    #     else:
    #         output_size = 20
    #     root_dir = f'./datasets/{dataset}.pkl'
    #     records = pd.DataFrame(
    #         columns=['model', 'flatten', 'times', 'train_loss', 'valid_loss', 'valid_accuracy', 'valid_f1',
    #                  'test_Accuracy', 'test_FPR', 'test_F1'])
    #     row = 0
    #     all_runs_records = pd.DataFrame()
    #     for model_type in single_branch_model_types:
    #         if model_type not in ["TCN", "SingleCnnGru", "CnnGru"]:
    #             flatten_methods = ["mean", "max"]
    #         else:
    #             flatten_methods = ["mean", "max", "last"]
    #
    #         for method in flatten_methods:
    #             metrics_list = []
    #             time = 1
    #             for run in range(3):
    #                 # Create a dictionary to store the metrics during testing
    #                 record = ClassificationModel(
    #                     dataset_name=dataset,
    #                     dataset_path=root_dir,
    #                     model_type=model_type,
    #                     feature_set='acc',
    #                     batch_size_train=16,
    #                     batch_size_valid=32,
    #                     batch_size_test=32,
    #                     position='waist',
    #                     input_size=3,
    #                     output_size=output_size,
    #                     flatten_method=method,  # Changed to the current flatten method
    #                     num_channels=(64,) * 5 + (128,) * 2,
    #                     kernel_size=2,  # 2
    #                     kernel_size1=2,  # 2
    #                     kernel_size2=3,  # 3
    #                     dropout=0.5,
    #                     learning_rate=0.001,
    #                     num_epochs=200,
    #                     model_save_path=f"./models/{dataset}_multi_class/flatten_multi_time/{model_type}_time{time}/{method}",
    #                     # Changed to reflect the flatten method
    #                     augmenter=None,
    #                     aug_name=method
    #                 ).run()
    #
    #                 metrics_list.append(record)
    #
    #                 records = pd.concat([records, pd.DataFrame([record])], ignore_index=True)
    #                 records.iloc[row, 0] = model_type
    #                 records.iloc[row, 1] = method
    #                 records.iloc[row, 2] = time
    #                 time += 1
    #                 row += 1
    #             average_metrics = pd.DataFrame(metrics_list).mean().to_dict()
    #             average_metrics['model'] = model_type
    #             average_metrics['flatten'] = method
    #             all_runs_records = pd.concat([all_runs_records, pd.DataFrame([average_metrics])], ignore_index=True)
    #
    #     records.to_csv(f'./experiment_results/flatten/{dataset}_multi_class_each_time.csv', index=False)
    #     all_runs_records.to_csv(f'./experiment_results/flatten/{dataset}_multi_class_average.csv', index=False)


    '''
        Train multi-branch models with 'mean' flatten method
    '''

    # for dataset in datasets:
    #     root_dir = f'./datasets/{dataset}.pkl'
    #     records = pd.DataFrame(columns=['model', 'flatten', 'train_loss', 'valid_loss', 'valid_accuracy', 'valid_f1',
    #                                     'valid_TPR', 'valid_FPR', 'valid_FP', 'valid_FN', 'valid_TP', 'valid_TN',
    #                                     'test_Accuracy', 'test_TN', 'test_FP', 'test_FN', 'test_TP', 'test_TPR',
    #                                     'test_FPR', 'test_F1'])
    #     row = 0
    #     for model_type in multi_branch_model_types:
    #         if dataset == 'upfall':
    #             output_size = 11
    #         else:
    #             output_size = 20
    #         # Create a dictionary to store the metrics during testing
    #         record = ClassificationModel(
    #             dataset_name=dataset,
    #             dataset_path=root_dir,
    #             model_type=model_type,
    #             feature_set='acc',
    #             batch_size_train=16,
    #             batch_size_valid=32,
    #             batch_size_test=32,
    #             position='waist',
    #             input_size=3,
    #             output_size=output_size,
    #             flatten_method="mean",  # Changed to the current flatten method
    #             num_channels=(64,) * 5 + (128,) * 2,
    #             kernel_size=2,  # 2
    #             kernel_size1=2,  # 2
    #             kernel_size2=3,  # 3
    #             dropout=0.5,
    #             learning_rate=0.001,
    #             num_epochs=200,
    #             model_save_path=f"./models/{dataset}_multi_class/multibranch/{model_type}/mean",
    #             # Changed to reflect the flatten method
    #             augmenter=None,
    #             aug_name="mean"
    #         ).run()
    #
    #         records = pd.concat([records, pd.DataFrame([record])], ignore_index=True)
    #         records.iloc[row, 0] = model_type
    #         records.iloc[row, 1] = "mean"
    #
    #         row += 1
    #
    #     records.to_csv(f'./experiment_results/multibranch/{dataset}_multi_class.csv', index=False)

    '''
        Train model with different feature sets
    '''

    # for dataset in datasets:
    #     if dataset == 'upfall':
    #         output_size = 11
    #     else:
    #         output_size = 20
    #     root_dir = f'./datasets/{dataset}.pkl'
    #     records = pd.DataFrame(columns=['model', 'flatten', 'feature_set', 'times', 'train_loss', 'valid_loss',
    #                                     'valid_accuracy', 'valid_f1', 'test_Accuracy', 'test_FPR', 'test_F1'])
    #     row = 0
    #     all_runs_records = pd.DataFrame()
    #     model_type = "Resnet1D"
    #     method = 'mean'
    #     for feature_set in ['acc', 'norm', 'acc_norm']:
    #         if feature_set == 'norm':
    #             input_size = 1
    #         elif feature_set == 'acc':
    #             input_size = 3
    #         elif feature_set == 'acc_norm':
    #             input_size = 4
    #         metrics_list = []
    #         time = 1
    #         for run in range(3):
    #             # Create a dictionary to store the metrics during testing
    #             record = ClassificationModel(
    #                 dataset_name=dataset,
    #                 dataset_path=root_dir,
    #                 model_type=model_type,
    #                 feature_set=feature_set,
    #                 batch_size_train=16,
    #                 batch_size_valid=32,
    #                 batch_size_test=32,
    #                 position='waist',
    #                 input_size=input_size,
    #                 output_size=output_size,
    #                 flatten_method=method,  # Changed to the current flatten method
    #                 num_channels=(64,) * 5 + (128,) * 2,
    #                 kernel_size=2,  # 2
    #                 kernel_size1=2,  # 2
    #                 kernel_size2=3,  # 3
    #                 dropout=0.5,
    #                 learning_rate=0.001,
    #                 num_epochs=200,
    #                 model_save_path=f"./models/{dataset}_multi_class/feature_multi_time/{model_type}/{feature_set}_time{time}",
    #                 # Changed to reflect the flatten method
    #                 augmenter=None,
    #                 aug_name=feature_set
    #             ).run()
    #             metrics_list.append(record)
    #
    #             records = pd.concat([records, pd.DataFrame([record])], ignore_index=True)
    #             records.iloc[row, 0] = model_type
    #             records.iloc[row, 1] = method
    #             records.iloc[row, 2] = feature_set
    #             records.iloc[row, 3] = time
    #             time += 1
    #             row += 1
    #         average_metrics = pd.DataFrame(metrics_list).mean().to_dict()
    #         average_metrics['model'] = model_type
    #         average_metrics['flatten'] = method
    #         average_metrics['feature_set'] = feature_set
    #         all_runs_records = pd.concat([all_runs_records, pd.DataFrame([average_metrics])], ignore_index=True)
    #
    #     records.to_csv(f'./experiment_results/feature/{dataset}_multi_class_each_time.csv', index=False)
    #     all_runs_records.to_csv(f'./experiment_results/feature/{dataset}_multi_class_average.csv', index=False)

    '''
        Train model with different augmentation sets
    '''
    augmentations = {
        'timewarp': aug.Timewarp(sigma=0.2, knot=4, p=0.5),
        'jitter': aug.Jitter(sigma=0.05, p=0.5),
        'scale': aug.Scale(sigma=0.1, p=0.5),
        'magnitudeWarp': aug.MagnitudeWarp(sigma=0.2, knot=4, p=0.5),
        'rotation': aug.Rotation(angle_range=90, p=0.5),
        'permutation': aug.Permutation(n_perm=4, min_seg_length=10, p=0.5),
    }
    for dataset in datasets:
        root_dir = f'./datasets/{dataset}.pkl'
        records = pd.DataFrame(
            columns=['model', 'augmentation', 'times', 'train_loss', 'valid_loss', 'valid_accuracy', 'valid_f1',
                     'test_Accuracy', 'test_FPR', 'test_F1'])
        row = 0
        all_runs_records = pd.DataFrame()
        for model_type in ["Resnet1D", "MultibranchResnet1D"]:
            method = 'mean'
            if dataset == 'upfall':
                output_size = 11
            else:
                output_size = 20
            for name, augmentation in augmentations.items():
                augmenter = aug.Augmenter([augmentation])
                metrics_list = []
                time = 1
                for run in range(3):
                    # Create a dictionary to store the metrics during testing
                    record = ClassificationModel(
                        dataset_name=dataset,
                        dataset_path=root_dir,
                        model_type=model_type,
                        feature_set='acc',
                        batch_size_train=16,
                        batch_size_valid=32,
                        batch_size_test=32,
                        position='waist',
                        input_size=3,
                        output_size=output_size,
                        flatten_method=method,  # Changed to the current flatten method
                        num_channels=(64,) * 5 + (128,) * 2,
                        kernel_size=2,  # 2
                        kernel_size1=2,  # 2
                        kernel_size2=3,  # 3
                        dropout=0.5,
                        learning_rate=0.001,
                        num_epochs=200,
                        model_save_path=f"./models/{dataset}_multi_class/augmentation/{model_type}/{name}_time{time}",
                        # Changed to reflect the flatten method
                        augmenter=augmenter,
                        aug_name=name
                    ).run()
                    metrics_list.append(record)
                    records = pd.concat([records, pd.DataFrame([record])], ignore_index=True)
                    records.iloc[row, 0] = model_type
                    records.iloc[row, 1] = name
                    records.iloc[row, 2] = time
                    time += 1
                    row += 1
            average_metrics = pd.DataFrame(metrics_list).mean().to_dict()
            average_metrics['model'] = model_type
            average_metrics['augmentation'] = name
            all_runs_records = pd.concat([all_runs_records, pd.DataFrame([average_metrics])], ignore_index=True)

        records.to_csv(f'./experiment_results/augmentation/{dataset}_multi_class_each_time.csv', index=False)
        all_runs_records.to_csv(f'./experiment_results/augmentation/{dataset}_average.csv', index=False)


    '''
        Train model with different angles
    '''
    #
    # for dataset in datasets:
    #     root_dir = f'./datasets/{dataset}.pkl'
    #     records = pd.DataFrame(columns=['model', 'flatten', 'augmentation', 'train_loss', 'valid_loss', 'valid_accuracy', 'valid_f1',
    #                                     'valid_TPR', 'valid_FPR', 'valid_FP', 'valid_FN', 'valid_TP', 'valid_TN',
    #                                     'test_Accuracy', 'test_TN', 'test_FP', 'test_FN', 'test_TP', 'test_TPR',
    #                                     'test_FPR', 'test_F1'])
    #     row = 0
    #     model_type = "Resnet1D"
    #     method = 'mean'
    #     for angle in range(10, 61, 10):
    #         augmentations = {
    #             # 'None': None,
    #             # 'timewarp': aug.Timewarp(sigma=0.2, knot=4, p=0.5),
    #             # 'jitter': aug.Jitter(sigma=0.05, p=0.5),
    #             # 'scale': aug.Scale(sigma=0.1, p=0.5),
    #             # 'magnitudeWarp': aug.MagnitudeWarp(sigma=0.2, knot=4, p=0.5),
    #             'rotation': aug.Rotation(angle_range=angle, p=0.5)
    #             # 'permutation': aug.Permutation(n_perm=4, min_seg_length=10, p=0.5),
    #         }
    #         if dataset == 'upfall':
    #             output_size = 11
    #         else:
    #             output_size = 20
    #         for name, augmentation in augmentations.items():
    #             if name =='None':
    #                 augmenter = None
    #             else:
    #                 augmenter = aug.Augmenter([augmentation])
    #
    #             # Create a dictionary to store the metrics during testing
    #             record = ClassificationModel(
    #                 dataset_name=dataset,
    #                 dataset_path=root_dir,
    #                 model_type=model_type,
    #                 feature_set='acc',
    #                 batch_size_train=16,
    #                 batch_size_valid=32,
    #                 batch_size_test=32,
    #                 position='waist',
    #                 input_size=3,
    #                 output_size=output_size,
    #                 flatten_method=method,  # Changed to the current flatten method
    #                 num_channels=(64,) * 5 + (128,) * 2,
    #                 kernel_size=2,  # 2
    #                 kernel_size1=2,  # 2
    #                 kernel_size2=3,  # 3
    #                 dropout=0.5,
    #                 learning_rate=0.001,
    #                 num_epochs=200,
    #                 model_save_path=f"./models/{dataset}_multi_class/angles/{angle}",
    #                 # Changed to reflect the flatten method
    #                 augmenter=augmenter,
    #                 aug_name=name
    #             ).run()
    #
    #             records = pd.concat([records, pd.DataFrame([record])], ignore_index=True)
    #             records.iloc[row, 0] = model_type
    #             records.iloc[row, 1] = method
    #             records.iloc[row, 2] = name
    #
    #             row += 1
    #
    #         records.to_csv(f'./experiment_results/angles/{dataset}_multi_class.csv', index=False)


    # flatten_methods = ["last", "mean", "max"]
    #
    # for dataset in datasets:
    #     root_dir = f'./datasets/{dataset}.pkl'
    #     row = 0
    #     records = pd.DataFrame(columns=['model', 'flatten', 'train_loss', 'valid_loss', 'valid_accuracy', 'valid_f1',
    #                                     'valid_TPR', 'valid_FPR', 'valid_FP', 'valid_FN', 'valid_TP', 'valid_TN',
    #                                     'test_Accuracy', 'test_TN', 'test_FP', 'test_FN', 'test_TP', 'test_TPR',
    #                                     'test_FPR', 'test_F1'])
    #     for size in [3, 5, 7]:
    #         for method in flatten_methods:
    #             # Create a dictionary to store the metrics during testing
    #             record = ClassificationModel(
    #                 dataset_name=dataset,
    #                 dataset_path=root_dir,
    #                 model_type="SingleCnnGru",
    #                 batch_size_train=16,
    #                 batch_size_valid=32,
    #                 batch_size_test=32,
    #                 position='waist',
    #                 input_size=3,
    #                 output_size=1,
    #                 flatten_method=method,  # Changed to the current flatten method
    #                 num_channels=(64,) * 5 + (128,) * 2,
    #                 kernel_size=size,  # 2
    #                 kernel_size1=2,  # 2
    #                 kernel_size2=3,  # 3
    #                 dropout=0.5,
    #                 learning_rate=0.001,
    #                 num_epochs=30,
    #                 model_save_path=f"./models/{dataset}/flatten/SingleCnnGru/size{size}_{method}",
    #                 # Changed to reflect the flatten method
    #                 augmenter=None,
    #                 aug_name=method
    #             ).run()
    #
    #             records = pd.concat([records, pd.DataFrame([record])], ignore_index=True)
    #             records.iloc[row, 0] = "SingleCnnGru"
    #             records.iloc[row, 1] = method
    #
    #             row += 1
    #
    #     records.to_csv(f'./experiment_results/flatten/{dataset}_single_cnn_gru.csv', index=False)
