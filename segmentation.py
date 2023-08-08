import pandas as pd
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from tcn import TemporalConvNet, TCNSeg
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
from data_load_and_EarlyStop import EarlyStopping, BasicArrayDataset, ResampleArrayDataset, ResampleArrayDatasetSeg, \
    BasicArrayDatasetSeg
import utils.augmentation as aug


def process_data_for_segmentation(root_dir, window_length, subwindow_length):
    sample_rate = 50  # 采样频率
    samples_per_subwindow = subwindow_length * sample_rate  # 每个子窗口的样本数
    num_subwindows = int(window_length // subwindow_length)  # 一个窗口中的子窗口数

    windows = []
    labels = []

    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            arr = np.load(file_path)[:, :, 1:]  # 加载数据
            fall = folder.endswith('_fall')  # 检查是否为跌倒

            for window in arr:
                subwindow_labels = []
                subwindows = []
                for i in range(num_subwindows):
                    start = i * samples_per_subwindow  # 子窗口的开始
                    end = start + samples_per_subwindow  # 子窗口的结束
                    subwindow = window[start:end, :]  # 提取子窗口

                    subwindows.append(subwindow)

                    # 如果这个子窗口在跌倒事件发生的3秒内，则标记为跌倒
                    # 否则，标记为非跌倒
                    if fall and i < 1:  # 这里我们检查跌倒是否在前3秒发生
                        subwindow_labels.append(1)
                    else:
                        subwindow_labels.append(0)

                window = np.concatenate(subwindows, axis=0)  # 按照第0个维度（即行）进行拼接
                windows.append(window)
                labels.append(subwindow_labels)

    windows = np.array(windows)
    labels = np.array(labels)

    fall_indices = np.where(np.sum(labels, axis=1) > 0)[0]
    non_fall_indices = np.where(np.sum(labels, axis=1) == 0)[0]

    np.random.seed(123)
    np.random.shuffle(fall_indices)
    np.random.shuffle(non_fall_indices)

    train_fall_indices = fall_indices[:int(0.7 * len(fall_indices))]
    valid_fall_indices = fall_indices[int(0.7 * len(fall_indices)):int(0.9 * len(fall_indices))]
    test_fall_indices = fall_indices[int(0.9 * len(fall_indices)):]

    train_non_fall_indices = non_fall_indices[:int(0.7 * len(non_fall_indices))]
    valid_non_fall_indices = non_fall_indices[int(0.7 * len(non_fall_indices)):int(0.9 * len(non_fall_indices))]
    test_non_fall_indices = non_fall_indices[int(0.9 * len(non_fall_indices)):]

    train_indices = np.concatenate([train_fall_indices, train_non_fall_indices])
    valid_indices = np.concatenate([valid_fall_indices, valid_non_fall_indices])
    test_indices = np.concatenate([test_fall_indices, test_non_fall_indices])

    # Split dataset into train, valid, and test
    windows_train = windows[train_indices]
    labels_train = labels[train_indices]
    windows_valid = windows[valid_indices]
    labels_valid = labels[valid_indices]
    windows_test = windows[test_indices]
    labels_test = labels[test_indices]

    return windows_train, labels_train, windows_valid, labels_valid, windows_test, labels_test


def calculate_metrics(actuals, predictions):
    # Flatten the arrays if they have more than one dimension
    actuals = np.asarray(actuals).flatten()
    predictions = np.asarray(predictions).flatten()

    # Calculate the confusion matrix
    cm = confusion_matrix(actuals, predictions)

    # The confusion matrix returns the values in the format: [[TN, FP], [FN, TP]]
    TN, FP, FN, TP = cm.ravel()

    # Calculate the F1 score
    f1 = f1_score(actuals, predictions, average='weighted', zero_division=0)

    return TP, TN, FP, FN, f1


class SegmentationModel:
    def __init__(self, dataset_path, batch_size_train, batch_size_valid, window_length, subwindow_length,
                 batch_size_test, input_size, output_size,
                 num_channels, kernel_size, dropout, learning_rate, num_epochs, model_save_path, augmenter, aug_name):
        self.augmenter = augmenter
        self.dataset_path = dataset_path
        self.batch_size_train = batch_size_train
        self.batch_size_valid = batch_size_valid
        self.batch_size_test = batch_size_test
        self.input_size = input_size
        self.output_size = output_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.model_save_path = model_save_path
        self.aug_name = aug_name
        self.window_length = window_length
        self.subwindow_length = subwindow_length

    def run(self):
        # Load data
        windows_train, labels_train, windows_valid, labels_valid, windows_test, labels_test = process_data_for_segmentation(
            self.dataset_path, self.window_length, self.subwindow_length)
        train_set = ResampleArrayDatasetSeg(windows_train, labels_train, augmenter=self.augmenter)
        train_loader = DataLoader(train_set, batch_size=self.batch_size_train, shuffle=True)
        valid_set = BasicArrayDatasetSeg(windows_valid, labels_valid)
        valid_loader = DataLoader(valid_set, batch_size=self.batch_size_valid, shuffle=False)
        test_set = BasicArrayDatasetSeg(windows_test, labels_test)
        test_loader = DataLoader(test_set, batch_size=self.batch_size_test, shuffle=False)

        window_size = self.dataset_path.split('window_sec')[1]

        if not os.path.exists(self.model_save_path):
            # If the directory doesn't exist, create it
            os.makedirs(self.model_save_path)
        early_stopping = EarlyStopping(self.model_save_path, window_size)

        # Instantiate the TCN model
        model = TCNSeg(self.input_size, self.output_size, self.num_channels, self.kernel_size, self.dropout)

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
            valid_f1 = f1_score(valid_labels, valid_outputs, average='weighted', zero_division=0)

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

        csv_file_name = f'./KFall_train_records/{window_size}_{self.subwindow_length}training_metrics_{self.aug_name}.csv'

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

        FP, FN, TP, TN, F1 = calculate_metrics(actuals, predictions)
        TPR = TP / (TP + FN)  # Sensitivity
        FPR = FP / (FP + TN)  # false positive rate

        print('Test Accuracy: {:.4f}, F1 score: {:.4f}, TPR: {:.4f}, FPR: {:.4f}'.format(accuracy, F1, TPR, FPR))
        return window_size, accuracy, F1, TPR, FPR


if __name__ == "__main__":
    root_dir = 'C:/Repository/master/Processed_Dataset/KFall/KFall_window_sec30'

    for subwindow_length in [2, 3, 4, 5]:
        test_metrics = {
            'window_size': [],
            'sub_window_size': [],
            'test_accuracy': [],
            'test_F1': [],
            'test_TPR': [],
            'test_FPR': [],
        }

        window_length = 30

        # Create a dictionary to store the metrics during testing
        window_size, accuracy, F1, TPR, FPR = SegmentationModel(
            dataset_path=root_dir,
            batch_size_train=8,
            batch_size_valid=16,
            batch_size_test=16,
            input_size=9,
            num_channels=(64,) * 3 + (128,) * 2,
            kernel_size=2,
            dropout=0.5,
            learning_rate=0.01,
            num_epochs=30,
            window_length=window_length,
            subwindow_length=subwindow_length,
            output_size=int(window_length // subwindow_length),
            model_save_path=f"./seg_model_test",
            augmenter=None,
            aug_name="test"
        ).run()

        test_metrics['window_size'].append(window_size)
        test_metrics['sub_window_size'].append(subwindow_length)
        test_metrics['test_accuracy'].append(accuracy)
        test_metrics['test_F1'].append(F1)
        test_metrics['test_TPR'].append(TPR)
        test_metrics['test_FPR'].append(FPR)

    df_metrics = pd.DataFrame(test_metrics)
    df_metrics.to_csv(f'./seg_test_metrics.csv', index=False)
