import os
import re

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from data_load_and_EarlyStop import EarlyStopping, ResampleArrayDataset, BasicArrayDataset
from tcn import TCN


def extract_task(folder_name):
    match = re.search(r'KFall_task(\d+)', folder_name)
    if match:
        return int(match.group(1))
    else:
        return None


def multiclass_load_dataset(root_dir):
    train_dict = {}
    valid_dict = {}
    test_dict = {}

    windows = []
    labels = []
    tasks = []

    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if folder == 'KFall_fall':
            task = 0
        else:
            task = extract_task(folder)
        for file in os.listdir(folder_path):
            if task not in train_dict:
                train_dict[task] = []
                valid_dict[task] = []
                test_dict[task] = []

            file_path = os.path.join(folder_path, file)

            arr = np.load(file_path)[:, :, 1:]
            windows.append(arr)
            labels += [task] * len(arr)
            tasks.append(task)

    windows = np.concatenate(windows)
    labels = np.array(labels)

    np.random.seed(123)

    for task in set(tasks):
        task_idx = np.where(labels == task)[0]
        np.random.shuffle(task_idx)

        train_task_idx = task_idx[:int(0.7 * len(task_idx))]
        valid_task_idx = task_idx[int(0.7 * len(task_idx)):int(0.9 * len(task_idx))]
        test_task_idx = task_idx[int(0.9 * len(task_idx)):]

        # split dataset into train, valid and test
        windows_train = windows[train_task_idx]
        windows_valid = windows[valid_task_idx]
        windows_test = windows[test_task_idx]

        # append train into train_dict(s)
        train_dict[task].append(windows_train)

        # append valid into valid dict
        valid_dict[task].append(windows_valid)

        # append test into test dict
        test_dict[task].append(windows_test)

    # return result
    train_dict = {key: np.concatenate(value) for key, value in train_dict.items()}
    valid_dict = {key: np.concatenate(value) for key, value in valid_dict.items()}
    test_dict = {key: np.concatenate(value) for key, value in test_dict.items()}

    return train_dict, valid_dict, test_dict


def cal_tp_tn_fp_fn(confusionmatrix):
    FP = confusionmatrix.sum(axis=0) - np.diag(confusionmatrix)
    FN = confusionmatrix.sum(axis=1) - np.diag(confusionmatrix)
    TP = np.diag(confusionmatrix)
    TN = confusionmatrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    return TP, TN, FP, FN

def plot_confusion(y_true, y_pred, labels):
    sns.set()
    f, ax = plt.subplots()
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(matrix, annot=True, ax=ax)  # 画热力图
    ax.set_xlabel('predict')  # x轴
    ax.set_ylabel('true')  # y轴
    plt.show()
    return matrix


class classificationModel:
    def __init__(self, dataset_path, batch_size_train, batch_size_valid, batch_size_test, input_size, output_size,
                 num_channels, kernel_size, dropout, learning_rate, num_epochs, model_save_path):
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

    def run(self):
        # Load data
        train, valid, test = multiclass_load_dataset(self.dataset_path)
        train_set = BasicArrayDataset(train)
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
        model = TCN(self.input_size, self.output_size, self.num_channels, self.kernel_size, self.dropout)

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

        csv_file_name = f'./KFall_train_records/{window_size}_cla_training_metrics.csv'

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
        confusion = plot_confusion(actuals, predictions, np.unique(actuals))

        FP, FN, TP, TN = cal_tp_tn_fp_fn(confusion)
        TPR = TP[0] / (TP[0] + FN[0])  # Sensitivity
        FPR = FP[0] / (FP[0] + TN[0])  # false positive rate
        F1 = float(f1_score(actuals, predictions, average='weighted'))

        print('Test Accuracy: {:.4f}, F1 score: {:.4f}, TPR: {:.4f}, FPR: {:.4f}'.format(accuracy, F1, TPR, FPR))
        return window_size, accuracy, F1, TPR, FPR

if __name__ == "__main__":
    root_dir = 'C:/Repository/master/Processed_Dataset/KFall'
    test_metrics = {
        'window_size': [],
        'test_accuracy': [],
        'test_F1': [],
        'test_TPR': [],
        'test_FPR': [],
    }
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        # Create a dictionary to store the metrics during testing
        window_size, accuracy, F1, TPR, FPR = classificationModel(
            dataset_path=folder_path,
            batch_size_train=16,
            batch_size_valid=32,
            batch_size_test=32,
            input_size=9,
            output_size=1,
            num_channels=(64,) * 3 + (128,) * 2,
            kernel_size=2,
            dropout=0.2,
            learning_rate=0.001,
            num_epochs=100,
            model_save_path="./cla_model"
        ).run()

        test_metrics['window_size'].append(window_size)
        test_metrics['test_accuracy'].append(accuracy)
        test_metrics['test_F1'].append(F1)
        test_metrics['test_TPR'].append(TPR)
        test_metrics['test_FPR'].append(FPR)

    df_metrics = pd.DataFrame(test_metrics)
    df_metrics.to_csv('./cla_test_metrics.csv', index=False)
