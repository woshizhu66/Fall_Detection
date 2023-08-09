import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class BasicArrayDataset(Dataset):
    def __init__(self, label_data_dict: dict, augmenter=None, float_precision: str = 'float32'):
        """
        Initialize a dataset for basic array-based data with corresponding labels.

        Args:
            label_data_dict (dict): A dictionary where keys are labels (integers)
                                   and values are data arrays [n_samples, ..., n_channels].
            augmenter: Augmenter object
            float_precision (str): Data type to which data arrays will be converted (default is 'float32').
        """
        print('Label distribution:')
        for k, v in label_data_dict.items():
            print(k, ':', len(v))

        self.float_precision = float_precision
        self.augmenter = augmenter

        self.num_classes = len(label_data_dict)
        self.data = []
        self.label = []
        for label, data in label_data_dict.items():
            self.data.append(data)
            self.label.append([label] * len(data))

        self.data = np.concatenate(self.data)
        self.label = np.concatenate(self.label)

        self.label = self.label.astype(np.int64)

        self.augmenter = augmenter

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        # data = data.transpose(1, 0)

        if self.augmenter is not None:
            data, non_label = self.augmenter.apply(data)

        return data.astype(self.float_precision), label

    def __len__(self) -> int:
        return len(self.label)


class BasicArrayDatasetSeg(Dataset):
    """
    Initialize a dataset for segmented array-based data with corresponding labels.

    Args:
        windows: List of data windows [n_samples, ..., n_channels].
        labels: Corresponding labels for each window.
        augmenter: Augmenter object to apply data augmentation.
        float_precision (str): Data type to which data arrays will be converted (default is 'float32').
    """
    def __init__(self, windows, labels, augmenter=None, float_precision: str = 'float32'):
        self.augmenter = augmenter
        self.windows = windows
        self.labels = labels
        self.float_precision = float_precision

    def __getitem__(self, index):
        window = self.windows[index]
        label = self.labels[index]

        if self.augmenter is not None:
            window, label = self.augmenter.apply(window, label)

        return window.astype(self.float_precision), label

    def __len__(self) -> int:
        return len(self.windows)


class ResampleArrayDataset(Dataset):
    def __init__(self, label_data_dict: dict, augmenter=None, shuffle: bool = True,
                 float_precision: str = 'float32'):
        """
            Initialize a resampling dataset for array-based data with corresponding labels.

            Args:
                label_data_dict (dict): A dictionary where keys are labels (integers)
                                        and values are data arrays [n_samples, ..., n_channels].
                shuffle (bool): Whether to shuffle data after each epoch.
                augmenter: Augmenter object
                float_precision (str): Data type to which data arrays will be converted (default is 'float32').
        """
        self.num_classes = len(label_data_dict)
        self.shuffle = shuffle
        self.float_precision = float_precision
        self.augmenter = augmenter
        self.label_data_dict = label_data_dict
        self.label_pick_idx = {}

        print('Label distribution:')
        for cls, arr in self.label_data_dict.items():
            print(cls, ':', len(arr))
            self.label_pick_idx[cls] = 0

        # calculate dataset size
        self.dataset_len = sum(len(arr) for arr in self.label_data_dict.values())
        self.mean_class_len = self.dataset_len / len(self.label_data_dict)

    def __getitem__(self, index):
        label = int(index // self.mean_class_len)
        data = self.label_data_dict[label][self.label_pick_idx[label]]

        self.label_pick_idx[label] += 1
        if self.label_pick_idx[label] == len(self.label_data_dict[label]):
            self.label_pick_idx[label] = 0
            self._shuffle_class_index(label)

        # data = data.transpose(1, 0)

        if self.augmenter is not None:
            data, non_label = self.augmenter.apply(data)

        return data.astype(self.float_precision), label

    def _shuffle_class_index(self, cls: int):
        if self.shuffle:
            self.label_data_dict[cls] = self.label_data_dict[cls][
                torch.randperm(len(self.label_data_dict[cls]))
            ]

    def __len__(self) -> int:
        return self.dataset_len


class ResampleArrayDatasetSeg(Dataset):
    """
    Initialize a segmented resampling dataset for array-based data with corresponding labels.

    Args:
        windows: List of data windows [n_samples, ..., n_channels].
        labels: Corresponding labels for each window.
        augmenter: Augmenter object
        float_precision (str): Data type to which data arrays will be converted (default is 'float32').
    """
    def __init__(self, windows, labels, augmenter=None, float_precision: str = 'float32'):

        self.augmenter = augmenter
        self.windows, self.labels = self.oversample_minority(windows, labels)
        self.float_precision = float_precision

    def __getitem__(self, index):
        window = self.windows[index]
        label = self.labels[index]

        if self.augmenter is not None:
            window, label = self.augmenter.apply(window, label)

        return window.astype(self.float_precision), label

    def oversample_minority(self, windows, labels):
        """
        Oversample the minority class to balance the dataset.

        Args:
            windows: Data windows of the minority class.
            labels: Labels corresponding to the minority class windows.

        Returns:
            tuple: Oversampled data windows and corresponding labels.
        """
        minority_idx = np.where(np.sum(labels, axis=1) > 0)[0]
        majority_idx = np.where(np.sum(labels, axis=1) == 0)[0]

        # Calculate the number of replications so that the number of minority classes
        # is roughly equal to the number of majority classes
        replication_factor = len(majority_idx) // len(minority_idx)

        # Randomly select minority class samples for replication
        minority_idx_to_replicate = np.random.choice(minority_idx, size=len(minority_idx) * replication_factor)
        minority_windows = windows[minority_idx_to_replicate]
        minority_labels = labels[minority_idx_to_replicate]

        oversampled_windows = np.concatenate([windows[majority_idx], minority_windows])
        oversampled_labels = np.concatenate([labels[majority_idx], minority_labels])

        return oversampled_windows, oversampled_labels

    def __len__(self) -> int:
        return len(self.windows)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, save_path, window_size, patience=10, verbose=False, delta=0):
        """
        Args:
            save_path :
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.window_size = window_size

    def __call__(self, epoch, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch, val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch, val_loss, model)
            self.counter = 0

    def save_checkpoint(self, epoch, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, f'model_size{self.window_size}_epoch{epoch}.pth')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss
