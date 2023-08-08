import os

import numpy as np

from trajectory_cal import whole_trajectory


def trajectory_load_dataset(root_dir):
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
            trajectories = whole_trajectory(arr)
            if folder.endswith('_fall'):
                file_label = 1
            else:
                file_label = 0
            windows.append(trajectories)
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