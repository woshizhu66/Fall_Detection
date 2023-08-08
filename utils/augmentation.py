import numpy as np
from scipy.interpolate import CubicSpline
from tqdm import tqdm
from transforms3d.axangles import axangle2mat
from typing import Union, List, Tuple


def gen_random_curves(length: int, num_curves: int, sigma=0.2, knot=4):
    """
    Generate random curves using CubicSpline interpolation.

    Args:
        length:
        num_curves:
        sigma:
        knot:

    Returns:
        array shape [length, num curves]
    """
    xx = np.arange(0, length, (length - 1) / (knot + 1))
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, num_curves))
    x_range = np.arange(length)
    curves = np.array([CubicSpline(xx, yy[:, i])(x_range) for i in range(num_curves)]).T
    return curves


def format_range(x: any, start_0: bool) -> np.ndarray:
    """
    Turn an arbitrary input into a range. For example:
        1 to [-1, 1] or [0, 1]
        None to [0, 0]
        [-1, 1] will be kept intact

    Args:
        x: any input
        start_0: if input x is a scalar and this is True, the starting point of the range will be 0,
            otherwise it will be -abs(x)

    Returns:
        a np array of 2 element [range start, range end]
    """
    if x is None:
        x = [0, 0]
    if isinstance(x, float) or isinstance(x, int):
        x = abs(x)
        x = [0 if start_0 else -x, x]
    assert len(x) == 2, 'x must be a scalar or a list/tuple of 2 numbers'
    return np.array(x)


class Augmentation:
    def __init__(self, p=1):
        """
        Initialize the base Augmentation class.

        Args:
            p (float): Probability of applying the augmentation.
        """
        self.p = p

    def apply(self, data):
        """
        Apply the augmentation to the given data.

        Args:
            data (np.ndarray)

        Returns:
            np.ndarray: Augmented data.
        """
        if (np.random.rand() < self.p) or (self.p >= 1):  # Only augment the data with probability p
            return self.augment(data)
        else:
            return data

    def augment(self, data):
        """
        Abstract method to apply the augmentation.

        Args:
            data (np.ndarray): Input data to be augmented.

        Returns:
            np.ndarray: Augmented data.
        """
        raise NotImplementedError


class Augmenter:
    def __init__(self, augmentations):
        """
        Initialize the Augmenter class.

        Args:
            augmentations (list): List of augmentation instances.
        """
        self.augmentations = augmentations

    def apply(self, data, label=None):
        """
        Apply a list of augmentations to the data and label (if provided).

        Args:
            data (np.ndarray): Input data to be augmented.
            label: Label associated with the data, if available.

        Returns:
            np.ndarray: Augmented data.
            label: Augmented label (if available).
        """
        for augment in self.augmentations:
            if label is not None:
                if augment == 'RandSampleSeg' or augment == 'TimewarpSeg' or augment == 'PermutationSeg':
                    data, label = augment.apply(data, label)
                else:
                    data = augment.apply(data)
            else:
                data = augment.apply(data)
        return data, label


class Jitter(Augmentation):
    def __init__(self, sigma=0.05, p=1):
        """
        Args:
            sigma (float): Standard deviation of the noise added to the data.
            p (float): Probability of applying the augmentation.
        """
        super().__init__(p)
        self.sigma = sigma

    def augment(self, data):
        if np.random.rand() < self.p:
            noise = np.random.normal(loc=0, scale=self.sigma, size=data.shape)
            data = data + noise
        return data


class Scale(Augmentation):
    def __init__(self, sigma=0.1, p=1):
        """
        Args:
            sigma (float): Standard deviation of the scaling factor.
            p (float): Probability of applying the augmentation.
        """
        super().__init__(p)
        self.sigma = sigma

    def augment(self, data):
        factor = np.random.normal(loc=1.0, scale=self.sigma, size=(1, data.shape[1]))  # shape=(1,3)
        noise = np.matmul(np.ones((data.shape[0], 1)), factor)
        return data * noise


class MagnitudeWarp(Augmentation):
    def __init__(self, sigma=0.2, knot=4, p=1.0):
        """
        Args:
            sigma (float): Standard deviation of the magnitude warp curves.
            knot (int): Number of knots for generating the random curves.
            p (float): Probability of applying the augmentation.
        """
        super().__init__(p)
        self.sigma = sigma
        self.knot = knot

    def augment(self, data):
        return data * gen_random_curves(data.shape[0], data.shape[1], self.sigma)


class Rotation(Augmentation):
    def __init__(self, p: float, angle_range: Union[list, tuple, float] = None) -> None:
        """
        Rotate tri-axial data in a random axis.

        Args:
            p: probability to apply this augmenter each time it is called
            angle_range: (degree) the angle is randomised within this range;
                if this is a list, randomly pick an angle in this range;
                if it's a float, the range is [-float, float]
        """
        super().__init__(p)

        self.angle_range = format_range(angle_range, start_0=False) / 180 * np.pi

    def augment(self, org_data: np.ndarray) -> np.ndarray:
        """
        Apply augmentation methods in self.list_aug_func
        :param org_data:
            shape (time step, channel) channel must be divisible by 3,
            otherwise bugs may occur
        :return: array shape (time step, channel)
        """
        assert (len(org_data.shape) >= 2) and (org_data.shape[-1] % 3 == 0), \
            f"expected data shape: [*, any length, channel%3==0], got {org_data.shape}"

        angle = np.random.uniform(low=self.angle_range[0], high=self.angle_range[1])
        direction_vector = np.random.uniform(-1, 1, size=3)

        # transpose data to shape [channel, time step]
        data = org_data.T

        # for every 3 channels
        for i in range(0, data.shape[-2], 3):
            data[i:i + 3, :] = self.rotate(data[i:i + 3, :], angle, direction_vector)

        # transpose back to [time step, channel]
        data = data.T
        return data

    @staticmethod
    def rotate(data, angle: float, axis: np.ndarray):
        """
        Rotate data array

        Args:
            data: data array, shape [3, n]
            angle: a random angle in radian
            axis: a 3-d vector, the axis to rotate around

        Returns:
            rotated data of the same format as the input
        """
        rot_mat = axangle2mat(axis, angle)
        data = np.matmul(rot_mat, data)
        return data


class Timewarp(Augmentation):
    def __init__(self, sigma=0.2, knot=4, p=1.0):
        super().__init__(p)
        self.sigma = sigma
        self.knot = knot

    def distort_time_steps(self, length: int, num_curves: int):
        tt = gen_random_curves(length, num_curves, self.sigma, self.knot)
        tt_cum = np.cumsum(tt, axis=0)

        # Make the last value equal length
        t_scale = (length - 1) / tt_cum[-1]

        tt_cum *= t_scale
        return tt_cum

    def augment(self, org_data: np.ndarray) -> np.ndarray:
        # create new timestamp for all channels
        tt_new = self.distort_time_steps(org_data.shape[-2], 1).squeeze()
        x_range = np.arange(org_data.shape[0])
        data = np.array([
            np.interp(x_range, tt_new, org_data[:, i]) for i in range(org_data.shape[-1])
        ]).T
        return data

class TimewarpSeg(Augmentation):
    def __init__(self, sigma=0.2, knot=4, p=1.0):
        super().__init__(p)
        self.sigma = sigma
        self.knot = knot

    def distort_time_steps(self, length: int, num_curves: int):
        tt = gen_random_curves(length, num_curves, self.sigma, self.knot)
        tt_cum = np.cumsum(tt, axis=0)

        # Make the last value equal length
        t_scale = (length - 1) / tt_cum[-1]

        tt_cum *= t_scale
        return tt_cum

    def augment(self, org_data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # create new timestamp for all channels
        tt_new = self.distort_time_steps(org_data.shape[-2], 1).squeeze()
        x_range = np.arange(org_data.shape[0])
        data = np.array([
            np.interp(x_range, tt_new, org_data[:, i]) for i in range(org_data.shape[-1])
        ]).T

        labels = np.interp(x_range, tt_new, labels)
        return data, labels


class Permutation(Augmentation):
    def __init__(self, n_perm=4, min_seg_length=10, p=1):
        super().__init__(p)
        self.n_perm = n_perm
        self.min_seg_length = min_seg_length

    def augment(self, data):
        data_new = np.zeros(data.shape)
        idx = np.random.permutation(self.n_perm)
        bWhile = True
        while bWhile:
            segs = np.zeros(self.n_perm + 1, dtype=int)
            segs[1:-1] = np.sort(
                np.random.randint(self.min_seg_length, data.shape[0] - self.min_seg_length, self.n_perm - 1))
            segs[-1] = data.shape[0]
            if np.min(segs[1:] - segs[0:-1]) > self.min_seg_length:
                bWhile = False
        pp = 0
        for ii in range(self.n_perm):
            x_temp = data[segs[idx[ii]]:segs[idx[ii] + 1], :]
            data_new[pp:pp + len(x_temp), :] = x_temp
            pp += len(x_temp)
        return data_new

class PermutationSeg(Augmentation):
    def __init__(self, n_perm=4, min_seg_length=10, p=1):
        super().__init__(p)
        self.n_perm = n_perm
        self.min_seg_length = min_seg_length

    def augment(self, data, labels):
        data_new = np.zeros(data.shape)
        labels_new = np.zeros(labels.shape)
        idx = np.random.permutation(self.n_perm)
        bWhile = True
        while bWhile:
            segs = np.zeros(self.n_perm + 1, dtype=int)
            segs[1:-1] = np.sort(
                np.random.randint(self.min_seg_length, data.shape[0] - self.min_seg_length, self.n_perm - 1))
            segs[-1] = data.shape[0]
            if np.min(segs[1:] - segs[0:-1]) > self.min_seg_length:
                bWhile = False
        pp = 0
        for ii in range(self.n_perm):
            x_temp = data[segs[idx[ii]]:segs[idx[ii] + 1], :]
            y_temp = labels[segs[idx[ii]]:segs[idx[ii] + 1]]
            data_new[pp:pp + len(x_temp), :] = x_temp
            labels_new[pp:pp + len(y_temp)] = y_temp
            pp += len(x_temp)
        return data_new, labels_new

class RandSample(Augmentation):
    def __init__(self, n_sample=1000, p=0.5):
        super().__init__(p)
        self.n_sample = n_sample

    def rand_sample_time_steps(self, length: int, num_features: int):
        tt = np.zeros((self.n_sample, num_features), dtype=int)
        tt[1:-1, :] = np.sort(np.random.randint(1, length - 1, (self.n_sample - 2, num_features)))
        tt[-1, :] = length - 1
        return tt

    def augment(self, org_data: np.ndarray) -> np.ndarray:
        tt_new = self.rand_sample_time_steps(org_data.shape[-2], 1).squeeze()
        x_range = np.arange(org_data.shape[0])
        data = np.array([
            np.interp(x_range, tt_new, org_data[tt_new, i]) for i in range(org_data.shape[-1])
        ]).T
        return data

class RandSampleSeg(Augmentation):
    def __init__(self, n_sample=1000, p=0.5):
        super().__init__(p)
        self.n_sample = n_sample

    def rand_sample_time_steps(self, length: int, num_features: int):
        tt = np.zeros((self.n_sample, num_features), dtype=int)
        tt[1:-1, :] = np.sort(np.random.randint(1, length - 1, (self.n_sample - 2, num_features)))
        tt[-1, :] = length - 1
        return tt

    def augment(self, org_data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        tt_new = self.rand_sample_time_steps(org_data.shape[-2], 1).squeeze()
        x_range = np.arange(org_data.shape[0])
        data = np.array([
            np.interp(x_range, tt_new, org_data[tt_new, i]) for i in range(org_data.shape[-1])
        ]).T

        labels = labels[tt_new]
        return data, labels