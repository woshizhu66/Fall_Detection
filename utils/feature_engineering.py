import math

import numpy as np


def calculate_trajectory(accelerometer_data, euler_angles, time):
    """
    Calculate the trajectory of an object based on accelerometer data and euler angles.

    Args:
        accelerometer_data (np.ndarray): Accelerometer data of shape (num_steps, 3) in units of m/s^2.
        euler_angles (np.ndarray): Euler angles (yaw, pitch, roll) of shape (num_steps, 3) in degrees.
        time (np.ndarray): Time values corresponding to each step in seconds.

    Returns:
        position (np.ndarray): Calculated position trajectory of shape (num_steps, 3) in meters.
    """
    g = 9.8  # gravity in m/s^2

    dt = np.gradient(time)  # time step

    num_steps = len(time)

    velocity = np.zeros((num_steps, 3))
    position = np.zeros((num_steps, 3))

    for t in range(1, num_steps):
        angles_rad = np.radians(euler_angles[t])  # convert angles to radians

        # Rotation matrices around x, y, and z axes
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(angles_rad[0]), -np.sin(angles_rad[0])],
                        [0, np.sin(angles_rad[0]), np.cos(angles_rad[0])]])

        R_y = np.array([[np.cos(angles_rad[1]), 0, np.sin(angles_rad[1])],
                        [0, 1, 0],
                        [-np.sin(angles_rad[1]), 0, np.cos(angles_rad[1])]])

        R_z = np.array([[np.cos(angles_rad[2]), -np.sin(angles_rad[2]), 0],
                        [np.sin(angles_rad[2]), np.cos(angles_rad[2]), 0],
                        [0, 0, 1]])

        # Rotation matrix from euler angles
        R = np.dot(R_z, np.dot(R_y, R_x))

        # Acceleration in m/s^2
        accel = accelerometer_data[t] * g
        accel_earth = np.dot(R, accel)  # convert to earth frame

        velocity[t] = velocity[t - 1] + accel_earth * dt[t]
        position[t] = position[t - 1] + velocity[t - 1] * dt[t] + 0.5 * accel_earth * dt[t] ** 2

    return position


def whole_trajectory(data):
    """
    Calculate the trajectory for each window in the input data.

    Args:
        data (np.ndarray): Input data of shape (num_windows, window_length, num_features).

    Returns:
        trajectories (np.ndarray): Calculated trajectories of shape (num_windows, window_length, 3).
    """
    num_windows, window_length, _ = data.shape

    # Create an array to store trajectory data
    trajectories = np.zeros((num_windows, window_length, 3))

    for i in range(num_windows):
        # Extract accelerometer data and euler angles, and convert accelerometer data from g to m/s^2
        accelerometer_data = data[i, :, 1:4] * 9.81  # indices 1:4 correspond to acc_x, acc_y, acc_z
        euler_angles = data[i, :, 7:]  # indices 7:10 correspond to euler_x, euler_y, euler_z
        time = data[i, :, 0]  # the first column is time

        trajectories[i] = calculate_trajectory(accelerometer_data, euler_angles, time)

    return trajectories


def cal_acc_norm(data):
    """
        Calculate the norm of accelerometer data for each window in the input data.

        Args:
            data (np.ndarray): Input data of shape (num_windows, window_length, num_features).

        Returns:
            norm (np.ndarray): Norm of accelerometer data for each window.
        """
    num_windows, window_length, _ = data.shape
    norm = []
    for i in range(num_windows):
        accelerometer_data = data[i, :, 1:4]  # indices 1:4 correspond to acc_x, acc_y, acc_z
        # Calculate the norm
        acceleration_norm = np.linalg.norm(accelerometer_data, axis=1)
        norm.append(acceleration_norm)
    return np.array(norm)


def cal_gyr_norm(data):
    """
    Calculate the norm of gyroscope data for each window in the input data.

    Args:
        data (np.ndarray): Input data of shape (num_windows, window_length, num_features).

    Returns:
        norm (np.ndarray): Norm of gyroscope data for each window.
    """
    num_windows, window_length, _ = data.shape
    norm = []
    for i in range(num_windows):
        gyr_data = data[i, :, 4:7]
        # Calculate the norm
        gyr_norm = np.linalg.norm(gyr_data, axis=1)
        norm.append(gyr_norm)
    return np.array(norm)


# Calculate roll and pitch using the accelerometer data
def calculate_roll_pitch(acc_x, acc_y, acc_z):
    roll = np.arctan2(acc_y, np.sqrt(acc_x ** 2 + acc_z ** 2))
    pitch = np.arctan2(-acc_x, np.sqrt(acc_y ** 2 + acc_z ** 2))
    return roll, pitch



# Calculate yaw using the magnetometer data and roll & pitch
def calculate_yaw(mag_x, mag_y, mag_z, roll, pitch):
    # Correct the magnetic readings for tilt (using pitch and roll)
    mx = mag_x * np.cos(pitch) + mag_z * np.sin(pitch)
    my = mag_x * np.sin(roll) * np.sin(pitch) + mag_y * np.cos(roll) - mag_z * np.sin(roll) * np.cos(pitch)
    mz = -mag_x * np.cos(roll) * np.sin(pitch) + mag_y * np.sin(roll) + mag_z * np.cos(roll) * np.cos(pitch)

    yaw = np.arctan2(my, mx)
    return yaw
