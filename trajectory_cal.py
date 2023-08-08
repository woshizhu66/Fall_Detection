import numpy as np


def calculate_trajectory(accelerometer_data, euler_angles, time):
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
    num_windows, window_length, _ = data.shape

    # 创建一个数组用于存储轨迹数据
    trajectories = np.zeros((num_windows, window_length, 3))

    for i in range(num_windows):
        # 提取加速度计数据和欧拉角数据，同时将加速度计数据从 g 转换为 m/s^2
        accelerometer_data = data[i, :, 1:4] * 9.81  # indices 1:4 correspond to acc_x, acc_y, acc_z
        euler_angles = data[i, :, 7:]  # indices 7:10 correspond to euler_x, euler_y, euler_z
        time = data[i, :, 0]  # the first column is time

        # 计算轨迹并存储结果
        trajectories[i] = calculate_trajectory(accelerometer_data, euler_angles, time)

    return trajectories


def cal_acc_norm(data):
    num_windows, window_length, _ = data.shape
    norm = []
    for i in range(num_windows):
        accelerometer_data = data[i, :, 1:4]  # indices 1:4 correspond to acc_x, acc_y, acc_z
        # Calculate the norm
        acceleration_norm = np.linalg.norm(accelerometer_data, axis=1)
        norm.append(acceleration_norm)
    return np.array(norm)


def cal_gyr_norm(data):
    num_windows, window_length, _ = data.shape
    norm = []
    for i in range(num_windows):
        gyr_data = data[i, :, 4:7]
        # Calculate the norm
        gyr_norm = np.linalg.norm(gyr_data, axis=1)
        norm.append(gyr_norm)
    return np.array(norm)
