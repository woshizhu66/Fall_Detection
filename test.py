import pandas as pd
import math

# IMU更新的Python版本
Kp = 3.5
Ki = 0.05
halfT = 0.0078125 / 2  # 你的数据的采样间隔的一半
exInt, eyInt, ezInt = 0, 0, 0
q0 = 0
q1 = 1
q2 = 0
q3 = 0  # 初始化为Y轴向下，Z轴向前

def IMUupdate(gx, gy, gz, ax, ay, az):
    global q0, q1, q2, q3, exInt, eyInt, ezInt
    norm = math.sqrt(ax*ax + ay*ay + az*az)
    ax /= norm
    ay /= norm
    az /= norm

    # 计算预期向量
    vx = 2*(q1*q3 - q0*q2)
    vy = 2*(q0*q1 + q2*q3)
    vz = q0*q0 - q1*q1 - q2*q2 + q3*q3

    # 计算误差
    ex = ay*vz - az*vy
    ey = az*vx - ax*vz
    ez = ax*vy - ay*vx

    # 累积误差
    exInt += ex * Ki
    eyInt += ey * Ki
    ezInt += ez * Ki

    # 对陀螺数据进行补偿
    gx += Kp*ex + exInt
    gy += Kp*ey + eyInt
    gz += Kp*ez + ezInt

    # 更新四元数
    q0 += (-q1*gx - q2*gy - q3*gz) * halfT
    q1 += (q0*gx + q2*gz - q3*gy) * halfT
    q2 += (q0*gy - q1*gz + q3*gx) * halfT
    q3 += (q0*gz + q1*gy - q2*gx) * halfT

    norm = math.sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)
    q0 /= norm
    q1 /= norm
    q2 /= norm
    q3 /= norm

    roll = math.atan2(2*q2*q3 + 2*q0*q1, -2*q1*q1 - 2*q2*q2 + 1) * 57.3
    pitch = math.asin(2*q1*q3 - 2*q0*q2) * 57.3
    yaw = -math.atan2(2*q1*q2 + 2*q0*q3, -2*q2*q2 - 2*q3*q3 + 1) * 57.3

    return roll, pitch, yaw

# 读取数据
df = pd.read_csv('D:/IMU/IMU/SUB01_1.txt', sep='\t')

euler_angles = []

# 遍历每一帧的数据并计算欧拉角
for index, row in df.iterrows():
    roll, pitch, yaw = IMUupdate(row['GYR ML [deg/s]'], row['GYR SI [deg/s]'], row['GYR AP [deg/s]'],
                                 row['ACC ML [g]'], row['ACC SI [g]'],  row['ACC AP [g]'])

    euler_angles.append([roll, pitch, yaw])

# 转换成DataFrame，并可以输出为CSV
euler_df = pd.DataFrame(euler_angles, columns=['Roll', 'Pitch', 'Yaw'])
euler_df['acc_x']=df['ACC ML [g]'].copy()
euler_df['acc_y']=df['ACC SI [g]'].copy()
euler_df['acc_z']=df['ACC AP [g]'].copy()
euler_df['time']=df['Time [s]'].copy()

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
        angles_rad = np.radians(euler_angles.iloc[t,:])  # convert angles to radians

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
        accel = accelerometer_data.iloc[t,:] * g
        accel_earth = np.dot(R, accel)  # convert to earth frame
        accel_earth[2] -= 9.8

        velocity[t] = velocity[t - 1] + accel_earth * dt[t]
        position[t] = position[t - 1] + velocity[t - 1] * dt[t] + 0.5 * accel_earth * dt[t] ** 2

    return position



import numpy as np
trajectory=calculate_trajectory(euler_df.iloc[:,0:3],euler_df.iloc[:,3:6], euler_df['time'])
print(trajectory)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from JSAnimation.IPython_display import display_animation

# 假设你有以下的三维数据点
x = trajectory[:,0]
y = trajectory[:,1]
z = trajectory[:,2]

fig = plt.figure("3D Surface", facecolor="lightgray")
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z)

# 设置坐标轴的标签
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# 显示图形
plt.show()
