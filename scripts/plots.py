# import pickle
# import matplotlib.pyplot as plt
# import random
# import numpy as np
# import pywt
# import numpy as np
# import scipy.signal
# from scipy.signal import butter, filtfilt


# def remove_outliers(data, jump_threshold=10):
#     smoothed_data = data.copy()
#     for i in range(1, len(smoothed_data) - 2):
#         if abs(smoothed_data[i] - smoothed_data[i-1]) > jump_threshold and \
#            abs(smoothed_data[i] - smoothed_data[i+2]) > jump_threshold:
#             smoothed_data[i] = (smoothed_data[i-1] + smoothed_data[i+2]) / 2
#     return smoothed_data

# file_path = '/Users/srahmani/Downloads/waymo_all.pkl'

# with open(file_path, 'rb') as file:
#     data = pickle.load(file)

# ## For multiple trajectory pickle files
# random_pair = random.choice(data)
# leader_v = random_pair['leader_v']
# leader_t = random_pair['leader_t']
# follower_v = random_pair['follower_v']
# follower_t = random_pair['follower_t']

# ### For one trajectory pickle file
# # leader_v = data['leader_v']
# # leader_t = data['leader_t']
# # follower_v = data['follower_v']
# # follower_t = data['follower_t']

# def robust_low_pass_filter(data, cutoff=0.5, fs=10.0, order=4, jump_threshold=1):
#     # Remove outliers
#     data_no_outliers = remove_outliers(data, jump_threshold)
#     # Apply low-pass filter
#     nyquist = 0.5 * fs
#     normal_cutoff = cutoff / nyquist
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     y = filtfilt(b, a, data_no_outliers)
#     return y

# # Apply the robust low-pass filter to the leader and follower data
# leader_v_smooth_lp = robust_low_pass_filter(leader_v, cutoff=0.5, fs=10.0, order=4, jump_threshold=1)
# follower_v_smooth_lp = robust_low_pass_filter(follower_v, cutoff=0.5, fs=10.0, order=4, jump_threshold=1)

# # Plotting the original data and the smoothed data using Low Pass Filter
# plt.figure(figsize=(6, 6))

# plt.scatter(leader_t, leader_v, label='Original Leader Trajectory', color='blue', s=10, alpha=0.5)
# plt.scatter(follower_t, follower_v, label='Original Follower Trajectory', color='red', s=10, alpha=0.5)
# plt.plot(leader_t, leader_v_smooth_lp, label='Smoothed Leader Trajectory (LPF)', color='blue')
# plt.plot(follower_t, follower_v_smooth_lp, label='Smoothed Follower Trajectory (LPF)', color='red')

# plt.xlabel('Time (s)')
# plt.ylabel('Speed (m/s)')
# plt.title('Leader and Follower Trajectories Over Time (Robust Low Pass Filter)')
# plt.legend()
# plt.grid(True)
# plt.show()

import pickle
import matplotlib.pyplot as plt
import random
import numpy as np
import scipy.signal
from scipy.signal import butter, filtfilt

# Function to remove outliers
# def remove_outliers(data, jump_threshold=10):
#     smoothed_data = data.copy()
#     for i in range(1, len(smoothed_data) - 2):
#         if abs(smoothed_data[i] - smoothed_data[i-1]) > jump_threshold and \
#            abs(smoothed_data[i] - smoothed_data[i+2]) > jump_threshold:
#             smoothed_data[i] = (smoothed_data[i-1] + smoothed_data[i+2]) / 2
#     return smoothed_data
def remove_outliers(data, jump_threshold=2, offset:int=10):
    smoothed_data = data.copy()

    # remove the jumps in the beginning of speed profile
    for index in [4,3,2,1,0]:
        if abs(smoothed_data[index] - smoothed_data[index: index + offset].mean()) > jump_threshold:
            smoothed_data[index] = smoothed_data[index: index + offset].mean()

    # remove the jumps in the middle of speed profile
    for i in range(1, len(smoothed_data) - 2):
        if abs(smoothed_data[i] - smoothed_data[i-1]) > jump_threshold and \
           abs(smoothed_data[i] - smoothed_data[i+2]) > jump_threshold:
            smoothed_data[i] = (smoothed_data[i-1] + smoothed_data[i+2]) / 2
    return smoothed_data


# Load the pickle file
file_path = '../data/waymo_all.pkl'
with open(file_path, 'rb') as file:
    data = pickle.load(file)

## ID Selection
random_number = random.randint(0, len(data)) # If you want to select a random trajectory
print(random_number)
random_number = 315 # If you want to select a specific trajectory
# Trajectory Selection
random_pair = data[random_number]
leader_v = random_pair['leader_v']
leader_t = random_pair['leader_t']
follower_v = random_pair['follower_v']
follower_t = random_pair['follower_t']

# Function to apply a robust low-pass filter
def robust_low_pass_filter(data, cutoff=0.5, fs=10.0, order=4, jump_threshold=1):
    data_no_outliers = remove_outliers(data, jump_threshold)
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data_no_outliers)
    return y

# Apply the robust low-pass filter to the leader and follower data
leader_v_smooth_lp = robust_low_pass_filter(leader_v, cutoff=0.5, fs=10.0, order=4, jump_threshold=1)
follower_v_smooth_lp = robust_low_pass_filter(follower_v, cutoff=0.5, fs=10.0, order=4, jump_threshold=1)

# plotting the raw data seperately
plt.figure(figsize=(10, 6))

# Plot original data with scatter and line
plt.scatter(leader_t, leader_v, label='Leader - Raw', color='blue', s=10, alpha=1)
# plt.plot(leader_t, leader_v, color='skyblue', alpha=0.5)
plt.scatter(follower_t, follower_v, label='Follower - Raw', color='red', s=10, alpha=1)
# plt.plot(follower_t, follower_v, color='lightcoral', alpha=0.5)

# Plot smoothed data
# plt.plot(leader_t, leader_v_smooth_lp, label='Smoothed Leader Trajectory (LPF)', color='blue', linewidth=2)
# plt.plot(follower_t, follower_v_smooth_lp, label='Smoothed Follower Trajectory (LPF)', color='red', linewidth=2)

plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.title('Leader and Follower Trajectories Over Time (Robust Low Pass Filter)')
plt.legend()
plt.grid(True)
plt.show()


# Plotting the original and smoothed data
plt.figure(figsize=(10, 6))

# Plot original data with scatter and line
plt.scatter(leader_t, leader_v, label='Leader - Raw', color='skyblue', s=10, alpha=1)
plt.plot(leader_t, leader_v, color='skyblue', alpha=0.5)
plt.scatter(follower_t, follower_v, label='Follower - Raw', color='lightcoral', s=10, alpha=1)
plt.plot(follower_t, follower_v, color='lightcoral', alpha=0.5)

# Plot smoothed data
plt.plot(leader_t, leader_v_smooth_lp, label='Leader - Smoothed (LPF)', color='blue', linewidth=2)
plt.plot(follower_t, follower_v_smooth_lp, label='Follower - Smoothed (LPF)', color='red', linewidth=2)

plt.xlabel('Time (s)', fontsize=18)
plt.ylabel('Speed (m/s)', fontsize=18)
plt.title('Leader and Follower Trajectories', fontsize=20)
plt.legend()
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True)
plt.show()


