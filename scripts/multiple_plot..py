import pickle
import matplotlib.pyplot as plt
import random
import numpy as np
import pywt
import scipy.signal
from scipy.signal import butter, filtfilt

# Function to remove outliers
def remove_outliers(data, jump_threshold=2, offset=10):
    smoothed_data = data.copy()
    for index in [4, 3, 2, 1, 0]:
        if abs(smoothed_data[index] - smoothed_data[index: index + offset].mean()) > jump_threshold:
            smoothed_data[index] = smoothed_data[index: index + offset].mean()
    for i in range(1, len(smoothed_data) - 2):
        if abs(smoothed_data[i] - smoothed_data[i-1]) > jump_threshold and \
           abs(smoothed_data[i] - smoothed_data[i+2]) > jump_threshold:
            smoothed_data[i] = (smoothed_data[i-1] + smoothed_data[i+2]) / 2
    return smoothed_data

# Load the provided pickle file
file_path = '../data/lyft_all.pkl'

with open(file_path, 'rb') as file:
    data = pickle.load(file)

# For multiple trajectory pickle files
random_pair = random.choice(data)
leader_v = random_pair['leader_v']
leader_t = random_pair['leader_t']
follower_v = random_pair['follower_v']
follower_t = random_pair['follower_t']

# Function to apply a low-pass filter with outlier removal
def robust_low_pass_filter(data, cutoff=0.5, fs=10.0, order=4, jump_threshold=1):
    data_no_outliers = remove_outliers(data, jump_threshold)
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data_no_outliers)
    return y

# Apply the robust low-pass filter
leader_v_smooth_lp = robust_low_pass_filter(leader_v, cutoff=0.5, fs=10.0, order=4, jump_threshold=1)
follower_v_smooth_lp = robust_low_pass_filter(follower_v, cutoff=0.5, fs=10.0, order=4, jump_threshold=1)

# Apply Savitzky-Golay filter
from scipy.signal import savgol_filter
leader_v_smooth_sg = savgol_filter(leader_v, 33, 3)
follower_v_smooth_sg = savgol_filter(follower_v, 33, 3)

# Function to apply Wavelet Transform for denoising
def apply_wavelet_denoising(data, wavelet='db4', level=1):
    coeff = pywt.wavedec(data, wavelet, mode='symmetric')
    sigma = np.median(np.abs(coeff[-level])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(data)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='soft') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='symmetric')

# Apply Wavelet Denoising
leader_v_smooth_wv = apply_wavelet_denoising(leader_v)
follower_v_smooth_wv = apply_wavelet_denoising(follower_v)

# Ensure lengths match
leader_v_smooth_wv = leader_v_smooth_wv[:len(leader_t)]
follower_v_smooth_wv = follower_v_smooth_wv[:len(follower_t)]

# Function to apply Kalman Filter
def apply_kalman_filter(data):
    from filterpy.kalman import KalmanFilter
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([0., 0.])
    kf.F = np.array([[1., 1.], [0., 1.]])
    kf.H = np.array([[1., 0.]])
    kf.P *= 1000.
    kf.R = 5
    kf.Q = 0.1
    filtered_data = []
    for z in data:
        kf.predict()
        kf.update(z)
        filtered_data.append(kf.x[0])
    return filtered_data

# Apply Kalman Filter
leader_v_smooth_kf = apply_kalman_filter(leader_v)
follower_v_smooth_kf = apply_kalman_filter(follower_v)

# Plotting
plt.rcParams.update({'font.size': 14})
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Robust Low Pass Filter plot
axs[0, 0].scatter(leader_t, leader_v, label='Original Leader', color='blue', s=10, alpha=0.5)
axs[0, 0].scatter(follower_t, follower_v, label='Original Follower', color='red', s=10, alpha=0.5)
axs[0, 0].plot(leader_t, leader_v_smooth_lp, label='Smoothed Leader (LPF)', color='blue')
axs[0, 0].plot(follower_t, follower_v_smooth_lp, label='Smoothed Follower (LPF)', color='red')
axs[0, 0].set_title('Robust Low Pass Filter')
axs[0, 0].set_xlabel('Time (s)')
axs[0, 0].set_ylabel('Speed (m/s)')
axs[0, 0].grid(True)

# Savitzky-Golay Filter plot
axs[0, 1].scatter(leader_t, leader_v, label='Original Leader', color='blue', s=10, alpha=0.5)
axs[0, 1].scatter(follower_t, follower_v, label='Original Follower', color='red', s=10, alpha=0.5)
axs[0, 1].plot(leader_t, leader_v_smooth_sg, label='Smoothed Leader (SG)', color='blue')
axs[0, 1].plot(follower_t, follower_v_smooth_sg, label='Smoothed Follower (SG)', color='red')
axs[0, 1].set_title('Savitzky-Golay Filter')
axs[0, 1].set_xlabel('Time (s)')
axs[0, 1].set_ylabel('Speed (m/s)')
axs[0, 1].grid(True)

# Wavelet Denoising plot
axs[1, 0].scatter(leader_t, leader_v, label='Original Leader', color='blue', s=10, alpha=0.5)
axs[1, 0].scatter(follower_t, follower_v, label='Original Follower', color='red', s=10, alpha=0.5)
axs[1, 0].plot(leader_t, leader_v_smooth_wv, label='Smoothed Leader (Wavelet)', color='blue')
axs[1, 0].plot(follower_t, follower_v_smooth_wv, label='Smoothed Follower (Wavelet)', color='red')
axs[1, 0].set_title('Wavelet Denoising')
axs[1, 0].set_xlabel('Time (s)')
axs[1, 0].set_ylabel('Speed (m/s)')
axs[1, 0].grid(True)

# Kalman Filter plot
axs[1, 1].scatter(leader_t, leader_v, label='Original Leader', color='blue', s=10, alpha=0.5)
axs[1, 1].scatter(follower_t, follower_v, label='Original Follower', color='red', s=10, alpha=0.5)
axs[1, 1].plot(leader_t, leader_v_smooth_kf, label='Smoothed Leader (KF)', color='blue')
axs[1, 1].plot(follower_t, follower_v_smooth_kf, label='Smoothed Follower (KF)', color='red')
axs[1, 1].set_title('Kalman Filter')
axs[1, 1].set_xlabel('Time (s)')
axs[1, 1].set_ylabel('Speed (m/s)')
axs[1, 1].grid(True)

# Legend outside the plot
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.1))
# handles, labels = axs[0, 0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.1, -0.05))


plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()
