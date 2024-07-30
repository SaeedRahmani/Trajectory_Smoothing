import pickle
import matplotlib.pyplot as plt
import random
import numpy as np
import pywt
import numpy as np
import scipy.signal
from scipy.signal import butter, filtfilt


def remove_outliers(data, jump_threshold=10):
    smoothed_data = data.copy()
    for i in range(1, len(smoothed_data) - 2):
        if abs(smoothed_data[i] - smoothed_data[i-1]) > jump_threshold and \
           abs(smoothed_data[i] - smoothed_data[i+2]) > jump_threshold:
            smoothed_data[i] = (smoothed_data[i-1] + smoothed_data[i+2]) / 2
    return smoothed_data

# def detect_and_replace_outliers(data, threshold=1):
#     clean_data = data.copy()
#     for i in range(1, len(data)):
#         if abs(data[i] - data[i - 1]) > threshold:
#             clean_data[i] = np.median(data[max(0, i-5):min(len(data), i+5)])
#     return clean_data

# Load the provided pickle file
file_path = '/Users/srahmani/Downloads/lyft_all.pkl'

with open(file_path, 'rb') as file:
    data = pickle.load(file)

## For multiple trajectory pickle files
random_pair = random.choice(data)
leader_v = random_pair['leader_v']
leader_t = random_pair['leader_t']
follower_v = random_pair['follower_v']
follower_t = random_pair['follower_t']

### For one trajectory pickle file
# leader_v = data['leader_v']
# leader_t = data['leader_t']
# follower_v = data['follower_v']
# follower_t = data['follower_t']

### If you want to apply the outlier detection to all methods
# leader_v = remove_outliers(leader_v, jump_threshold=1)
# follower_v = remove_outliers(follower_v, jump_threshold=1)

# Function to apply a low-pass filter with outlier removal
def robust_low_pass_filter(data, cutoff=0.5, fs=10.0, order=4, jump_threshold=1):
    # Remove outliers
    data_no_outliers = remove_outliers(data, jump_threshold)
    # Apply low-pass filter
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data_no_outliers)
    return y

# Apply the robust low-pass filter to the leader and follower data
leader_v_smooth_lp = robust_low_pass_filter(leader_v, cutoff=0.5, fs=10.0, order=4, jump_threshold=1)
follower_v_smooth_lp = robust_low_pass_filter(follower_v, cutoff=0.5, fs=10.0, order=4, jump_threshold=1)

# Plotting the original data and the smoothed data using Low Pass Filter
plt.figure(figsize=(12, 6))

plt.scatter(leader_t, leader_v, label='Original Leader Trajectory', color='blue', s=10, alpha=0.5)
plt.scatter(follower_t, follower_v, label='Original Follower Trajectory', color='red', s=10, alpha=0.5)
plt.plot(leader_t, leader_v_smooth_lp, label='Smoothed Leader Trajectory (LPF)', color='blue')
plt.plot(follower_t, follower_v_smooth_lp, label='Smoothed Follower Trajectory (LPF)', color='red')

plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.title('Leader and Follower Trajectories Over Time (Robust Low Pass Filter)')
plt.legend()
plt.grid(True)
plt.show()


### Applying Savitzky-Golay filter with window size 51 and polynomial order 3
from scipy.signal import savgol_filter
leader_v_smooth_sg = savgol_filter(leader_v, 33, 3)
follower_v_smooth_sg = savgol_filter(follower_v, 33, 3)

plt.figure(figsize=(12, 6))
plt.scatter(leader_t, leader_v, label='Original Leader Trajectory', color='blue', s=10, alpha=0.5)
plt.scatter(follower_t, follower_v, label='Original Follower Trajectory', color='red', s=10, alpha=0.5)
plt.plot(leader_t, leader_v_smooth_sg, label='Smoothed Leader Trajectory (SG)', color='blue')
plt.plot(follower_t, follower_v_smooth_sg, label='Smoothed Follower Trajectory (SG)', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.title('Leader and Follower Trajectories Over Time (Savitzky-Golay)')
plt.legend()
plt.grid(True)
plt.show()


### Applying 3-sigma method to the data
import numpy as np
import matplotlib.pyplot as plt

def apply_3sigma_method(signal, window_size=30, threshold=0.15):
        """
        Apply 3-sigma smoothing to a signal.
        
        Parameters:
            - signal: The input signal to be smoothed.
            - window_size: The size of the smoothing window.
            - threshold: The threshold in terms of standard deviations.
        
        Returns:
            - smoothed_signal: The smoothed signal.
        """
        smoothed_signal = np.zeros_like(signal)
        for i in range(len(signal)):
            start = max(0, i - window_size//2)
            end = min(len(signal), i + window_size//2 + 1)
            window = signal[start:end]
            mean = np.mean(window)
            std = np.std(window)
            if np.abs(signal[i] - mean) > threshold * std:
                smoothed_signal[i] = mean
            else:
                smoothed_signal[i] = signal[i]
        return smoothed_signal

leader_v_smooth_3sigma = apply_3sigma_method(leader_v)
follower_v_smooth_3sigma = apply_3sigma_method(follower_v)

# Plotting the original data and the smoothed data using 3-sigma method
plt.figure(figsize=(12, 6))

plt.scatter(leader_t, leader_v, label='Original Leader Trajectory', color='blue', s=10, alpha=0.5)
plt.scatter(follower_t, follower_v, label='Original Follower Trajectory', color='red', s=10, alpha=0.5)
plt.plot(leader_t, leader_v_smooth_3sigma, label='Smoothed Leader Trajectory (3-Sigma)', color='blue')
plt.plot(follower_t, follower_v_smooth_3sigma, label='Smoothed Follower Trajectory (3-Sigma)', color='red')

plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.title('Leader and Follower Trajectories Over Time (3-Sigma Method)')
plt.legend()
plt.grid(True)
plt.show()

# Function to apply Wavelet Transform for denoising
def apply_wavelet_denoising(data, wavelet='db4', level=1):
    coeff = pywt.wavedec(data, wavelet, mode='symmetric')
    sigma = np.median(np.abs(coeff[-level])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(data)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='soft') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='symmetric')

# Apply Wavelet Denoising to the data
leader_v_smooth_wv = apply_wavelet_denoising(leader_v)
follower_v_smooth_wv = apply_wavelet_denoising(follower_v)

# Ensure that the lengths of the smoothed data and time arrays are equal
leader_v_smooth_wv = leader_v_smooth_wv[:len(leader_t)]
follower_v_smooth_wv = follower_v_smooth_wv[:len(follower_t)]

# Plotting the original data and the smoothed data using Wavelet Denoising
plt.figure(figsize=(12, 6))

plt.scatter(leader_t, leader_v, label='Original Leader Trajectory', color='blue', s=10, alpha=0.5)
plt.scatter(follower_t, follower_v, label='Original Follower Trajectory', color='red', s=10, alpha=0.5)
plt.plot(leader_t, leader_v_smooth_wv, label='Smoothed Leader Trajectory (Wavelet)', color='blue')
plt.plot(follower_t, follower_v_smooth_wv, label='Smoothed Follower Trajectory (Wavelet)', color='red')

plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.title('Leader and Follower Trajectories Over Time (Wavelet Denoising)')
plt.legend()
plt.grid(True)
plt.show()


# # Apply low pass filter to the data
# def apply_low_pass_filter(data, cutoff=0.5, fs=10.0, order=4):
#     nyquist = 0.5 * fs
#     normal_cutoff = cutoff / nyquist
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     y = filtfilt(b, a, data)
#     return y

# # Apply Low Pass Filter to the data
# leader_v_smooth_lp = apply_low_pass_filter(leader_v, cutoff=0.5, fs=10.0, order=4)
# follower_v_smooth_lp = apply_low_pass_filter(follower_v, cutoff=0.5, fs=10.0, order=4)

# # Plotting the original data and the smoothed data using Low Pass Filter
# plt.figure(figsize=(12, 6))

# plt.scatter(leader_t, leader_v, label='Original Leader Trajectory', color='blue', s=10, alpha=0.5)
# plt.scatter(follower_t, follower_v, label='Original Follower Trajectory', color='red', s=10, alpha=0.5)
# plt.plot(leader_t, leader_v_smooth_lp, label='Smoothed Leader Trajectory (LPF)', color='blue')
# plt.plot(follower_t, follower_v_smooth_lp, label='Smoothed Follower Trajectory (LPF)', color='red')

# plt.xlabel('Time (s)')
# plt.ylabel('Speed (m/s)')
# plt.title('Leader and Follower Trajectories Over Time (Low Pass Filter)')
# plt.legend()
# plt.grid(True)
# plt.show()

# Apply Kalman Filter 1
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter

# Function to apply Kalman Filter
def apply_kalman_filter(data):
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([0., 0.])  # Initial state (location and velocity)
    kf.F = np.array([[1., 1.],
                     [0., 1.]])  # State transition matrix
    kf.H = np.array([[1., 0.]])  # Measurement function
    kf.P *= 1000.  # Covariance matrix
    kf.R = 5  # Measurement noise
    kf.Q = 0.1  # Process noise

    filtered_data = []
    for z in data:
        kf.predict()
        kf.update(z)
        filtered_data.append(kf.x[0])

    return filtered_data

# Apply Kalman Filter to the data
leader_v_smooth_kf = apply_kalman_filter(leader_v)
follower_v_smooth_kf = apply_kalman_filter(follower_v)

# Plotting the original data and the smoothed data using Kalman Filter
plt.figure(figsize=(12, 6))

plt.scatter(leader_t, leader_v, label='Original Leader Trajectory', color='blue', s=10, alpha=0.5)
plt.scatter(follower_t, follower_v, label='Original Follower Trajectory', color='red', s=10, alpha=0.5)
plt.plot(leader_t, leader_v_smooth_kf, label='Smoothed Leader Trajectory (KF)', color='blue')
plt.plot(follower_t, follower_v_smooth_kf, label='Smoothed Follower Trajectory (KF)', color='red')

plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.title('Leader and Follower Trajectories Over Time (Kalman Filter 1)')
plt.legend()
plt.grid(True)
plt.show()

### Aplly Kalman Filter 2
from pykalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
import random

# Define the updated Kalman filter function
def apply_kalman_filter(z, t):
    # Ensure z and t have the same length
    min_len = min(len(z), len(t))
    z = z[:min_len]
    t = t[:min_len]

    # State transition model
    dt = np.diff(t)
    F = np.array([[1, 0], [0, 1]])
    
    # Measurement model
    H = np.array([[1, 0]])
    
    # Covariance of the process noise
    Q = np.array([[0.01, 0], [0, 0.01]])
    
    # Covariance of the observation noise
    R = np.array([[0.1]])
    
    # Initial state
    x = np.array([[z[0]], [0]])
    
    # Initial estimate covariance
    P = np.array([[1, 0], [0, 1]])
    
    # Storage for filtered states
    filtered_state = []
    
    for i in range(len(z)):
        if i > 0:
            # Predict
            F[0, 1] = dt[i-1]
            x = F @ x
            P = F @ P @ F.T + Q
        
        # Update
        y = z[i] - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(2) - K @ H) @ P
        
        filtered_state.append(x[0, 0])
    
    return np.array(filtered_state)

# Apply Kalman Filter to the data
leader_v_smooth_kf = apply_kalman_filter(leader_v, leader_t)
follower_v_smooth_kf = apply_kalman_filter(follower_v, follower_t)

# Plotting the original data and the smoothed data using Kalman Filter
plt.figure(figsize=(12, 6))

plt.scatter(leader_t, leader_v, label='Original Leader Trajectory', color='blue', s=10, alpha=0.5)
plt.scatter(follower_t, follower_v, label='Original Follower Trajectory', color='red', s=10, alpha=0.5)
plt.plot(leader_t, leader_v_smooth_kf, label='Smoothed Leader Trajectory (KF)', color='blue')
plt.plot(follower_t, follower_v_smooth_kf, label='Smoothed Follower Trajectory (KF)', color='red')

plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.title('Leader and Follower Trajectories Over Time (Kalman Filter 2)')
plt.legend()
plt.grid(True)
plt.show()

