import numpy as np
import torch
from scipy.stats import skew, kurtosis
from scipy import signal
import pywt


def get_data(ecg_data_path = r'ECG.txt', label_path = r'label_profusion.txt'):
    """
    返回:
    ECG_data:  len(ECG_data) = 4065000 = 125 * 30 * 1084, 每个浮点数代表一个时间点上的心脏电压值。
    sleep_labels:  len(sleep_labels) = 1084, 代表ECG每个epoch对应的睡眠分期标签, 其中睡眠分期标签的映射如下":"w:0 ;N1:1; N2:2;N3:3;R:4;
    """
    ECG_data = open(ecg_data_path).read()         #设立data列表变量，python 文件流，.read读文件
    ECG_data = ECG_data.split( )                         #以空格为分隔符，返回数值列表data
    ECG_data = [float(s) for s in ECG_data]              #将列表data中的数值强制转换为float类型
    # len(ECG_data) = 4065000 = 125 * 30 * 1084

    sleep_labels = open(label_path).read()         #设立data列表变量，python 文件流，.read读文件
    sleep_labels = sleep_labels.split( )                         #以空格为分隔符，返回数值列表data
    sleep_labels = [int(s) for s in sleep_labels]              #将列表data中的数值强制转换为float类型
    # len(sleep_labels) = 1084
    return ECG_data, sleep_labels

import numpy as np
from scipy import signal

def rawdata_remove_noise(ecg_data, sampling_rate=125):
    # 使用滤波器去除高频噪声
    nyquist = 0.5 * sampling_rate
    low = 0.25
    high = 100.0
    low_cutoff = low / nyquist
    high_cutoff = min(high, 0.499 * sampling_rate) / nyquist
    b, a = signal.butter(1, [low_cutoff, high_cutoff], btype='band')
    filtered_ecg = signal.filtfilt(b, a, ecg_data)
    return filtered_ecg

def rawdata_normalize(ecg_data):
    # 归一化数据，将数据缩放到 [0, 1] 范围
    normalized_ecg = (((ecg_data - np.min(ecg_data)) / (np.max(ecg_data) - np.min(ecg_data))) - 0.5) * 2
    return normalized_ecg

def rawdata_z_score_normalize(ecg_data):
    mean = np.mean(ecg_data)
    std = np.std(ecg_data)
    normalized_ecg = (ecg_data - mean) / std
    return normalized_ecg

def feature_minmax_normalize(features):
    min_vals = features.min(axis=0, keepdims=True)
    max_vals = features.max(axis=0, keepdims=True)
    normalized_features = (features - min_vals) / (max_vals - min_vals)
    return normalized_features

def feature_z_score_normalize(features):
    mean_vals = features.mean(axis=0, keepdims=True)
    std_devs = features.std(axis=0, keepdims=True)
    normalized_features = (features - mean_vals) / std_devs
    return normalized_features

def calculate_slope(x, y):
    N = len(x)
    sum_xy = np.sum(x * y)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x_squared = np.sum(x**2)

    slope = (N * sum_xy - sum_x * sum_y) / (N * sum_x_squared - sum_x**2)

    return slope

def extract_features(raw_signal):
    features = []
    
    # 时域特征
    features.append(np.mean(raw_signal))
    features.append(np.std(raw_signal))
    features.append(np.min(raw_signal))
    features.append(np.max(raw_signal))
    features.append(calculate_slope(np.array(list(range(1,125*30+1,1)))/1000,raw_signal))  # 斜度2
    # features.append((raw_signal[-1] - raw_signal[0]) / len(raw_signal))  # 斜度
    features.append(np.sum(np.square(raw_signal)))  # 能量
    
    # 频域特征
    f, Pxx = signal.periodogram(raw_signal)
    features.append(np.sum(Pxx))  # 功率谱密度总和
    
    # 小波变换系数
    coeffs = pywt.wavedec(raw_signal, 'db1', level=5)
    for c in coeffs:
        features.extend([np.mean(c), np.std(c), skew(c), kurtosis(c)])
    
    return features

def get_features(ECG_data):
    """ECG_data's length is 4065000"""
    feature_list = []

    # 对每个epoch提取特征
    for i in range(len(ECG_data) // 125 // 30):
        feature_list.append(extract_features(ECG_data[i*125*30:(i+1)*125*30]))
    
    return feature_list