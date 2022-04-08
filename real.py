# 필요 라이브러리 선언
import sys
import scipy.io as sio
import scipy.io.wavfile
import librosa
import librosa.display
import pandas as pd
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from pylab import rcParams
import numpy as np
from scipy.signal import butter, lfilter, freqz, filtfilt
import matplotlib.pyplot as plt
rcParams['figure.figsize'] = 14, 6
from scipy import signal as sp
from scipy import signal
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
import tensorflow.keras as keras
import tensorflow.keras
from IPython.display import Audio, IFrame, display
import pygame
from scipy.io import wavfile

# 음성처리용 함수 선언
def wiener_filter(input):
    sr, data = sio.wavfile.read(input)
    sr = sr*2
    y1,sr = librosa.load(input, mono=True, sr=sr, offset=0, duration=10)
    
    # BPF 함수 선언
    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a
    def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y
    
    # 위너필터
    yw1 = sp.wiener(y1, mysize=33)
    
    # BPF 통과
    ywb1 = butter_bandpass_filter(yw1, 20, 1000, sr/2)
    
    
    # ann모델 선언 및 사용
    ft = librosa.stft(ywb1)
    ft_t = np.transpose(np.abs(ft))
    ft_t_in = np.copy(ft_t)
    ft_t_in[:,20:500] = 0
    ft_t_out = np.copy(ft_t)

    data_size = len(ft_t[0])

    layer_0_1 = keras.layers.Input(shape=(data_size,))
    layer_1_1 = keras.layers.Dense(500, activation='relu')(layer_0_1)
    layer_1_2 = keras.layers.Dropout(0.1)(layer_1_1)
    layer_2_1 = keras.layers.Dense(500, activation='sigmoid')(layer_1_2)
    layer_2_2 = keras.layers.Dropout(0.1)(layer_2_1)
    layer_0_and_2 = keras.layers.Concatenate()([layer_0_1, layer_2_2])
    layer_3_1 = keras.layers.Dense(500, activation='sigmoid')(layer_0_and_2)
    layer_1_and_3 = keras.layers.Concatenate()([layer_1_2, layer_3_1])
    layer_4_1 = keras.layers.Dense(500, activation='relu')(layer_1_and_3)
    layer_2_and_4 = keras.layers.Concatenate()([layer_2_2, layer_4_1])
    layer_5_1 = keras.layers.Dense(500, activation='relu')(layer_2_and_4)
    layer_3_and_5 = keras.layers.Concatenate()([layer_3_1, layer_5_1])
    layer_6_1 = keras.layers.Dense(500, activation='sigmoid')(layer_3_and_5)
    layer_4_and_6 = keras.layers.Concatenate()([layer_4_1, layer_6_1])
    layer_7_1 = keras.layers.Dense(500, activation='relu')(layer_4_and_6)
    layer_5_and_7 = keras.layers.Concatenate()([layer_5_1, layer_7_1])
    layer_8_1 = keras.layers.Dense(data_size, activation='relu')(layer_5_and_7)

    filler = keras.models.Model(layer_0_1, layer_8_1)
    filler.compile(optimizer='rmsprop', loss='logcosh')

    filler.fit(ft_t_in, ft_t_out,
               epochs=50, batch_size=100,
               shuffle=True, verbose=0,
               validation_data=(ft_t_in, ft_t_out))

    ft_t = np.transpose(np.abs(ft))
    ft_t_in = np.copy(ft_t)
    ft_t_in[:,20:500] = 0

    ft_ann_pred = np.transpose(filler.predict(ft_t_in))

    ft_ann = (np.abs(ft) - ft_ann_pred) * (ft / np.abs(ft))
    ywba1 = librosa.istft(ft_ann)

    # int형에 맞도록 정규화
    ywba1 = np.int16(ywba1/np.max(np.abs(ywba1)) * 32767)
    
    # 파일 쓰기
    wavfile.write('output.wav',sr,ywba1)
    return ywba1

# 함수 실행
wiener_filter(sys.argv[1])