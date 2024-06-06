import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.signal import stft
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from scipy.signal import butter, lfilter, cheby1
import pywt
import sklearn	
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score	
from scipy.stats import skew,kurtosis
import scipy.signal
from scipy.signal import welch,periodogram
from scipy.integrate import quad




def MAV(data):
    spectral_amplitude = np.abs(np.fft.fft(data))
    return np.mean(np.abs(spectral_amplitude))

# zero crossing rate
def ZCR(data):
	return ((data[:-1] * data[1:]) < 0).sum()


#waveform length
def WL(data):
	return  (abs(data[:-1]-data[1:])).sum()


#root mean square
def RMS(data):
    return np.sqrt(np.mean(data**2))
	
#slope sign changes
def SSC(data, alpha):
	return  ((data[:-1]-data[1:])*(data[:1]-data[1:]) > alpha).sum()

def Energy(data):
    spectral_amplitude = np.abs(np.fft.fft(data))
    spectral_energy = np.sum(spectral_amplitude)
    return skew(spectral_amplitude)

def spectralPower(data):
    spectrum = np.abs(np.fft.fft(data))**2
    total_power = np.sum(spectrum)
    return total_power

def HJ(data):
    fft_result = np.fft.fft(data)
    spectral_amplitude = np.abs(fft_result)
    dominant_frequencies_indices = np.argsort(spectral_amplitude)[::-1][:512]
    dominant_frequencies_values = np.fft.fftfreq(len(data))[dominant_frequencies_indices]
    spectral_energy = np.sum(spectral_amplitude)

    bandwidth = np.sum(spectral_amplitude[dominant_frequencies_indices]) / spectral_energy
    return bandwidth
def Skewness(data):
    spectral_amplitude = np.abs(np.fft.fft(data))
    return np.std(spectral_amplitude)


def ftj(signal):
    cutoff = 30
    order = 10
    fs = 512

    b, a = butter(order, cutoff / (0.5 * fs), btype='low')

    filtered_signal = lfilter(b, a, signal)

    return filtered_signal

def wavelet(signal):
    fs = 512  # Frecvență de eșantionare (Hz)
    T = 2  # Durata semnalului (secunde)
    t = np.arange(0, T, 1/fs)  # Vector de timp

    # Creați un wavelet mexican hat (Ricker wavelet)
    wavelet = pywt.ContinuousWavelet('mexh')

    # Generați coeficienții wavelet pentru semnal
    coeffs,_ = pywt.cwt(signal, scales=np.arange(1, 25), wavelet=wavelet)

    features = []
    
    # Calcularea energiei totale în coeficienții wavelet
    total_energy = np.sum(np.abs(coeffs) ** 2)

    peaks = np.max(coeffs, axis=1)
    
    
    
    
    return kurtosis(coeffs)



def svm_classifier_function(X, Y):
	
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    svm_classifier = SVC(kernel='linear', C=1)


    svm_classifier.fit(X_train, y_train)


    y_pred = svm_classifier.predict(X_test)


    accuracy = accuracy_score(y_test, y_pred)
    print(f'Acuratețe: {accuracy * 100:.2f}%')
	



def emg_bandpass_filter(data: np.ndarray, edges: list[float], sample_rate: float, poles: int = 5):
    sos = scipy.signal.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data


def ampl_filter(data,threshold):
    for i in range(len(data)):
        if data[i] < threshold:
            data[i] = 0
    return data


def MNF(signal):
    # Calculează densitatea spectrală de putere (PSD) folosind metoda Welch
    fs = 512
    f, Pxx = periodogram(signal, fs=fs, window='hann')

    # Calculează Mean Frequency (MNF)
    mnf = np.sum(f * Pxx) / np.sum(Pxx)

    return mnf



def iemg(data):
    spectral_amplitude = np.abs(np.fft.fft(data))
    spectral_energy = np.sum(spectral_amplitude)
    return kurtosis(spectral_amplitude)


