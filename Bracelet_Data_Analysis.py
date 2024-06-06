import socket
import struct
import time
import numpy as np
import threading
import tkinter as tk
from tkinter import Entry, Button, Label, messagebox
from scipy.signal import welch,periodogram
from scipy.stats import linregress
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.signal import stft
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout,BatchNormalization,LSTM,Attention, Conv2D, MaxPooling2D, SimpleRNN,TimeDistributed
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from scipy.signal import butter, lfilter, cheby1
import pywt
import sklearn		
from loadData import loadData 
from processData import processData  	
from tensorflow.keras.callbacks import ModelCheckpoint	
from tensorflow.keras.optimizers import Adam	
from tensorflow.keras.regularizers import l2
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score,classification_report



UDP_IP = "192.168.4.1"
UDP_PORT = 1234
BUFFER_SIZE = 1024
NUM_CHANNELS = 8

class BraceletInterface:
    def __init__(self, master):
        self.master = master
        master.title("Bracelet Data Analysis")

        self.start_time = 0
        self.elapsed_time = 0
        self.channels = [[] for _ in range(NUM_CHANNELS)]  
        self.mnf_values = [[] for _ in range(NUM_CHANNELS)]
        self.rms_values = [[] for _ in range(NUM_CHANNELS)]

        self.channel_labels = []
        for i in range(NUM_CHANNELS):
            label = Label(master, text=f"Channel {i+1}: RMS - MNF", padx=20, pady=10)
            label.pack()
            self.channel_labels.append(label)

        self.start_button = Button(master, text="Start", command=self.start_reading, padx=20, pady=10)
        self.start_button.pack()

        self.save_button = Button(master, text="Save", command=self.save_data, padx=20, pady=10)
        self.save_button.pack()

        self.filename_label = Label(master, text="Filename:", padx=20, pady=10)
        self.filename_label.pack()

        self.filename_entry = Entry(master)
        self.filename_entry.pack()
        self.filename_entry.insert(0, "bracelet_data")

        self.time_label = Label(master, text="Time (seconds):", padx=20, pady=10)
        self.time_label.pack()

        self.time_entry = Entry(master)
        self.time_entry.pack()
        self.time_entry.insert(0, "5")

        self.predict_button = Button(master, text="Predict", command=self.predict, padx=20, pady=10)
        self.predict_button.pack()

    def start_reading(self):
        self.start_time = time.time()
        self.elapsed_time = 0
        self.channels = [[] for _ in range(NUM_CHANNELS)]
        self.mnf_values = [[] for _ in range(NUM_CHANNELS)]
        self.rms_values = [[] for _ in range(NUM_CHANNELS)]

        threading.Thread(target=self.read_data).start()

    def read_data(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect((UDP_IP, UDP_PORT))
        start_cmd = "START".encode("utf-8")
        retry_cmd = "RETRANSMIT".encode("utf-8")

        sock.settimeout(0.002)

        while self.elapsed_time <= int(self.time_entry.get()):
            current_time = time.time()
            self.elapsed_time = current_time - self.start_time

            try:
                sock.send(start_cmd)
                data, addr = sock.recvfrom(BUFFER_SIZE)
                struct_format = f'{BUFFER_SIZE // NUM_CHANNELS}B'
                unpacked_data = struct.unpack(struct_format * NUM_CHANNELS, data)
                for i in range(NUM_CHANNELS):
                    self.channels[i].extend(unpacked_data[i * BUFFER_SIZE // NUM_CHANNELS:(i + 1) * BUFFER_SIZE // NUM_CHANNELS])

            except socket.timeout:
                print('2 x timeout')
                sock.send(retry_cmd)
                try:
                    data, addr = sock.recvfrom(BUFFER_SIZE)
                    struct_format = f'{BUFFER_SIZE // NUM_CHANNELS}B'
                    unpacked_data = struct.unpack(struct_format * NUM_CHANNELS, data)
                    for i in range(NUM_CHANNELS):
                        self.channels[i].extend(unpacked_data[i * BUFFER_SIZE // NUM_CHANNELS:(i + 1) * BUFFER_SIZE // NUM_CHANNELS])

                except:
                    print('2 x err')
                    pass

            time.sleep(0.001)  

            for i in range(NUM_CHANNELS):
                if self.channels[i]:
                    f, psd = periodogram(self.channels[i], fs=512)
                    mnf = np.sum(f * psd) / np.sum(psd)
                    rms = np.sqrt(np.mean(np.array(self.channels[i]) ** 2))
                    self.mnf_values[i].append(mnf)
                    self.rms_values[i].append(rms)
                    self.update_label(i, f"Channel {i+1}: RMS - {rms:.2f} | MNF - {mnf:.2f}")

            # Perform linear regression and check muscle fatigue
            for i in range(NUM_CHANNELS):
                if self.rms_values[i] and self.mnf_values[i]:
                    rms_slope, _, _, _, _ = linregress(range(len(self.rms_values[i])), self.rms_values[i])
                    mnf_slope, _, _, _, _ = linregress(range(len(self.mnf_values[i])), self.mnf_values[i])
                    if rms_slope > 0 and mnf_slope < 0:
                        self.update_label(i, self.channel_labels[i].cget("text") + " | The muscle is tired.")
                    else:
                        self.update_label(i, self.channel_labels[i].cget("text") + " | The muscle is not tired.")

        sock.close()

    def update_label(self, channel, text):
        self.channel_labels[channel].config(text=text)

    def save_data(self):
        filename = self.filename_entry.get() + ".npy"
        np.save(filename, np.array(self.channels))
        print(f"Data saved to {filename}")
    def loadData_predict(self,data):
        
        channel1 = data[0].astype(int)-128
        channel2 = data[1].astype(int)-128
        channel3 = data[2].astype(int)-128
        channel4 = data[3].astype(int)-128
        channel5 = data[4].astype(int)-128
        channel6 = data[5].astype(int)-128
        channel7 = data[6].astype(int)-128
        channel8 = data[7].astype(int)-128
        #lis = [utils.emg_bandpass_filter(channel1,[20, 400],5000), utils.emg_bandpass_filter(channel2,[20, 400],5000), utils.emg_bandpass_filter(channel3,[20, 400],5000), utils.emg_bandpass_filter(channel4,[20, 400],5000), utils.emg_bandpass_filter(channel5,[20, 400],5000), utils.emg_bandpass_filter(channel6,[20, 400],5000), utils.emg_bandpass_filter(channel7,[20, 400],5000), utils.emg_bandpass_filter(channel8,[20, 400],5000)]
        #lis = [utils.ampl_filter(channel1,30),utils.ampl_filter(channel2,30),utils.ampl_filter(channel3,30),utils.ampl_filter(channel4,30),utils.ampl_filter(channel5,30),utils.ampl_filter(channel6,30),utils.ampl_filter(channel7,30),utils.ampl_filter(channel8,30)]
        dataStore = [[channel1,channel2,channel3,channel4,channel5,channel6,channel7,channel8]]
        return dataStore
    def predict(self):
        filename = "bracelet_data.npy"
        try:
            data = np.load(filename)
            dataStore = self.loadData_predict(data)
            fs = 512
            labels = [0] 
            window_length = 2
            overlap = 1
            process_Data = processData(dataStore,labels,fs,overlap,window_length)
            X,Y = process_Data.extractArmFeatures()
            X = np.array(X,dtype=np.float32)
            Y = np.array(Y)
            shape = X.shape
            data_reshaped = X.reshape(-1, shape[-1])
            scaler = sklearn.preprocessing.StandardScaler()
            data_scaled = scaler.fit_transform(data_reshaped)
            X = data_scaled.reshape(shape)
            X_reshaped = X.reshape(X.shape[0], -1)
            X_test = X
            y_test = Y
            print(X_test[0].shape)
            model = tf.keras.models.load_model('final_model_8features_3classesarm.keras')
            y_pred = model.predict(X_test)
            print(y_pred)
            exer1 = 0
            exer2 = 0
            exer3 = 0
            for i in y_pred:
                if np.argmax(i) == 0:
                    exer1+=1
                elif np.argmax(i) == 1:
                    exer2+=1
                elif np.argmax(i) == 2:
                    exer3+=1
            rez = [exer1,exer2,exer3]
            max = np.argmax(rez)

            if max == 0:
                messagebox.showinfo("Prediction","Exercise #1")
            elif max == 1:
                messagebox.showinfo("Prediction","Exercise #2")
            elif max == 2:
                messagebox.showinfo("Prediction","Exercise #3")

        except FileNotFoundError:
            messagebox.showerror("Error", "File not found.")

if __name__ == "__main__":
    root = tk.Tk()
    app = BraceletInterface(root)
    root.mainloop()
