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
import os
import random
import utils




class loadData():

    def __init__(self,dataDirectory):
        self.dataDirectory = dataDirectory

    def load_data(self,filename):
        data = np.load(filename)
        return data

# functie ce returneaza datele din fisiere pentru 3 clase
    def loadData_threeClasses(self):
        dataStore = []
        labels = []
        for filename in os.listdir(self.dataDirectory):
            if filename.endswith('.npy'):
                file_data = self.load_data(os.path.join(self.dataDirectory, filename))
                parts = filename.split('_')
                print(parts)
                greutate = parts[2]
                if greutate == "greutateusoara.npy":
                    channel1 = file_data[0].astype(int)-128
                    channel2 = file_data[1].astype(int)-128
                    channel3 = file_data[2].astype(int)-128
                    channel4 = file_data[3].astype(int)-128
                    lis = [channel1[5120:20480], channel2[5120:20480], channel3[5120:20480], channel4[5120:20480]]
                    dataStore.append(lis)
                    labels.append(0)
                elif greutate == "greutatemedie.npy":
                    channel1 = file_data[0].astype(int)-128
                    channel2 = file_data[1].astype(int)-128
                    channel3 = file_data[2].astype(int)-128
                    channel4 = file_data[3].astype(int)-128
                    lis = [channel1[5120:20480], channel2[5120:20480], channel3[5120:20480], channel4[5120:20480]]
                    dataStore.append(lis)
                    labels.append(1)
                elif greutate == "greutategrea.npy":
                    channel1 = file_data[0].astype(int)-128
                    channel2 = file_data[1].astype(int)-128
                    channel3 = file_data[2].astype(int)-128
                    channel4 = file_data[3].astype(int)-128
                    lis = [channel1[5120:20480], channel2[5120:20480], channel3[5120:20480], channel4[5120:20480]]
                    dataStore.append(lis)
                    labels.append(2)
        return dataStore,labels
    

    def loadData_twoClasses(self):
        dataStore = []
        labels = []
        for filename in os.listdir(self.dataDirectory):
            if filename.endswith('.npy'):
                file_data = self.load_data(os.path.join(self.dataDirectory, filename))
                parts = filename.split('_')
                print(parts)
                greutate = parts[2]
                if greutate == "greutategrea.npy":
                    channel1 = file_data[0].astype(int)-128
                    channel2 = file_data[1].astype(int)-128
                    channel3 = file_data[2].astype(int)-128
                    channel4 = file_data[3].astype(int)-128
                    lis = [channel1[:7680], channel2[:7680], channel3[:7680], channel4[:7680]]  # 0s to 15s non fatigue 2
                    dataStore.append(lis)
                    labels.append(0)
                    lis = [channel1[7680:15360], channel2[7680:15360], channel3[7680:15360], channel4[7680:15360]]
                    dataStore.append(lis)
                    labels.append(1)
                
                
                
        return dataStore,labels
    

    def loadData_sixClasses(self):
        dataStore = []
        labels = []
        for filename in os.listdir(self.dataDirectory):
            if filename.endswith('.npy'):
                file_data = self.load_data(os.path.join(self.dataDirectory, filename))
                parts = filename.split('_')
                print(parts)
                greutate = parts[2]
                if greutate == "greutateusoara.npy":
                    channel1 = file_data[0].astype(int)-128
                    channel2 = file_data[1].astype(int)-128
                    channel3 = file_data[2].astype(int)-128
                    channel4 = file_data[3].astype(int)-128
                    lis = [channel1[:7680], channel2[:7680], channel3[:7680], channel4[:7680]]  # 0s to 15s non fatigue 0
                    dataStore.append(lis)
                    labels.append(0)
                    lis = [channel1[20480:28160], channel2[20480:28160], channel3[20480:28160], channel4[20480:28160]]  # 40s to 55s fatigue 1
                    dataStore.append(lis)
                    labels.append(1)
                elif greutate == "greutatemedie.npy":
                    channel1 = file_data[0].astype(int)-128
                    channel2 = file_data[1].astype(int)-128
                    channel3 = file_data[2].astype(int)-128
                    channel4 = file_data[3].astype(int)-128
                    lis = [channel1[:7680], channel2[:7680], channel3[:7680], channel4[:7680]]  # 0s to 15s non fatigue 2
                    dataStore.append(lis)
                    labels.append(2)
                    lis = [channel1[12800:20480], channel2[12800:20480], channel3[12800:20480], channel4[12800:20480]]  # 25s to 40s fatigue 3
                    dataStore.append(lis)
                    labels.append(3)
                elif greutate == "greutategrea.npy":
                    channel1 = file_data[0].astype(int)-128
                    channel2 = file_data[1].astype(int)-128
                    channel3 = file_data[2].astype(int)-128
                    channel4 = file_data[3].astype(int)-128
                    lis = [channel1[:7680], channel2[:7680], channel3[:7680], channel4[:7680]]  # 0s to 15s non fatigue 4
                    dataStore.append(lis)
                    labels.append(4)
                    lis = [channel1[7680:15360], channel2[7680:15360], channel3[7680:15360], channel4[7680:15360]]  # 15s to 30s fatigue 5
                    dataStore.append(lis)
                    labels.append(5)
        return dataStore,labels
    




    def loadData_twoClasses_leg(self):
        dataStore = []
        labels = []
        for filename in os.listdir(self.dataDirectory):
            if filename.endswith('.npy'):
                file_data = self.load_data(os.path.join(self.dataDirectory, filename))
                parts = filename.split('_')
                print(parts)
                cl = parts[2]
                if cl == "3":
                    channel1 = file_data[0].astype(int)-128
                    channel2 = file_data[1].astype(int)-128
                    channel3 = file_data[2].astype(int)-128
                    channel4 = file_data[3].astype(int)-128
                    lis = [channel1[:20480], channel2[:20480], channel3[:20480], channel4[:20480]]    # 0 to 40s
                    dataStore.append(lis)
                    labels.append(0)
                    lis = [channel1[20480:40960], channel2[20480:40960], channel3[20480:40960], channel4[20480:40960]]    # 40 to 80s
                    dataStore.append(lis)
                    labels.append(1)
        return dataStore,labels

    def loadData_twoClasses_firstarmmovement(self):
        dataStore = []
        labels = []
        for filename in os.listdir(self.dataDirectory):
            if filename.endswith('.npy'):
                file_data = self.load_data(os.path.join(self.dataDirectory, filename))
                parts = filename.split('_')
                print(parts)
                cl = parts[2]
                if cl == "0":
                    channel1 = file_data[0].astype(int)-128
                    channel2 = file_data[1].astype(int)-128
                    channel3 = file_data[2].astype(int)-128
                    channel4 = file_data[3].astype(int)-128
                    channel5 = file_data[4].astype(int)-128
                    channel6 = file_data[5].astype(int)-128
                    channel7 = file_data[6].astype(int)-128
                    channel8 = file_data[7].astype(int)-128
                    lis = [channel1[:15360], channel2[:15360], channel3[:15360], channel4[:15360], channel5[:15360], channel6[:15360], channel7[:15360], channel8[:15360]]    # 0s to 30s
                    dataStore.append(lis)
                    labels.append(0)
                    lis = [channel1[15360:], channel2[15360:], channel3[15360:], channel4[15360:], channel5[15360:], channel6[15360:], channel7[15360:], channel8[15360:]]    # 30s to 60s
                    dataStore.append(lis)
                    labels.append(1)
        return dataStore,labels

    def loadData_twoClasses_secondarmmovement(self):
        dataStore = []
        labels = []
        for filename in os.listdir(self.dataDirectory):
            if filename.endswith('.npy'):
                file_data = self.load_data(os.path.join(self.dataDirectory, filename))
                parts = filename.split('_')
                print(parts)
                cl = parts[2]
                if cl == "1":
                    channel1 = file_data[0].astype(int)-128
                    channel2 = file_data[1].astype(int)-128
                    channel3 = file_data[2].astype(int)-128
                    channel4 = file_data[3].astype(int)-128
                    channel5 = file_data[4].astype(int)-128
                    channel6 = file_data[5].astype(int)-128
                    channel7 = file_data[6].astype(int)-128
                    channel8 = file_data[7].astype(int)-128
                    lis = [channel1[5120:15360], channel2[5120:15360], channel3[5120:15360], channel4[5120:15360], channel5[5120:15360], channel6[5120:15360], channel7[5120:15360], channel8[5120:15360]]    # 10s to 30s
                    dataStore.append(lis)
                    labels.append(0)
                    lis = [channel1[20480:], channel2[20480:], channel3[20480:], channel4[20480:], channel5[20480:], channel6[20480:], channel7[20480:], channel8[20480:]]    # 40s to 60s
                    dataStore.append(lis)
                    labels.append(1)
        return dataStore,labels
    
    def loadData_twoClasses_thirdarmmovement(self):
        dataStore = []
        labels = []
        for filename in os.listdir(self.dataDirectory):
            if filename.endswith('.npy'):
                file_data = self.load_data(os.path.join(self.dataDirectory, filename))
                parts = filename.split('_')
                print(parts)
                cl = parts[2]
                if cl == "2":
                    channel1 = file_data[0].astype(int)-128
                    channel2 = file_data[1].astype(int)-128
                    channel3 = file_data[2].astype(int)-128
                    channel4 = file_data[3].astype(int)-128
                    channel5 = file_data[4].astype(int)-128
                    channel6 = file_data[5].astype(int)-128
                    channel7 = file_data[6].astype(int)-128
                    channel8 = file_data[7].astype(int)-128
                    lis = [channel1[5120:15360], channel2[5120:15360], channel3[5120:15360], channel4[5120:15360], channel5[5120:15360], channel6[5120:15360], channel7[5120:15360], channel8[5120:15360]]    # 10s to 30s
                    dataStore.append(lis)
                    labels.append(0)
                    lis = [channel1[20480:], channel2[20480:], channel3[20480:], channel4[20480:], channel5[20480:], channel6[20480:], channel7[20480:], channel8[20480:]]    # 40s to 60s
                    dataStore.append(lis)
                    labels.append(1)
        return dataStore,labels
    


    def loadData_armthreeClasses(self):
        dataStore = []
        labels = []
        for filename in os.listdir(self.dataDirectory):
            if filename.endswith('.npy'):
                file_data = self.load_data(os.path.join(self.dataDirectory, filename))
                parts = filename.split('_')
                print(parts)
                cl = parts[2]
                if cl == "0":
                    channel1 = file_data[0].astype(int)-128
                    channel2 = file_data[1].astype(int)-128
                    channel3 = file_data[2].astype(int)-128
                    channel4 = file_data[3].astype(int)-128
                    channel5 = file_data[4].astype(int)-128
                    channel6 = file_data[5].astype(int)-128
                    channel7 = file_data[6].astype(int)-128
                    channel8 = file_data[7].astype(int)-128
                    #lis = [utils.emg_bandpass_filter(channel1,[20, 400],5000), utils.emg_bandpass_filter(channel2,[20, 400],5000), utils.emg_bandpass_filter(channel3,[20, 400],5000), utils.emg_bandpass_filter(channel4,[20, 400],5000), utils.emg_bandpass_filter(channel5,[20, 400],5000), utils.emg_bandpass_filter(channel6,[20, 400],5000), utils.emg_bandpass_filter(channel7,[20, 400],5000), utils.emg_bandpass_filter(channel8,[20, 400],5000)]
                    #lis = [utils.ampl_filter(channel1,30),utils.ampl_filter(channel2,30),utils.ampl_filter(channel3,30),utils.ampl_filter(channel4,30),utils.ampl_filter(channel5,30),utils.ampl_filter(channel6,30),utils.ampl_filter(channel7,30),utils.ampl_filter(channel8,30)]
                    lis = [channel1,channel2,channel3,channel4,channel5,channel6,channel7,channel8]

                    #random.shuffle(lis)
                    dataStore.append(lis)
                    labels.append(0)
                elif cl == "1":
                    channel1 = file_data[0].astype(int)-128
                    channel2 = file_data[1].astype(int)-128
                    channel3 = file_data[2].astype(int)-128
                    channel4 = file_data[3].astype(int)-128
                    channel5 = file_data[4].astype(int)-128
                    channel6 = file_data[5].astype(int)-128
                    channel7 = file_data[6].astype(int)-128
                    channel8 = file_data[7].astype(int)-128
                    #lis = [utils.emg_bandpass_filter(channel1,[20, 400],5000), utils.emg_bandpass_filter(channel2,[20, 400],5000), utils.emg_bandpass_filter(channel3,[20, 400],5000), utils.emg_bandpass_filter(channel4,[20, 400],5000), utils.emg_bandpass_filter(channel5,[20, 400],5000), utils.emg_bandpass_filter(channel6,[20, 400],5000), utils.emg_bandpass_filter(channel7,[20, 400],5000), utils.emg_bandpass_filter(channel8,[20, 400],5000)]
                    lis = [channel1,channel2,channel3,channel4,channel5,channel6,channel7,channel8]
                    #lis = [utils.ampl_filter(channel1,30),utils.ampl_filter(channel2,30),utils.ampl_filter(channel3,30),utils.ampl_filter(channel4,30),utils.ampl_filter(channel5,30),utils.ampl_filter(channel6,30),utils.ampl_filter(channel7,30),utils.ampl_filter(channel8,30)]

                    #random.shuffle(lis)
                    dataStore.append(lis)
                    labels.append(1)
                elif cl == "2":
                    channel1 = file_data[0].astype(int)-128
                    channel2 = file_data[1].astype(int)-128
                    channel3 = file_data[2].astype(int)-128
                    channel4 = file_data[3].astype(int)-128
                    channel5 = file_data[4].astype(int)-128
                    channel6 = file_data[5].astype(int)-128
                    channel7 = file_data[6].astype(int)-128
                    channel8 = file_data[7].astype(int)-128
                    #lis = [utils.emg_bandpass_filter(channel1,[20, 400],5000), utils.emg_bandpass_filter(channel2,[20, 400],5000), utils.emg_bandpass_filter(channel3,[20, 400],5000), utils.emg_bandpass_filter(channel4,[20, 400],5000), utils.emg_bandpass_filter(channel5,[20, 400],5000), utils.emg_bandpass_filter(channel6,[20, 400],5000), utils.emg_bandpass_filter(channel7,[20, 400],5000), utils.emg_bandpass_filter(channel8,[20, 400],5000)]
                    lis = [channel1,channel2,channel3,channel4,channel5,channel6,channel7,channel8]
                    #lis = [utils.ampl_filter(channel1,30),utils.ampl_filter(channel2,20),utils.ampl_filter(channel3,30),utils.ampl_filter(channel4,30),utils.ampl_filter(channel5,30),utils.ampl_filter(channel6,30),utils.ampl_filter(channel7,30),utils.ampl_filter(channel8,30)]

                    #random.shuffle(lis)
                    dataStore.append(lis)
                    labels.append(2)
        return dataStore,labels
    

    
    



    def loadData_elbow(self):
        dataStore = []
        labels = []
        for filename in os.listdir(self.dataDirectory):
            if filename.endswith('.txt'):
                

                file_data = os.path.join(self.dataDirectory, filename)
                parts = filename.split('_')
                print(parts)
                cl = parts[3]

                if cl == "0.txt":
                    channel1 = []
                    channel2 = []
                    with open(file_data, 'r') as file:
                        for line in file.readlines():
            
                            values = line.split()
                            channel1.append(float(values[0]))
                            channel2.append(float(values[1]))
                    
                    lis = [np.array(channel1),np.array(channel2)]
                    dataStore.append(lis)
                    labels.append(0)
                elif cl == "1360.txt":
                    channel1 = []
                    channel2 = []
                    with open(file_data, 'r') as file:
                        for line in file.readlines():
            
                            values = line.split()
                            channel1.append(float(values[0]))
                            channel2.append(float(values[1]))
                            
                    
                    lis = [np.array(channel1),np.array(channel2)]
                    dataStore.append(lis)
                    labels.append(1)
                elif cl == "2270.txt":
                    channel1 = []
                    channel2 = []
                    with open(file_data, 'r') as file:
                        for line in file.readlines():
            
                            values = line.split()
                            channel1.append(float(values[0]))
                            channel2.append(float(values[1]))
                    
                    lis = [np.array(channel1),np.array(channel2)]
                    dataStore.append(lis)
                    labels.append(2)
        return dataStore,labels
    


    def loadData_elbow_jasa(self):
        dataStore = []
        labels = []
        for filename in os.listdir(self.dataDirectory):
            if filename.endswith('.txt'):
                

                file_data = os.path.join(self.dataDirectory, filename)
                parts = filename.split('_')
                print(parts)
                cl = parts[3]

                if cl == "2270.txt":
                    channel1 = []
                    with open(file_data, 'r') as file:
                        for line in file.readlines():
            
                            values = line.split()
                            channel1.append(float(values[0]))
                    
                    lis = np.array(channel1)
                    dataStore.append(lis)
        return dataStore

