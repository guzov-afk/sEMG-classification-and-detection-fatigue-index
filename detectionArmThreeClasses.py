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
import seaborn as sns



# antrenare
data_directory = 'C:/Users/Alexandru Guzu/Desktop/Disertatie_2024/data_2345/database'
load_Data = loadData(data_directory)
dataStore, labels = load_Data.loadData_armthreeClasses()




fs = 512 
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
X_train = X
y_train = Y



# validare 
data_directory = 'C:/Users/Alexandru Guzu/Desktop/Disertatie_2024/data_2345/valid_data'
load_Data = loadData(data_directory)
dataStore, labels = load_Data.loadData_armthreeClasses()



fs = 512 
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
X_valid = X
y_valid = Y

#testare
data_directory = 'C:/Users/Alexandru Guzu/Desktop/Disertatie_2024/data_2345/test_data'
load_Data = loadData(data_directory)
dataStore, labels = load_Data.loadData_armthreeClasses()
fs = 512 
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




# y_train = to_categorical(y_train, num_classes=3)
# y_valid = to_categorical(y_valid, num_classes=3)
# y_test = to_categorical(y_test, num_classes=3)


print(X_train.shape)
print(y_train.shape)

var = 5

scaler = sklearn.preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, var*8)) 
X_valid_scaled = scaler.fit_transform(X_valid.reshape(-1, var*8))
pca = PCA(n_components=3) 
X_train_pca = pca.fit_transform(X_train_scaled)

X_valid_pca = pca.fit_transform(X_valid_scaled)

# Plotează datele transformate PCA în 3D
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], X_train_pca[:, 2], c=y_train, cmap='viridis', alpha=0.7)

# Adăugarea de etichete și titluri
ax.set_title('Scatter plot of data transformed using PCA')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
plt.colorbar(scatter, label='Class')
plt.show()




print(X_train_pca.shape)
print(X_valid_pca.shape)

print(X_train_pca)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# # Redimensionați datele pentru a se potrivi cu forma de intrare a modelului KNN
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

# # Inițializați și antrenați modelul KNN
knn_model = KNeighborsClassifier(n_neighbors=500)  # Specificați numărul de vecini
knn_model.fit(X_train_flattened, y_train)

# # Evaluați modelul pe setul de testare
y_pred = knn_model.predict(X_test_flattened)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on test set: {accuracy}')

report = classification_report(y_test, y_pred)
print(report)



conf_matrix = confusion_matrix(y_test, y_pred)


plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('k-NN Confusion Matrix')
plt.show()



from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report

# Redimensionați datele pentru a se potrivi cu forma de intrare a modelului SVM
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

# Inițializați și antrenați modelul SVM
svm_model = SVC(C=10,kernel='rbf',gamma='auto')  # Specificați kernel-ul și alte parametri după necesități
svm_model.fit(X_train_flattened, y_train)

# Evaluați modelul pe setul de testare
y_pred = svm_model.predict(X_test_flattened)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on test set: {accuracy}')

report = classification_report(y_test, y_pred)
print(report)

conf_matrix = confusion_matrix(y_test, y_pred)


plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('SVM Confusion Matrix')
plt.show()



y_train = to_categorical(y_train, num_classes=3)
y_valid = to_categorical(y_valid, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)


model = Sequential()

# model.add(Conv2D(128, kernel_size=(2, 2), activation='relu', input_shape=(3, 8, 1)))
# model.add(MaxPooling2D(pool_size=(2, 2)))




model.add(Flatten(input_shape=(var,8,1)))


model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(16,activation='relu'))
model.add(Dense(3, activation='softmax'))  




model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


checkpoint = ModelCheckpoint('best_model_8features_3classesarm.h5', monitor='val_loss', save_best_only=True, mode='min')

history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_valid,y_valid), callbacks=[checkpoint])

loss = history.history['loss']
val_loss = history.history['val_loss']
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Plotarea pierderii
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotarea acurateței
plt.subplot(1, 2, 2)
plt.plot(accuracy, label='Train Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Afișarea graficelor
plt.tight_layout()
plt.show()

# Evaluarea modelului pe setul de testare
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss on test set: {loss}')
print(f'Accuracy on test set: {accuracy}')

# Salvarea modelului antrenat
model.save('final_model_8features_3classesarm.keras')

y_pred = model.predict(X_test)

y_test_classes = np.argmax(y_test, axis=1)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculați raportul de clasificare, care include acuratețea pentru fiecare clasă
report = classification_report(y_test_classes, y_pred_classes)
print(report)

conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)

# Vizualizați matricea de confuzie
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('MLP Confusion Matrix')
plt.show()






















