# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 10:36:33 2019

@author: F1-PC
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 11:02:44 2019

@author: F1-PC
"""
def plot_history(net_history):
    history = net_history.history
    import matplotlib.pyplot as plt
    losses = history['loss']
    val_losses = history['val_loss']
    accuracies = history['acc']
    val_accuracies = history['val_acc']
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(losses)
    plt.plot(val_losses)
    plt.legend(['loss', 'val_loss'])
    
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(accuracies)
    plt.plot(val_accuracies)
    plt.legend(['acc', 'val_acc'])
    
import pandas 
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
from keras.optimizers import SGD
from keras.losses import categorical_crossentropy
 

i_path="D:/data/1N.xlsx"
i_i=pandas.ExcelFile(i_path)
df1=np.array(i_i.parse('Sheet2'))

X_train=df1[26150:58919,0:27]
Y_train=df1[26150:58919,27:29]

X_test=df1[9766:26150,0:27]
Y_test=df1[9766:26150,27:29]
X_valid=df1[0:9472,0:27]
Y_valid=df1[0:9472,27:29]

X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
X_test = X_test.astype('float32')
Y_train = Y_train.astype('float32')
Y_valid = Y_valid.astype('float32')
Y_test = Y_test.astype('float32')

data_dim = 16
timesteps = 8
nb_classes = 2
batch_size = 64
 

model = Sequential()
model.add(LSTM(60, return_sequences=True, stateful=True,activation='relu',batch_input_shape=(batch_size, 27, 1)))
model.add(LSTM(50, return_sequences=True, stateful=True, activation='relu'))
model.add(LSTM(40, return_sequences=True, stateful=True))
model.add(LSTM(30, return_sequences=True, stateful=True))
model.add(LSTM(20, stateful=True))
#model.add(Dense(10, activation='relu'))
#model.add(Dense(2, activation='sigmoid'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer=SGD(lr=0.001),loss = categorical_crossentropy ,metrics=['accuracy'])

X_train = X_train.reshape(32768,27,1)
X_test = X_test.reshape(16384,27,1)
X_valid = X_valid.reshape(9472,27,1)
network_history = model.fit(X_train, Y_train,batch_size=batch_size,nb_epoch=1500,validation_data=(X_valid, Y_valid)) 
#network_history = model.fit(X_train, Y_train,batch_size=batch_size,nb_epoch=1500,validation_split=0.2) 
print(batch_size)
plot_history(network_history)

# Evaluation
test_loss, test_acc = model.evaluate(X_test, Y_test,batch_size=batch_size)
#test_loss, test_acc = model.evaluate(X_valid, Y_valid ,batch_size=batch_size)
test_labels_p = model.predict(X_test,batch_size=batch_size)
print(test_labels_p)
import numpy as np
test_labels_p = np.argmax(test_labels_p, axis=1)

# Change layers config
model.layers[0].name = 'Layer_0'
model.layers[0].trainable = False
model.layers[0].get_config()
