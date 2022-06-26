import os
import numpy as np
import pandas as pd
import math
import csv
import sys
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Input, Dense, Lambda, Layer, Activation
from keras.layers.normalization.batch_normalization_v1 import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras import metrics, optimizers
from keras.callbacks import Callback
import keras

def sampling(args):
    z_mean, z_log_var = args

    epsilon = K.random_normal(shape=K.shape(z_mean), mean=0., stddev=1.0)

    z = z_mean + K.exp(z_log_var / 2) * epsilon
    return z

def vae_loss(x_input, x_decoded):
    reconstruction_loss = original_dim * metrics.mse(x_input, x_decoded)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    print(K.get_value(beta))
    return K.mean(reconstruction_loss + (K.get_value(beta) * kl_loss))

def reconstruction_loss(x_input, x_decoded):
    return metrics.mse(x_input, x_decoded)

def kl_loss(x_input, x_decoded):
    return - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)


class WarmUpCallback(Callback):
    def __init__(self, beta, kappa):
        self.beta = beta
        self.kappa = kappa

    # Behavior on each epoch
    def on_epoch_end(self, epoch, logs={}):
        if K.get_value(self.beta) <= 1:
            K.set_value(self.beta, K.get_value(self.beta) + self.kappa)

input_filename = "~.csv"
output_filename = "result.csv"
data_df = pd.read_csv(input_filename)
conditon_df = data_df.iloc[:,(len(data_df.columns)-1)]
input_df=data_df.iloc[:,1:(len(data_df.columns)-1)]

original_dim = input_df.shape[1]
intermediate1_dim = 120
intermediate2_dim = 60

latent_dim = 2  


batch_size = 20
learning_rate = 0.0005
beta = K.variable(1)
kappa = 0

test_data_size = 20
epochs = 10
fold_count = 5

# Separate data to training and test sets
input_df_training = input_df.iloc[:-1 * test_data_size, :]
input_df_test = input_df.iloc[-1 * test_data_size:, :]

print("INPUT DF")
print(input_df_training.shape)
print(input_df_training.index)
print("TEST DF")
print(input_df_test.shape)
print(input_df_test.index)

# Define encoder 
x = Input(shape=(original_dim,))
net = Dense(intermediate1_dim)(x)
net2 = BatchNormalization()(net)
net3 = Activation('relu')(net2)
net4 = Dense(intermediate2_dim)(net3)
net5 = BatchNormalization()(net4)
net6 = Activation('relu')(net5)
z_mean = Dense(latent_dim)(net6)
z_log_var = Dense(latent_dim)(net6)

# Sample from mean and var
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

decoder_h = Dense(intermediate2_dim, activation='relu')
decoder_h2 = Dense(intermediate1_dim, activation='relu')
decoder_mean = Dense(original_dim)

h_decoded = decoder_h(z)
h_decoded2 = decoder_h2(h_decoded)
x_decoded_mean = decoder_mean(h_decoded2)

# VAE model
vae = Model(x, x_decoded_mean)

adam = tf.keras.optimizers.Adam(lr=learning_rate)
vae.compile(optimizer=adam, loss=vae_loss, metrics=[reconstruction_loss, kl_loss])
vae.summary()

# Train from only training data
history = vae.fit(np.array(input_df_training), np.array(input_df_training),
                  shuffle=True,
                  epochs=epochs,
                  batch_size=batch_size,
                  verbose=2,
                  validation_data=(np.array(input_df_test), np.array(input_df_test)),
                  callbacks=[WarmUpCallback(beta, kappa)])
plt.figure(figsize=(11,11))
plt.rc('font',family="Times New Roman")
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('VAE Model Loss',fontsize=22)
plt.ylabel('loss',fontsize=20,fontweight="bold")
plt.xlabel('epoch',fontsize=20,fontweight="bold")
plt.legend(['train', 'test'], loc='upper left')
plt.yticks(fontproperties = 'Times New Roman', size = 16)
plt.xticks(fontproperties = 'Times New Roman', size = 16)
fig = plt.gcf()
fig.set_size_inches(14.5, 8.5)
plt.savefig('~',dpi=400)
plt.show()

plt.figure(figsize=(11,11))
plt.plot(history.history['reconstruction_loss'])
plt.plot(history.history['val_reconstruction_loss'])
plt.title('VAE Model Reconstruction Error',fontsize=22,fontweight="bold")
plt.ylabel('reconstruction error',fontsize=20,fontweight="bold")
plt.xlabel('epoch',fontsize=20,fontweight="bold")
plt.legend(['train', 'test'], loc='upper left')
plt.yticks(fontproperties = 'Times New Roman', size = 16)
plt.xticks(fontproperties = 'Times New Roman', size = 16)
fig = plt.gcf()
fig.set_size_inches(14.5, 8.5)
plt.savefig('~',dpi=400)
plt.show()

# DEFINE ENCODER 
encoder = Model(x, z_mean)

# SAVE THE ENCODER
from keras.models import model_from_json

# model_json = encoder.to_json()
# with open("encoder" + str(fold_count) + ".json", "w") as json_file:
#     json_file.write(model_json)
#
# encoder.save_weights("encoder" + str(fold_count) + ".h5")
from keras.models import save_model,load_model
encoder.save("encoder.h5")
print("Saved model to disk")

# DEFINE DECODER
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_h_decoded2 = decoder_h2(_h_decoded)
_x_decoded_mean = decoder_mean(_h_decoded2)
decoder = Model(decoder_input, _x_decoded_mean)

# Encode test data into the latent representation - and save output

train_encoded = encoder.predict(input_df_training, batch_size=batch_size)
train_encoded_df = pd.DataFrame(train_encoded, index=input_df_training.index)
test_encoded = encoder.predict(input_df_test, batch_size=batch_size)
test_encoded_df = pd.DataFrame(test_encoded, index=input_df_test.index)

# How well does the model reconstruct the input data
test_reconstructed = decoder.predict(np.array(test_encoded_df))
test_reconstructed_df = pd.DataFrame(test_reconstructed, index=input_df_test.index, columns=input_df_test.columns)

recons_error = mean_squared_error(np.array(input_df_test), np.array(test_reconstructed_df))

print("TEST RECONSTRUCTION ERROR: " + str(recons_error))

result_df = train_encoded_df.append(test_encoded_df)  
result_all = pd.concat([result_df,conditon_df], axis=1, join="inner")  
result_all.to_csv('~.csv')