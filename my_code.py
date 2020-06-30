# importing libraries 
import matplotlib.pyplot as plt
import numpy as np
import random
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dense, Input 

#Loading data
(X_train, _), (X_test, _) = mnist.load_data()

#Normalizing
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255.0
X_test /= 255.0

#Reshaping
aspet_ratio = 28
X_train = np.reshape(X_train, (len(X_train), aspet_ratio, aspet_ratio, 1))
X_test = np.reshape(X_test, (len(X_test), aspet_ratio, aspet_ratio, 1))

#Adding noise 
noise = 0.5
x_train_noisy = X_train + noise * np.random.normal(loc = 0.0, scale = 1.0, size = X_train.shape) 
x_test_noisy = X_test + noise * np.random.normal(loc = 0.0, scale = 1.0, size = X_test.shape) 

# clip to be in (low_limit, upper_limit)
a_min = 0.0
a_max = 1.0
x_train_noisy = np.clip(x_train_noisy, a_min, a_max)
x_test_noisy = np.clip(x_test_noisy, a_min, a_max)

#Creating encoding and decoding layers
input_img = Input(shape=(aspet_ratio, aspet_ratio, 1))

def build_encoding_layers(input_img):
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    return encoded

def build_decoding_layers(encoding_layers):
    x = Conv2D(32, (3, 3), activation = 'relu', padding='same')(encoding_layers)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation = 'relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation = 'sigmoid', padding = 'same')(x)
    return decoded

#Making set of pictures with and without noise
encoded = build_encoding_layers(input_img)
decoded = build_decoding_layers(encoded)

#Creating model
autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)
autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')

#Fitting
autoencoder.fit(x_train_noisy, X_train, epochs=1, shuffle=True)

#Predicting test 
encoded_imgs = encoder.predict(x_test_noisy)
decoded_imgs = autoencoder.predict(x_test_noisy)

#Visualization
def visualize(images, amount, total_rows, current_row, new_shape, figsize=None):
    for i in range(1, amount):
        if figsize is not None: plt.figure(figsize=figsize)
        ax1 = plt.subplot(total_rows, amount, i + amount*current_row)
        plt.imshow(images[i].reshape(new_shape))
        plt.gray()
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)

amount = 10
plt.figure(figsize=(15, 9))
visualize(images= X_test, amount= amount, total_rows= 3, current_row= 0, new_shape= (aspet_ratio, aspet_ratio))
visualize(images= x_test_noisy, amount= amount, total_rows= 3, current_row= 1, new_shape= (aspet_ratio, aspet_ratio))
visualize(images= decoded_imgs, amount= amount, total_rows= 3, current_row= 2, new_shape= (aspet_ratio, aspet_ratio))
plt.show()

