import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import fashion_mnist

# Load and normalize dataset
(x_train, _), (x_test, _) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape( (len(x_train), 28, 28, 1))
x_test = x_test.reshape( (len(x_test), 28, 28, 1))

# Add noise
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Build autoencoder
input_img = Input(shape=(28, 28, 1))

# Encoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)  # Specify filters, kernel_size, activation, and padding
x = MaxPooling2D((2, 2), padding='same')(x)  # Specify pool_size and padding
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)  # Specify filters, kernel_size, activation, and padding
encoded = MaxPooling2D((2, 2), padding='same')(x)  # Specify pool_size and padding

# Decoder
x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)  # Specify filters, kernel_size, activation, and padding
x = UpSampling2D((2, 2))(x)  # Specify size
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)  # Specify filters, kernel_size, activation, and padding
x = UpSampling2D((2, 2))(x)  # Specify size
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)  # Specify filters, kernel_size, activation, and padding

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train_noisy, x_train,
                epochs=5,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))

# Visualize
decoded_imgs = autoencoder.predict(x_test_noisy)
