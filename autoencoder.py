import keras
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense

input_size = 784
encoding_dim = 32

(X_train, _),(X_test, _) = mnist.load_data()

X_train = X_train.astype("float32") / 255.
X_test = X_test.astype("float32") / 255.

X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test),np.prod(X_test.shape[1:])))

print('X_train shape:',X_train.shape)
print('X_test shape:',X_test.shape)

input_img = Input(shape=(input_size,))

encoded = Dense(encoding_dim, activation='relu')(input_img)

decoded = Dense(input_size,activation='sigmoid')(encoded)

autoencoder = Model(inputs=input_img,outputs=decoded)

autoencoder.compile(optimizer='adam',loss='binary_crossentropy')

autoencoder.fit(X_train,X_train,epochs=20,batch_size=128,shuffle=True,validation_data=(X_test,X_test))

decoded_imgs = autoencoder.predict(X_test)

n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2,n,i+1)
    plt.imshow(X_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2,n,i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()