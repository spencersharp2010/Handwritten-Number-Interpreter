import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras import regularizers

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
index = 7777
print(y_train[index])
plt.imshow(x_train[index], cmap='Greys')
plt.show()

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train: ', x_train.shape[0])
print('Number of images in x_test: ', x_test.shape[0])

# Creating a Sequential Model and adding the layers
model = Sequential()
#model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
#model.add(Dense(300, input_dim=x_train.shape[0], activation=tf.nn.sigmoid, kernel_initializer='random_uniform', bias_initializer='zeros'))
model.add(Dense(300, input_dim=x_train.shape[0], activation=tf.nn.sigmoid, kernel_initializer='random_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l1(0.00001)))
#model.add(Dropout(0.2))
#model.add(Dense(10,activation=tf.nn.softmax, kernel_initializer='random_uniform', bias_initializer='zeros' ))
model.add(Dense(10,activation=tf.nn.softmax, kernel_initializer='random_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l1(0.00001)))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x=x_train,y=y_train, epochs=30, batch_size=1000, validation_split=0.17, shuffle=True)

_, result = model.evaluate(x_test, y_test)

print( "the accuracy of the model using test data is: " + str(result) )

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy with regularization')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss with regularization')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

image_index = 4443
plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
plt.show()
pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
print(pred.argmax())