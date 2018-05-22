from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images[0])
# print(mnist.load_data())


# The Network Architecture

from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation="relu", input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation="softmax"))

# The Compilation step

# print(train_images.shape)
# (60000, 28, 28) => 3D
network.compile(optimizer="rmsprop", loss='categorical_crossentropy', metrics=["accuracy"])
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32')/255
# print(train_images.shape)
#  (60000, 784) => 2D
print(train_images.ndim)
# Preparing the image data - reshape it from 2D into 1DD - something the network expects
test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32')/255

# Preparing the labels

from keras.utils import to_categorical

# print("pre-categorical", train_labels)

# This cause the 1D numpy array of length X that contains N number of categories to transform into a 2D numpy array of X x N size

train_labels = to_categorical(train_labels)

# print("post-categorical", train_labels.shape)

test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs = 5, batch_size= 128)

# Now we apply the model on the test set

test_loss, test_acc = network.evaluate(test_images, test_labels)
print("test acc: {}, test _loss: {}".format(test_acc, test_loss))



