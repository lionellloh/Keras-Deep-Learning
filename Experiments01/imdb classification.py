from keras.datasets import imdb

# Take the top 10000 most used words. If a word is not inside the toip 10,000 words then it will not
# be mapped properly(?)
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print("shape of train data: {} \n shape of train_labels: {}".format(train_data.shape, train_labels.shape))



# print(train_data)

# [list([1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941, 4, 173, 36, 256, 5, 25, 100, 43, 838, 112, 50, 670, 2, 9, 35, 480, 284, 5, 150, 4, 172, 112, 167, 2, 336, 385, 39, 4, 172, 4536, 1111, 17, 546, 38, 13, 447, 4, 192, 50, 16, 6, 147, 2025, 19, 14, 22, 4, 1920, 4613, 469, 4, 22, 71, 87, 12, 16, 43, 530, 38, 76, 15, 13, 1247, 4, 22, 17, 515, 17, 12, 16, 626, 18, 2, 5, 62, 386, 12, 8, 316, 8, 106, 5, 4, 2223, 5244, 16, 480, 66, 3785, 33, 4, 130, 12, 16, 38, 619, 5, 25, 124, 51, 36, 135, 48, 25, 1415, 33, 6, 22, 12, 215, 28, 77, 52, 5, 14, 407, 16, 82, 2, 8, 4, 107, 117, 5952, 15, 256, 4, 2, 7, 3766, 5, 723, 36, 71, 43, 530, 476, 26, 400, 317, 46, 7, 4, 2, 1029, 13, 104, 88, 4, 381, 15, 297, 98, 32, 2071, 56, 26, 141, 6, 194, 7486, 18, 4, 226, 22, 21, 134, 476, 26, 480, 5, 144, 30, 5535, 18, 51, 36, 28, 224, 92, 25, 104, 4, 226, 65, 16, 38, 1334, 88, 12, 16, 283, 5, 16, 4472, 113, 103, 32, 15, 16, 5345, 19, 178, 32])
#  list([1, 194, 1153, 194, 8255, 78, 228, 5, 6, 1463, 4369, 5012, 134, 26, 4, 715, 8, 118, 1634, 14, 394, 20, 13, 119, 954, 189, 102, 5, 207, 110, 3103, 21, 14, 69, 188, 8, 30, 23, 7, 4, 249, 126, 93, 4, 114, 9, 2300, 1523, 5, 647, 4, 116, 9, 35, 8163, 4, 229, 9, 340, 1322, 4, 118, 9, 4, 130, 4901, 19, 4, 1002, 5, 89, 29, 952, 46, 37, 4, 455, 9, 45, 43, 38, 1543, 1905, 398, 4, 1649, 26, 6853, 5, 163, 11, 3215, 2, 4, 1153, 9, 194, 775, 7, 8255, 2, 349, 2637, 148, 605, 2, 8003, 15, 123, 125, 68, 2, 6853, 15, 349, 165, 4362, 98, 5, 4, 228, 9, 43, 2, 1157, 15, 299, 120, 5, 120, 174, 11, 220, 175, 136, 50, 9, 4373, 228, 8255, 5, 2, 656, 245, 2350, 5, 4, 9837, 131, 152, 491, 18, 2, 32, 7464, 1212, 14, 9, 6, 371, 78, 22, 625, 64, 1382, 9, 8, 168, 145, 23, 4, 1690, 15, 16, 4, 1355, 5, 28, 6, 52, 154, 462, 33, 89, 78, 285, 16, 145, 95])
#  list([1, 14, 47, 8, 30, 31, 7, 4, 249, 108, 7, 4, 5974, 54, 61, 369, 13, 71, 149, 14, 22, 112, 4, 2401, 311, 12, 16, 3711, 33, 75, 43, 1829, 296, 4, 86, 320, 35, 534, 19, 263, 4821, 1301, 4, 1873, 33, 89, 78, 12, 66, 16, 4, 360, 7, 4, 58, 316, 334, 11, 4, 1716, 43, 645, 662, 8, 257, 85, 1200, 42, 1228, 2578, 83, 68, 3912, 15, 36, 165, 1539, 278, 36, 69, 2, 780, 8, 106, 14, 6905, 1338, 18, 6, 22, 12, 215, 28, 610, 40, 6, 87, 326, 23, 2300, 21, 23, 22, 12, 272, 40, 57, 31, 11, 4, 22, 47, 6, 2307, 51, 9, 170, 23, 595, 116, 595, 1352, 13, 191, 79, 638, 89, 2, 14, 9, 8, 106, 607, 624, 35, 534, 6, 227, 7, 129, 113])
#  ...
# train_data looks like this... I am not sure why it has the list data type specified in front

# print(train_data[0])

# Trying to convert the integer list into a words.

word_index = imdb.get_word_index()
# print("word_index{}".format(word_index))

reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])
# print(reverse_word_index)

"""train_data[0] is a list of integers where each integer represents a word. 
this list of encoded words are from the first review"""
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
# print(decoded_review)

# one-hot encode the integer sequences into a binary matrix
import numpy as np

"""The arguments below are rather tricky. Sequences represent a 1d numpy array that contains 25,0000 lists.
Each list has a certain number of elements that correspond to the number of words. The reason why it is not 2D
is because the number of words for each review is inconsistent. Hence, it cannot be formatted as a matrix.

To prove this: let's do a little experiment to check out the number of elements in some of the 25,000 reviews.

If you run the print statement, it will show:

Number of words in the 1st review: 218, 2nd review: 189 and 3rd review: 141
"""

print("Number of words in the 1st review: {}, 2nd review: {} and 3rd review: {}".format(len(train_data[0]), len(train_data[1]), len(train_data[2])))


def vectorize_sequences(sequences, dimensions = 10000):
    """Create a matrix filled with zeros of a shape (m, n) where m is the number of reviews and n is the
    index of the 10,000 most frequently used words. I believe that the number of non-zero elements
    in nth row represents the number of unique words that are in the top 10,000 word in the nth review"""
    results = np.zeros((len(sequences), dimensions))
    for i, sequence in enumerate(sequences):
        results[i][sequence] = 1.

    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

print("train data's shape: {} and test data's shape: {}".format(x_train.shape, x_test.shape))
# train data's shape: (25000, 10000) and test data's shape: (25000, 10000)

# Follow up by vectorizing your labels


y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


print(y_train.shape)

from keras import models
from keras import layers

model = models.Sequential()
# First layer is the input layer with the input_shape specified
model.add(layers.Dense(16, activation="relu", input_shape =(10000,)))
# Hidden Layer starts
model.add(layers.Dense(16, activation="relu"))


# outputs the probability which reflects the sentiment
model.add(layers.Dense(1, activation = "sigmoid"))

model.summary()

model.compile(loss = "binary_crossentropy", optimizer = "rmsprop", metrics=["accuracy"])

# The following code allows you to configure the arguments of the optimizer function to
# a greater resolution

from keras import optimizers

model.compile(loss = "binary_crossentropy", optimizer=optimizers.rmsprop(lr = 0.001),
              metrics = ["accuracy"])

"""Splitting your data. Note that you should never train your model against your test set. 
You should keep optimizing it with the test set. Test data is purely for checking!
 If you were to ever cheat and optimize so that your model performs well on the test set, 
 it might fail on the real world unseen data."""

x_val = x_train[:10000]
partial_x = x_train[10000:]
y_val = y_train[:10000]
partial_y = y_train[10000:]

print(partial_x)

# The batch_size parameter is the number of samples for each epoch of training
model.compile(loss = "binary_crossentropy", optimizer = "rmsprop", metrics = ["accuracy"])
history = model.fit(partial_x,
                    partial_y,
                    epochs = 20,
                    batch_size= 512,
                    validation_data=(x_val, y_val))

# History is the object that is generated
history_dict = history.history

# print(history.history['acc']z)
