import csv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.preprocessing.image import ImageDataGenerator
import keras.preprocessing.image as image
import numpy as np
train = []
XSize = 28 * 28

print("Getting training data...\n")
# Get training data
with open("28x28Info/sign_mnist_train/sign_mnist_train.csv") as File:
    reader = csv.DictReader(File)
    for row in reader:
        train.append(row)


Y = []
X = []
print("Formatting data...\n")
for currTrain in train:
    Xtemp = []
    Ytemp = [0 for i in range(0,26)]
    Ytemp[int(currTrain["label"])] = 1;
    # Format training data
    for i in range(0, 28):
        Xtemp2 = []
        for j in range(1, 29):
            Xtemp2.append([int(currTrain["pixel" + str(28 * i + j)])/255])
        Xtemp.append(Xtemp2)
    X.append(Xtemp)
    Y.append(Ytemp)
X = np.array(X)
Y = np.array(Y)


datagen = ImageDataGenerator(width_shift_range=[-5,5])


# from matplotlib import pyplot as plt
# plt.imshow(np.reshape(X[201], (28,28)), cmap='gray')
# plt.show()
#




print("Creating model...\n")
model = Sequential()
model.add(Conv2D(16, (3,3), input_dim=3, input_shape=(28,28,1), activation='relu')) # input layer, 28x28 nodes for each pixel
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Dense(150, activation='relu')) # hidden layer 1, to detect certain pixel patterns
model.add(Flatten())
model.add(Dense(70, activation='relu')) # hidden layer 2, to detect larger pixel patterns
model.add(Dense(26, activation='sigmoid')) # output layer, answer

# categorical crossentropy since we have 26 classes we'd like to test
# adam optimizer
# want to see the accuracy, so we set accuracy metric
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


print("Training model...\n")

# model.fit(X,Y, epochs=30, batch_size=100)
model.fit(datagen.flow(X, Y, batch_size=100), epochs=10)


print("Saving model...\n")
model.save('handModel')


# (150, 50) = best!
