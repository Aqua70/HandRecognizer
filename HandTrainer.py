import csv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
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
    for i in range(1, XSize+1):
        Xtemp.append(int(currTrain["pixel" + str(i)])/255)
    X.append(Xtemp)
    Y.append(Ytemp)

X = np.array(X)
Y = np.array(Y)
print("Creating model...\n")
model = Sequential()
model.add(Dense(XSize, input_dim=XSize, activation='relu')) # input layer, 28x28 nodes for each pixel
model.add(Dense(150, activation='relu')) # hidden layer 1, to detect certain pixel patterns
model.add(Dense(70, activation='relu')) # hidden layer 2, to detect larger pixel patterns
model.add(Dense(26, activation='sigmoid')) # output layer, answer

# categorical crossentropy since we have 26 classes we'd like to test
# adam optimizer
# want to see the accuracy, so we set accuracy metric
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


print("Training model...\n")
model.fit(X, Y, epochs=30, batch_size=100)


print("Saving model...\n")
model.save('handModel')


# (150, 50) = best!
