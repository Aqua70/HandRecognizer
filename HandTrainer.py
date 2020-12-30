import csv
from keras.models import Sequential
from keras.layers import Dense
import tf-nightly

train = []
XSize = 28 * 28
YSize = 1

model = Sequential()
model.add(Dense(XSize,input_dim=XSize, activation='relu'))
model.add(Dense(XSize/4, activation='relu'))
model.add(Dense(XSize/8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Get training data
with open("28x28Info/sign_mnist_test/sign_mnist_test.csv") as File:
    reader = csv.DictReader(File)
    for row in reader:
        train.append(row)


for currTrain in train:
    Y = int(currTrain["label"])
    X = []
    # Format training data
    for i in range(1, XSize + 1):
        X.append(int(currTrain["pixel" + str(i)])/255)


