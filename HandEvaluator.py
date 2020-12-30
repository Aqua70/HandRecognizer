import os
import csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras.models
import numpy as np
test = []
XSize = 28 * 28
model = keras.models.load_model('handModel')

print("Getting test data...\n")
with open("28x28Info/sign_mnist_test/sign_mnist_test.csv") as file:
    reader = csv.DictReader(file)
    for row in reader:
        test.append(row)

Y = []
X = []
ans = []
print("Formatting data...\n")
for currTest in test:
    Xtemp = []
    Ytemp = [0 for i in range(0,26)]
    Ytemp[int(currTest["label"])] = 1;
    # Format training data
    for i in range(1, XSize + 1):
        Xtemp.append(int(currTest["pixel" + str(i)])/255)
    X.append(Xtemp)
    Y.append(Ytemp)
    ans.append(int(currTest["label"]))

print("Predicting...\n")
predictions = np.argmax(model.predict(X), axis=-1)
# summarize the first 5 cases
totalPredictions = len(X)
totalCorrect = 0
for i in range(len(predictions)):
    if predictions[i] == ans[i]:
        totalCorrect += 1

print("%d%% rate of correctness" % (totalCorrect/totalPredictions * 100))
