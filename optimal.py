import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold 

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from matplotlib import pyplot as plt
import seaborn as sn

from constants import SEED
from constants import K_SPLITS

import pandas as pd
import numpy as np
from typing import Final
import json


np.random.seed(SEED)
tf.random.set_seed(SEED)

xs = np.load("data/final/xs.npy")
ys = np.load("data/final/ys.npy")

xs, test_xs, ys, test_ys = train_test_split(xs,ys, test_size=0.3, shuffle=True)



model = Sequential()

model.add(Dense(12, activation="relu", input_dim=12))
model.add(Dense(64, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.summary()

model.compile(optimizer=Adam(lr=0.001), loss="binary_crossentropy", metrics=["binary_accuracy"])

print("\n\n-----------------------------------------\n")
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(xs, ys)
print("Random Forest Classifier:", clf.score(test_xs, test_ys))


lgr = LogisticRegression(random_state=0)
lgr.fit(xs, ys)
print("Logistic Regression:", lgr.score(test_xs, test_ys))



dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(xs, ys)
print("Decision Tree Classifier:", dtc.score(test_xs, test_ys))

k_splits=K_SPLITS
kf = KFold(n_splits=k_splits, random_state=None)
acc_score = []
history = {"loss": [], "accuracy":[]}

for train_ind, test_ind in kf.split(xs):
    train_xs, val_xs = xs[train_ind], xs[test_ind]
    train_ys, val_ys = ys[train_ind], ys[test_ind]

    hist = model.fit(train_xs, train_ys, epochs=15, batch_size=32, verbose=0).history
    predictions = model.predict(val_xs)

    predictions = [[round(x) for x in i] for i in predictions]
    
    acc = accuracy_score(predictions, val_ys)
    acc_score.append(acc)

    loss = hist["loss"]
    accuracy = hist["binary_accuracy"]
    history["loss"] += loss
    history["accuracy"] += accuracy

avg_acc_score = sum(acc_score)/k_splits
 
print('\n-----------------------------------------\nKeras Model K-Fold Accuracies - {}'.format(acc_score))
print('Keras Model Average Accuracy : {}'.format(avg_acc_score))

results = model.evaluate(test_xs, test_ys)
print("Keras Model Test Loss:", results[0])
print("Keras Model Test Accuracy:", results[1])

predictions = model.predict(test_xs)
predictions = [[round(x) for x in i] for i in predictions]
conf_matrix = confusion_matrix(test_ys, predictions)
confusion_df = pd.DataFrame(conf_matrix)
plt.figure(figsize=(10,7))
sn.heatmap(confusion_df, annot=True)
plt.show()


model.save("model/optimal/model.h5")

with open("model/optimal/confusion_matrix.npy", "wb") as handler:
    np.save(handler, np.array(conf_matrix).astype(int))

pd.DataFrame.from_dict(history).to_csv("model/optimal/history.csv", index=False)

stats = {
    "accuracy": results[1],
    "loss": results[0]
}
with open("model/optimal/results.json", "w") as handler:
    json.dump(stats, handler)