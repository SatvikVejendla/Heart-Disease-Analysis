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

import pandas as pd
import numpy as np
from typing import Final
import json



np.random.seed(SEED)
tf.random.set_seed(SEED)

df = pd.read_csv("data/heart.csv")

ys = df["output"].values
xs = df.drop(["output"], axis=1).values


xs = xs.astype("float32")
ys = ys.astype("float32")


xs, test_xs, ys, test_ys = train_test_split(xs,ys, test_size=0.3, shuffle=True)


model = Sequential()

model.add(Dense(256, activation="relu", input_dim=13))
model.add(Dense(64, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.summary()

model.compile(optimizer=Adam(lr=0.001), loss="binary_crossentropy", metrics=["binary_accuracy"])
history = model.fit(xs, ys, epochs=50, batch_size=30, verbose=1).history

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

with open("model/optimized/confusion_matrix.npy", "wb") as handler:
    np.save(handler, np.array(conf_matrix).astype(int))

model.save("model/uncleaned/model.h5")
pd.DataFrame.from_dict(history).to_csv("model/uncleaned/history.csv", index=False)


stats = {
    "accuracy": results[1],
    "loss": results[0]
}
with open("model/uncleaned/results.json", "w") as handler:
    json.dump(stats, handler)