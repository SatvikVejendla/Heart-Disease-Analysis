from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import tensorflow as tf

from constants import SEED


np.random.seed(SEED)
tf.random.set_seed(SEED)

df = pd.read_csv("data/cleaned.csv")


ys = df["output"].values
xs = df.drop(["output"], axis=1).values


xs = xs.astype("float32")
ys = ys.astype("float32")


sm = SMOTE()
xs, ys = sm.fit_resample(xs, ys)


scaler = StandardScaler()

xs = scaler.fit_transform(xs)


with open("data/final/xs.npy", "wb") as handler:
    np.save(handler, np.array(xs).astype(np.float32))
    

with open("data/final/ys.npy", "wb") as handler:
    np.save(handler, np.array(ys).astype(np.float32))