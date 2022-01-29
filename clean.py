import pandas as pd
import numpy as np
import tensorflow as tf


from constants import SEED

np.random.seed(SEED)
tf.random.set_seed(SEED)

df = pd.read_csv("data/heart.csv")


df = df.drop(["fbs"], axis=1)
df.drop_duplicates()


def remove_outlier(col):
    global df
    values = df[col]

    q25 = np.percentile(values, 25)
    q75 = np.percentile(values, 75)
    iqr = q75 - q25
    delta_iqr = 1.5 * iqr

    lower_bound = q25 - delta_iqr
    upper_bound = q75 + delta_iqr
    for ind, val in df[col].items():
        if(val < lower_bound or val > upper_bound):
            df = df.drop(ind)


continuous_attrs = ["age", "chol", "thalachh", "trtbps"]

for i in continuous_attrs:
    remove_outlier(i)

df.to_csv("data/cleaned.csv", index=False)
