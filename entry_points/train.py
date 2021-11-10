import numpy as np
import pandas as pd
import tensorflow as tf
import utils as utils

df = pd.read_csv('./train_values.csv', nrows=10000)
df.pop('building_id')
df['sentences'] = df.apply(utils.convert_to_sentence, columns= df.columns, axis=1)

if __name__=="__main__":
    print(df.head())