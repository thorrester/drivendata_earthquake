import numpy as np
import pandas as pd
import tensorflow as tf
import utils as utils
from sklearn.model_selection import train_test_split
from model import NetTraining

# Read data
df = pd.read_csv("./train_values.csv", nrows=10000)
labels = pd.read_csv("./train_labels.csv", nrows=10000)
df.pop("building_id")

# Convert to sentences
df["sentences"] = df.apply(utils.convert_to_sentence, columns=df.columns, axis=1)

# Create X and y
X = df.pop("sentences")
y = labels.pop("damage_grade")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# Fit tokenizer
tokenizer = utils.WordpieceTokenizer()
tokenizer.fit(X_train)

X_train_tran = tokenizer.transform(X_train)
X_test_tran = tokenizer.transform(X_test)

train_data = (X_train_tran, y_train)
test_data = (X_test_tran, y_test)

# Train model
model = NetTraining(
    train_data=train_data,
    test_data=test_data,
    epochs=20,
    batch_size=32,
    max_tokens=20_000,
    maxlen=tokenizer.vocab_size,
    embed_dim=32,
    num_class=3,
)

if __name__ == "__main__":
    print(X_train_tran)
