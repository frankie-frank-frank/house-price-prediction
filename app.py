import pandas as pd
import seaborn as sns
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Normalization, Dense, InputLayer
from tensorflow.keras.losses import MeanSquaredError, Huber, MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
import matplotlib.pyplot as plt

data = pd.read_csv("california_housing_test.csv", ",")
data.head()

data.shape

sns.pairplot(data[["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]], diag_kind="kde")

tensor_data = tf.constant(data)
tensor_data = tf.cast(tensor_data, tf.float32)
print(tensor_data.shape)

X = tensor_data[:, :-1]
y = tensor_data[:, -1]
y = tf.expand_dims(y, axis = -1)
print(X[:5])
print(y[:5])

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
DATASET_SIZE = len(X)

X_train = X[:int(DATASET_SIZE * TRAIN_RATIO)]
y_train = y[:int(DATASET_SIZE * TRAIN_RATIO)]

X_val = X[int(DATASET_SIZE * TRAIN_RATIO):int(DATASET_SIZE * (TRAIN_RATIO + VAL_RATIO))]
y_val = y[int(DATASET_SIZE * TRAIN_RATIO):int(DATASET_SIZE * (TRAIN_RATIO + VAL_RATIO))]

X_test = X[int(DATASET_SIZE * (TRAIN_RATIO + VAL_RATIO)):]
y_test = y[int(DATASET_SIZE * (TRAIN_RATIO + VAL_RATIO)):]

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size = 30, reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_dataset = val_dataset.shuffle(buffer_size = 30, reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = test_dataset.shuffle(buffer_size = 30, reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)

for x,y in train_dataset:
  print(x, y)
  break;

normalizer = Normalization()
normalizer.adapt(X_train)
normalizer(X_train)[:5]

model = tf.keras.Sequential([InputLayer(input_shape = (8,))])
model.add(normalizer)
model.add(Dense(128, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(1))
model.summary()

tf.keras.utils.plot_model(model, to_file = "model.png", show_shapes=True)

# model.compile(loss = Huber(delta = 0.2))
model.compile(optimizer = Adam(learning_rate=0.1),loss = MeanAbsoluteError(), metrics = RootMeanSquaredError())

history = model.fit(train_dataset, validation_data=val_dataset, epochs = 100, verbose = 1)

history.history

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train', 'val'])
plt.show()

plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('model performance')
plt.ylabel('rmse')
plt.xlabel('epochs')
plt.legend(['train', 'val'])
plt.show()

model.evaluate(X_test, y_test)

model.predict(tf.expand_dims(X_test[0], axis = 0))

y_test[0]

y_true = list(y_test[:,0].numpy())

y_pred = list(model.predict(X_test)[:,0])
print(len(y_pred))

ind = np.arange(300)
plt.figure(figsize=(40, 20))

width = 0.1

plt.bar(ind, y_pred, width, label="Predicted House Price")
plt.bar(ind + width, y_true, width, label="Actual House Price")

plt.xlabel("Actual vs Predicted House Price")
plt.ylabel("House prices")

