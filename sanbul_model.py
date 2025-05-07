import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
tf.random.set_seed(42)

df = pd.read_csv('sanbul2district-divby100.csv')

X = df.drop(['burned_area', 'month', 'day'], axis=1)
y = df['burned_area']

scaler = StandardScaler()
fires_prepared = scaler.fit_transform(X)
fires_labels = y.values

X_train, X_valid, y_train, y_valid = train_test_split(
    fires_prepared, fires_labels, test_size=0.2, random_state=42)

X_test, y_test = X_valid, y_valid

model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(loss='mean_squared_error', optimizer=keras.optimizers.SGD(learning_rate=1e-3))
model.fit(X_train, y_train, epochs=200, validation_data=(X_valid, y_valid))

model.save('fires_model.keras')

X_new = X_test[:3]
print(np.round(model.predict(X_new), 2))
