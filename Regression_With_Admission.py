import app
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow	import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import r2_score

df = pd.read_csv('admissions_data.csv')
#print(df.head())
#print(df.columns)
#print(df.describe())
#print(df.dtypes)

labels = df.iloc[:, -1]
#print(labels)
features = df.iloc[:, 0:-1]
#print(features)
#print(df['University Rating'])
features.drop(['University Rating', 'Serial No.'], axis=1, inplace=True)
#print(features.columns)

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.transform(features_test)

model = Sequential()
input = layers.InputLayer(input_shape=(features.shape[1],))
model.add(input)
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1))

#print(model.summary())

opt = Adam(learning_rate=0.05)
model.compile(loss='mse', metrics=['mae'], optimizer=opt)

stop=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=40)


history = model.fit(features_train_scaled, labels_train, epochs=500, batch_size=30, verbose=1, validation_split=0.2, callbacks=[stop])

res_mse, res_mae = model.evaluate(features_test_scaled, labels_test)
print(res_mse, res_mae)

y_pred = model.predict(features_test_scaled)
print('R^2: ', r2_score(labels_test, y_pred))


fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['mae'])
ax1.plot(history.history['val_mae'])
ax1.set_title('model mae')
ax1.set_ylabel('MAE')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'validation'], loc='upper left')

ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('model loss')
ax2.set_ylabel('loss')
ax2.set_xlabel('epoch')
ax2.legend(['train', 'validation'], loc='upper left')
 
fig.tight_layout()
fig.savefig('static/images/my_plots.png')