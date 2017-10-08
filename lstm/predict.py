import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

from tensorflow.contrib import learn
from sklearn.metrics import mean_squared_error

LOG_DIR = 'resources/logs/'
TIMESTEPS = 1
RNN_LAYERS = [{'num_units': 4}]
DENSE_LAYERS = None
TRAINING_STEPS = 100
PRINT_STEPS = TRAINING_STEPS / 10
BATCH_SIZE = 100

regressor = learn.SKCompat(learn.Estimator(model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS),
                            model_dir=LOG_DIR))

validation_monitor = learn.monitors.ValidationMonitor(X_valed, y_valed, every_n_steps=PRINT_STEPS,
eval_steps=1,
early_stopping_rounds=1000)

regressor.fit(X_trained, y_trained,
              monitors=[validation_monitor],
              batch_size=BATCH_SIZE,
              steps=TRAINING_STEPS)

predicted = regressor.predict(X_tested,as_iterable=False)
score = mean_squared_error(predicted, y_test)
print("MSE: %f" % score)

plot_predicted, = plt.plot(predicted, label='predicted')
plot_test, = plt.plot(y_test, label='test')
plt.legend(handles=[plot_predicted, plot_test])
plt.show()

Windows = [11, 18, 30, 48, 78, 126, 203, 329]

print("Start computing...")

n = train.shape[1] - 1 #  550
Visits = np.zeros(train.shape[0])
for i, row in train.iterrows():
    M = []
    start = row[1:].nonzero()[0]
    if len(start) == 0:
        continue
    if n - start[0] < Windows[0]:
        Visits[i] = row.iloc[start[0]+1:].median()
        continue
    for W in Windows:
        if W > n-start[0]:
            break
        M.append(row.iloc[-W:].median())
    Visits[i] = np.median(M)

Visits[np.where(Visits < 1)] = 0.
train['Visits'] = Visits

print("Printing results...")
test = pd.read_csv("key_1.csv")
test['Page'] = test.Page.apply(lambda x: x[:-11])

test = test.merge(train[['Page','Visits']], on='Page', how='left')
test[['Id','Visits']].to_csv('bfill_0830.csv', index=False)